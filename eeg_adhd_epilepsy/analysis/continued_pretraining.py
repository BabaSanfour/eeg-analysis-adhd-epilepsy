#!/usr/bin/env python3
"""Continued self-supervised pretraining per fold, scored at the child level.

For one outer fold of a cohort's children, a foundation model goes on with its
self-supervised pretraining (coco-pipe ``continue_pretraining``) on every saved window of
the database (EO and EC, every recording, labels unused) except the windows of that fold's
test children, so no test child is seen in any form. The frozen embeddings of the cohort's
windows from the continued weights and from the original weights are then each decoded at
the child level (mean embedding in EO, in EC and EC - EO, fold-local PCA, logistic
regression; :func:`child_level.linear_scores`), trained on the fold's training children
and scored on its test children. ``--combine`` gathers the folds' scores and compares the
two with :func:`ensemble.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from eeg_adhd_epilepsy.analysis.child_level import CONDITIONS, child_blocks, linear_scores
from eeg_adhd_epilepsy.analysis.ensemble import evaluate

logger = logging.getLogger(__name__)

BATCH_SIZE = {"cbramod": 256, "luna": 256, "reve": 64}
# The saved epochs hold volts; the models expect microvolts, as the cohort is loaded
# (``units: uV``) and the embeddings were extracted.
MICROVOLTS = 1e6


def child_folds(labels: pd.Series, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds of the (sorted) children, the same for every model and job."""
    labels = labels.sort_index()
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=0)
    return list(splitter.split(np.zeros(len(labels)), labels))


def load_cohort(args):
    """The cohort's windows (EO and EC), their child, subject and condition, and labels."""
    from eeg_adhd_epilepsy.analysis.dataset import build_dataset
    from eeg_adhd_epilepsy.analysis.utils.decoding import (
        build_loader_args,
        prepare_decoding_scope,
        resolve_decoding_paths,
    )
    from eeg_adhd_epilepsy.utils.config import resolve_cli_config

    config = resolve_cli_config(
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        matched=args.matched,
        bids_root=args.bids_root,
        desc=args.desc,
        metadata=args.metadata,
        derivative_root=str(args.output_dir),
    )
    metadata = resolve_decoding_paths(config, input_mode="foundation")[3]
    loader_args = build_loader_args(
        config, input_mode="raw", layout_mode="sensor", segment_duration=10.0, overlap=0.0
    )
    loader_args.representation = "epoch"
    if loader_args.units != "uV":
        raise ValueError(f"The cohort must be loaded in microvolts, not {loader_args.units}.")
    eval_spec = next(e for e in config["evals"] if e["name"] == args.target)
    parts, labels = [], {}
    for condition in CONDITIONS:
        container = build_dataset(loader_args, metadata, condition, target_col=None)
        selected, y, _, meta, _ = prepare_decoding_scope(
            container,
            eval_spec,
            scope=condition,
            group_col=eval_spec.get("group_col", "patient_group_id"),
            session_col=config.get("session_col", "session"),
            subject_col=config["subject_col"],
            requested_splits=5,
        )
        child = meta["group_id"].astype(str).to_numpy()
        # binary targets are encoded with the positive class as 1
        label = (np.asarray(y) == 1).astype(int)
        labels.update(zip(child, label))
        subject = meta[config["subject_col"]].astype(str).to_numpy()
        parts.append(
            (
                np.asarray(selected.X, dtype=np.float32),
                child,
                subject,
                np.full(len(child), condition),
            )
        )
        channels = [str(c) for c in selected.coords["channel"]]
        sfreq = float(selected.meta["sfreq"])
    X, child, subject, condition = (np.concatenate(p) for p in zip(*parts))
    return X, child, subject, condition, pd.Series(labels), channels, sfreq, config


def load_pool(config, exclude_subjects: set[str], exclude_children: set[str], channels: list[str]):
    """Every saved window (EO and EC) except the excluded subjects' and children's (any
    recording of a test child, in the cohort or not), with its subject."""
    from eeg_adhd_epilepsy.analysis.dataset import (
        consolidated_epochs_path,
        read_consolidated_epochs,
    )

    X, subjects = [], []
    for condition in CONDITIONS:
        path = consolidated_epochs_path(
            Path(config["bids_root"]),
            config.get("desc", "base"),
            condition,
            config.get("task", "clinical"),
        )
        container = read_consolidated_epochs(path)
        subject = np.asarray(container.coords["subject"]).astype(str)
        keep = ~np.isin(subject, sorted(exclude_subjects))
        if "patient_group_id" in container.coords:
            group = np.asarray(container.coords["patient_group_id"]).astype(str)
            keep &= ~np.isin(group, sorted(exclude_children))
        order = [list(map(str, container.coords["channel"])).index(c) for c in channels]
        X.append(np.asarray(container.X, dtype=np.float32)[keep][:, order] * MICROVOLTS)
        subjects.append(subject[keep])
    return np.concatenate(X), np.concatenate(subjects)


def embed(model: str, X, channels, sfreq, checkpoint_dir=None, device="cuda") -> np.ndarray:
    from coco_pipe.decoding.foundation_models import FrozenBackboneTransformer

    backend_kwargs = {"checkpoint_dir": str(checkpoint_dir)} if checkpoint_dir else None
    frozen = FrozenBackboneTransformer(
        model, sfreq=sfreq, ch_names=channels, device=device, backend_kwargs=backend_kwargs
    )
    return np.asarray(frozen.fit(X).transform(X), dtype=np.float32).reshape(len(X), -1)


def run_fold(args) -> Path:
    from coco_pipe.decoding.foundation_models import continue_pretraining

    X, child, subject, condition, labels, channels, sfreq, config = load_cohort(args)
    # children with both conditions, as in child_level
    both = pd.Series(condition).groupby(child).nunique()
    labels = labels.loc[both.index[both == 2]].sort_index()
    train, test = child_folds(labels, args.n_folds)[args.fold]
    test_children = set(labels.index[test])
    excluded = set(subject[np.isin(child, sorted(test_children))])
    out = args.output_dir / args.model / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    pool, pool_subject = load_pool(config, excluded, test_children, channels)
    logger.info(
        "fold %d: %d pool windows (%d subjects excluded)", args.fold, len(pool), len(excluded)
    )
    continue_pretraining(
        args.model,
        pool,
        channels,
        sfreq,
        output_dir=out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=BATCH_SIZE.get(args.model, 64),
        groups=pool_subject,
        device="auto",
        seed=args.fold,
    )
    del pool

    scores = pd.DataFrame({"y": labels.iloc[test]})
    for name, checkpoint in (("original", None), ("continued", out)):
        Z = embed(args.model, X, channels, sfreq, checkpoint)
        blocks = child_blocks(Z, child, condition, args.model)
        scores[name] = linear_scores(blocks, labels, [(train, test)]).iloc[test]
    scores.to_csv(out / "scores.csv", index_label="child")
    return out


def combine(args) -> dict:
    """Out-of-fold scores of all folds, original vs continued weights, per model."""
    summary = {}
    for model_dir in sorted(p for p in args.output_dir.iterdir() if p.is_dir()):
        files = sorted(model_dir.glob("fold*/scores.csv"))
        if len(files) < args.n_folds:
            logger.warning("%s: %d of %d folds", model_dir.name, len(files), args.n_folds)
        frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        for name in ("original", "continued"):
            children = frame.rename(columns={name: "score"})[["child", "y", "score"]]
            summary[f"{model_dir.name}_{name}"] = {
                **evaluate(children, n_permutations=args.n_permutations),
                "n_folds": len(files),
            }
    text = json.dumps(summary, indent=2)
    (args.output_dir / "summary.json").write_text(text + "\n")
    print(text)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--combine", action="store_true", help="Score the saved folds.")
    parser.add_argument("--cohort_config")
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--desc", default="base", choices=["base", "baserejected"])
    parser.add_argument("--target", default="med_adhd_vs_ctrl")
    parser.add_argument("--model", choices=sorted(BATCH_SIZE))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bids_root", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--analysis_config", default="configs/analyses/decoding/foundation_explore.yaml"
    )
    parser.add_argument("--n_permutations", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.combine:
        combine(args)
        return
    if args.cohort_config is None or args.model is None or args.fold is None:
        parser.error("--cohort_config, --model and --fold are needed to run a fold.")
    print(run_fold(args))


if __name__ == "__main__":
    main()

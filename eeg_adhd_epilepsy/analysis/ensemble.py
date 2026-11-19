#!/usr/bin/env python3
"""Child-level ensemble of saved decoding fits.

Every fit of a decoding run saves out-of-fold predictions: each child is predicted by a
model that never trained on it. This averages those predictions per child over fits
(models, train modes, conditions), with a fixed rule and no selection, so the ensemble is
out-of-fold too:

1. per fit, a child's score is its mean positive-class probability over its rows (epochs or
   recordings), and the scores are ranked (0-1) so fits with different calibration weigh
   the same;
2. a child's ensemble score is the mean of its ranks over the fits that predicted it.

It is scored on children, the unit of a diagnosis: the AUC (threshold-free) with a label
permutation p-value and a bootstrap interval, and the balanced accuracy of a split at the
median score. Fits are found as ``<run>/artifacts/fits/<fit>/result.joblib`` under a
cohort's decoding folder; ``--include`` / ``--exclude`` filter their paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def child_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per child of a fit's predictions: its label and mean positive-class probability."""
    proba = sorted(column for column in predictions if column.startswith("y_proba_"))
    if not proba:
        raise ValueError("The predictions hold no class probabilities (y_proba_*).")
    group = "group_id" if "group_id" in predictions else "Group"
    children = predictions.groupby(group).agg(y=("y_true", "first"), score=(proba[-1], "mean"))
    return children.rename_axis("child").reset_index()


def ensemble(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean rank per child over ``fits`` (name -> :func:`child_scores` frame)."""
    ranked = []
    for name, children in fits.items():
        ranked.append(children.assign(rank=children["score"].rank(pct=True), fit=name))
    stacked = pd.concat(ranked, ignore_index=True)
    labels = stacked.groupby("child")["y"].nunique()
    if (labels > 1).any():
        raise ValueError(f"Children with two labels across fits: {list(labels[labels > 1].index)}")
    return (
        stacked.groupby("child")
        .agg(y=("y", "first"), score=("rank", "mean"), n_fits=("fit", "nunique"))
        .reset_index()
    )


def evaluate(
    children: pd.DataFrame, n_permutations: int = 2000, n_bootstrap: int = 1000, seed: int = 0
) -> dict[str, float]:
    """AUC (permutation p, bootstrap 95 % interval) and median-split balanced accuracy."""
    rng = np.random.default_rng(seed)
    y, score = children["y"].to_numpy(), children["score"].to_numpy()
    auc = roc_auc_score(y, score)
    null = np.array([roc_auc_score(rng.permutation(y), score) for _ in range(n_permutations)])
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], score[idx]))
    return {
        "n_children": int(len(y)),
        "positive_share": float(y.mean()),
        "auc": float(auc),
        "auc_ci_low": float(np.percentile(boots, 2.5)),
        "auc_ci_high": float(np.percentile(boots, 97.5)),
        "auc_p": float((1 + (null >= auc).sum()) / (n_permutations + 1)),
        "balanced_accuracy_median_split": float(
            balanced_accuracy_score(y, (score > np.median(score)).astype(int))
        ),
    }


def load_fits(
    cohort_root: Path, target: str, include: str | None = None, exclude: str | None = None
) -> dict[str, pd.DataFrame]:
    """:func:`child_scores` of every saved fit of ``target`` under ``cohort_root``."""
    from coco_pipe.decoding import ExperimentResult

    fits = {}
    for path in sorted(Path(cohort_root).glob("*/artifacts/fits/*/result.joblib")):
        name = str(path.parent.relative_to(cohort_root))
        if f"_{target}_" not in path.parent.name:
            continue
        if (include and not re.search(include, name)) or (exclude and re.search(exclude, name)):
            continue
        fits[name] = child_scores(ExperimentResult.load(path).get_predictions())
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cohort_root",
        required=True,
        type=Path,
        help="A cohort's decoding folder, e.g. results/decoding_base/<dataset_name>.",
    )
    parser.add_argument("--target", required=True, help="Eval name, e.g. med_adhd_vs_ctrl.")
    parser.add_argument("--include", help="Regex a fit path must match (run/artifacts/fits/fit).")
    parser.add_argument("--exclude", help="Regex a fit path must not match.")
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="JSON summary (with the fit list).")
    args = parser.parse_args()
    fits = load_fits(args.cohort_root, args.target, args.include, args.exclude)
    if not fits:
        parser.error("No saved fit matches.")
    summary = {**evaluate(ensemble(fits), n_permutations=args.n_permutations), "fits": sorted(fits)}
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

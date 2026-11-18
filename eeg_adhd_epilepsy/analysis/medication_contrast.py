#!/usr/bin/env python3
"""Does stimulant medication bring ADHD children's EEG towards that of controls?

Children (one first recording each) are controls, unmedicated ADHD or medicated ADHD
(methylphenidate or lisdexamfetamine). Each child is described by its mean frozen
foundation-model embedding per condition (EO, EC), and every contrast is decoded the same
way: a logistic regression per model and condition (fold-local scaling, PCA,
:func:`child_level.linear_scores`), rank-averaged over those that the child has, out of
fold over repeated stratified 5-fold splits.

Groups of different size or make-up are never compared directly. Where a contrast is
set against a smaller group (``reference``), the larger group is drawn many times at the
reference's size with the same make-up in every cell of :data:`COVARIATES` (epilepsy,
autism, age group, sex), so the draws differ from the reference only in medication.

Contrasts:

- ``unmedicated_vs_control`` against ``medicated_vs_control`` draws matched to the
  unmedicated children: if medication normalises the EEG, unmedicated children separate
  from controls better than most draws.
- ``axis``: a classifier trained on unmedicated ADHD vs controls places the medicated
  children (never seen in training) on that axis; the normalisation index is their mean
  score's distance from the controls', relative to the unmedicated children's (0: like
  controls, 1: like unmedicated ADHD), per matched draw.
- ``medicated_vs_unmedicated``: the two ADHD groups directly, medicated drawn matched.
- ``lisdexamfetamine_vs_control`` against ``methylphenidate_vs_control`` draws matched to
  the lisdexamfetamine children, and ``methylphenidate_vs_lisdexamfetamine`` directly.

Each contrast also gets the AUC of the covariates alone, its floor.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_adhd_epilepsy.analysis.child_level import (
    CONDITIONS,
    _load_scope,
    late_fusion,
    linear_scores,
)

logger = logging.getLogger(__name__)

COVARIATES = ("epilepsy", "autism", "age_group", "sex")
DRUGS = {"Methylphenidate": "methylphenidate", "Lisdexamfetamine": "lisdexamfetamine"}


def child_groups(meta: pd.DataFrame) -> pd.Series:
    """control, unmedicated or medicated, per child."""
    adhd = pd.to_numeric(meta["adhd"], errors="coerce") == 1
    medicated = pd.to_numeric(meta["psychostimulant"], errors="coerce") == 1
    group = np.where(~adhd, "control", np.where(medicated, "medicated", "unmedicated"))
    return pd.Series(group, index=meta.index, name="group")


def matched_draw(
    pool: pd.DataFrame, reference: pd.DataFrame, rng: np.random.Generator, covariates=COVARIATES
) -> pd.Index:
    """Children of ``pool``, as many as ``reference`` has in every covariate cell.

    Drops the last covariate (and so on) while a cell of the reference has more children
    than the pool has in it.
    """
    for k in range(len(covariates), 0, -1):
        cols = list(covariates[:k])
        need = reference.groupby(cols).size()
        have = pool.groupby(cols).size().reindex(need.index, fill_value=0)
        if (have >= need).all():
            break
    else:
        raise ValueError("The pool is smaller than the reference.")
    chosen = []
    for cell, n in need.items():
        cell = cell if isinstance(cell, tuple) else (cell,)
        members = pool.index[(pool[cols] == pd.Series(cell, index=cols)).all(axis=1)]
        chosen.extend(rng.choice(members, size=n, replace=False))
    return pd.Index(chosen)


def contrast_scores(
    blocks: dict[str, pd.DataFrame], y: pd.Series, n_repeats: int = 3, seed: int = 0
) -> pd.Series:
    """Out-of-fold scores of ``y``'s children, late-fused over blocks, mean over repeats."""
    runs = []
    for repeat in range(n_repeats):
        per_block = {}
        for name, block in blocks.items():
            have = y.index.intersection(block.index)
            ys = y.loc[have]
            if ys.nunique() < 2 or ys.value_counts().min() < 5:
                continue
            folds = list(StratifiedKFold(5, shuffle=True, random_state=seed + repeat).split(ys, ys))
            per_block[name] = linear_scores({name: block}, ys, folds, seed=seed)
        runs.append(late_fusion(per_block).reindex(y.index))
    return pd.concat(runs, axis=1).mean(axis=1)


def covariate_block(meta: pd.DataFrame) -> pd.DataFrame:
    """The covariates as one-hot columns, the floor every contrast is set against."""
    return pd.get_dummies(meta[list(COVARIATES)].astype(str)).astype(float)


def auc(y: pd.Series, score: pd.Series) -> float:
    """AUC of ``y``'s children that have a score."""
    score = score.reindex(y.index)
    keep = score.notna()
    return float(roc_auc_score(y[keep], score[keep]))


def axis_scores(
    blocks: dict[str, pd.DataFrame], y: pd.Series, others: pd.Index, seed: int = 0
) -> pd.Series:
    """Out-of-fold scores of ``y``'s children and, averaged over the folds' models, of the
    ``others`` (never in training), late-fused over blocks."""
    per_block = {}
    for name, block in blocks.items():
        have = y.index.intersection(block.index)
        ys, extra = y.loc[have], others.intersection(block.index)
        out = pd.Series(np.nan, index=have.append(extra))
        extra_scores = []
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(ys, ys):
            X = block.loc[have].to_numpy()
            steps = [StandardScaler()] + ([PCA(10, random_state=seed)] if X.shape[1] > 10 else [])
            pipe = make_pipeline(*steps).fit(X[train])
            model = LogisticRegressionCV(
                Cs=[1e-3, 1e-2, 1e-1, 1.0, 10.0],
                cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                scoring="roc_auc",
                class_weight="balanced",
                max_iter=5000,
            ).fit(pipe.transform(X[train]), ys.iloc[train])
            out.loc[have[test]] = model.decision_function(pipe.transform(X[test]))
            extra_scores.append(
                model.decision_function(pipe.transform(block.loc[extra].to_numpy()))
            )
        out.loc[extra] = np.mean(extra_scores, axis=0)
        per_block[name] = out
    return late_fusion(per_block)


def normalisation_index(score: pd.Series, control, unmedicated, medicated) -> float:
    """Where the medicated children's mean score lies, 0 at the controls', 1 at the
    unmedicated children's."""
    c, u = score.loc[control].mean(), score.loc[unmedicated].mean()
    return float((score.loc[medicated].mean() - c) / (u - c))


def _against_draws(blocks, meta, reference, pool, others, n_draws, n_repeats, seed, n_jobs):
    """AUC of ``reference`` vs ``others`` and of matched draws of ``pool`` vs ``others``."""
    y = pd.Series(1, index=reference).combine_first(pd.Series(0, index=others)).astype(int)
    ref_auc = auc(y, contrast_scores(blocks, y, n_repeats, seed))
    cov_auc = auc(y, contrast_scores({"covariates": covariate_block(meta)}, y, n_repeats, seed))

    def one(draw):
        rng = np.random.default_rng(seed + 1000 + draw)
        chosen = matched_draw(meta.loc[pool], meta.loc[reference], rng)
        yd = pd.Series(1, index=chosen).combine_first(pd.Series(0, index=others)).astype(int)
        return auc(yd, contrast_scores(blocks, yd, n_repeats, seed))

    draws = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in range(n_draws)))
    return {
        "reference_auc": ref_auc,
        "covariates_auc": cov_auc,
        "draws_auc_median": float(np.median(draws)),
        "draws_auc_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        # one-sided: how often a matched draw separates at least as well as the reference
        "p_draws_at_least_reference": float((1 + (draws >= ref_auc).sum()) / (1 + len(draws))),
        "n_reference": int(len(reference)),
        "n_others": int(len(others)),
        "n_draws": int(len(draws)),
    }, draws


def _direct(blocks, meta, positive, pool, n_draws, n_repeats, seed, n_jobs):
    """Matched draws of ``pool`` (label 0) vs ``positive`` (label 1): AUC per draw, and
    of the covariates alone."""

    def one(draw):
        rng = np.random.default_rng(seed + 2000 + draw)
        chosen = matched_draw(meta.loc[pool], meta.loc[positive], rng)
        y = pd.Series(1, index=positive).combine_first(pd.Series(0, index=chosen)).astype(int)
        return (
            auc(y, contrast_scores(blocks, y, n_repeats, seed)),
            auc(y, contrast_scores({"covariates": covariate_block(meta)}, y, n_repeats, seed)),
        )

    draws = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in range(n_draws)))
    return {
        "auc_median": float(np.median(draws[:, 0])),
        "auc_95": [float(np.quantile(draws[:, 0], 0.025)), float(np.quantile(draws[:, 0], 0.975))],
        "share_of_draws_above_chance": float((draws[:, 0] > 0.5).mean()),
        "covariates_auc_median": float(np.median(draws[:, 1])),
        "n_positive": int(len(positive)),
        "n_draws": int(len(draws)),
    }, draws[:, 0]


def run(blocks, meta, n_draws=100, n_repeats=3, seed=0, n_jobs=1) -> tuple[dict, pd.DataFrame]:
    group = child_groups(meta)
    control, unmedicated, medicated = (
        group.index[group == g] for g in ("control", "unmedicated", "medicated")
    )
    drug = meta["psychostimulant_category"].astype(str)
    mph, ldx = (medicated.intersection(drug.index[drug == d]) for d in DRUGS)
    logger.info(
        "%d controls, %d unmedicated, %d medicated (%d methylphenidate, %d lisdexamfetamine)",
        len(control),
        len(unmedicated),
        len(medicated),
        len(mph),
        len(ldx),
    )
    args = (n_draws, n_repeats, seed, n_jobs)
    summary, draws = {}, {}
    summary["unmedicated_vs_control"], draws["medicated_vs_control"] = _against_draws(
        blocks, meta, unmedicated, medicated, control, *args
    )
    summary["lisdexamfetamine_vs_control"], draws["methylphenidate_vs_control"] = _against_draws(
        blocks, meta, ldx, mph, control, *args
    )
    summary["medicated_vs_unmedicated"], draws["medicated_vs_unmedicated"] = _direct(
        blocks, meta, unmedicated, medicated, *args
    )
    summary["methylphenidate_vs_lisdexamfetamine"], draws["methylphenidate_vs_lisdexamfetamine"] = (
        _direct(blocks, meta, ldx, mph, *args)
    )

    y = pd.Series(1, index=unmedicated).combine_first(pd.Series(0, index=control)).astype(int)
    score = axis_scores(blocks, y, medicated, seed)
    rng = np.random.default_rng(seed + 3000)
    index = []
    for _ in range(n_draws):
        chosen = matched_draw(meta.loc[medicated], meta.loc[unmedicated], rng)
        index.append(normalisation_index(score, control, unmedicated, chosen))
    index = np.array(index)
    ym = pd.Series(1, index=medicated).combine_first(pd.Series(0, index=control)).astype(int)
    summary["axis"] = {
        "unmedicated_vs_control_auc": auc(y, score),
        "medicated_vs_control_auc": auc(ym, score),
        "normalisation_index_all_medicated": normalisation_index(
            score, control, unmedicated, medicated
        ),
        "normalisation_index_matched_median": float(np.median(index)),
        "normalisation_index_matched_95": [
            float(np.quantile(index, 0.025)),
            float(np.quantile(index, 0.975)),
        ],
    }
    draws["normalisation_index"] = index
    return summary, pd.DataFrame(draws)


def load_children(args) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Mean embedding per child, model and condition, and each child's covariates."""
    from eeg_adhd_epilepsy.utils.config import resolve_cli_config

    blocks, rows = {}, []
    columns = ["adhd", "psychostimulant", "psychostimulant_category", *COVARIATES]
    for model in args.models:
        config = resolve_cli_config(
            cohort_config=args.cohort_config,
            analysis_config=args.embeddings_config,
            bids_root=args.bids_root,
            desc=args.desc,
            metadata=args.metadata,
            derivative_root=str(args.output_dir),
            embedding_derivative_root=args.embedding_derivative_root,
            embedding_model_key=model,
            representation="recording",
        )
        eval_spec = next(e for e in config["evals"] if e["name"] == args.target)
        for condition in CONDITIONS:
            container, child, _ = _load_scope(config, "foundation_embeddings", condition, eval_spec)
            X = np.asarray(container.X, dtype=np.float32).reshape(len(child), -1)
            blocks[f"{model}_{condition.split('_')[0]}"] = (
                pd.DataFrame(X, index=child).groupby(level=0).mean()
            )
            rows.append(
                pd.DataFrame(
                    {c: np.asarray(container.coords[c]).astype(str) for c in columns}, index=child
                )
            )
    meta = pd.concat(rows)
    meta = meta[~meta.index.duplicated()]
    return blocks, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort_config", default="configs/contrasts/adhd_medication.yaml")
    parser.add_argument("--target", default="adhd_vs_ctrl")
    parser.add_argument("--desc", default="base", choices=["base", "baserejected"])
    parser.add_argument("--models", nargs="+", default=["cbramod", "labram", "reve", "luna"])
    parser.add_argument("--bids_root", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embedding_derivative_root", required=True)
    parser.add_argument(
        "--embeddings_config", default="configs/analyses/decoding/foundation_embeddings.yaml"
    )
    parser.add_argument("--n_draws", type=int, default=100)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks, meta = load_children(args)
    summary, draws = run(blocks, meta, args.n_draws, args.n_repeats, n_jobs=args.n_jobs)
    draws.to_csv(args.output_dir / "draws.csv", index=False)
    text = json.dumps(summary, indent=2)
    (args.output_dir / "summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

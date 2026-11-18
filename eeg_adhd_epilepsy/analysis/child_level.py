#!/usr/bin/env python3
"""Child-level decoding: one example per child, built from all of its windows.

Window-level training gives every window its child's label, so a model lowers its loss by
recognising the training children. Here each child is one example, and its label is used once:

- **mean**: the child's mean frozen embedding in EO, in EC and their difference (EC - EO);
  each block is standardised and reduced by PCA inside the fold, then a logistic regression.
- **attention**: a gated-attention network (Ilse et al. 2018) pools all of a child's EO and
  EC windows into one vector and classifies it, with one loss per child; the attention weights
  say which windows carry the decision.
- **descriptors**: the child's recording descriptors in EO, in EC and their difference, either
  summarised (each measure averaged over sensors) or reduced by fold-local PCA.
- **early fusion**: the descriptor and embedding blocks in one logistic regression.
- **late fusion**: the rank average of the separate models' out-of-fold scores.

Every method uses the same children and the same stratified 5-fold split, so all scores are
out-of-fold and comparable; each is scored with :func:`ensemble.evaluate` (child-level AUC
with permutation p and bootstrap interval, balanced accuracy at the median split).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_adhd_epilepsy.analysis.ensemble import evaluate

logger = logging.getLogger(__name__)

CONDITIONS = ("EO_baseline", "EC_baseline")


# ---------------------------------------------------------------- child-level blocks


def child_blocks(X: np.ndarray, child: np.ndarray, condition: np.ndarray, name: str) -> dict:
    """Per-child mean of ``X`` in EO and in EC (each child that has it), and EC - EO."""
    frame = pd.DataFrame(np.asarray(X, dtype=float))
    frame["child"], frame["condition"] = np.asarray(child).astype(str), np.asarray(condition)
    means = {
        c: g.drop(columns="condition").groupby("child").mean()
        for c, g in frame.groupby("condition")
    }
    eo, ec = means[CONDITIONS[0]], means[CONDITIONS[1]]
    both = eo.index.intersection(ec.index)
    return {f"{name}_EO": eo, f"{name}_EC": ec, f"{name}_EC-EO": ec.loc[both] - eo.loc[both]}


def linear_scores(
    blocks: dict[str, pd.DataFrame],
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_components: int | None | dict[str, int | None] = 10,
    seed: int = 0,
) -> pd.Series:
    """Out-of-fold scores of a logistic regression on standardised (and PCA-reduced) blocks.

    Each block is scaled, and reduced to ``n_components`` (``None``: kept whole; a dict gives
    it per block name) on the training children only; the inverse regularisation is chosen
    by inner CV on the AUC.
    """
    scores = pd.Series(np.nan, index=y.index)
    for train, test in folds:
        parts_train, parts_test = [], []
        for name, block in blocks.items():
            block = block.loc[y.index].to_numpy()
            k = n_components.get(name) if isinstance(n_components, dict) else n_components
            steps = [StandardScaler()]
            if k is not None and block.shape[1] > k:
                steps.append(PCA(k, random_state=seed))
            pipe = make_pipeline(*steps)
            parts_train.append(pipe.fit_transform(block[train]))
            parts_test.append(pipe.transform(block[test]))
        model = LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1.0, 10.0],
            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
        ).fit(np.hstack(parts_train), y.iloc[train])
        scores.iloc[test] = model.predict_proba(np.hstack(parts_test))[:, 1]
    return scores


def late_fusion(scores: dict[str, pd.Series]) -> pd.Series:
    """Mean rank (0-1) over the given out-of-fold score columns."""
    return pd.concat([s.rank(pct=True) for s in scores.values()], axis=1).mean(axis=1)


# ---------------------------------------------------------------- attention pooling


def attention_scores(
    X: np.ndarray,
    child: np.ndarray,
    condition: np.ndarray,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: tuple[int, ...] = (0, 1, 2),
    n_components: int = 32,
    max_windows: int = 128,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-fold scores of a gated-attention network over each child's windows.

    Per fold, windows are standardised and reduced by PCA on the training children, and a
    condition flag (EO / EC) is appended; 20 % of the training children stop the training
    (validation AUC). Scores are averaged over ``seeds``. Also returns the attention weight
    of every window of the test children (mean over seeds).
    """
    import torch

    child = np.asarray(child).astype(str)
    is_ec = (np.asarray(condition) == CONDITIONS[1]).astype(np.float32)[:, None]
    rows = pd.Series(np.arange(len(child))).groupby(child).apply(np.asarray)
    scores = pd.Series(np.nan, index=y.index)
    weights = []
    for train, test in folds:
        train_children, test_children = y.index[train], y.index[test]
        fit_rows = np.concatenate(rows.loc[train_children].to_numpy())
        pipe = make_pipeline(StandardScaler(), PCA(min(n_components, X.shape[1]), random_state=0))
        sample = fit_rows[np.random.default_rng(0).permutation(len(fit_rows))[:50000]]
        pipe.fit(X[sample])
        Z = np.hstack([pipe.transform(X).astype(np.float32), is_ec])
        bags = {c: torch.from_numpy(Z[rows[c]]) for c in y.index}
        fold_scores, fold_weights = [], []
        for seed in seeds:
            model = _fit_attention(bags, y, train_children, seed, max_windows, max_epochs, patience)
            logits, attn = _predict(model, bags, test_children)
            fold_scores.append(torch.sigmoid(logits).numpy())
            fold_weights.append(attn)
        scores.loc[test_children] = np.mean(fold_scores, axis=0)
        for i, c in enumerate(test_children):
            weights.append(
                pd.DataFrame(
                    {
                        "child": c,
                        "window": rows[c],
                        "condition": np.asarray(condition)[rows[c]],
                        "attention": np.mean([w[i] for w in fold_weights], axis=0),
                    }
                )
            )
    return scores, pd.concat(weights, ignore_index=True)


def _attention_net(n_features: int, hidden: int = 64, dropout: float = 0.25):
    import torch
    from torch import nn

    class GatedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            self.V, self.U = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
            self.w, self.out = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

        def forward(self, x, mask):
            h = self.embed(x)
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = a.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            return self.out((a.unsqueeze(-1) * h).sum(dim=1)).squeeze(-1), a

    return GatedAttention()


def _batch(bags: dict, children, rng=None, max_windows: int | None = None):
    import torch

    tensors = []
    for c in children:
        bag = bags[c]
        if rng is not None and max_windows and len(bag) > max_windows:
            bag = bag[torch.from_numpy(rng.choice(len(bag), max_windows, replace=False))]
        tensors.append(bag)
    lengths = torch.tensor([len(t) for t in tensors])
    x = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return x, torch.arange(x.shape[1])[None, :] < lengths[:, None]


def _predict(model, bags: dict, children) -> tuple[Any, list[np.ndarray]]:
    import torch

    model.eval()
    with torch.no_grad():
        x, mask = _batch(bags, children)
        logits, attn = model(x, mask)
    return logits, [attn[i, : int(mask[i].sum())].numpy() for i in range(len(children))]


def _fit_attention(bags, y, children, seed, max_windows, max_epochs, patience, batch_size=32):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit, valid = train_test_split(
        np.asarray(children), test_size=0.2, stratify=y.loc[children], random_state=seed
    )
    y_fit = torch.tensor(y.loc[fit].to_numpy(), dtype=torch.float32)
    positive = float(y_fit.mean())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - positive) / positive))
    model = _attention_net(next(iter(bags.values())).shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, waited = -np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(len(fit))
        for start in range(0, len(fit), batch_size):
            idx = order[start : start + batch_size]
            x, mask = _batch(bags, fit[idx], rng, max_windows)
            optimiser.zero_grad()
            loss_fn(model(x, mask)[0], y_fit[idx]).backward()
            optimiser.step()
        logits, _ = _predict(model, bags, valid)
        auc = roc_auc_score(y.loc[valid], logits.numpy())
        if auc > best:
            best, waited = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------- data


def _load_scope(config: dict, input_mode: str, condition: str, eval_spec: dict):
    """Rows, children and binary labels of one condition for ``eval_spec``."""
    from eeg_adhd_epilepsy.analysis.dataset import build_dataset
    from eeg_adhd_epilepsy.analysis.utils.decoding import (
        build_loader_args,
        prepare_decoding_scope,
        resolve_decoding_paths,
    )

    metadata = resolve_decoding_paths(config, input_mode=input_mode)[3]
    loader_args = build_loader_args(config, input_mode=input_mode, layout_mode="flat")
    container = build_dataset(loader_args, metadata, condition, target_col=None)
    selected, y, _, meta, _ = prepare_decoding_scope(
        container,
        eval_spec,
        scope=condition,
        group_col=eval_spec.get("group_col", config.get("group_col", "patient_group_id")),
        session_col=config.get("session_col", "session"),
        subject_col=config["subject_col"],
        requested_splits=5,
    )
    label = (np.asarray(y).astype(str) == str(eval_spec["positive_class"])).astype(int)
    return selected, meta["group_id"].astype(str).to_numpy(), label


def load_inputs(args) -> tuple[pd.Series, dict, dict, dict]:
    """Labels per child, embedding windows per model, embedding and descriptor blocks."""
    from eeg_adhd_epilepsy.utils.config import resolve_cli_config

    common = dict(
        cohort_config=args.cohort_config,
        matched=args.matched,
        bids_root=args.bids_root,
        desc=args.desc,
        metadata=args.metadata,
        derivative_root=str(args.output_dir),
    )
    labels, windows, blocks = {}, {}, {}
    for model in args.models:
        config = resolve_cli_config(
            analysis_config=args.embeddings_config,
            embedding_derivative_root=args.embedding_derivative_root,
            embedding_model_key=model,
            representation="epoch",
            **common,
        )
        eval_spec = next(e for e in config["evals"] if e["name"] == args.target)
        parts = []
        for condition in CONDITIONS:
            container, child, label = _load_scope(
                config, "foundation_embeddings", condition, eval_spec
            )
            labels.update(zip(child, label))
            X = np.asarray(container.X, dtype=np.float32).reshape(len(child), -1)
            parts.append((X, child, np.full(len(child), condition)))
            logger.info(
                "%s %s: %d windows, %d children", model, condition, len(child), len(set(child))
            )
        X, child, condition = (np.concatenate(p) for p in zip(*parts))
        windows[model] = (X, child, condition)
        blocks.update(child_blocks(X, child, condition, model))
    descriptors = {}
    if args.descriptor_table_path:
        config = resolve_cli_config(
            analysis_config=args.descriptors_config,
            descriptor_table_path=str(args.descriptor_table_path),
            representation="recording",
            **common,
        )
        eval_spec = next(e for e in config["evals"] if e["name"] == args.target)
        frames, child, condition, measure = [], [], [], {}
        for cond in CONDITIONS:
            container, ids, label = _load_scope(config, "descriptors", cond, eval_spec)
            columns = container.coords["feature"]
            frames.append(pd.DataFrame(container.X, columns=columns))
            measure.update(zip(columns, container.coords["feature_measure"]))
            labels.update(zip(ids, label))
            child.append(ids)
            condition.append(np.full(len(ids), cond))
        flat = pd.concat(frames, ignore_index=True)
        # no imputation: a column missing in one condition, or with a NaN, is dropped
        flat = flat.loc[:, flat.notna().all()]
        child, condition = np.concatenate(child), np.concatenate(condition)
        summary = flat.T.groupby(pd.Series(measure).loc[flat.columns].to_numpy()).mean().T
        descriptors["summary"] = child_blocks(summary.to_numpy(), child, condition, "desc")
        descriptors["pca"] = child_blocks(flat.to_numpy(), child, condition, "desc")
    return pd.Series(labels, name="y"), windows, blocks, descriptors


# ---------------------------------------------------------------- run


CONDITION_SETS = {
    "EO": ("EO",),
    "EC": ("EC",),
    "EC-EO": ("EC-EO",),
    "EO+EC": ("EO", "EC"),
    "EO+EC+diff": ("EO", "EC", "EC-EO"),
}


def run(labels, windows, blocks, descriptors, n_permutations=1000, seeds=(0, 1, 2)):
    """Every combination, each on every child that has its inputs; returns scores, summary
    and attention weights.

    The children are split once into 5 stratified folds; a method uses the children that
    have all of its blocks (no imputation), trains on those of the other folds and scores
    those of each fold. For each condition set (EO, EC, EC - EO, EO + EC, EO + EC +
    difference) and each source (one model's embeddings, the descriptors): a logistic
    regression; then each model and all models together, alone or with the descriptors, by
    early fusion (one logistic regression on all blocks, children with all of them) and late
    fusion (per child, the mean rank of the sources' scores it has). Attention pooling runs
    per model on EO, EC and EO + EC bags. The headline is fixed in advance: per child, the
    mean rank over every model's EO and EC scores, the summary descriptors' EO and EC scores
    and every model's EO + EC attention score, so every child with any input is scored.
    """
    models = list(windows)
    sources = {
        m: {k.removeprefix(f"{m}_"): v for k, v in blocks.items() if k.startswith(f"{m}_")}
        for m in models
    }
    for kind, desc in descriptors.items():
        sources[f"desc-{kind}"] = {k.removeprefix("desc_"): v for k, v in desc.items()}
    for source, frames in sources.items():
        logger.info("%s children: %s", source, {c: len(f) for c, f in frames.items()})
    labels = labels.sort_index().astype(int)
    fold = pd.Series(-1, index=labels.index)
    splitter = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        fold.iloc[test] = k
    logger.info("%d children (%.2f positive)", len(labels), labels.mean())

    def fit(fused: dict, widths) -> pd.Series:
        children = labels.index
        for frame in fused.values():
            children = children.intersection(frame.index)
        y, f = labels.loc[children], fold.loc[children].to_numpy()
        folds = [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(5)]
        return linear_scores(fused, y, folds, widths)

    def width(source):  # the summary descriptors enter whole, the rest reduced by PCA
        return None if source == "desc-summary" else 10

    def pick(source, conditions):
        return {f"{source}_{c}": sources[source][c] for c in conditions}

    groups = [[m] for m in models] + ([models] if len(models) > 1 else [])
    kinds = list(descriptors)
    scores, single = {}, {}
    for cname, conditions in CONDITION_SETS.items():
        for source in sources:
            single[source, cname] = fit(pick(source, conditions), width(source))
            scores[f"mean|{cname}|{source}"] = single[source, cname]
        for kind in [None, *kinds]:
            for group in groups:
                if kind is None and len(group) == 1:
                    continue
                parts = group + ([f"desc-{kind}"] if kind else [])
                label = (group[0] if len(group) == 1 else "all") + (f"+desc-{kind}" if kind else "")
                fused = {k: v for part in parts for k, v in pick(part, conditions).items()}
                widths = {k: width(part) for part in parts for k in pick(part, conditions)}
                scores[f"early|{cname}|{label}"] = fit(fused, widths)
                scores[f"late|{cname}|{label}"] = late_fusion({p: single[p, cname] for p in parts})
        logger.info("%s done", cname)

    attention = {}
    bags = {"EO": CONDITIONS[:1], "EC": CONDITIONS[1:], "EO+EC": CONDITIONS}
    for model in models:
        X, child, condition = windows[model]
        child = np.asarray(child).astype(str)
        for bag, conditions in bags.items():
            rows = np.isin(child, labels.index) & np.isin(condition, conditions)
            have = labels.index.intersection(pd.unique(child[rows]))
            y, f = labels.loc[have], fold.loc[have].to_numpy()
            folds = [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(5)]
            scores[f"attention|{bag}|{model}"], weights = attention_scores(
                X[rows], child[rows], condition[rows], y, folds, seeds=seeds
            )
            if bag == "EO+EC":
                attention[model] = weights
        logger.info("attention %s done", model)

    headline = {f"{s}-{c}": single[s, c] for s in sources if s != "desc-pca" for c in ("EO", "EC")}
    headline |= {f"attention-{m}": scores[f"attention|EO+EC|{m}"] for m in models}
    scores["late|headline|every-input"] = late_fusion(headline)

    table = pd.DataFrame(scores).assign(y=labels).dropna(how="all", subset=list(scores))
    summary = {}
    for name in scores:
        rows = table[name].notna()
        frame = pd.DataFrame(
            {"child": table.index[rows], "y": table["y"][rows], "score": table[name][rows]}
        )
        summary[name] = evaluate(frame, n_permutations=n_permutations)
    return table, summary, attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort_config", required=True)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--desc", default="base", choices=["base", "baserejected"])
    parser.add_argument("--target", default="med_adhd_vs_ctrl")
    parser.add_argument("--models", nargs="+", default=["cbramod", "labram", "reve", "luna"])
    parser.add_argument("--bids_root", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embedding_derivative_root", required=True)
    parser.add_argument(
        "--descriptor_table_path", default=None, help="sensor_recording_features.parquet"
    )
    parser.add_argument(
        "--embeddings_config", default="configs/analyses/decoding/foundation_embeddings.yaml"
    )
    parser.add_argument("--descriptors_config", default="configs/analyses/decoding/classical.yaml")
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_threads", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.n_threads:
        import torch

        torch.set_num_threads(args.n_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, summary, attention = run(*load_inputs(args), args.n_permutations, tuple(args.seeds))
    table.to_csv(args.output_dir / "scores.csv", index_label="child")
    for model, frame in attention.items():
        frame.to_csv(args.output_dir / f"attention_{model}.csv", index=False)
    text = json.dumps(summary, indent=2)
    (args.output_dir / "summary.json").write_text(text + "\n")
    for name, r in sorted(summary.items(), key=lambda item: -item[1]["auc"]):
        auc = (
            f"AUC {r['auc']:.3f} [{r['auc_ci_low']:.3f}-{r['auc_ci_high']:.3f}] p {r['auc_p']:.4f}"
        )
        ba = r["balanced_accuracy_median_split"]
        print(f"{name:45s} {auc} BA {ba:.3f} n {r['n_children']}")


if __name__ == "__main__":
    main()

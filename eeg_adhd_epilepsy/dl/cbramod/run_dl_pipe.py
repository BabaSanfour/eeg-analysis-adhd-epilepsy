#!/usr/bin/env python3
"""
Deep-learning pipeline runner.
Explicitly runs:
1. Permutation-based Feature Selection (Recursive Drop)
2. Cross-Validation (on selected features)
3. Subject Aggregation (with threshold tuning)
"""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

from coco_pipe.io import load as load_tabular

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_embedder(x: Any) -> np.ndarray:
    return np.asarray(x)

def _build_embed_fn(model_name: str, model_cfg: Dict[str, Any]) -> Callable[[Any], np.ndarray]:
    name = (model_name or "").lower()
    mode = (model_cfg or {}).get("mode", "precomputed")

    if name == "cbramod":
        if mode in {"precomputed", "identity"}:
            return _identity_embedder
        from coco_pipe.fm.cbramod.embedder import CBRAModEmbedder
        weights_path = model_cfg.get("ckpt_path") or model_cfg.get("weights_path")
        device = model_cfg.get("device", "cuda")
        return CBRAModEmbedder(weights_path=weights_path, device=device)

    raise ValueError(f"Unknown foundation model '{model_name}'.")

def _resolve_base_classifier(cfg: Dict[str, Any]) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    name = (cfg.get("base_classifier") or "logreg").lower()
    if name in {"logreg", "logistic"}:
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name.startswith("hist"):
        return HistGradientBoostingClassifier(
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth"),
            max_iter=cfg.get("max_iter", 200),
            l2_regularization=cfg.get("l2_regularization", 0.0),
            class_weight=cfg.get("class_weight", "balanced"),
        )
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=cfg.get("min_samples_leaf", 2),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=cfg.get("n_jobs", -1),
        )
    raise ValueError(f"Unknown base_classifier '{name}'")

def _load_embeddings_df(cfg, analysis_cfg):
    embed_cfg = analysis_cfg.get("embedding", {})
    data_path = embed_cfg.get("data_path")
    if not data_path:
        raise ValueError("No data_path provided in embedding config.")
    
    logger.info("Loading data from %s", data_path)
    df = load_tabular("tabular", data_path)
    return df, embed_cfg

def _attach_targets(df, analysis_cfg, embed_cfg):
    target_cols = analysis_cfg.get("target_columns") or ["Epilepsy"]
    target_path = analysis_cfg.get("target_path") or embed_cfg.get("target_path")
    
    if target_path:
        targets_df = load_tabular("tabular", target_path)
        if "participant_id" in targets_df.columns:
            targets_df = targets_df.rename(columns={"participant_id": "subject"})
        
        merge_keys = [k for k in ["subject", "segment"] if k in df.columns and k in targets_df.columns]
        if merge_keys:
            df = df.merge(targets_df[merge_keys + target_cols], on=merge_keys, how="inner")

    if "subject" in df.columns:
        groups = df["subject"]
    elif "group_id" in df.columns:
        groups = df["group_id"]
        df = df.rename(columns={"group_id": "subject"})
    else:
        logger.warning("No 'subject' or 'group_id' column found. Using index as groups (may leak).")
        groups = pd.Series(range(len(df)))

    y = df[target_cols[0]]
    drop_cols = target_cols + ["subject", "segment", "n_channels", "group_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    if X.isna().any().any():
        X = X.fillna(X.median())
        
    return X, y, groups

def _aggregate_subject_probs(groups, y_true, y_prob, agg: str = "mean", quantile: float = 0.75):
    """Aggregate per-segment probabilities to per-subject scores."""
    df_res = pd.DataFrame({
        "subject": groups.values,
        "y_true": y_true.values,
        "y_prob": y_prob,
    })

    def _agg_series(x):
        if agg == "mean":
            return x.mean()
        if agg == "median":
            return x.median()
        if agg == "quantile":
            return x.quantile(quantile)
        return x.mean()

    subject_agg = df_res.groupby("subject").agg(
        y_true=("y_true", "first"),
        y_prob=("y_prob", _agg_series),
    )
    return subject_agg

def _tune_threshold(y_true, y_prob, metric_fn=balanced_accuracy_score):
    """Simple grid search for threshold that maximizes the metric."""
    thresholds = np.linspace(0.1, 0.9, 17)
    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = metric_fn(y_true, preds)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score

def _calculate_subject_metrics(y_true, y_prob, groups, agg="mean", quantile=0.75):
    """
    Aggregate segment predictions to subjects, tune threshold for balanced accuracy,
    and report subject-level AUC and balanced accuracy.
    """
    subject_agg = _aggregate_subject_probs(groups, y_true, y_prob, agg=agg, quantile=quantile)

    sub_auc = roc_auc_score(subject_agg["y_true"], subject_agg["y_prob"])
    best_t, sub_bal_acc = _tune_threshold(subject_agg["y_true"], subject_agg["y_prob"])

    logger.info(
        "Subject-Level Results (%s): AUC=%.4f, Balanced Acc=%.4f at threshold=%.2f",
        agg, sub_auc, sub_bal_acc, best_t,
    )
    return sub_auc, sub_bal_acc, best_t

def _run_single_analysis(cfg, analysis_cfg):
    embeddings_df, embed_cfg = _load_embeddings_df(cfg, analysis_cfg)
    X_df, y, groups = _attach_targets(embeddings_df, analysis_cfg, embed_cfg)
    
    logger.info("Data loaded: X=%s, Groups=%d", X_df.shape, groups.nunique())

    clf = _resolve_base_classifier(analysis_cfg)
    if analysis_cfg.get("use_scaler", True):
        estimator = make_pipeline(StandardScaler(), clf)
    else:
        estimator = clf

    # Setup Cross-Validation
    cv_strategy = analysis_cfg.get("cv_kwargs", {}).get("cv_strategy", "stratified_group")
    n_splits = analysis_cfg.get("cv_kwargs", {}).get("n_splits", 5)
    random_state = analysis_cfg.get("random_state", 42)

    if "group" in cv_strategy:
        if "stratified" in cv_strategy:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = GroupKFold(n_splits=n_splits)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # --- 1. FEATURE SELECTION (Permutation-Based, Subsampled) ---
    logger.info("Running Feature Selection (Permutation Importance, subsampled)...")
    
    # Optionally subsample rows for importance to avoid OOM
    fs_sample_size = analysis_cfg.get("fs_perm_sample_size", 50000)
    rng = np.random.default_rng(42)
    if len(X_df) > fs_sample_size:
        idx = rng.choice(len(X_df), size=fs_sample_size, replace=False)
        X_fs = X_df.iloc[idx]
        y_fs = y.iloc[idx]
    else:
        X_fs, y_fs = X_df, y

    est_select = clone(estimator)
    est_select.fit(X_fs, y_fs)
    
    perm_result = permutation_importance(
        est_select,
        X_fs,
        y_fs,
        n_repeats=analysis_cfg.get("fs_perm_repeats", 2),
        random_state=random_state,
        n_jobs=1,
    )
    
    keep_mask = perm_result.importances_mean > 0.0
    selected_feats = X_df.columns[keep_mask].tolist()
    
    logger.info(f"Feature Selection: Dropped {len(X_df.columns) - len(selected_feats)} features. Keeping {len(selected_feats)}.")
    
    if len(selected_feats) < 2:
        logger.warning("Selection dropped almost all features! Reverting to full set.")
        selected_feats = X_df.columns.tolist()
        
    X_selected = X_df[selected_feats]
    
    imp_df = pd.DataFrame({
        'feature': X_df.columns, 
        'importance': perm_result.importances_mean
    }).sort_values(by='importance', ascending=False)

    # --- 2. FINAL CROSS-VALIDATION (On Selected Features) ---
    metrics = ["balanced_accuracy", "accuracy", "roc_auc", "f1"]
    logger.info(f"Running Final Cross-Validation on {len(selected_feats)} features...")
    
    cv_results = cross_validate(
        estimator, X_selected, y, groups=groups, cv=cv, scoring=metrics, 
        n_jobs=analysis_cfg.get("n_jobs", -1)
    )

    # --- 3. SUBJECT AGGREGATION ---
    logger.info("Running Subject Aggregation...")
    y_pred_probs = cross_val_predict(
        estimator, X_selected, y, groups=groups, cv=cv, method="predict_proba",
        n_jobs=analysis_cfg.get("n_jobs", -1)
    )
    pos_probs = y_pred_probs[:, 1]
    agg_mode = analysis_cfg.get("subject_agg", "mean")
    agg_quantile = analysis_cfg.get("subject_agg_quantile", 0.75)
    
    sub_auc, sub_bal_acc, sub_thresh = _calculate_subject_metrics(
        y, pos_probs, groups, agg=agg_mode, quantile=agg_quantile
    )

    # --- 4. PACKAGE RESULTS ---
    results = {
        "model_name": analysis_cfg.get("base_classifier", "classifier"),
        "top_features": ", ".join(imp_df.head(10)["feature"].tolist()),
        "n_features_selected": len(selected_feats),
        
        # Subject Metrics
        "subject_auc": sub_auc,
        "subject_balanced_accuracy": sub_bal_acc,
        "subject_threshold": sub_thresh,
        "subject_agg": agg_mode,
    }
    
    # Add Segment Metrics
    for k, v in cv_results.items():
        if k.startswith("test_"):
            metric_name = k.replace("test_", "")
            results[f"segment_{metric_name}_mean"] = float(np.mean(v))
            results[f"segment_{metric_name}_std"] = float(np.std(v))
        else:
            results[f"segment_{k}_mean"] = float(np.mean(v))

    return results

def _save_results_csv(all_results: Dict[str, Any], out_path: str):
    rows = []
    for aid, data in all_results.items():
        row = {"analysis_id": aid}
        for k, v in data.items():
            if isinstance(v, (int, float, np.float64, np.float32)):
                row[k] = round(float(v), 4)
            else:
                row[k] = str(v)
        rows.append(row)
    
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info(f"Summary saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    
    all_results = {}
    for analysis in cfg.get("analyses", []):
        analysis_cfg = deepcopy(cfg.get("defaults", {}))
        analysis_cfg.update(analysis)
        try:
            results = _run_single_analysis(cfg, analysis_cfg)
            all_results[analysis_cfg["id"]] = results
        except Exception as e:
            logger.error(f"Analysis {analysis_cfg.get('id')} failed: {e}", exc_info=True)

    res_dir = cfg.get("results_dir", "data/results/dl")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"{cfg.get('global_experiment_id', 'dl_run')}_summary.csv")
    _save_results_csv(all_results, csv_path)

if __name__ == "__main__":
    main()

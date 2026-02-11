#!/usr/bin/env python3
"""
Deep-learning pipeline runner that mirrors ml/run_ml_pipe.py but targets
foundation models (CBraMod, REVE, ToTo, etc.).
"""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold, cross_val_score

from coco_pipe.fm.pipeline import (
    FoundationClassificationPipeline,
    FoundationRegressionPipeline,
)
from coco_pipe.io import load as load_tabular

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key_set(df: pd.DataFrame, keys: List[str]) -> set:
    """Return a set of key tuples (or strings for single key) for intersection checks."""
    if len(keys) == 1:
        return set(df[keys[0]].astype(str))
    return set(tuple(x) for x in df[keys].astype(str).itertuples(index=False, name=None))


def _identity_embedder(x: Any) -> np.ndarray:
    """Treat incoming X as already-embedded and coerce to numpy."""
    return np.asarray(x)


def _build_embed_fn(model_name: str, model_cfg: Dict[str, Any]) -> Callable[[Any], np.ndarray]:
    """
    Return an embedding function for the requested foundation model.
    """
    name = (model_name or "").lower()
    mode = (model_cfg or {}).get("mode", "precomputed")

    if name == "cbramod":
        if mode in {"precomputed", "identity"}:
            return _identity_embedder
        # Optional: use the CBraMod backbone directly if raw patches are provided.
        from coco_pipe.fm.cbramod.embedder import CBRAModEmbedder

        weights_path = model_cfg.get("ckpt_path") or model_cfg.get("weights_path")
        if not weights_path:
            raise ValueError("CBraMod backbone mode requires ckpt_path/weights_path")
        device = model_cfg.get("device", "cuda")
        return CBRAModEmbedder(weights_path=weights_path, device=device)

    if name in {"reve", "toto"}:
        raise NotImplementedError(f"Embedder for foundation model '{name}' is not implemented yet.")

    raise ValueError(f"Unknown foundation model '{model_name}'.")


def _resolve_base_classifier(cfg: Dict[str, Any]) -> Any:
    """Instantiate a downstream classifier based on config."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    name = (cfg.get("base_classifier") or "logreg").lower()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
    if name in {"hgb", "histgb", "hist_gradient_boosting", "hist_gradient_boosting_classifier"}:
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
            random_state=cfg.get("random_state", 42),
        )
    raise ValueError(f"Unknown base_classifier '{name}'")


def _load_embeddings_df(
    cfg: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load or generate embeddings according to analysis_cfg.embedding.
    """
    embed_cfg = analysis_cfg.get("embedding", {}) or {}
    source = embed_cfg.get("source")
    if source is None:
        source = "precomputed" if embed_cfg.get("data_path") or analysis_cfg.get("data_path") or cfg.get("data_path") else "bids"
    source = source.lower()

    if source == "precomputed":
        data_path = embed_cfg.get("data_path") or analysis_cfg.get("data_path") or cfg.get("data_path")
        if not data_path:
            raise ValueError("embedding.source=precomputed but no data_path provided.")
        logger.info("Loading precomputed embeddings from %s", data_path)
        df = load_tabular("tabular", data_path)
        
        # Temporal aggregation logic
        agg_sec = embed_cfg.get("temporal_aggregation_seconds")
        if agg_sec and agg_sec > 1:
            seg_dur = embed_cfg.get("segment_duration", 1)
            rows_per_block = int(agg_sec / seg_dur)
            if rows_per_block > 1:
                logger.info(f"Aggregating embeddings: {agg_sec}s blocks (pooling {rows_per_block} rows). Prev shape: {df.shape}")
                
                if "subject" in df.columns and "segment" in df.columns:
                    # Sort to ensure contiguous segments
                    df = df.sort_values(["subject", "segment"]).reset_index(drop=True)
                    
                    # Create block ID
                    df["_block"] = df["segment"] // rows_per_block
                    
                    # Identify feature cols (exclude meta)
                    meta = ["subject", "_block", "segment", "n_channels", "session", "run"]
                    feats = [c for c in df.columns if c not in meta]
                    
                    # Group and average
                    # preservation of subject is key
                    grouped = df.groupby(["subject", "_block"])
                    df_agg = grouped[feats].mean().reset_index()
                    
                    # Restore n_channels if present
                    if "n_channels" in df.columns:
                        meta_agg = grouped["n_channels"].first().reset_index(drop=True)
                        df_agg["n_channels"] = meta_agg
                        
                    # Rename block -> segment
                    df_agg = df_agg.rename(columns={"_block": "segment"})
                    
                    logger.info(f"Aggregation complete. New shape: {df_agg.shape}")
                    df = df_agg
                else:
                    logger.warning("Skipping aggregation: 'subject' or 'segment' column missing.")

        return df, embed_cfg

    if source == "bids":
        model = (analysis_cfg.get("foundation_model") or cfg.get("foundation_model") or "cbramod").lower()
        if model != "cbramod":
            raise NotImplementedError("BIDS -> embeddings is only implemented for CBraMod for now.")
        from eeg_adhd_epilepsy_psychostimulant.dl.cbramod_embed import compute_cbramod_embeddings_to_df

        derivatives_dir = embed_cfg.get("derivatives_dir")
        ckpt_path = embed_cfg.get("ckpt_path") or embed_cfg.get("weights_path")
        if not derivatives_dir or not ckpt_path:
            raise ValueError("BIDS embedding requires derivatives_dir and ckpt_path.")
        segment_duration = embed_cfg.get("segment_duration", 1)
        processing = embed_cfg.get("processing")
        subjects = embed_cfg.get("subjects")
        cache_path = embed_cfg.get("cache_path")
        overwrite = embed_cfg.get("overwrite", False)

        df = compute_cbramod_embeddings_to_df(
            derivatives_dir=Path(derivatives_dir),
            ckpt_path=Path(ckpt_path),
            subjects=subjects,
            segment_duration=segment_duration,
            processing=processing,
            overwrite=overwrite,
            cache_path=Path(cache_path) if cache_path else None,
        )
        return df, embed_cfg

    raise ValueError(f"Unknown embedding.source '{source}'")


def _attach_targets(
    df: pd.DataFrame,
    analysis_cfg: Dict[str, Any],
    embed_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.Series]:
    """
    Ensure target columns are present and aligned.
    Returns (X, y, groups) where groups usually contains subject IDs.
    """
    target_cols = analysis_cfg.get("target_columns") or []
    if not target_cols:
        raise ValueError("Each analysis must specify target_columns.")

    # 1. Merge Targets if missing
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        target_path = (analysis_cfg.get("target_path") or embed_cfg.get("target_path"))
        if not target_path:
            raise ValueError(f"Targets {missing} not found in embeddings and no target_path provided.")
        
        logger.info("Merging targets from %s", target_path)
        targets_df = load_tabular("tabular", target_path)
        
        # Standardize 'subject' column
        if "subject" not in targets_df.columns:
            if "Study ID" in targets_df.columns:
                targets_df = targets_df.copy()
                def _format_subject(val: Any) -> str | None:
                    if pd.isna(val): return None
                    try: return f"sub-{int(float(str(val).strip())):04d}"
                    except: return None
                targets_df["subject"] = targets_df["Study ID"].apply(_format_subject)
                targets_df = targets_df.dropna(subset=["subject"])
            elif "participant_id" in targets_df.columns:
                targets_df = targets_df.rename(columns={"participant_id": "subject"})

        # Identify merge keys
        possible_keys = ["subject", "segment", "session"]
        merge_keys = [col for col in possible_keys if col in df.columns and col in targets_df.columns]
        
        if not merge_keys:
            raise ValueError("Could not find common merge keys (subject/segment) between embeddings and targets.")
            
        df = df.merge(targets_df[merge_keys + target_cols], on=merge_keys, how="inner")
        df = df.dropna(subset=target_cols)

    # 2. Extract Groups (Subject ID) BEFORE dropping columns
    if "subject" in df.columns:
        groups = df["subject"]
    else:
        logger.warning("No 'subject' column found for grouping; CV might leak data if using segments.")
        groups = pd.Series(range(len(df))) # Fallback (bad for segments)

    # 3. Prepare X and y
    drop_keys = [k for k in ("subject", "segment", "session") if k in df.columns]
    y = df[target_cols[0]] if len(target_cols) == 1 else df[target_cols]
    X = df.drop(columns=target_cols + drop_keys)

    # 4. Clean / Impute X
    na_drop_frac = float(analysis_cfg.get("na_drop_frac", 0.3))
    near_const_std = float(analysis_cfg.get("near_const_std", 1e-6))

    if na_drop_frac < 1.0:
        na_frac = X.isna().mean()
        drop_cols = na_frac[na_frac > na_drop_frac].index.tolist()
        if drop_cols:
            X = X.drop(columns=drop_cols)

    std = X.std(numeric_only=True)
    const_cols = std[std < near_const_std].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    if X.isna().sum().sum() > 0:
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)
        
        remaining_rows_na = X.isna().any(axis=1)
        if remaining_rows_na.any():
            valid_idx = ~remaining_rows_na
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            groups = groups.loc[valid_idx]

    return X, y, groups


def _optuna_search_classifier(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    groups: pd.Series,
    analysis_cfg: Dict[str, Any],
    embed_fn: Callable[[Any], np.ndarray],
) -> Dict[str, Any]:
    """Optuna search with support for GroupKFold."""
    try:
        import optuna
    except Exception:
        return {}

    base_name = (analysis_cfg.get("base_classifier") or "hist_gradient_boosting").lower()
    n_trials = int(analysis_cfg.get("optuna_trials", 30))
    random_state = analysis_cfg.get("random_state", 42)
    cv_strategy = analysis_cfg.get("cv_kwargs", {}).get("cv_strategy", "stratified")
    n_splits = analysis_cfg.get("cv_kwargs", {}).get("n_splits", 5)

    X_emb = embed_fn(X)
    if isinstance(y, pd.DataFrame):
        y_arr = y.iloc[:, 0].values
    else:
        y_arr = y.values
    
    groups_arr = groups.values if groups is not None else None

    if "group" in cv_strategy:
        if "stratified" in cv_strategy:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = GroupKFold(n_splits=n_splits)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        if base_name.startswith("hist"):
            from sklearn.ensemble import HistGradientBoostingClassifier
            params = dict(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                max_iter=trial.suggest_int("max_iter", 100, 500),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-5, 1e-1, log=True),
                class_weight="balanced",
            )
            clf = HistGradientBoostingClassifier(**params)
        elif base_name in {"rf", "random_forest"}:
            from sklearn.ensemble import RandomForestClassifier
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                class_weight="balanced",
                n_jobs=analysis_cfg.get("n_jobs", -1),
                random_state=random_state,
            )
            clf = RandomForestClassifier(**params)
        else:
            from sklearn.linear_model import LogisticRegression
            params = dict(
                C=trial.suggest_float("C", 1e-3, 10, log=True),
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
            )
            clf = LogisticRegression(**params)

        scores = cross_val_score(
            clf, X_emb, y_arr, groups=groups_arr, cv=cv,
            scoring="balanced_accuracy",
            n_jobs=analysis_cfg.get("n_jobs", -1),
            error_score="raise",
            verbose=2,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna best balanced_accuracy=%.4f with params=%s", study.best_value, study.best_params)

    return {
        "optuna_best_score": study.best_value,
        "optuna_best_params": study.best_params,
        "optuna_trials": n_trials,
        "optuna_base_classifier": base_name,
    }


def _save_prepared_Xy(cfg: Dict[str, Any], analysis_cfg: Dict[str, Any], 
                      X: pd.DataFrame, y: pd.Series | pd.DataFrame, groups: pd.Series) -> None:
    """Persist the exact X/y/groups used for the analysis."""
    try:
        results_dir = cfg.get("results_dir", ".")
        os.makedirs(os.path.join(results_dir, "prepared"), exist_ok=True)
        base_file = analysis_cfg.get("results_file") or f"{cfg.get('global_experiment_id')}_{analysis_cfg.get('id')}"
        csv_path = os.path.join(results_dir, "prepared", f"{base_file}_Xy.csv")
        
        df_to_save = X.copy()
        
        # Add target
        target_name = y.name if hasattr(y, "name") else "target"
        if target_name:
            df_to_save[target_name] = y
        else:
            df_to_save = df_to_save.join(y)
            
        # Add groups (subject)
        if groups is not None:
            df_to_save["group_id"] = groups

        df_to_save.to_csv(csv_path, index=False)
        logger.info("Saved prepared X/y to %s", csv_path)
    except Exception as exc:
        logger.warning("Could not save prepared X/y: %s", exc)


def _run_single_analysis(cfg: Dict[str, Any], analysis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one analysis (classification/regression)."""
    embeddings_df, embed_cfg_used = _load_embeddings_df(cfg, analysis_cfg)
    
    # Unpack groups here
    X_df, y, groups = _attach_targets(embeddings_df, analysis_cfg, embed_cfg_used)
    
    foundation_model = analysis_cfg.get("foundation_model") or cfg.get("foundation_model") or "cbramod"
    embed_fn = _build_embed_fn(foundation_model, analysis_cfg.get("embedding", {}))

    task = analysis_cfg.get("task", "classification").lower()
    analysis_type = analysis_cfg.get("analysis_type", "baseline")
    logger.info(
        "Running analysis id=%s model=%s task=%s on X=%s, groups=%d",
        analysis_cfg.get("id"), foundation_model, task, X_df.shape, groups.nunique()
    )

    common_kwargs = dict(
        embed_fn=embed_fn,
        metrics=analysis_cfg.get("metrics"),
        hp_search_params=analysis_cfg.get("hp_search_params"),
        use_scaler=analysis_cfg.get("use_scaler", False),
        base_classifier=_resolve_base_classifier(analysis_cfg),
        random_state=analysis_cfg.get("random_state", cfg.get("random_state", 42)),
        n_jobs=analysis_cfg.get("n_jobs", cfg.get("n_jobs", -1)),
        cv_kwargs=analysis_cfg.get("cv_kwargs", cfg.get("cv_kwargs")),
        verbose=analysis_cfg.get("verbose", True),
    )
    common_kwargs["groups"] = groups

    if task == "classification":
        pipeline = FoundationClassificationPipeline(X=X_df, y=y, **common_kwargs)
    elif task == "regression":
        pipeline = FoundationRegressionPipeline(X=X_df, y=y, **common_kwargs)
    else:
        raise ValueError(f"Unknown task '{task}'")

    results: Dict[str, Any] = {}
    
    # Run Optuna with groups
    if analysis_cfg.get("optuna_trials"):
        results.update(
            _optuna_search_classifier(
                X=X_df, y=y, groups=groups,
                analysis_cfg=analysis_cfg, embed_fn=embed_fn,
            )
        )

    # Run Pipeline
    results.update(
        pipeline.run(
            analysis_type=analysis_type,
            n_features=analysis_cfg.get("n_features"),
            direction=analysis_cfg.get("direction", "forward"),
            search_type=analysis_cfg.get("search_type", "grid"),
            n_iter=analysis_cfg.get("n_iter", 50),
            scoring=analysis_cfg.get("scoring"),
        )
    )

    _save_prepared_Xy(cfg, analysis_cfg, X_df, y, groups)
    return results


def _save_results_csv(all_results: Dict[str, Any], out_path: str) -> None:
    """Flatten results dictionary and save as a human-readable CSV."""
    rows = []
    
    for analysis_id, res_data in all_results.items():
        row = {"analysis_id": analysis_id}
        
        # Flatten dictionary
        if isinstance(res_data, dict):
            for k, v in res_data.items():
                # Handle arrays (like cv scores) -> Mean/Std
                if k.startswith("test_") and hasattr(v, "mean"):
                    row[f"{k}_mean"] = float(v.mean())
                    row[f"{k}_std"] = float(v.std())
                # Handle best params (dict) -> string
                elif isinstance(v, dict):
                    row[k] = str(v)
                # Scalars (accuracy numbers, best scores)
                elif isinstance(v, (int, float, str)):
                    row[k] = v
                # Lists
                elif isinstance(v, list):
                    row[k] = str(v)
        
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info("Saved results summary CSV to %s", out_path)
    else:
        logger.warning("No results to save to CSV.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run foundation-model DL analyses from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    defaults = cfg.get("defaults", {}) or {}
    all_results: Dict[str, Any] = {}

    for analysis in cfg.get("analyses", []):
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)
        try:
            results = _run_single_analysis(cfg, analysis_cfg)
            all_results[analysis_cfg.get("id")] = results
        except Exception as e:
            logger.error(f"Analysis {analysis_cfg.get('id')} failed: {e}", exc_info=True)

    out_dir = cfg.get("results_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV summary only (skip pickle to keep results human-readable)
    csv_path = os.path.join(out_dir, f"{cfg.get('global_experiment_id', 'dl_run')}_summary.csv")
    _save_results_csv(all_results, csv_path)
    logger.info("Results saved to CSV (pickle output disabled): %s", csv_path)


if __name__ == "__main__":
    main()

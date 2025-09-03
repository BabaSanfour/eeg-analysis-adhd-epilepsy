#!/usr/bin/env python3
import argparse
import logging
import os
from copy import deepcopy
import inspect

import yaml
import pandas as pd
import numpy as np
import itertools

from coco_pipe.io import load, select_features, balance_dataset
from coco_pipe.ml.pipeline import MLPipeline
from coco_pipe.io import clean_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(X, y, groups, analysis_cfg):
    """
    Run one or more MLPipeline runs according to analysis_cfg.

    Supports three multivariate variants (all / per-sensor / per-feature) plus
    univariate mode inside each slice handled by MLPipeline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix, shape (n_samples, n_sensors * n_features) if DataFrame,
        or generic 2D array.
    y : pd.Series or np.ndarray
        Target vector or matrix.
    groups : pd.Series or np.ndarray
        Group labels for cross-validation (optional).
    analysis_cfg : dict
        Must include keys:
          - task, analysis_type, models, metrics, cv_kwargs, n_features, direction,
            search_type, n_iter, scoring, n_jobs, save_intermediate,
            results_dir, results_file, mode
        And for slicing:
          - spatial_units : list of sensor names or "all"
          - feature_names : list of feature names or "all"
          - analysis_unit: one of "all", "sensor", "feature"
          - sep           : string separator in column names
          - reverse       : bool, if True swap sensor/feature in names
    """
    # 1) Prepare X as DataFrame for easy column-based slicing
    X_df = X
    y_arr = y.values if hasattr(y, "values") else y
    groups_arr = groups.values if hasattr(groups, "values") else groups

    # 2) Extract slicing parameters
    spatial_units = analysis_cfg.get("spatial_units", "all")
    feature_names = analysis_cfg.get("feature_names", "all")
    sep = analysis_cfg.get("sep", "_")
    reverse = analysis_cfg.get("reverse", False)
    unit = analysis_cfg.get("analysis_unit", "all")  # "all", "sensor", or "feature"

    # 3) Build base config for MLPipeline (exclude X, y, groups)
    base_cfg = {
        "task":            analysis_cfg.get("task"),
        "analysis_type":   analysis_cfg.get("analysis_type"),
        "models":          analysis_cfg.get("models"),
        "metrics":         analysis_cfg.get("metrics"),
        "cv_strategy":     analysis_cfg.get("cv_kwargs", {}).get("cv_strategy"),
        "n_splits":        analysis_cfg.get("cv_kwargs", {}).get("n_splits"),
        "cv_kwargs":       analysis_cfg.get("cv_kwargs"),
        "n_features":      analysis_cfg.get("n_features"),
        "direction":       analysis_cfg.get("direction"),
        "search_type":     analysis_cfg.get("search_type"),
        "n_iter":          analysis_cfg.get("n_iter"),
        "scoring":         analysis_cfg.get("scoring"),
        "n_jobs":          analysis_cfg.get("n_jobs"),
        "save_intermediate": analysis_cfg.get("save_intermediate", True),
        "results_dir":     analysis_cfg.get("results_dir"),
        "results_file":    analysis_cfg.get("results_file"),
        "mode":            analysis_cfg.get("mode", "multivariate"),
    }
    # Drop any None values so defaults inside MLPipeline apply
    base_cfg = {k: v for k, v in base_cfg.items() if v is not None}

    logger.info(
        "Launching '%s' (%s, unit=%s) on data %s",
        base_cfg["task"],
        base_cfg["mode"],
        unit,
        X_df.shape,
    )

    # Helper to run one slice of X through MLPipeline
    def _run_slice(X_sub, label):
        logger.info("  • pipeline slice=%r, shape=%s", label, X_sub.shape)
        X_arr = X_sub
        # copy base config and update results_file to include slice label
        cfg_slice = deepcopy(base_cfg)
        # amend results filename if specified
        if "results_file" in cfg_slice:
            base_name, ext = os.path.splitext(cfg_slice["results_file"])
            cfg_slice["results_file"] = f"{base_name}_{label}{ext}"
        pipeline = MLPipeline(
            X=X_arr,
            y=y_arr,
            groups=groups_arr,
            config=cfg_slice
        )
        return pipeline.run()

    # 4) Build map of label → DataFrame slice
    slice_map = {}

    if unit == "all":
        slice_map["all"] = X_df

    elif unit == "sensor":
        # one run per sensor name in spatial_units
        for sensor in spatial_units:
            if not reverse:
                cols = [c for c in X_df.columns if c.startswith(f"{sensor}{sep}")]
            else:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{sensor}")]
            if not cols:
                logger.warning("No columns found for sensor=%r", sensor)
            else:
                slice_map[sensor] = X_df[cols]

    elif unit == "feature":
        # one run per feature name in feature_names
        for feat in feature_names:
            if not reverse:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{feat}")]
            else:
                cols = [c for c in X_df.columns if f"{feat}{sep}" in c]
            if not cols:
                logger.warning("No columns found for feature=%r", feat)
            else:
                slice_map[feat] = X_df[cols]

    else:
        raise ValueError(
            "Invalid analysis_unit %r; must be one of 'all', 'sensor', 'feature'",
            unit
        )

    if not slice_map:
        raise ValueError(f"No valid slices generated for unit={unit!r}")

    # 5) Run each slice and collect results
    results = {
        label: _run_slice(X_sub, label)
        for label, X_sub in slice_map.items()
    }

    # 6) If only the "all" slice was requested, unwrap the dict
    if list(slice_map.keys()) == ["all"]:
        return results["all"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ML analyses as defined in a YAML config")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML file with defaults + analyses"
    )
    args = parser.parse_args()

    # 0) Load global config and data
    cfg = yaml.safe_load(open(args.config, "r"))
    df = load("tabular", cfg["data_path"])
    all_results = {}
    defaults = cfg.get("defaults", {})

    # 1) Loop over each analysis block
    for analysis in cfg["analyses"]:
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)
        feature_names = analysis_cfg.get("feature_names", "all")
        feature_names = list(feature_names)
        feature_names = [f"feature-{feat}" for feat in feature_names]
        # 2) Feature selection & target extraction
        X, y = select_features(
            df,
            target_columns=analysis_cfg["target_columns"],
            covariates=analysis_cfg.get("covariates"),
            spatial_units=analysis_cfg.get("spatial_units"),
            feature_names=feature_names,
            row_filter=analysis_cfg.get("row_filter"),
            sep=analysis_cfg.get("sep", ".spaces-"),
            reverse=analysis_cfg.get("reverse", False),
            verbose=True,
        )

        logger.info(
            "Analysis %r selected %d features × %d samples, target=%r",
            analysis["id"], X.shape[1], X.shape[0], getattr(y, "name", None)
        )
        logger.info("  First features: %s", X.columns.tolist()[:5])

        # 2b) Optional class balancing
        bal_cfg = analysis_cfg.get("balance")
        if bal_cfg:
            tcols = analysis_cfg.get("target_columns")
            target_col = (
                (tcols[0] if isinstance(tcols, (list, tuple)) else tcols)
                if tcols is not None else (getattr(y, "name", None) or "target")
            )

            df_bal = X.copy()
            df_bal[target_col] = y

            covs = bal_cfg.get("covariates", analysis_cfg.get("covariates"))
            if covs:
                try:
                    df_bal = df_bal.join(df.loc[X.index, covs])
                except Exception as e:
                    logger.warning("Could not join covariates for balancing: %s", e)

            logger.info("Class distribution before balance:\n%s", df_bal[target_col].value_counts())
            prefer_flag = bal_cfg.get("prefer_clean_rows", bal_cfg.get("prefer_clean", True))
            bd_kwargs = dict(
                df=df_bal,
                target=target_col,
                strategy=bal_cfg.get("strategy", "undersample"),
                covariates=covs,
                n_bins=bal_cfg.get("qbins", bal_cfg.get("n_bins", 5)),
                binning=bal_cfg.get("binning", "quantile"),
                random_state=bal_cfg.get("seed", analysis_cfg.get("cv_kwargs", {}).get("random_state", 42)),
                grid_balance=analysis_cfg.get("grid_balance"),
                require_full_grid=analysis_cfg.get("require_full_grid"),
                prefer_clean=prefer_flag,
                prefer_clean_rows=prefer_flag,
            )
            sig = inspect.signature(balance_dataset)
            bd_kwargs = {k: v for k, v in bd_kwargs.items() if k in sig.parameters}
            balanced = balance_dataset(**bd_kwargs)
            logger.info("Class distribution after balance:\n%s", balanced[target_col].value_counts())

            y = balanced[target_col]
            X = balanced[X.columns]
            
            # 1.5) Optional: clean features based on config
            clean_cfg = analysis_cfg.get("clean_cfg") or analysis_cfg.get("clean_features")
            if clean_cfg:
                if isinstance(clean_cfg, dict):
                    mode = clean_cfg.get("mode", "any")
                    sep_clean = clean_cfg.get("sep", "_")
                    reverse = clean_cfg.get("reverse", False)
                    verbose_clean = clean_cfg.get("verbose", True)
                    min_abs_value = clean_cfg.get("min_abs_value", 1e-5)
                    min_abs_fraction = clean_cfg.get("min_abs_fraction", 0.1)
                else:
                    mode, sep_clean, reverse, verbose_clean = "any", "_", False, True
                    min_abs_value, min_abs_fraction = None, 0.0
                # Build kwargs and filter by signature for compatibility
                cf_kwargs = dict(
                    mode=mode,
                    sep=sep_clean,
                    reverse=reverse,
                    verbose=verbose_clean,
                    min_abs_value=min_abs_value,
                    min_abs_fraction=min_abs_fraction,
                )
                try:
                    sig_cf = inspect.signature(clean_features)
                    cf_kwargs = {k: v for k, v in cf_kwargs.items() if k in sig_cf.parameters}
                except Exception:
                    pass
                X, report = clean_features(X, **cf_kwargs)
                logger.info(
                    "Cleaned features (mode=%s): dropped %d columns; %d -> %d",
                    report.get("mode"),
                    len(report.get("dropped_columns", [])),
                    report.get("n_before"),
                    report.get("n_after"),
                )
        # 2d) Sync analysis_cfg.feature_names to remaining columns (useful for per-feature slicing)
        orig_feats = analysis_cfg.get("feature_names")
        if isinstance(orig_feats, (list, tuple)):
            sep_name = analysis_cfg.get("sep", "_")
            rev = analysis_cfg.get("reverse", False)
            present = set()
            for col in X.columns:
                if sep_name in col:
                    head = col.split(sep_name)[0] if rev else col.split(sep_name)[-1]
                    if head.startswith("feature-"):
                        head = head[len("feature-") :]
                    present.add(head)
            filtered = [f for f in orig_feats if f in present]
            if len(filtered) != len(orig_feats):
                analysis_cfg["feature_names"] = filtered
                logger.info(
                    "Filtered feature_names based on X columns: %d -> %d",
                    len(orig_feats), len(filtered)
                )
        # 3) Persist the exact X/y used for this analysis to CSV (after cleaning/balancing)
        try:
            tcols = analysis_cfg.get("target_columns")
            target_name = (
                (tcols[0] if isinstance(tcols, (list, tuple)) else tcols)
                if tcols is not None else (getattr(y, "name", None) or "target")
            )
            df_to_save = X.copy()
            df_to_save[target_name] = y
            out_root = cfg.get("results_dir", ".")
            os.makedirs(os.path.join(out_root, "prepared"), exist_ok=True)
            base_file = analysis_cfg.get("results_file") or f"{cfg['global_experiment_id']}_{analysis['id']}"
            csv_path = os.path.join(out_root, "prepared", f"{base_file}_Xy.csv")
            df_to_save.to_csv(csv_path, index=False)
            logger.info("Saved prepared X/y to %s", csv_path)
        except Exception as e:
            logger.warning("Could not save prepared X/y CSV: %s", e)

        # 4) Run the configured analysis
        groups = None

        results = run_analysis(X, y, groups, analysis_cfg)
        all_results[analysis["id"]] = results

    # 4) Save aggregated results
    out_dir = cfg.get("results_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg['global_experiment_id']}.pkl")
    logger.info("Saving all results to %r", out_path)
    pd.to_pickle(all_results, out_path)


if __name__ == "__main__":
    main()

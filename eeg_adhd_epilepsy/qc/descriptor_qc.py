"""Descriptor QC integrated into extraction and merge stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.descriptors.qc import (
    add_family_diagnostics,
    aggregate_family_qc,
    classify_descriptor_columns,
    compute_family_constant_summary,
    compute_family_missingness,
    summarize_failures,
)
from coco_pipe.io import read_json
from coco_pipe.io.quality import (
    compute_subject_outlier_burden as compute_shared_subject_outlier_burden,
)
from coco_pipe.io.quality import (
    make_qc_flag,
    resolve_qc_status,
)
from coco_pipe.report.descriptor_qc import (
    generate_descriptor_dataset_report,
    generate_descriptor_subject_report,
)

import eeg_adhd_epilepsy.io.bids as bids_io
import eeg_adhd_epilepsy.viz.descriptor_qc as viz_descriptor_qc
from eeg_adhd_epilepsy.qc.utils import DEFAULT_DESCRIPTOR_THRESHOLDS

LOGGER = logging.getLogger(__name__)


def run_descriptor_subject_qc(
    shard_root: Path,
    reports_root: Path,
    subject: str,
    session: str,
    condition: str,
    sensor_epoch_df: pd.DataFrame,
    sensor_subject_df: pd.DataFrame,
    sensor_epoch_feature_columns_path: Path,
    sensor_subject_feature_columns_path: Path,
    pooled_epoch_df: pd.DataFrame | None,
    pooled_subject_df: pd.DataFrame | None,
    pooled_epoch_feature_columns_path: Path | None,
    pooled_subject_feature_columns_path: Path | None,
    failure_df: pd.DataFrame,
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    qc_dir = shard_root / "qc"
    report_dir = bids_io.get_subject_session_stage_dir(
        reports_root=reports_root,
        subject_id=subject,
        session_id=session,
        stage="descriptor_qc",
        create_dir=True,
    )
    families_config = (
        (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    )
    expected_families = [
        family
        for family, config_key in (
            ("band", "bands"),
            ("param", "parametric"),
            ("complexity", "complexity"),
        )
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]
    feature_cols = [
        column
        for column in read_json(sensor_epoch_feature_columns_path)
        if any(
            str(column).startswith(f"{family}_") or f"_{family}_" in str(column)
            for family in expected_families
        )
        and str(column) in sensor_epoch_df.columns
    ]
    subject_feature_cols = [
        column
        for column in read_json(sensor_subject_feature_columns_path)
        if any(
            str(column).startswith(f"{family}_") or f"_{family}_" in str(column)
            for family in expected_families
        )
        and str(column) in sensor_subject_df.columns
    ]
    feature_df = (
        sensor_epoch_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
        if feature_cols
        else pd.DataFrame()
    )
    feature_missingness_df = compute_family_missingness(sensor_epoch_df, feature_cols)
    constant_df = compute_family_constant_summary(
        sensor_epoch_df,
        feature_cols,
        tol=DEFAULT_DESCRIPTOR_THRESHOLDS.near_constant_std_tol,
    )
    family_summary_df = aggregate_family_qc(
        sensor_epoch_df,
        feature_cols,
        failures_df=failure_df,
        tol=DEFAULT_DESCRIPTOR_THRESHOLDS.near_constant_std_tol,
    ).rename(
        columns={
            "missing_rate_mean": "missing_rate",
            "nonfinite_rate_mean": "nonfinite_rate",
        }
    )
    family_summary_df = add_family_diagnostics(
        family_summary_df,
        feature_missingness_df,
        feature_df,
    )
    failure_summaries = summarize_failures(failure_df)

    metrics = {
        "n_epochs": int(len(sensor_epoch_df)),
        "n_sensor_epoch_features": int(len(feature_cols)),
        "n_sensor_subject_rows": int(len(sensor_subject_df)),
        "n_sensor_subject_features": int(len(subject_feature_cols)),
        "n_pooled_epoch_rows": int(len(pooled_epoch_df)) if pooled_epoch_df is not None else 0,
        "n_pooled_subject_rows": int(len(pooled_subject_df))
        if pooled_subject_df is not None
        else 0,
        "n_failures_total": int(len(failure_df)),
        "nan_rate_sensor_epoch": float(feature_missingness_df["missing_rate"].mean())
        if not feature_missingness_df.empty
        else 0.0,
        "max_feature_missingness": float(feature_missingness_df["missing_rate"].max())
        if not feature_missingness_df.empty
        else 0.0,
        "n_all_nan_features": int(constant_df["is_all_nan"].sum()) if not constant_df.empty else 0,
        "n_constant_features": int(constant_df["is_constant"].sum())
        if not constant_df.empty
        else 0,
    }
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
        metrics[f"failure_rate_{family}"] = row.get("failure_rate", 0.0)

    flags: list[dict[str, Any]] = []
    if sensor_epoch_df.empty or sensor_subject_df.empty or not feature_cols:
        flags.append(
            make_qc_flag(
                "fail",
                "integrity_missing_outputs",
                "Sensor descriptor outputs are empty or missing.",
                scope="sensor",
            )
        )
    actual_families = set(
        feature_missingness_df.get("family", pd.Series(dtype=str)).dropna().astype(str)
    )
    for family in expected_families:
        if family not in actual_families:
            flags.append(
                make_qc_flag(
                    "fail",
                    "missing_expected_family",
                    f"Expected family '{family}' has no extracted columns.",
                    scope=family,
                )
            )
    if (
        metrics["nan_rate_sensor_epoch"] >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_nan_rate
        or metrics["max_feature_missingness"]
        >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_feature_missingness
    ):
        flags.append(
            make_qc_flag(
                "fail",
                "high_missingness",
                "Feature missingness is too high.",
                value=metrics["max_feature_missingness"],
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_feature_missingness,
            )
        )
    elif (
        metrics["nan_rate_sensor_epoch"] >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_nan_rate
        or metrics["max_feature_missingness"]
        >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_feature_missingness
    ):
        flags.append(
            make_qc_flag(
                "warn",
                "elevated_missingness",
                "Feature missingness is elevated.",
                value=metrics["max_feature_missingness"],
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_feature_missingness,
            )
        )
    if metrics["n_all_nan_features"] > 0:
        flags.append(
            make_qc_flag(
                "fail",
                "all_nan_features_present",
                "At least one descriptor feature is entirely NaN.",
                value=metrics["n_all_nan_features"],
                threshold=0,
            )
        )
    constant_fraction = metrics["n_constant_features"] / max(len(feature_cols), 1)
    if constant_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_zero_variance_fraction:
        flags.append(
            make_qc_flag(
                "fail",
                "many_constant_features",
                "Too many descriptor features are constant.",
                value=constant_fraction,
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_zero_variance_fraction,
            )
        )
    elif constant_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_zero_variance_fraction:
        flags.append(
            make_qc_flag(
                "warn",
                "constant_features_present",
                "Some descriptor features are constant.",
                value=constant_fraction,
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_zero_variance_fraction,
            )
        )
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
        failure_rate = float(row.get("failure_rate") or 0.0)
        if failure_rate >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "fail",
                    f"{family}_family_failure_high",
                    "Family failure rate is high.",
                    value=failure_rate,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_family_failure_rate,
                    scope=family,
                )
            )
        elif failure_rate >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "warn",
                    f"{family}_family_failure_warn",
                    "Family failure rate is elevated.",
                    value=failure_rate,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_family_failure_rate,
                    scope=family,
                )
            )
        if family == "band" and float(row.get("band_rel_out_of_range_rate") or 0.0) > 0:
            flags.append(
                make_qc_flag(
                    "warn",
                    "relative_power_out_of_range",
                    "Some relative band power features are outside [0, 1].",
                    value=row.get("band_rel_out_of_range_rate"),
                    threshold=0.0,
                    scope=family,
                )
            )
        if family == "param":
            r2_p05 = row.get("param_r_squared_p05")
            if pd.notna(r2_p05) and float(r2_p05) < 0.2:
                flags.append(
                    make_qc_flag(
                        "fail",
                        "param_low_r_squared",
                        "Parametric fits show very low r-squared.",
                        value=r2_p05,
                        threshold=0.2,
                        scope=family,
                    )
                )
            elif pd.notna(r2_p05) and float(r2_p05) < 0.5:
                flags.append(
                    make_qc_flag(
                        "warn",
                        "param_low_r_squared",
                        "Parametric fits show low r-squared.",
                        value=r2_p05,
                        threshold=0.5,
                        scope=family,
                    )
                )
        if family == "complexity" and int(row.get("n_constant_features") or 0) > 0:
            flags.append(
                make_qc_flag(
                    "warn",
                    "complexity_measure_collapse",
                    "Some complexity features are constant.",
                    value=row.get("n_constant_features"),
                    threshold=0,
                    scope=family,
                )
            )

    qc_status = resolve_qc_status(flags)
    report_path = bids_io.get_subject_session_stage_report_path(
        reports_root=reports_root,
        subject_id=subject,
        session_id=session,
        stage="descriptor_qc",
        report_stem=f"sub-{subject}_ses-{session}_{condition}",
        create_dir=True,
    )
    summary_row = {
        "subject": subject,
        "session": session,
        "condition": condition,
        "qc_status": qc_status,
        **metrics,
        "report_path": str(report_path),
    }

    overview_df = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Session": session,
                "Condition": condition,
                "Epochs": metrics["n_epochs"],
                "Sensor Features": metrics["n_sensor_epoch_features"],
                "Sensor Subject Features": metrics["n_sensor_subject_features"],
                "Pooled Outputs Present": bool(
                    pooled_epoch_df is not None and pooled_subject_df is not None
                ),
                "Failures": metrics["n_failures_total"],
                "QC Status": qc_status,
            }
        ]
    )
    flags_df = pd.DataFrame(flags)
    figure_paths = viz_descriptor_qc.save_subject_descriptor_qc_figures(
        figures_dir=report_dir / "figures",
        family_summary_df=family_summary_df,
        failure_summary_df=failure_summaries["combined"],
        feature_missingness_df=feature_missingness_df,
        epoch_feature_df=feature_df,
    )

    qc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_row]).to_csv(qc_dir / "summary_row.csv", index=False)
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        qc_dir / "summary_metrics.csv", index=False
    )
    flags_df.to_csv(qc_dir / "flags.csv", index=False)
    failure_summaries["combined"].to_csv(qc_dir / "failure_summary.csv", index=False)
    feature_missingness_df.to_csv(qc_dir / "feature_missingness.csv", index=False)
    family_summary_df.to_csv(qc_dir / "family_summary.csv", index=False)
    generate_descriptor_subject_report(
        output_path=report_path,
        overview_df=overview_df,
        flags_df=flags_df,
        failure_summary_df=failure_summaries["combined"],
        feature_missingness_df=feature_missingness_df,
        family_summary_df=family_summary_df,
        figure_paths=figure_paths,
        asset_urls="inline",
    )
    return summary_row


def run_descriptor_dataset_qc(
    derivative_root: Path,
    reports_root: Path,
    merged_sensor_epoch_df: pd.DataFrame | None,
    merged_sensor_subject_df: pd.DataFrame,
    merged_sensor_epoch_feature_columns_path: Path | None,
    merged_sensor_subject_feature_columns_path: Path,
    merged_pooled_epoch_df: pd.DataFrame | None,
    merged_pooled_subject_df: pd.DataFrame | None,
    merged_pooled_epoch_feature_columns_path: Path | None,
    merged_pooled_subject_feature_columns_path: Path | None,
    shard_qc_rows_df: pd.DataFrame | None,
    merged_failures_df: pd.DataFrame | None,
    config_snapshot: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del (
        derivative_root,
        merged_pooled_epoch_df,
        merged_pooled_subject_df,
        merged_pooled_epoch_feature_columns_path,
        merged_pooled_subject_feature_columns_path,
    )
    qc_dir = Path(merged_sensor_subject_df.attrs.get("qc_dir", ""))
    if not str(qc_dir):
        raise ValueError(
            "Merged sensor subject table must carry qc_dir in attrs for descriptor QC."
        )
    summary_dir = bids_io.get_stage_summary_dir(reports_root, "descriptor_qc", create_dir=True)

    families_config = (
        (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    )
    expected_families = [
        family
        for family, config_key in (
            ("band", "bands"),
            ("param", "parametric"),
            ("complexity", "complexity"),
        )
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]
    subject_feature_cols = [
        column
        for column in read_json(merged_sensor_subject_feature_columns_path)
        if any(
            str(column).startswith(f"{family}_") or f"_{family}_" in str(column)
            for family in expected_families
        )
        and str(column) in merged_sensor_subject_df.columns
    ]
    epoch_feature_cols = [
        column
        for column in (
            read_json(merged_sensor_epoch_feature_columns_path)
            if merged_sensor_epoch_feature_columns_path is not None
            else []
        )
        if any(
            str(column).startswith(f"{family}_") or f"_{family}_" in str(column)
            for family in expected_families
        )
        and merged_sensor_epoch_df is not None
        and str(column) in merged_sensor_epoch_df.columns
    ]
    feature_missingness_df = compute_family_missingness(
        merged_sensor_subject_df,
        subject_feature_cols,
    )
    constant_df = compute_family_constant_summary(
        merged_sensor_subject_df,
        subject_feature_cols,
        tol=DEFAULT_DESCRIPTOR_THRESHOLDS.near_constant_std_tol,
    )
    low_variance_df = constant_df[
        constant_df["is_constant"]
        | (
            constant_df["std"].fillna(np.inf)
            <= DEFAULT_DESCRIPTOR_THRESHOLDS.near_constant_std_tol * 10
        )
    ].copy()
    failure_df = merged_failures_df if merged_failures_df is not None else pd.DataFrame()
    failure_summaries = summarize_failures(failure_df)
    subject_feature_df = (
        merged_sensor_subject_df.loc[:, subject_feature_cols].replace([np.inf, -np.inf], np.nan)
        if subject_feature_cols
        else pd.DataFrame()
    )
    family_summary_df = aggregate_family_qc(
        merged_sensor_subject_df,
        subject_feature_cols,
        failures_df=failure_df,
        tol=DEFAULT_DESCRIPTOR_THRESHOLDS.near_constant_std_tol,
    ).rename(
        columns={
            "missing_rate_mean": "missing_rate",
            "nonfinite_rate_mean": "nonfinite_rate",
        }
    )
    family_summary_df = add_family_diagnostics(
        family_summary_df,
        feature_missingness_df,
        subject_feature_df,
    )
    outlier_df = compute_shared_subject_outlier_burden(
        merged_sensor_subject_df,
        subject_feature_cols,
    )
    distribution_rows: list[dict[str, Any]] = []
    for row in classify_descriptor_columns(subject_feature_cols).to_dict("records"):
        column = str(row["column"])
        numeric = pd.to_numeric(merged_sensor_subject_df[column], errors="coerce")
        distribution_rows.append(
            {
                "column": column,
                "family": row["family"],
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std()) if numeric.notna().any() else np.nan,
                "min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "median": float(numeric.median()) if numeric.notna().any() else np.nan,
                "max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
    distribution_df = pd.DataFrame(distribution_rows)

    condition_breakdown_rows: list[dict[str, Any]] = []
    if "condition" in merged_sensor_subject_df.columns:
        failure_has_condition = "condition" in failure_df.columns
        failure_has_family = "family" in failure_df.columns
        for condition, group in merged_sensor_subject_df.groupby("condition"):
            cond_missingness = (
                compute_family_missingness(group, subject_feature_cols)
                if subject_feature_cols
                else pd.DataFrame()
            )
            cond_failures = (
                failure_df[failure_df["condition"] == condition]
                if failure_has_condition
                else pd.DataFrame()
            )
            for family in expected_families:
                family_missingness = (
                    cond_missingness[cond_missingness["family"] == family]["missing_rate"]
                    if not cond_missingness.empty and "family" in cond_missingness.columns
                    else pd.Series(dtype=float)
                )
                n_failures = (
                    int((cond_failures["family"] == family).sum())
                    if not cond_failures.empty and failure_has_family
                    else 0
                )
                condition_breakdown_rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "n_subjects": int(group["subject"].nunique())
                        if "subject" in group.columns
                        else int(len(group)),
                        "mean_missing_rate": float(family_missingness.mean())
                        if not family_missingness.empty
                        else np.nan,
                        "n_failures": n_failures,
                    }
                )
    condition_breakdown_df = pd.DataFrame(condition_breakdown_rows)

    shard_summary_df = (
        shard_qc_rows_df.copy()
        if shard_qc_rows_df is not None
        else pd.DataFrame(columns=["subject", "session", "condition", "qc_status"])
    )
    metrics = {
        "n_shards": int(len(shard_summary_df)),
        "n_shards_pass": int(
            (shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "pass").sum()
        )
        if not shard_summary_df.empty
        else 0,
        "n_shards_warn": int(
            (shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "warn").sum()
        )
        if not shard_summary_df.empty
        else 0,
        "n_shards_fail": int(
            (shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "fail").sum()
        )
        if not shard_summary_df.empty
        else 0,
        "n_failures_total": int(len(failure_df)),
        "n_subjects": int(merged_sensor_subject_df["subject"].nunique())
        if "subject" in merged_sensor_subject_df.columns
        else int(len(merged_sensor_subject_df)),
        "n_conditions": int(merged_sensor_subject_df["condition"].nunique())
        if "condition" in merged_sensor_subject_df.columns
        else 0,
        "n_sensor_epoch_features": int(len(epoch_feature_cols)),
        "n_sensor_subject_features": int(len(subject_feature_cols)),
        "n_all_nan_features": int(constant_df["is_all_nan"].sum()) if not constant_df.empty else 0,
        "n_zero_variance_features": int(constant_df["is_constant"].sum())
        if not constant_df.empty
        else 0,
        "n_near_zero_variance_features": int(len(low_variance_df)),
        "max_feature_missingness": float(feature_missingness_df["missing_rate"].max())
        if not feature_missingness_df.empty
        else 0.0,
        "median_feature_missingness": float(feature_missingness_df["missing_rate"].median())
        if not feature_missingness_df.empty
        else 0.0,
        "n_subjects_high_outlier_burden": int(
            (
                outlier_df["outlier_fraction"]
                >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_subject_outlier_fraction
            ).sum()
        )
        if not outlier_df.empty
        else 0,
    }

    flags: list[dict[str, Any]] = []
    if metrics["n_shards_fail"] > 0:
        flags.append(
            make_qc_flag(
                "warn",
                "failed_shards_present",
                "Some descriptor shards failed QC.",
                value=metrics["n_shards_fail"],
                threshold=0,
            )
        )
    actual_families = set(
        feature_missingness_df.get("family", pd.Series(dtype=str)).dropna().astype(str)
    )
    for family in expected_families:
        if family not in actual_families:
            flags.append(
                make_qc_flag(
                    "fail",
                    "missing_expected_family",
                    f"Expected family '{family}' has no merged columns.",
                    scope=family,
                )
            )
    if metrics["max_feature_missingness"] >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_feature_missingness:
        flags.append(
            make_qc_flag(
                "fail",
                "high_global_missingness",
                "Global descriptor missingness is too high.",
                value=metrics["max_feature_missingness"],
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_feature_missingness,
            )
        )
    elif (
        metrics["max_feature_missingness"] >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_feature_missingness
    ):
        flags.append(
            make_qc_flag(
                "warn",
                "global_missingness_warn",
                "Global descriptor missingness is elevated.",
                value=metrics["max_feature_missingness"],
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_feature_missingness,
            )
        )
    zero_variance_fraction = metrics["n_zero_variance_features"] / max(len(subject_feature_cols), 1)
    if zero_variance_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_zero_variance_fraction:
        flags.append(
            make_qc_flag(
                "fail",
                "many_zero_variance_features",
                "Too many merged features are zero-variance.",
                value=zero_variance_fraction,
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_zero_variance_fraction,
            )
        )
    elif zero_variance_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_zero_variance_fraction:
        flags.append(
            make_qc_flag(
                "warn",
                "near_zero_variance_features",
                "Merged features include near-zero variance columns.",
                value=zero_variance_fraction,
                threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_zero_variance_fraction,
            )
        )
    if not outlier_df.empty:
        max_outlier_fraction = float(outlier_df["outlier_fraction"].max())
        if max_outlier_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_subject_outlier_fraction:
            flags.append(
                make_qc_flag(
                    "fail",
                    "subject_outlier_burden_high",
                    "A subject has high descriptor outlier burden.",
                    value=max_outlier_fraction,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_subject_outlier_fraction,
                )
            )
        elif max_outlier_fraction >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_subject_outlier_fraction:
            flags.append(
                make_qc_flag(
                    "warn",
                    "subject_outlier_burden_warn",
                    "A subject has elevated descriptor outlier burden.",
                    value=max_outlier_fraction,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_subject_outlier_fraction,
                )
            )
    for row in family_summary_df.to_dict("records"):
        failure_rate = float(row.get("failure_rate") or 0.0)
        family = str(row["family"])
        if failure_rate >= DEFAULT_DESCRIPTOR_THRESHOLDS.fail_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "fail",
                    "family_failure_concentration",
                    "Family failure concentration is high.",
                    value=failure_rate,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.fail_family_failure_rate,
                    scope=family,
                )
            )
        elif failure_rate >= DEFAULT_DESCRIPTOR_THRESHOLDS.warn_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "warn",
                    "family_failure_concentration",
                    "Family failure concentration is elevated.",
                    value=failure_rate,
                    threshold=DEFAULT_DESCRIPTOR_THRESHOLDS.warn_family_failure_rate,
                    scope=family,
                )
            )

    qc_status = resolve_qc_status(flags)
    report_path = summary_dir / "descriptor_qc_dataset_summary.html"
    overview_df = pd.DataFrame(
        [
            {
                "Subjects": metrics["n_subjects"],
                "Shards": metrics["n_shards"],
                "Conditions": metrics["n_conditions"],
                "Sensor Subject Rows": len(merged_sensor_subject_df),
                "Sensor Epoch Rows": len(merged_sensor_epoch_df)
                if merged_sensor_epoch_df is not None
                else 0,
                "Sensor Subject Features": metrics["n_sensor_subject_features"],
                "Sensor Epoch Features": metrics["n_sensor_epoch_features"],
                "Failures": metrics["n_failures_total"],
                "QC Status": qc_status,
            }
        ]
    )
    flags_df = pd.DataFrame(flags)
    figure_paths = viz_descriptor_qc.save_dataset_descriptor_qc_figures(
        figures_dir=summary_dir / "figures",
        shard_summary_df=shard_summary_df,
        failure_family_df=failure_summaries["by_family"],
        failure_channel_df=failure_summaries["by_channel"],
        feature_missingness_df=feature_missingness_df,
        low_variance_df=low_variance_df,
    )

    qc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in {
                **metrics,
                "qc_status": qc_status,
                "report_path": str(report_path),
            }.items()
        ]
    ).to_csv(qc_dir / "dataset_summary_metrics.csv", index=False)
    flags_df.to_csv(qc_dir / "dataset_flags.csv", index=False)
    shard_summary_df.to_csv(qc_dir / "shard_qc_summary.csv", index=False)
    failure_summaries["by_family"].to_csv(qc_dir / "failure_summary_by_family.csv", index=False)
    failure_summaries["by_channel"].to_csv(qc_dir / "failure_summary_by_channel.csv", index=False)
    feature_missingness_df.to_csv(qc_dir / "feature_missingness.csv", index=False)
    distribution_df.to_csv(qc_dir / "feature_distribution_summary.csv", index=False)
    low_variance_df.to_csv(qc_dir / "low_variance_features.csv", index=False)
    if not condition_breakdown_df.empty:
        condition_breakdown_df.to_csv(qc_dir / "condition_breakdown.csv", index=False)

    manifest_df = None
    if manifest:
        manifest_df = pd.DataFrame(
            [
                {"field": key, "value": value}
                for key, value in manifest.items()
                if not isinstance(value, (list, dict))
            ]
        )

    generate_descriptor_dataset_report(
        output_path=report_path,
        overview_df=overview_df,
        shard_summary_df=shard_summary_df,
        flags_df=flags_df,
        failure_family_df=failure_summaries["by_family"],
        failure_channel_df=failure_summaries["by_channel"],
        feature_missingness_df=feature_missingness_df,
        low_variance_df=low_variance_df,
        family_summary_df=family_summary_df,
        figure_paths=figure_paths,
        manifest_df=manifest_df,
        condition_breakdown_df=condition_breakdown_df,
        asset_urls="inline",
    )

    return {
        "qc_status": qc_status,
        "report_path": str(report_path),
        **metrics,
    }

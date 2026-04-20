"""Descriptor QC integrated into extraction and merge stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import eeg_adhd_epilepsy.io.bids as bids_io
import eeg_adhd_epilepsy.reports.descriptor_qc as report_descriptor_qc
import eeg_adhd_epilepsy.viz.descriptor_qc as viz_descriptor_qc

LOGGER = logging.getLogger(__name__)

_WARN_NAN_RATE = 0.01
_FAIL_NAN_RATE = 0.20
_WARN_FEATURE_MISSINGNESS = 0.20
_FAIL_FEATURE_MISSINGNESS = 0.50
_WARN_FAMILY_FAILURE_RATE = 0.05
_FAIL_FAMILY_FAILURE_RATE = 0.25
_NEAR_CONSTANT_STD_TOL = 1e-12
_WARN_ZERO_VARIANCE_FRACTION = 0.01
_FAIL_ZERO_VARIANCE_FRACTION = 0.05
_WARN_SUBJECT_OUTLIER_FRACTION = 0.10
_FAIL_SUBJECT_OUTLIER_FRACTION = 0.25
_STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def _feature_bucket_df(feature_cols: list[str], config_snapshot: dict[str, Any]) -> pd.DataFrame:
    families_config = (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    expected_families = [
        family
        for family, config_key in (("band", "bands"), ("param", "parametric"), ("complexity", "complexity"))
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        family = next(
            (
                candidate
                for candidate in expected_families
                if column.startswith(f"{candidate}_") or f"_{candidate}_" in column
            ),
            None,
        )
        if family is None:
            continue
        scope_marker = "_chgrp-" if "_chgrp-" in column else "_ch-" if "_ch-" in column else ""
        scope = "sensor_group" if scope_marker == "_chgrp-" else "sensor" if scope_marker == "_ch-" else ""
        sensor = column.split(scope_marker, 1)[1] if scope_marker else ""
        band_subtype = None
        if family == "band":
            for token, label in (
                ("band_abs_", "band_abs"),
                ("_abs_", "band_abs"),
                ("band_corr_abs_", "band_corr_abs"),
                ("_corr_abs_", "band_corr_abs"),
                ("band_log_abs_", "band_log_abs"),
                ("_log_abs_", "band_log_abs"),
                ("band_rel_", "band_rel"),
                ("_rel_", "band_rel"),
                ("band_corr_rel_", "band_corr_rel"),
                ("_corr_rel_", "band_corr_rel"),
                ("band_ratio_", "band_ratio"),
                ("_ratio_", "band_ratio"),
                ("agg_band_ratio_", "agg_band_ratio"),
                ("agg_ratio_", "agg_band_ratio"),
                ("band_corr_ratio_", "band_corr_ratio"),
                ("_corr_ratio_", "band_corr_ratio"),
                ("agg_band_corr_ratio_", "agg_band_corr_ratio"),
                ("agg_corr_ratio_", "agg_band_corr_ratio"),
            ):
                if token in column or column.startswith(token):
                    band_subtype = label
                    break
            if band_subtype is None:
                band_subtype = "band_other"
        complexity_measure = None
        if family == "complexity":
            remainder = column[len("complexity_"):] if column.startswith("complexity_") else column.split("_complexity_", 1)[1]
            complexity_measure = remainder.split(scope_marker, 1)[0] if scope_marker else remainder
        parametric_metric = None
        if family == "param":
            remainder = column[len("param_"):] if column.startswith("param_") else column.split("_param_", 1)[1]
            parametric_metric = remainder.split(scope_marker, 1)[0] if scope_marker else remainder
        rows.append(
            {
                "column": column,
                "family": family,
                "scope": scope,
                "sensor": sensor,
                "band_subtype": band_subtype,
                "complexity_measure": complexity_measure,
                "parametric_metric": parametric_metric,
            }
        )
    return pd.DataFrame(rows)


def _make_qc_flag(
    level: str,
    code: str,
    message: str,
    value: float | int | str | None = None,
    threshold: float | int | str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    return {
        "level": level,
        "code": code,
        "message": message,
        "value": value,
        "threshold": threshold,
        "scope": scope or "",
    }


def compute_feature_missingness(
    df: pd.DataFrame,
    feature_cols: list[str],
    config_snapshot: dict[str, Any],
) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "family",
                "scope",
                "sensor",
                "missing_count",
                "missing_rate",
                "nonfinite_count",
                "nonfinite_rate",
                "band_subtype",
                "complexity_measure",
                "parametric_metric",
            ]
        )
    feature_bucket_df = _feature_bucket_df(feature_cols, config_snapshot)
    rows: list[dict[str, Any]] = []
    for row in feature_bucket_df.to_dict("records"):
        column = str(row["column"])
        series = df[column]
        missing_mask = series.isna()
        numeric = pd.to_numeric(series, errors="coerce")
        nonfinite_mask = numeric.notna() & ~np.isfinite(numeric)
        nonfinite_rate = float(nonfinite_mask.mean()) if len(series) else 0.0
        rows.append(
            {
                "column": column,
                "family": row["family"],
                "scope": row["scope"],
                "sensor": row["sensor"],
                "missing_count": int(missing_mask.sum()),
                "missing_rate": float(missing_mask.mean()) if len(series) else 0.0,
                "nonfinite_count": int(nonfinite_mask.fillna(False).sum()),
                "nonfinite_rate": nonfinite_rate,
                "band_subtype": row["band_subtype"],
                "complexity_measure": row["complexity_measure"],
                "parametric_metric": row["parametric_metric"],
            }
        )
    return pd.DataFrame(rows)


def compute_constant_feature_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    tol: float,
    config_snapshot: dict[str, Any],
) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame(columns=["column", "family", "std", "is_all_nan", "is_constant"])
    feature_bucket_df = _feature_bucket_df(feature_cols, config_snapshot)
    rows: list[dict[str, Any]] = []
    for row in feature_bucket_df.to_dict("records"):
        column = str(row["column"])
        numeric = pd.to_numeric(df[column], errors="coerce")
        std = float(numeric.std(skipna=True)) if numeric.notna().any() else float("nan")
        rows.append(
            {
                "column": column,
                "family": row["family"],
                "std": std,
                "is_all_nan": bool(numeric.isna().all()),
                "is_constant": bool(numeric.notna().any() and std <= tol),
            }
        )
    return pd.DataFrame(rows)


def compute_subject_outlier_burden(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if df.empty or not feature_cols:
        return pd.DataFrame(columns=["subject", "outlier_fraction", "n_outlier_features"])
    numeric_df = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    median = numeric_df.median(axis=0, skipna=True)
    mad = (numeric_df.sub(median, axis=1)).abs().median(axis=0, skipna=True)
    mad = mad.replace(0, np.nan)
    robust_z = numeric_df.sub(median, axis=1).abs().div(1.4826 * mad, axis=1)
    outlier_mask = robust_z > 5.0
    summary = pd.DataFrame(
        {
            "subject": df["subject"].astype(str).to_numpy() if "subject" in df.columns else np.arange(len(df)).astype(str),
            "outlier_fraction": outlier_mask.mean(axis=1).fillna(0.0).to_numpy(),
            "n_outlier_features": outlier_mask.sum(axis=1).astype(int).to_numpy(),
        }
    )
    return summary


def summarize_failures(failure_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def _group_frame(column: str) -> pd.DataFrame:
        if failure_df.empty or column not in failure_df.columns:
            return pd.DataFrame(columns=["value", "count"])
        grouped = (
            failure_df[column]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("value")
            .reset_index(name="count")
        )
        return grouped

    by_family = _group_frame("family")
    by_channel = _group_frame("channel_name")
    by_exception = _group_frame("exception_type")
    by_condition = _group_frame("condition")
    by_family_channel = (
        failure_df.fillna({"family": "unknown", "channel_name": "unknown"})
        .groupby(["family", "channel_name"], dropna=False)
        .size()
        .reset_index(name="count")
        if not failure_df.empty and {"family", "channel_name"}.issubset(failure_df.columns)
        else pd.DataFrame(columns=["family", "channel_name", "count"])
    )
    combined = pd.concat(
        [
            by_family.assign(group="family"),
            by_channel.assign(group="channel"),
            by_exception.assign(group="exception_type"),
            by_condition.assign(group="condition"),
        ],
        ignore_index=True,
    ) if any(not frame.empty for frame in (by_family, by_channel, by_exception, by_condition)) else pd.DataFrame(columns=["value", "count", "group"])
    return {
        "by_family": by_family,
        "by_channel": by_channel,
        "by_exception_type": by_exception,
        "by_condition": by_condition,
        "by_family_channel": by_family_channel,
        "combined": combined,
    }


def _family_summary(
    feature_missingness_df: pd.DataFrame,
    constant_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    if feature_missingness_df.empty or "family" not in feature_missingness_df.columns:
        return pd.DataFrame(columns=["family", "n_features", "missing_rate", "missing_rate_max", "nonfinite_rate", "n_all_nan_features", "n_constant_features", "failure_count", "failure_rate"])
    rows: list[dict[str, Any]] = []
    for family in sorted(feature_missingness_df["family"].dropna().astype(str).unique()):
        family_missingness = feature_missingness_df[feature_missingness_df["family"] == family]
        family_constants = constant_df[constant_df["family"] == family]
        family_failures = failure_df[failure_df["family"].astype(str) == family] if not failure_df.empty and "family" in failure_df.columns else pd.DataFrame()
        family_cols = family_missingness["column"].tolist()
        row: dict[str, Any] = {
            "family": family,
            "n_features": int(len(family_cols)),
            "missing_rate": float(family_missingness["missing_rate"].mean()) if not family_missingness.empty else 0.0,
            "missing_rate_max": float(family_missingness["missing_rate"].max()) if not family_missingness.empty else 0.0,
            "nonfinite_rate": float(family_missingness["nonfinite_rate"].mean()) if not family_missingness.empty else 0.0,
            "n_all_nan_features": int(family_constants["is_all_nan"].sum()) if not family_constants.empty else 0,
            "n_constant_features": int(family_constants["is_constant"].sum()) if not family_constants.empty else 0,
            "failure_count": int(len(family_failures)),
            "failure_rate": float(len(family_failures) / max(len(feature_df), 1)),
        }
        if family == "band":
            band_abs_cols = [column for column in family_cols if "_band_abs_" in f"_{column}_" or "band_abs_" in column or "band_corr_abs_" in column]
            rel_cols = [column for column in family_cols if "band_rel_" in column]
            corr_rel_cols = [column for column in family_cols if "band_corr_rel_" in column]
            ratio_cols = [column for column in family_cols if "ratio_" in column]
            row["band_abs_negative_rate"] = float((feature_df[band_abs_cols] < 0).stack().mean()) if band_abs_cols else 0.0
            row["band_rel_out_of_range_rate"] = float(((feature_df[rel_cols] < 0) | (feature_df[rel_cols] > 1)).stack().mean()) if rel_cols else 0.0
            row["band_corr_rel_out_of_range_rate"] = float(((feature_df[corr_rel_cols] < 0) | (feature_df[corr_rel_cols] > 1)).stack().mean()) if corr_rel_cols else 0.0
            row["band_ratio_nan_rate"] = float(feature_df[ratio_cols].isna().stack().mean()) if ratio_cols else 0.0
        elif family == "param":
            r2_cols = [column for column in family_cols if "param_r_squared_" in column]
            fit_error_cols = [column for column in family_cols if "param_fit_error_" in column]
            peak_cols = [column for column in family_cols if "peak" in column]
            alpha_peak_cols = [column for column in family_cols if "alpha_peak_freq" in column]
            row["param_r_squared_median"] = float(pd.to_numeric(feature_df[r2_cols].stack(), errors="coerce").median()) if r2_cols else np.nan
            row["param_r_squared_p05"] = float(pd.to_numeric(feature_df[r2_cols].stack(), errors="coerce").quantile(0.05)) if r2_cols else np.nan
            row["param_fit_error_median"] = float(pd.to_numeric(feature_df[fit_error_cols].stack(), errors="coerce").median()) if fit_error_cols else np.nan
            row["param_fit_error_p95"] = float(pd.to_numeric(feature_df[fit_error_cols].stack(), errors="coerce").quantile(0.95)) if fit_error_cols else np.nan
            row["param_peak_count_missing_rate"] = float(feature_df[peak_cols].isna().stack().mean()) if peak_cols else np.nan
            row["param_alpha_peak_freq_missing_rate"] = float(feature_df[alpha_peak_cols].isna().stack().mean()) if alpha_peak_cols else np.nan
        elif family == "complexity":
            row["complexity_measure_missingness_max"] = row["missing_rate_max"]
            row["complexity_measure_missingness_median"] = float(family_missingness["missing_rate"].median()) if not family_missingness.empty else 0.0
            row["complexity_nonfinite_rate"] = row["nonfinite_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


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
    families_config = (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    expected_families = [
        family
        for family, config_key in (("band", "bands"), ("param", "parametric"), ("complexity", "complexity"))
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]
    feature_cols = [
        column
        for column in json.loads(sensor_epoch_feature_columns_path.read_text(encoding="utf-8"))
        if any(str(column).startswith(f"{family}_") or f"_{family}_" in str(column) for family in expected_families)
        and str(column) in sensor_epoch_df.columns
    ]
    subject_feature_cols = [
        column
        for column in json.loads(sensor_subject_feature_columns_path.read_text(encoding="utf-8"))
        if any(str(column).startswith(f"{family}_") or f"_{family}_" in str(column) for family in expected_families)
        and str(column) in sensor_subject_df.columns
    ]
    feature_df = sensor_epoch_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan) if feature_cols else pd.DataFrame()
    feature_missingness_df = compute_feature_missingness(sensor_epoch_df, feature_cols, config_snapshot)
    constant_df = compute_constant_feature_summary(sensor_epoch_df, feature_cols, _NEAR_CONSTANT_STD_TOL, config_snapshot)
    family_summary_df = _family_summary(feature_missingness_df, constant_df, failure_df, feature_df)
    failure_summaries = summarize_failures(failure_df)

    metrics = {
        "n_epochs": int(len(sensor_epoch_df)),
        "n_sensor_epoch_features": int(len(feature_cols)),
        "n_sensor_subject_rows": int(len(sensor_subject_df)),
        "n_sensor_subject_features": int(len(subject_feature_cols)),
        "n_pooled_epoch_rows": int(len(pooled_epoch_df)) if pooled_epoch_df is not None else 0,
        "n_pooled_subject_rows": int(len(pooled_subject_df)) if pooled_subject_df is not None else 0,
        "n_failures_total": int(len(failure_df)),
        "nan_rate_sensor_epoch": float(feature_missingness_df["missing_rate"].mean()) if not feature_missingness_df.empty else 0.0,
        "max_feature_missingness": float(feature_missingness_df["missing_rate"].max()) if not feature_missingness_df.empty else 0.0,
        "n_all_nan_features": int(constant_df["is_all_nan"].sum()) if not constant_df.empty else 0,
        "n_constant_features": int(constant_df["is_constant"].sum()) if not constant_df.empty else 0,
    }
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
        metrics[f"failure_rate_{family}"] = row.get("failure_rate", 0.0)

    flags: list[dict[str, Any]] = []
    if sensor_epoch_df.empty or sensor_subject_df.empty or not feature_cols:
        flags.append(_make_qc_flag("fail", "integrity_missing_outputs", "Sensor descriptor outputs are empty or missing.", scope="sensor"))
    actual_families = set(feature_missingness_df.get("family", pd.Series(dtype=str)).dropna().astype(str))
    for family in expected_families:
        if family not in actual_families:
            flags.append(_make_qc_flag("fail", "missing_expected_family", f"Expected family '{family}' has no extracted columns.", scope=family))
    if metrics["nan_rate_sensor_epoch"] >= _FAIL_NAN_RATE or metrics["max_feature_missingness"] >= _FAIL_FEATURE_MISSINGNESS:
        flags.append(_make_qc_flag("fail", "high_missingness", "Feature missingness is too high.", value=metrics["max_feature_missingness"], threshold=_FAIL_FEATURE_MISSINGNESS))
    elif metrics["nan_rate_sensor_epoch"] >= _WARN_NAN_RATE or metrics["max_feature_missingness"] >= _WARN_FEATURE_MISSINGNESS:
        flags.append(_make_qc_flag("warn", "elevated_missingness", "Feature missingness is elevated.", value=metrics["max_feature_missingness"], threshold=_WARN_FEATURE_MISSINGNESS))
    if metrics["n_all_nan_features"] > 0:
        flags.append(_make_qc_flag("fail", "all_nan_features_present", "At least one descriptor feature is entirely NaN.", value=metrics["n_all_nan_features"], threshold=0))
    constant_fraction = metrics["n_constant_features"] / max(len(feature_cols), 1)
    if constant_fraction >= _FAIL_ZERO_VARIANCE_FRACTION:
        flags.append(_make_qc_flag("fail", "many_constant_features", "Too many descriptor features are constant.", value=constant_fraction, threshold=_FAIL_ZERO_VARIANCE_FRACTION))
    elif constant_fraction >= _WARN_ZERO_VARIANCE_FRACTION:
        flags.append(_make_qc_flag("warn", "constant_features_present", "Some descriptor features are constant.", value=constant_fraction, threshold=_WARN_ZERO_VARIANCE_FRACTION))
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
        failure_rate = float(row.get("failure_rate") or 0.0)
        if failure_rate >= _FAIL_FAMILY_FAILURE_RATE:
            flags.append(_make_qc_flag("fail", f"{family}_family_failure_high", "Family failure rate is high.", value=failure_rate, threshold=_FAIL_FAMILY_FAILURE_RATE, scope=family))
        elif failure_rate >= _WARN_FAMILY_FAILURE_RATE:
            flags.append(_make_qc_flag("warn", f"{family}_family_failure_warn", "Family failure rate is elevated.", value=failure_rate, threshold=_WARN_FAMILY_FAILURE_RATE, scope=family))
        if family == "band" and float(row.get("band_rel_out_of_range_rate") or 0.0) > 0:
            flags.append(_make_qc_flag("warn", "relative_power_out_of_range", "Some relative band power features are outside [0, 1].", value=row.get("band_rel_out_of_range_rate"), threshold=0.0, scope=family))
        if family == "param":
            r2_p05 = row.get("param_r_squared_p05")
            if pd.notna(r2_p05) and float(r2_p05) < 0.2:
                flags.append(_make_qc_flag("fail", "param_low_r_squared", "Parametric fits show very low r-squared.", value=r2_p05, threshold=0.2, scope=family))
            elif pd.notna(r2_p05) and float(r2_p05) < 0.5:
                flags.append(_make_qc_flag("warn", "param_low_r_squared", "Parametric fits show low r-squared.", value=r2_p05, threshold=0.5, scope=family))
        if family == "complexity" and int(row.get("n_constant_features") or 0) > 0:
            flags.append(_make_qc_flag("warn", "complexity_measure_collapse", "Some complexity features are constant.", value=row.get("n_constant_features"), threshold=0, scope=family))

    qc_status = "pass"
    for flag in flags:
        if _STATUS_ORDER[flag["level"]] > _STATUS_ORDER[qc_status]:
            qc_status = str(flag["level"])
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
                "Pooled Outputs Present": bool(pooled_epoch_df is not None and pooled_subject_df is not None),
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
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(qc_dir / "summary_metrics.csv", index=False)
    flags_df.to_csv(qc_dir / "flags.csv", index=False)
    failure_summaries["combined"].to_csv(qc_dir / "failure_summary.csv", index=False)
    feature_missingness_df.to_csv(qc_dir / "feature_missingness.csv", index=False)
    family_summary_df.to_csv(qc_dir / "family_summary.csv", index=False)
    report_descriptor_qc.generate_descriptor_subject_report(
        output_path=report_path,
        overview_df=overview_df,
        flags_df=flags_df,
        failure_summary_df=failure_summaries["combined"],
        feature_missingness_df=feature_missingness_df,
        family_summary_df=family_summary_df,
        figure_paths=figure_paths,
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
) -> dict[str, Any]:
    del derivative_root, merged_pooled_epoch_df, merged_pooled_subject_df, merged_pooled_epoch_feature_columns_path, merged_pooled_subject_feature_columns_path
    qc_dir = Path(merged_sensor_subject_df.attrs.get("qc_dir", ""))
    if not str(qc_dir):
        raise ValueError("Merged sensor subject table must carry qc_dir in attrs for descriptor QC.")
    summary_dir = bids_io.get_stage_summary_dir(reports_root, "descriptor_qc", create_dir=True)

    families_config = (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    expected_families = [
        family
        for family, config_key in (("band", "bands"), ("param", "parametric"), ("complexity", "complexity"))
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]
    subject_feature_cols = [
        column
        for column in json.loads(merged_sensor_subject_feature_columns_path.read_text(encoding="utf-8"))
        if any(str(column).startswith(f"{family}_") or f"_{family}_" in str(column) for family in expected_families)
        and str(column) in merged_sensor_subject_df.columns
    ]
    epoch_feature_cols = [
        column
        for column in (
            json.loads(merged_sensor_epoch_feature_columns_path.read_text(encoding="utf-8"))
            if merged_sensor_epoch_feature_columns_path is not None
            else []
        )
        if any(str(column).startswith(f"{family}_") or f"_{family}_" in str(column) for family in expected_families)
        and merged_sensor_epoch_df is not None
        and str(column) in merged_sensor_epoch_df.columns
    ]
    feature_missingness_df = compute_feature_missingness(merged_sensor_subject_df, subject_feature_cols, config_snapshot)
    constant_df = compute_constant_feature_summary(merged_sensor_subject_df, subject_feature_cols, _NEAR_CONSTANT_STD_TOL, config_snapshot)
    low_variance_df = constant_df[constant_df["is_constant"] | (constant_df["std"].fillna(np.inf) <= _NEAR_CONSTANT_STD_TOL * 10)].copy()
    failure_df = merged_failures_df if merged_failures_df is not None else pd.DataFrame()
    failure_summaries = summarize_failures(failure_df)
    family_summary_df = _family_summary(
        feature_missingness_df,
        constant_df,
        failure_df,
        merged_sensor_subject_df.loc[:, subject_feature_cols].replace([np.inf, -np.inf], np.nan) if subject_feature_cols else pd.DataFrame(),
    )
    outlier_df = compute_subject_outlier_burden(merged_sensor_subject_df, subject_feature_cols)
    distribution_rows: list[dict[str, Any]] = []
    for row in _feature_bucket_df(subject_feature_cols, config_snapshot).to_dict("records"):
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

    shard_summary_df = shard_qc_rows_df.copy() if shard_qc_rows_df is not None else pd.DataFrame(columns=["subject", "session", "condition", "qc_status"])
    metrics = {
        "n_shards": int(len(shard_summary_df)),
        "n_shards_pass": int((shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "pass").sum()) if not shard_summary_df.empty else 0,
        "n_shards_warn": int((shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "warn").sum()) if not shard_summary_df.empty else 0,
        "n_shards_fail": int((shard_summary_df.get("qc_status", pd.Series(dtype=str)) == "fail").sum()) if not shard_summary_df.empty else 0,
        "n_failures_total": int(len(failure_df)),
        "n_subjects": int(merged_sensor_subject_df["subject"].nunique()) if "subject" in merged_sensor_subject_df.columns else int(len(merged_sensor_subject_df)),
        "n_conditions": int(merged_sensor_subject_df["condition"].nunique()) if "condition" in merged_sensor_subject_df.columns else 0,
        "n_sensor_epoch_features": int(len(epoch_feature_cols)),
        "n_sensor_subject_features": int(len(subject_feature_cols)),
        "n_all_nan_features": int(constant_df["is_all_nan"].sum()) if not constant_df.empty else 0,
        "n_zero_variance_features": int(constant_df["is_constant"].sum()) if not constant_df.empty else 0,
        "n_near_zero_variance_features": int(len(low_variance_df)),
        "max_feature_missingness": float(feature_missingness_df["missing_rate"].max()) if not feature_missingness_df.empty else 0.0,
        "median_feature_missingness": float(feature_missingness_df["missing_rate"].median()) if not feature_missingness_df.empty else 0.0,
        "n_subjects_high_outlier_burden": int((outlier_df["outlier_fraction"] >= _WARN_SUBJECT_OUTLIER_FRACTION).sum()) if not outlier_df.empty else 0,
    }

    flags: list[dict[str, Any]] = []
    if metrics["n_shards_fail"] > 0:
        flags.append(_make_qc_flag("warn", "failed_shards_present", "Some descriptor shards failed QC.", value=metrics["n_shards_fail"], threshold=0))
    actual_families = set(feature_missingness_df.get("family", pd.Series(dtype=str)).dropna().astype(str))
    for family in expected_families:
        if family not in actual_families:
            flags.append(_make_qc_flag("fail", "missing_expected_family", f"Expected family '{family}' has no merged columns.", scope=family))
    if metrics["max_feature_missingness"] >= _FAIL_FEATURE_MISSINGNESS:
        flags.append(_make_qc_flag("fail", "high_global_missingness", "Global descriptor missingness is too high.", value=metrics["max_feature_missingness"], threshold=_FAIL_FEATURE_MISSINGNESS))
    elif metrics["max_feature_missingness"] >= _WARN_FEATURE_MISSINGNESS:
        flags.append(_make_qc_flag("warn", "global_missingness_warn", "Global descriptor missingness is elevated.", value=metrics["max_feature_missingness"], threshold=_WARN_FEATURE_MISSINGNESS))
    zero_variance_fraction = metrics["n_zero_variance_features"] / max(len(subject_feature_cols), 1)
    if zero_variance_fraction >= _FAIL_ZERO_VARIANCE_FRACTION:
        flags.append(_make_qc_flag("fail", "many_zero_variance_features", "Too many merged features are zero-variance.", value=zero_variance_fraction, threshold=_FAIL_ZERO_VARIANCE_FRACTION))
    elif zero_variance_fraction >= _WARN_ZERO_VARIANCE_FRACTION:
        flags.append(_make_qc_flag("warn", "near_zero_variance_features", "Merged features include near-zero variance columns.", value=zero_variance_fraction, threshold=_WARN_ZERO_VARIANCE_FRACTION))
    if not outlier_df.empty:
        max_outlier_fraction = float(outlier_df["outlier_fraction"].max())
        if max_outlier_fraction >= _FAIL_SUBJECT_OUTLIER_FRACTION:
            flags.append(_make_qc_flag("fail", "subject_outlier_burden_high", "A subject has high descriptor outlier burden.", value=max_outlier_fraction, threshold=_FAIL_SUBJECT_OUTLIER_FRACTION))
        elif max_outlier_fraction >= _WARN_SUBJECT_OUTLIER_FRACTION:
            flags.append(_make_qc_flag("warn", "subject_outlier_burden_warn", "A subject has elevated descriptor outlier burden.", value=max_outlier_fraction, threshold=_WARN_SUBJECT_OUTLIER_FRACTION))
    for row in family_summary_df.to_dict("records"):
        failure_rate = float(row.get("failure_rate") or 0.0)
        family = str(row["family"])
        if failure_rate >= _FAIL_FAMILY_FAILURE_RATE:
            flags.append(_make_qc_flag("fail", "family_failure_concentration", "Family failure concentration is high.", value=failure_rate, threshold=_FAIL_FAMILY_FAILURE_RATE, scope=family))
        elif failure_rate >= _WARN_FAMILY_FAILURE_RATE:
            flags.append(_make_qc_flag("warn", "family_failure_concentration", "Family failure concentration is elevated.", value=failure_rate, threshold=_WARN_FAMILY_FAILURE_RATE, scope=family))

    qc_status = "pass"
    for flag in flags:
        if _STATUS_ORDER[flag["level"]] > _STATUS_ORDER[qc_status]:
            qc_status = str(flag["level"])
    report_path = summary_dir / "descriptor_qc_dataset_summary.html"
    overview_df = pd.DataFrame(
        [
            {
                "Subjects": metrics["n_subjects"],
                "Shards": metrics["n_shards"],
                "Conditions": metrics["n_conditions"],
                "Sensor Subject Rows": len(merged_sensor_subject_df),
                "Sensor Epoch Rows": len(merged_sensor_epoch_df) if merged_sensor_epoch_df is not None else 0,
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
            for key, value in {**metrics, "qc_status": qc_status, "report_path": str(report_path)}.items()
        ]
    ).to_csv(qc_dir / "dataset_summary_metrics.csv", index=False)
    flags_df.to_csv(qc_dir / "dataset_flags.csv", index=False)
    shard_summary_df.to_csv(qc_dir / "shard_qc_summary.csv", index=False)
    failure_summaries["by_family"].to_csv(qc_dir / "failure_summary_by_family.csv", index=False)
    failure_summaries["by_channel"].to_csv(qc_dir / "failure_summary_by_channel.csv", index=False)
    feature_missingness_df.to_csv(qc_dir / "feature_missingness.csv", index=False)
    distribution_df.to_csv(qc_dir / "feature_distribution_summary.csv", index=False)
    low_variance_df.to_csv(qc_dir / "low_variance_features.csv", index=False)
    report_descriptor_qc.generate_descriptor_dataset_report(
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
    )

    return {
        "qc_status": qc_status,
        "report_path": str(report_path),
        **metrics,
    }

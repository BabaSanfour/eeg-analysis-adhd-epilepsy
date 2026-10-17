"""Descriptor QC integrated into extraction and merge stages."""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.descriptors.qc import (
    add_family_diagnostics,
    aggregate_family_qc,
    compute_family_constant_summary,
    compute_family_missingness,
    summarize_failures,
)
from coco_pipe.io.quality import (
    make_qc_flag,
    resolve_qc_status,
)
from coco_pipe.report.descriptor_qc import (
    generate_descriptor_dataset_report,
    generate_descriptor_subject_report,
)

import eeg_adhd_epilepsy.io.report_paths as report_paths
import eeg_adhd_epilepsy.viz.descriptor_qc as viz_descriptor_qc
from eeg_adhd_epilepsy.qc.utils import DescriptorQCThresholds
from eeg_adhd_epilepsy.utils.formatting import reorder_columns_front

LOGGER = logging.getLogger(__name__)

# Descriptor family token -> config sub-key that enables it.
_FAMILY_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("band", "bands"),
    ("param", "parametric"),
    ("complexity", "complexity"),
)


_PEAK_FEATURE = re.compile(r"_alpha_peak_|_peak_[a-z]+_dom")


def _is_peak_feature(column: object) -> bool:
    """A peak feature (alpha peak, dominant peak): NaN where specparam found no such
    peak, a result, not missing data, so it is left out of the missingness, all-NaN
    and constant checks (one epoch with a peak leaves a single value)."""
    return bool(_PEAK_FEATURE.search(str(column)))


def descriptor_qc_thresholds(config_snapshot: dict[str, Any]) -> DescriptorQCThresholds:
    """The QC thresholds of ``qc.thresholds`` in *config_snapshot*, defaults for the rest."""
    return DescriptorQCThresholds(**((config_snapshot.get("qc") or {}).get("thresholds") or {}))


def _split_failures(failure_df: pd.DataFrame, epoch_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Extraction failures of the epochs in *epoch_df*, and the number of epochs dropped
    by outlier rejection.

    Failures of epochs left out afterwards (rejected, or in a too-short recording) do not
    count.
    """
    if failure_df.empty or "family" not in failure_df.columns:
        return failure_df, 0
    rejection = failure_df["family"].astype(str).str.startswith("MAD_Rejection")
    failures = failure_df[~rejection]
    if "obs_id" in failures.columns and "obs_id" in epoch_df.columns:
        failures = failures[failures["obs_id"].astype(str).isin(epoch_df["obs_id"].astype(str))]
    return failures, int(failure_df.loc[rejection, "obs_id"].nunique())


def _expected_families(config_snapshot: dict[str, Any]) -> list[str]:
    """Family tokens enabled in *config_snapshot* (``band``/``param``/``complexity``)."""
    families_config = (
        (config_snapshot.get("families") or {}) if isinstance(config_snapshot, dict) else {}
    )
    return [
        family
        for family, config_key in _FAMILY_CONFIG_KEYS
        if bool((families_config.get(config_key) or {}).get("enabled"))
    ]


def _report_asset_urls(config_snapshot: dict[str, Any]) -> dict[str, str] | str | None:
    """Return configured report assets without triggering implicit downloads."""
    asset_urls = config_snapshot.get("report_asset_urls")
    return None if asset_urls == "cdn" else asset_urls


def _column_matches_family(column: object, families: Iterable[str]) -> bool:
    text = str(column)
    return any(text.startswith(f"{family}_") or f"_{family}_" in text for family in families)


def _select_family_feature_cols(
    feature_columns: Iterable[object],
    families: Sequence[str],
    present_columns: Iterable[object],
) -> list[object]:
    """Feature columns belonging to *families* that are also present in the table."""
    present = {str(column) for column in present_columns}
    return [
        column
        for column in feature_columns
        if _column_matches_family(column, families) and str(column) in present
    ]


def _missing_family_flags(
    feature_missingness_df: pd.DataFrame,
    expected_families: Sequence[str],
    *,
    noun: str,
) -> list[dict[str, Any]]:
    """Fail flag per expected family that produced no columns."""
    actual_families = set(
        feature_missingness_df.get("family", pd.Series(dtype=str)).dropna().astype(str)
    )
    return [
        make_qc_flag(
            "fail",
            "missing_expected_family",
            f"Expected family '{family}' has no {noun} columns.",
            scope=family,
        )
        for family in expected_families
        if family not in actual_families
    ]


def _missingness_flags(
    max_missingness: float,
    *,
    thresh: DescriptorQCThresholds,
    fail_code: str,
    warn_code: str,
    fail_msg: str,
    warn_msg: str,
    nan_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Fail/warn flag for feature missingness against the configured thresholds.

    When *nan_rate* is supplied (subject scope) the per-subject average NaN rate
    also contributes to the fail/warn decision; the flagged value is always the
    per-feature ``max_missingness`` for consistency with the threshold reported.
    """
    fail_hit = max_missingness >= thresh.fail_feature_missingness or (
        nan_rate is not None and nan_rate >= thresh.fail_nan_rate
    )
    warn_hit = max_missingness >= thresh.warn_feature_missingness or (
        nan_rate is not None and nan_rate >= thresh.warn_nan_rate
    )
    if fail_hit:
        return [
            make_qc_flag(
                "fail",
                fail_code,
                fail_msg,
                value=max_missingness,
                threshold=thresh.fail_feature_missingness,
            )
        ]
    if warn_hit:
        return [
            make_qc_flag(
                "warn",
                warn_code,
                warn_msg,
                value=max_missingness,
                threshold=thresh.warn_feature_missingness,
            )
        ]
    return []


def _zero_variance_flags(
    fraction: float,
    *,
    thresh: DescriptorQCThresholds,
    fail_code: str,
    warn_code: str,
    fail_msg: str,
    warn_msg: str,
) -> list[dict[str, Any]]:
    """Fail/warn flag for the fraction of constant (zero-variance) features."""
    if fraction >= thresh.fail_zero_variance_fraction:
        return [
            make_qc_flag(
                "fail",
                fail_code,
                fail_msg,
                value=fraction,
                threshold=thresh.fail_zero_variance_fraction,
            )
        ]
    if fraction >= thresh.warn_zero_variance_fraction:
        return [
            make_qc_flag(
                "warn",
                warn_code,
                warn_msg,
                value=fraction,
                threshold=thresh.warn_zero_variance_fraction,
            )
        ]
    return []


def _family_failure_flags(
    family_summary_df: pd.DataFrame,
    *,
    thresh: DescriptorQCThresholds,
    fail_code: Callable[[str], str],
    warn_code: Callable[[str], str],
    fail_msg: str,
    warn_msg: str,
) -> list[dict[str, Any]]:
    """Per-family fail/warn flags driven by each family's failure rate."""
    flags: list[dict[str, Any]] = []
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
        failure_rate = float(row.get("failure_rate") or 0.0)
        if failure_rate >= thresh.fail_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "fail",
                    fail_code(family),
                    fail_msg,
                    value=failure_rate,
                    threshold=thresh.fail_family_failure_rate,
                    scope=family,
                )
            )
        elif failure_rate >= thresh.warn_family_failure_rate:
            flags.append(
                make_qc_flag(
                    "warn",
                    warn_code(family),
                    warn_msg,
                    value=failure_rate,
                    threshold=thresh.warn_family_failure_rate,
                    scope=family,
                )
            )
    return flags


def run_descriptor_shard_qc(
    subject: str,
    session: str,
    condition: str,
    sensor_epoch_df: pd.DataFrame,
    sensor_subject_df: pd.DataFrame,
    sensor_epoch_feature_columns: Sequence[str],
    sensor_subject_feature_columns: Sequence[str],
    failure_df: pd.DataFrame,
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """QC of one shard: its summary row and flags, and the tables its report shows.

    Writes nothing: stage 05 keeps the summary and flags in the shard's ``shard.json``
    and puts every condition of a subject in one report
    (:func:`write_descriptor_subject_report`).
    """
    thresh = descriptor_qc_thresholds(config_snapshot)
    expected_families = _expected_families(config_snapshot)
    feature_cols = _select_family_feature_cols(
        sensor_epoch_feature_columns,
        expected_families,
        sensor_epoch_df.columns,
    )
    subject_feature_cols = _select_family_feature_cols(
        sensor_subject_feature_columns,
        expected_families,
        sensor_subject_df.columns,
    )
    feature_df = (
        sensor_epoch_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
        if feature_cols
        else pd.DataFrame()
    )
    feature_missingness_df = compute_family_missingness(sensor_epoch_df, feature_cols)
    constant_df = compute_family_constant_summary(
        sensor_epoch_df,
        feature_cols,
        tol=thresh.near_constant_std_tol,
    )
    extraction_failures, n_epochs_rejected = _split_failures(failure_df, sensor_epoch_df)
    # The family table, its missingness chart and the failure counts follow the checks:
    # no peak features, and rejected epochs are not failures.
    family_summary_df = aggregate_family_qc(
        sensor_epoch_df,
        [column for column in feature_cols if not _is_peak_feature(column)],
        failures_df=extraction_failures,
        tol=thresh.near_constant_std_tol,
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
    failure_summaries = summarize_failures(extraction_failures)
    checked_missingness = feature_missingness_df[
        ~feature_missingness_df["column"].map(_is_peak_feature)
    ]

    def peak_rate(token: str) -> float:
        """Share of channel-epochs where specparam found the peak behind *token*."""
        rates = feature_missingness_df.loc[
            feature_missingness_df["column"].str.contains(token, regex=False), "missing_rate"
        ]
        return float(1 - rates.mean()) if not rates.empty else np.nan

    metrics = {
        "n_epochs": int(len(sensor_epoch_df)),
        "n_epochs_rejected": n_epochs_rejected,
        "n_sensor_epoch_features": int(len(feature_cols)),
        "n_sensor_subject_rows": int(len(sensor_subject_df)),
        "n_sensor_subject_features": int(len(subject_feature_cols)),
        "n_failures_total": int(len(extraction_failures)),
        "nan_rate_sensor_epoch": float(checked_missingness["missing_rate"].mean())
        if not checked_missingness.empty
        else 0.0,
        "max_feature_missingness": float(checked_missingness["missing_rate"].max())
        if not checked_missingness.empty
        else 0.0,
        "n_all_nan_features": int(
            constant_df.loc[~constant_df["column"].map(_is_peak_feature), "is_all_nan"].sum()
        )
        if not constant_df.empty
        else 0,
        "n_constant_features": int(
            constant_df.loc[~constant_df["column"].map(_is_peak_feature), "is_constant"].sum()
        )
        if not constant_df.empty
        else 0,
        "peak_rate": peak_rate("_peak_freq_dom"),
        "alpha_peak_rate": peak_rate("_alpha_peak_freq"),
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
    flags.extend(_missing_family_flags(feature_missingness_df, expected_families, noun="extracted"))
    flags.extend(
        _missingness_flags(
            metrics["max_feature_missingness"],
            thresh=thresh,
            fail_code="high_missingness",
            warn_code="elevated_missingness",
            fail_msg="Feature missingness is too high.",
            warn_msg="Feature missingness is elevated.",
            nan_rate=metrics["nan_rate_sensor_epoch"],
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
    flags.extend(
        _zero_variance_flags(
            constant_fraction,
            thresh=thresh,
            fail_code="many_constant_features",
            warn_code="constant_features_present",
            fail_msg="Too many descriptor features are constant.",
            warn_msg="Some descriptor features are constant.",
        )
    )
    flags.extend(
        _family_failure_flags(
            family_summary_df,
            thresh=thresh,
            fail_code=lambda family: f"{family}_family_failure_high",
            warn_code=lambda family: f"{family}_family_failure_warn",
            fail_msg="Family failure rate is high.",
            warn_msg="Family failure rate is elevated.",
        )
    )
    for row in family_summary_df.to_dict("records"):
        family = str(row["family"])
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
            if pd.notna(r2_p05) and float(r2_p05) < thresh.fail_param_r_squared_p05:
                flags.append(
                    make_qc_flag(
                        "fail",
                        "param_low_r_squared",
                        "Parametric fits show very low r-squared.",
                        value=r2_p05,
                        threshold=thresh.fail_param_r_squared_p05,
                        scope=family,
                    )
                )
            elif pd.notna(r2_p05) and float(r2_p05) < thresh.warn_param_r_squared_p05:
                flags.append(
                    make_qc_flag(
                        "warn",
                        "param_low_r_squared",
                        "Parametric fits show low r-squared.",
                        value=r2_p05,
                        threshold=thresh.warn_param_r_squared_p05,
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

    summary_row = {
        "subject": subject,
        "session": session,
        "condition": condition,
        "qc_status": resolve_qc_status(flags),
        **metrics,
    }
    param_fit_cols = [
        column
        for column in feature_df.columns
        if "param_r_squared_" in column or "param_fit_error_" in column
    ]
    return {
        "summary": summary_row,
        "flags": flags,
        "family_summary": family_summary_df,
        "failure_summary": failure_summaries["combined"],
        "feature_missingness": checked_missingness[checked_missingness["missing_rate"] > 0],
        "param_fits": feature_df[param_fit_cols],
    }


def write_descriptor_subject_report(
    reports_root: Path,
    subject: str,
    session: str,
    shard_qcs: Sequence[dict[str, Any]],
    config_snapshot: dict[str, Any],
) -> Path:
    """One QC report for a subject's session, with every condition in *shard_qcs*.

    *shard_qcs* are :func:`run_descriptor_shard_qc` results; their tables are stacked
    with a ``condition`` column and their figures captioned with the condition.
    """
    report_path = report_paths.subject_report_dir(
        reports_root=reports_root,
        subject=subject,
        session=session,
        stage=report_paths.ReportStage.DESCRIPTOR_QC,
        create=True,
    ) / report_paths.descriptor_qc_report_name(subject, session)

    def stacked(key: str) -> pd.DataFrame:
        frames = [
            qc[key].assign(condition=qc["summary"]["condition"])
            for qc in shard_qcs
            if not qc[key].empty
        ]
        if not frames:
            return pd.DataFrame()
        return reorder_columns_front(pd.concat(frames, ignore_index=True), ["condition"])

    overview_df = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Session": session,
                "Condition": qc["summary"]["condition"],
                "Epochs": qc["summary"]["n_epochs"],
                "Rejected Epochs": qc["summary"]["n_epochs_rejected"],
                "Peak Found": round(qc["summary"]["peak_rate"], 3),
                "Alpha Peak Found": round(qc["summary"]["alpha_peak_rate"], 3),
                "Epoch Features": qc["summary"]["n_sensor_epoch_features"],
                "Recording Features": qc["summary"]["n_sensor_subject_features"],
                "Failures": qc["summary"]["n_failures_total"],
                "QC Status": qc["summary"]["qc_status"],
            }
            for qc in shard_qcs
        ]
    )
    flags_df = pd.DataFrame(
        [
            {"condition": qc["summary"]["condition"], **flag}
            for qc in shard_qcs
            for flag in qc["flags"]
        ]
    )
    # The figures live only inside the HTML report, which embeds them.
    with tempfile.TemporaryDirectory() as figures_dir:
        figure_paths: dict[str, Path] = {}
        for qc in shard_qcs:
            condition = qc["summary"]["condition"]
            figures = viz_descriptor_qc.save_subject_descriptor_qc_figures(
                figures_dir=Path(figures_dir) / condition,
                family_summary_df=qc["family_summary"],
                failure_summary_df=qc["failure_summary"],
                feature_missingness_df=qc["feature_missingness"],
                epoch_feature_df=qc["param_fits"],
            )
            figure_paths.update(
                {f"{condition}: {key.replace('_', ' ')}": path for key, path in figures.items()}
            )
        generate_descriptor_subject_report(
            output_path=report_path,
            overview_df=overview_df,
            flags_df=flags_df,
            failure_summary_df=stacked("failure_summary"),
            feature_missingness_df=stacked("feature_missingness"),
            family_summary_df=stacked("family_summary"),
            figure_paths=figure_paths,
            asset_urls=_report_asset_urls(config_snapshot),
        )
    return report_path


def run_descriptor_dataset_qc(
    *,
    reports_root: Path,
    merged_sensor_epoch_df: pd.DataFrame,
    merged_sensor_subject_df: pd.DataFrame,
    merged_sensor_epoch_feature_columns: Sequence[str],
    merged_sensor_subject_feature_columns: Sequence[str],
    shard_qc_rows_df: pd.DataFrame | None,
    merged_failures_df: pd.DataFrame | None,
    config_snapshot: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_dir = report_paths.summary_report_dir(
        reports_root, report_paths.ReportStage.DESCRIPTOR_QC, create=True
    )
    thresh = descriptor_qc_thresholds(config_snapshot)

    expected_families = _expected_families(config_snapshot)
    subject_feature_cols = _select_family_feature_cols(
        merged_sensor_subject_feature_columns,
        expected_families,
        merged_sensor_subject_df.columns,
    )
    epoch_feature_cols = _select_family_feature_cols(
        merged_sensor_epoch_feature_columns,
        expected_families,
        merged_sensor_epoch_df.columns,
    )
    # As in the shard QC: peak features (NaN where no peak was found) are not checked,
    # and rejected epochs are not failures.
    checked_cols = [column for column in subject_feature_cols if not _is_peak_feature(column)]
    feature_missingness_df = compute_family_missingness(
        merged_sensor_subject_df,
        subject_feature_cols,
    )
    checked_missingness = feature_missingness_df[
        ~feature_missingness_df["column"].map(_is_peak_feature)
    ]
    constant_df = compute_family_constant_summary(
        merged_sensor_subject_df,
        checked_cols,
        tol=thresh.near_constant_std_tol,
    )
    low_variance_df = constant_df[
        constant_df["is_constant"]
        | (constant_df["relative_std"].fillna(np.inf) <= thresh.near_constant_std_tol * 10)
    ].copy()
    extraction_failures, n_epochs_rejected = _split_failures(
        merged_failures_df if merged_failures_df is not None else pd.DataFrame(),
        merged_sensor_epoch_df,
    )
    failure_summaries = summarize_failures(extraction_failures)
    family_summary_df = aggregate_family_qc(
        merged_sensor_subject_df,
        checked_cols,
        failures_df=extraction_failures,
        tol=thresh.near_constant_std_tol,
    ).rename(
        columns={
            "missing_rate_mean": "missing_rate",
            "nonfinite_rate_mean": "nonfinite_rate",
        }
    )
    # Failures are per channel-epoch, so their rate is taken on the epoch table.
    epoch_failures = aggregate_family_qc(
        merged_sensor_epoch_df, epoch_feature_cols, failures_df=extraction_failures
    )[["family", "failure_count", "failure_rate"]]
    family_summary_df = family_summary_df.drop(columns=["failure_count", "failure_rate"]).merge(
        epoch_failures, on="family", how="left"
    )
    # The fit and range diagnostics read epoch values, as in the shard QC: a recording
    # column is a mean or a spread of them.
    family_summary_df = add_family_diagnostics(
        family_summary_df,
        compute_family_missingness(merged_sensor_epoch_df, epoch_feature_cols),
        merged_sensor_epoch_df.loc[:, epoch_feature_cols].replace([np.inf, -np.inf], np.nan),
    )
    condition_breakdown_rows: list[dict[str, Any]] = []
    if "condition" in merged_sensor_subject_df.columns:
        failure_has_condition = "condition" in extraction_failures.columns
        failure_has_family = "family" in extraction_failures.columns
        for condition, group in merged_sensor_subject_df.groupby("condition"):
            cond_missingness = (
                compute_family_missingness(group, checked_cols) if checked_cols else pd.DataFrame()
            )
            cond_failures = (
                extraction_failures[extraction_failures["condition"] == condition]
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
    shard_status = shard_summary_df.get("qc_status", pd.Series(dtype=str))
    metrics = {
        "n_shards": int(len(shard_summary_df)),
        "n_shards_pass": int((shard_status == "pass").sum()) if not shard_summary_df.empty else 0,
        "n_shards_warn": int((shard_status == "warn").sum()) if not shard_summary_df.empty else 0,
        "n_shards_fail": int((shard_status == "fail").sum()) if not shard_summary_df.empty else 0,
        "n_failures_total": int(len(extraction_failures)),
        "n_epochs_rejected": n_epochs_rejected,
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
        "max_feature_missingness": float(checked_missingness["missing_rate"].max())
        if not checked_missingness.empty
        else 0.0,
        "median_feature_missingness": float(checked_missingness["missing_rate"].median())
        if not checked_missingness.empty
        else 0.0,
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
    flags.extend(_missing_family_flags(feature_missingness_df, expected_families, noun="merged"))
    flags.extend(
        _missingness_flags(
            metrics["max_feature_missingness"],
            thresh=thresh,
            fail_code="high_global_missingness",
            warn_code="global_missingness_warn",
            fail_msg="Global descriptor missingness is too high.",
            warn_msg="Global descriptor missingness is elevated.",
        )
    )
    zero_variance_fraction = metrics["n_zero_variance_features"] / max(len(subject_feature_cols), 1)
    flags.extend(
        _zero_variance_flags(
            zero_variance_fraction,
            thresh=thresh,
            fail_code="many_zero_variance_features",
            warn_code="near_zero_variance_features",
            fail_msg="Too many merged features are zero-variance.",
            warn_msg="Merged features include near-zero variance columns.",
        )
    )
    flags.extend(
        _family_failure_flags(
            family_summary_df,
            thresh=thresh,
            fail_code=lambda _family: "family_failure_concentration",
            warn_code=lambda _family: "family_failure_concentration",
            fail_msg="Family failure concentration is high.",
            warn_msg="Family failure concentration is elevated.",
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
                "Recording Rows": len(merged_sensor_subject_df),
                "Epoch Rows": len(merged_sensor_epoch_df),
                "Recording Features": metrics["n_sensor_subject_features"],
                "Epoch Features": metrics["n_sensor_epoch_features"],
                "Failures": metrics["n_failures_total"],
                "Rejected Epochs": metrics["n_epochs_rejected"],
                "QC Status": qc_status,
            }
        ]
    )
    flags_df = pd.DataFrame(flags)
    manifest_df = (
        pd.DataFrame([{"field": key, "value": value} for key, value in manifest.items()])
        if manifest
        else None
    )
    missing_df = checked_missingness[checked_missingness["missing_rate"] > 0]

    with tempfile.TemporaryDirectory() as figures_dir:
        figure_paths = viz_descriptor_qc.save_dataset_descriptor_qc_figures(
            figures_dir=Path(figures_dir),
            shard_summary_df=shard_summary_df,
            failure_family_df=failure_summaries["by_family"],
            failure_channel_df=failure_summaries["by_channel"],
            feature_missingness_df=missing_df,
            low_variance_df=low_variance_df,
        )
        generate_descriptor_dataset_report(
            output_path=report_path,
            overview_df=overview_df,
            shard_summary_df=shard_summary_df,
            flags_df=flags_df,
            failure_family_df=failure_summaries["by_family"],
            failure_channel_df=failure_summaries["by_channel"],
            feature_missingness_df=missing_df,
            low_variance_df=low_variance_df,
            family_summary_df=family_summary_df,
            figure_paths=figure_paths,
            manifest_df=manifest_df,
            condition_breakdown_df=condition_breakdown_df,
            asset_urls=_report_asset_urls(config_snapshot),
        )

    return {
        "qc_status": qc_status,
        "report_path": str(report_path),
        **metrics,
    }

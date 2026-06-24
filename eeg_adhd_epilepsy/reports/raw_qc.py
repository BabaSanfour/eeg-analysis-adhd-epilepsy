"""Pre-base raw QC report generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from coco_pipe.report.core import Report, Section

from eeg_adhd_epilepsy.reports._common import (
    add_images as _add_images,
)
from eeg_adhd_epilepsy.reports._common import (
    add_optional_table as _add_optional_table,
)
from eeg_adhd_epilepsy.reports._common import (
    build_dataset_mean_metric_table,
    build_subject_overview_table,
)
from eeg_adhd_epilepsy.reports._common import (
    build_flag_reason_table as _build_flag_reason_table,
)
from eeg_adhd_epilepsy.reports._common import (
    format_value as _format_value,
)

# Human-readable labels for the coded flag reasons from ``evaluate_signal_qc_flag``.
REASON_LABELS = {
    "too_many_bad_channels": "Too many bad channels",
    "many_bad_channels": "Many bad channels",
    "amplitude_above_threshold": "Peak amplitude above threshold",
    "line_noise_residual": "Residual line noise",
    "high_hf_ratio": "High HF/LF ratio",
    "low_duration_retention": "Low clean-duration retention",
    "low_condition_retention": "Low condition-coverage retention",
}

OUTLIER_Z = 3.5
OUTLIER_METRICS: tuple[tuple[str, str], ...] = (
    ("amplitude_mean_uv", "Mean amplitude"),
    ("amplitude_max_uv", "Max amplitude"),
    ("pct_bad_channels", "Bad channels %"),
    ("line_noise_ratio", "Line-noise ratio"),
    ("hf_lf_ratio", "HF/LF ratio"),
    ("aperiodic_slope", "Aperiodic slope"),
)


def _humanize_reasons(reasons: object) -> str:
    text = str(reasons or "").strip()
    if not text:
        return "None"
    labels = [
        REASON_LABELS.get(code.strip(), code.strip()) for code in text.split(";") if code.strip()
    ]
    return "; ".join(labels) if labels else "None"


def _annotate_alpha_peak(value: object) -> str:
    v = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(v):
        return "not detected"
    band = "in band" if 8.0 <= v <= 13.0 else "out of band 8-13 Hz"
    return f"{v:.1f} Hz ({band})"


def _annotate_aperiodic_slope(value: object) -> str:
    v = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(v):
        return "n/a"
    if 1.0 <= v <= 2.0:
        note = "typical"
    elif 0.5 <= v <= 3.0:
        note = "plausible"
    else:
        note = "atypical"
    return f"{v:.2f} ({note})"


def build_usability_table(record: Mapping[str, object]) -> pd.DataFrame:
    reactivity = pd.to_numeric(record.get("alpha_reactivity"), errors="coerce")
    reactivity_str = f"{reactivity:.2f} (>1 expected)" if np.isfinite(reactivity) else "n/a"
    return pd.DataFrame(
        [
            {"Metric": "QC Status", "Value": record.get("subject_flag", "")},
            {
                "Metric": "Flag Reasons",
                "Value": _humanize_reasons(record.get("subject_flag_reasons")),
            },
            {
                "Metric": "Coverage vs raw",
                "Value": _format_value(record.get("coverage_pct"), suffix="%"),
            },
            {
                "Metric": "Mean amplitude",
                "Value": _format_value(record.get("amplitude_mean_uv"), suffix=" uV"),
            },
            {
                "Metric": "Max amplitude",
                "Value": _format_value(record.get("amplitude_max_uv"), suffix=" uV"),
            },
            {"Metric": "Flat channels", "Value": int(record.get("n_flat_channels", 0) or 0)},
            {"Metric": "Noisy channels", "Value": int(record.get("n_noisy_channels", 0) or 0)},
            {
                "Metric": "Bad channels",
                "Value": _format_value(record.get("pct_bad_channels"), suffix="%"),
            },
            {"Metric": "Line-noise ratio", "Value": _format_value(record.get("line_noise_ratio"))},
            {"Metric": "HF/LF ratio", "Value": _format_value(record.get("hf_lf_ratio"))},
            {
                "Metric": "Alpha peak",
                "Value": _annotate_alpha_peak(record.get("alpha_peak_hz")),
            },
            {"Metric": "Alpha reactivity (EC/EO)", "Value": reactivity_str},
            {
                "Metric": "Aperiodic slope",
                "Value": _annotate_aperiodic_slope(record.get("aperiodic_slope")),
            },
        ]
    )


def build_threshold_verdict_table(record: Mapping[str, object]) -> pd.DataFrame:
    """Per-metric value vs flagging threshold, so the usability call is auditable."""
    columns = ["Check", "Value", "Threshold", "Status"]
    thresholds = record.get("thresholds") or {}
    if not thresholds:
        return pd.DataFrame(columns=columns)

    def _status(value: object, limit: float) -> str:
        v = pd.to_numeric(value, errors="coerce")
        if not np.isfinite(v):
            return "n/a"
        return "exceeds" if float(v) > float(limit) else "ok"

    n_bad = float(record.get("n_flat_channels", 0) or 0) + float(
        record.get("n_noisy_channels", 0) or 0
    )
    n_borderline = int(thresholds["n_bad_borderline"])
    n_unusable = int(thresholds["n_bad_unusable"])
    bad_status = (
        "unusable" if n_bad >= n_unusable else "borderline" if n_bad >= n_borderline else "ok"
    )
    rows = [
        {
            "Check": "Bad channels (count)",
            "Value": f"{int(n_bad)}",
            "Threshold": f">={n_borderline} borderline, >={n_unusable} unusable",
            "Status": bad_status,
        },
        {
            "Check": "Peak amplitude (uV)",
            "Value": _format_value(record.get("amplitude_max_uv")),
            "Threshold": f"<= {thresholds['amplitude_max_uv']:g}",
            "Status": _status(record.get("amplitude_max_uv"), thresholds["amplitude_max_uv"]),
        },
        {
            "Check": "Line-noise ratio",
            "Value": _format_value(record.get("line_noise_ratio")),
            "Threshold": f"<= {thresholds['line_noise_ratio']:g}",
            "Status": _status(record.get("line_noise_ratio"), thresholds["line_noise_ratio"]),
        },
        {
            "Check": "HF/LF ratio",
            "Value": _format_value(record.get("hf_lf_ratio")),
            "Threshold": f"<= {thresholds['hf_lf_ratio']:g}",
            "Status": _status(record.get("hf_lf_ratio"), thresholds["hf_lf_ratio"]),
        },
    ]
    return pd.DataFrame(rows, columns=columns)


def build_channel_diagnostics_tables(
    channel_diagnostics: Mapping[str, object] | None,
) -> dict[str, pd.DataFrame]:
    if not channel_diagnostics:
        empty = pd.DataFrame(columns=["Channel"])
        return {
            "flat": empty,
            "noisy": empty,
            "top_amplitude": pd.DataFrame(columns=["Channel", "Amplitude PTP (uV)"]),
            "top_line_noise": pd.DataFrame(columns=["Channel", "Line Noise Ratio"]),
        }
    flat_df = pd.DataFrame({"Channel": list(channel_diagnostics.get("flat_channels", []))})
    noisy_df = pd.DataFrame({"Channel": list(channel_diagnostics.get("noisy_channels", []))})
    top_amp_df = pd.DataFrame(
        [
            {"Channel": channel, "Amplitude PTP (uV)": float(value)}
            for channel, value in channel_diagnostics.get("top_amplitude_channels", [])
        ]
    )
    top_line_df = pd.DataFrame(
        [
            {"Channel": channel, "Line Noise Ratio": float(value)}
            for channel, value in channel_diagnostics.get("top_line_noise_channels", [])
        ]
    )
    return {
        "flat": flat_df,
        "noisy": noisy_df,
        "top_amplitude": top_amp_df,
        "top_line_noise": top_line_df,
    }


def build_run_summary_table(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "Run": record.get("run_id", ""),
                "QC Status": record.get("subject_flag", ""),
                "QC Score (0=best)": _format_value(record.get("qc_score")),
                "Bad Channels (%)": _format_value(record.get("pct_bad_channels")),
                "Mean Amplitude (uV)": _format_value(record.get("amplitude_mean_uv")),
                "Max Amplitude (uV)": _format_value(record.get("amplitude_max_uv")),
                "Line Noise Ratio": _format_value(record.get("line_noise_ratio")),
                "HF/LF Ratio": _format_value(record.get("hf_lf_ratio")),
            }
        )
    return pd.DataFrame(rows).sort_values("Run") if rows else pd.DataFrame()


def build_dataset_summary_table(runs_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
    status_counts = (
        runs_df.get("subject_flag", pd.Series(dtype=str))
        .fillna("unknown")
        .astype(str)
        .value_counts()
    )
    return pd.DataFrame(
        [
            {
                "Total subject-sessions": int(len(subjects_df)),
                "Total runs": int(len(runs_df)),
                "Usable runs": int(status_counts.get("usable", 0)),
                "Borderline runs": int(status_counts.get("borderline", 0)),
                "Unusable runs": int(status_counts.get("unusable", 0)),
            }
        ]
    )


def build_noise_metrics_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    return build_dataset_mean_metric_table(
        runs_df,
        (
            ("Mean bad channels", "pct_bad_channels", "%"),
            ("Mean amplitude", "amplitude_mean_uv", " uV"),
            ("Mean max amplitude", "amplitude_max_uv", " uV"),
            ("Mean line-noise ratio", "line_noise_ratio", ""),
            ("Mean HF/LF ratio", "hf_lf_ratio", ""),
            ("Mean alpha peak", "alpha_peak_hz", " Hz"),
            ("Mean aperiodic slope", "aperiodic_slope", ""),
        ),
    )


def build_flag_reason_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    table = _build_flag_reason_table(
        runs_df, reasons_column="subject_flag_reasons", count_label="Runs"
    )
    if not table.empty and "Reason" in table.columns:
        table = table.copy()
        table["Reason"] = table["Reason"].map(lambda code: REASON_LABELS.get(str(code), str(code)))
    return table


def build_channel_failure_table(failure_rates: Mapping[str, float] | None) -> pd.DataFrame:
    """Per-electrode fraction of recordings in which it was flagged bad.

    High values point to systematic equipment/placement problems for a channel
    rather than per-recording noise.
    """
    columns = ["Channel", "Recordings flagged bad (%)"]
    if not failure_rates:
        return pd.DataFrame(columns=columns)
    rows = [
        {"Channel": channel, "Recordings flagged bad (%)": _format_value(rate * 100.0, suffix="%")}
        for channel, rate in sorted(failure_rates.items(), key=lambda kv: kv[1], reverse=True)
        if rate > 0
    ]
    return pd.DataFrame(rows, columns=columns)


def build_cohort_outlier_table(
    runs_df: pd.DataFrame, z_threshold: float = OUTLIER_Z
) -> pd.DataFrame:
    """Flag runs whose metrics are robust (MAD-based) outliers within the cohort."""
    columns = ["Run", "Metric", "Value", "Robust z"]
    if runs_df is None or runs_df.empty:
        return pd.DataFrame(columns=columns)
    label_col = "run_prefix" if "run_prefix" in runs_df.columns else "filepath"
    flagged: list[tuple[float, object, str, float, float]] = []
    for column, label in OUTLIER_METRICS:
        if column not in runs_df.columns:
            continue
        values = pd.to_numeric(runs_df[column], errors="coerce")
        median = values.median()
        mad = (values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        z_scores = 0.6745 * (values - median) / mad
        for idx, score in z_scores.items():
            if np.isfinite(score) and abs(score) >= z_threshold:
                run_label = runs_df.loc[idx, label_col] if label_col in runs_df.columns else idx
                flagged.append(
                    (abs(float(score)), run_label, label, float(values.loc[idx]), float(score))
                )
    flagged.sort(key=lambda item: item[0], reverse=True)
    rows = [
        {
            "Run": run_label,
            "Metric": metric,
            "Value": _format_value(value),
            "Robust z": _format_value(score),
        }
        for _, run_label, metric, value, score in flagged
    ]
    return pd.DataFrame(rows, columns=columns)


def build_dataset_report_tables(
    runs_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "dataset_summary_df": build_dataset_summary_table(runs_df, subjects_df),
        "noise_metrics_df": build_noise_metrics_table(runs_df),
        "flag_reason_df": build_flag_reason_table(runs_df),
    }


def generate_raw_qc_subject_report(
    record: Mapping[str, object],
    run_summary_df: pd.DataFrame,
    channel_diagnostics: Mapping[str, object],
    figure_paths: Mapping[str, Path],
    output_path: Path,
) -> Path:
    subject_id = record.get("subject_session_prefix", record.get("subject_id", "unknown"))
    report = Report(title=f"Raw QC Report - {subject_id}")

    overview = Section("Signal Overview", icon="🎛️")
    _add_optional_table(overview, build_subject_overview_table(record), "Subject Overview")
    report.add_section(overview)

    usability = Section("Usability", icon="🧪")
    _add_optional_table(usability, build_usability_table(record), "Signal Usability")
    _add_optional_table(usability, build_threshold_verdict_table(record), "Threshold Checks")
    usability.add_markdown(
        "Alpha peak is physiological at 8-13 Hz; aperiodic (1/f) slope is typically "
        "~1-2. Eyes-closed alpha reactivity (EC/EO) > 1 is the expected awake-resting pattern."
    )
    report.add_section(usability)

    if run_summary_df is not None and not run_summary_df.empty:
        runs = Section("Per-Run Summary", icon="🗂️")
        _add_optional_table(runs, run_summary_df, "Run QC Summary")
        runs.add_markdown("QC score is in [0, 1] where **0 is best** (~0.5 ≈ at threshold).")
        report.add_section(runs)

    diagnostics = Section("Per-Channel Diagnostics", icon="📡")
    diag_tables = build_channel_diagnostics_tables(channel_diagnostics)
    _add_optional_table(diagnostics, diag_tables["flat"], "Flat Channels")
    _add_optional_table(diagnostics, diag_tables["noisy"], "Noisy Channels")
    _add_optional_table(diagnostics, diag_tables["top_amplitude"], "Top Amplitude Channels")
    _add_optional_table(diagnostics, diag_tables["top_line_noise"], "Top Line-Noise Channels")
    report.add_section(diagnostics)

    figures = Section("Figures", icon="📈")
    _add_images(
        figures,
        figure_paths,
        (
            "amplitude_ptp_uv_topomap",
            "line_noise_ratio_topomap",
            "hf_lf_ratio_topomap",
            "segment_amplitude_mean_uv",
            "segment_line_noise_ratio",
            "segment_hf_lf_ratio",
        ),
    )
    report.add_section(figures)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path


def generate_raw_qc_dataset_report(
    tables: Mapping[str, pd.DataFrame],
    figure_paths: Mapping[str, Path],
    output_path: Path,
) -> Path:
    report = Report(title="Raw QC Dataset Report")

    summary_df = tables.get("dataset_summary_df", pd.DataFrame())
    if not summary_df.empty:
        report.add_summary_card(summary_df.iloc[0].to_dict())

    definition = Section("Overview", icon="🎯")
    definition.add_markdown(
        "Pre-base raw QC focuses on broad signal usability, channel quality, "
        "line-noise contamination, and high-frequency contamination."
    )
    report.add_section(definition)

    usability = Section("Usability Summary", icon="🧪")
    _add_optional_table(usability, tables.get("flag_reason_df", pd.DataFrame()), "Flag Reasons")
    _add_images(usability, figure_paths, ("flag_status", "flag_reasons"))
    report.add_section(usability)

    noise = Section("Noise and Artifact Metrics", icon="📉")
    _add_optional_table(noise, tables.get("noise_metrics_df", pd.DataFrame()), "Noise Metrics")
    _add_images(
        noise,
        figure_paths,
        (
            "amplitude_mean_uv",
            "amplitude_max_uv",
            "pct_bad_channels",
            "line_noise_ratio",
            "hf_lf_ratio",
            "alpha_peak_hz",
            "aperiodic_slope",
            "amplitude_ptp_uv_topomap",
            "line_noise_ratio_topomap",
            "hf_lf_ratio_topomap",
        ),
    )
    report.add_section(noise)

    segments = Section("Segment QC", icon="🧠")
    _add_images(
        segments,
        figure_paths,
        (
            "segment_amplitude_mean_uv",
            "segment_line_noise_ratio",
            "segment_hf_lf_ratio",
        ),
    )
    report.add_section(segments)

    electrodes = Section("Electrode Reliability", icon="📡")
    electrodes.add_markdown(
        "Fraction of recordings in which each electrode was flagged bad. "
        "Consistently failing channels suggest systematic hardware or placement issues."
    )
    _add_optional_table(
        electrodes, tables.get("channel_failure_df", pd.DataFrame()), "Consensus Bad Electrodes"
    )
    report.add_section(electrodes)

    outliers = Section("Cohort Outliers", icon="🚩")
    outliers.add_markdown(
        f"Runs whose metrics are robust (MAD-based) outliers at |z| ≥ {OUTLIER_Z} vs the cohort."
    )
    _add_optional_table(outliers, tables.get("outlier_df", pd.DataFrame()), "Metric Outliers")
    report.add_section(outliers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

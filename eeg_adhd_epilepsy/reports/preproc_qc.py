"""Shared post-preprocessing QC report generation."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from coco_pipe.report.core import ImageElement, Report, Section

from eeg_adhd_epilepsy.reports._common import (
    add_optional_table as _add_optional_table,
    build_dataset_mean_metric_table,
    build_flag_reason_table as _build_flag_reason_table,
    build_record_metric_table,
    format_value as _format_value,
)
from eeg_adhd_epilepsy.reports._common import add_images as _add_images_base
from eeg_adhd_epilepsy.utils.formatting import format_duration_hms

# preproc_qc figures are not individually captioned (unlike raw_qc/eeg_report).
_add_images = partial(_add_images_base, caption_from_key=False)


def build_stage_overview_table(record: Mapping[str, object], *, stage_display_name: str, previous_stage_label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Subject": record.get("subject_id", ""),
                "Session": record.get("session_id", ""),
                "Stage": stage_display_name,
                "Reference Stage": previous_stage_label,
                "Source Stage": record.get("source_stage", ""),
                "Raw Duration": format_duration_hms(record.get("raw_duration_sec", 0.0)),
                "Retained Duration": format_duration_hms(record.get("retained_duration_sec", 0.0)),
                "QC Status": record.get("qc_flag", ""),
            }
        ]
    )


def build_top_channels_table(
    channel_diagnostics: Mapping[str, object],
) -> pd.DataFrame:
    """Return the top-5 amplitude and top-5 line-noise channel ranking table.

    Flat/noisy channel names are surfaced in the residual-metrics table, so only
    the ranked problem-channel view is built here.
    """
    top_amp = list(channel_diagnostics.get("top_amplitude_channels") or [])
    top_noise = list(channel_diagnostics.get("top_line_noise_channels") or [])
    rank_rows = []
    max_rank = max(len(top_amp), len(top_noise), 0)
    for rank in range(max_rank):
        amp_ch, amp_val = top_amp[rank] if rank < len(top_amp) else ("", float("nan"))
        noise_ch, noise_val = top_noise[rank] if rank < len(top_noise) else ("", float("nan"))
        rank_rows.append({
            "Rank": rank + 1,
            "High Amplitude Channel": amp_ch,
            "Amplitude PTP (uV)": f"{amp_val:.1f}" if math.isfinite(amp_val) else "—",
            "High Line-Noise Channel": noise_ch,
            "Line-Noise Ratio": f"{noise_val:.3f}" if math.isfinite(noise_val) else "—",
        })
    rank_df = pd.DataFrame(rank_rows) if rank_rows else pd.DataFrame(
        columns=["Rank", "High Amplitude Channel", "Amplitude PTP (uV)",
                 "High Line-Noise Channel", "Line-Noise Ratio"]
    )
    return rank_df


def build_condition_comparison_table(segment_comparison: pd.DataFrame) -> pd.DataFrame:
    """Reformat segment_comparison into a human-readable pre-vs-post table."""
    if segment_comparison is None or segment_comparison.empty:
        return pd.DataFrame()

    def _fmt(series: pd.Series, digits: int = 2) -> pd.Series:
        return series.apply(lambda v: f"{v:.{digits}f}" if math.isfinite(float(v)) else "—")

    out = pd.DataFrame()
    out["Condition"] = segment_comparison.get("segment_type", pd.Series(dtype=str))

    if "n_usable_runs" in segment_comparison.columns:
        out["N Runs Usable"] = pd.to_numeric(segment_comparison["n_usable_runs"], errors="coerce").fillna(0).astype(int)
    if "total_duration_post_sec" in segment_comparison.columns:
        out["Mean Dur (s)"] = _fmt(pd.to_numeric(segment_comparison["total_duration_post_sec"], errors="coerce").fillna(float("nan")), 1)

    for label, pre_col, post_col in [
        ("Ampl uV",      "mean_amplitude_pre",        "mean_amplitude_post"),
        ("Line noise",   "mean_line_noise_pre",        "mean_line_noise_post"),
        ("HF/LF",        "mean_hf_lf_pre",            "mean_hf_lf_post"),
        ("Bad ch %",     "mean_pct_bad_channels_pre", "mean_pct_bad_channels_post"),
        ("Slope",        "mean_aperiodic_slope_pre",  "mean_aperiodic_slope_post"),
    ]:
        if pre_col in segment_comparison.columns:
            out[f"{label} (pre)"] = _fmt(pd.to_numeric(segment_comparison[pre_col], errors="coerce").fillna(float("nan")))
        if post_col in segment_comparison.columns:
            out[f"{label} (post)"] = _fmt(pd.to_numeric(segment_comparison[post_col], errors="coerce").fillna(float("nan")))

    return out.reset_index(drop=True)


def build_delta_table(record: Mapping[str, object], *, suffix: str, reference_label: str) -> pd.DataFrame:
    specs = (
        ("Mean amplitude", f"amplitude_mean_uv_delta_{suffix}", " uV"),
        ("Max amplitude", f"amplitude_max_uv_delta_{suffix}", " uV"),
        ("Bad channels", f"pct_bad_channels_delta_{suffix}", "%"),
        ("Line-noise ratio", f"line_noise_ratio_delta_{suffix}", ""),
        ("HF/LF ratio", f"hf_lf_ratio_delta_{suffix}", ""),
        ("Alpha peak", f"alpha_peak_hz_delta_{suffix}", " Hz"),
        ("Aperiodic slope", f"aperiodic_slope_delta_{suffix}", ""),
    )
    return build_record_metric_table(
        record, specs, value_col=f"Delta vs {reference_label}", skip_empty=True
    )


def build_retention_table(record: Mapping[str, object]) -> pd.DataFrame:
    """Retention table — condition-segment level only."""
    return pd.DataFrame(
        [
            {
                "Metric": "Clean time in condition segments",
                "Value": format_duration_hms(record.get("usable_condition_coverage_sec", 0.0)),
            },
            {
                "Metric": "Condition segment retention",
                "Value": _format_value(record.get("condition_coverage_retention_pct"), suffix="%"),
            },
        ]
    )



def build_residual_metrics_table(
    record: Mapping[str, object],
    channel_diagnostics: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    flat_names = ", ".join(
        str(ch) for ch in ((channel_diagnostics or {}).get("flat_channels") or [])
    ) or "None"
    noisy_names = ", ".join(
        str(ch) for ch in ((channel_diagnostics or {}).get("noisy_channels") or [])
    ) or "None"
    base = build_record_metric_table(
        record,
        (
            ("Mean amplitude", "amplitude_mean_uv", " uV"),
            ("Max amplitude", "amplitude_max_uv", " uV"),
        ),
    )
    extra = pd.DataFrame(
        [
            {"Metric": "Flat channels", "Value": f"{int(record.get('n_flat_channels', 0) or 0)}  \u2192  {flat_names}"},
            {"Metric": "Noisy channels", "Value": f"{int(record.get('n_noisy_channels', 0) or 0)}  \u2192  {noisy_names}"},
        ]
    )
    rest = build_record_metric_table(
        record,
        (
            ("Bad channels (%)", "pct_bad_channels", "%"),
            ("Line-noise ratio", "line_noise_ratio", ""),
            ("HF/LF ratio", "hf_lf_ratio", ""),
            ("Alpha peak", "alpha_peak_hz", " Hz"),
            ("Aperiodic slope", "aperiodic_slope", ""),
        ),
    )
    flag_reasons = pd.DataFrame(
        [{"Metric": "Flag reasons", "Value": record.get("qc_flag_reasons", "") or "None"}]
    )
    return pd.concat([base, extra, rest, flag_reasons], ignore_index=True)


def build_run_summary_table(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "Run": record.get("run_id", "") or record.get("source_stage", ""),
                "QC Status": record.get("qc_flag", ""),
                "Retention (%)": _format_value(record.get("duration_retention_pct")),
                "Bad Channels (%)": _format_value(record.get("pct_bad_channels")),
                "Line Noise": _format_value(record.get("line_noise_ratio")),
                "HF/LF": _format_value(record.get("hf_lf_ratio")),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary_table(
    runs_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    *,
    stage_display_name: str,
) -> pd.DataFrame:
    status_counts = runs_df.get("qc_flag", pd.Series(dtype=str)).fillna("unknown").astype(str).value_counts()
    return pd.DataFrame(
        [
            {
                "Stage": stage_display_name,
                "Subject-sessions": int(len(subjects_df)),
                "Records": int(len(runs_df)),
                "Usable": int(status_counts.get("usable", 0)),
                "Borderline": int(status_counts.get("borderline", 0)),
                "Unusable": int(status_counts.get("unusable", 0)),
            }
        ]
    )


def build_dataset_effect_table(runs_df: pd.DataFrame, *, suffix: str, reference_label: str) -> pd.DataFrame:
    specs = (
        ("Mean amplitude", f"amplitude_mean_uv_delta_{suffix}", " uV"),
        ("Max amplitude", f"amplitude_max_uv_delta_{suffix}", " uV"),
        ("Bad channels", f"pct_bad_channels_delta_{suffix}", "%"),
        ("Line-noise ratio", f"line_noise_ratio_delta_{suffix}", ""),
        ("HF/LF ratio", f"hf_lf_ratio_delta_{suffix}", ""),
        ("Alpha peak", f"alpha_peak_hz_delta_{suffix}", " Hz"),
        ("Aperiodic slope", f"aperiodic_slope_delta_{suffix}", ""),
    )
    return build_dataset_mean_metric_table(runs_df, specs, value_col=f"Mean delta vs {reference_label}")


def build_dataset_retention_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    duration_retention = pd.to_numeric(runs_df.get("duration_retention_pct"), errors="coerce")
    coverage_retention = pd.to_numeric(runs_df.get("condition_coverage_retention_pct"), errors="coerce")
    return pd.DataFrame(
        [
            {"Metric": "Mean recording retention", "Value": _format_value(duration_retention.mean(), suffix="%")},
            {"Metric": "Median recording retention", "Value": _format_value(duration_retention.median(), suffix="%")},
            {"Metric": "Mean condition coverage retention", "Value": _format_value(coverage_retention.mean(), suffix="%")},
            {"Metric": "Median condition coverage retention", "Value": _format_value(coverage_retention.median(), suffix="%")},
        ]
    )


def build_dataset_residual_metrics_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("Mean amplitude", "amplitude_mean_uv", " uV"),
        ("Max amplitude", "amplitude_max_uv", " uV"),
        ("Bad channels", "pct_bad_channels", "%"),
        ("Line-noise ratio", "line_noise_ratio", ""),
        ("HF/LF ratio", "hf_lf_ratio", ""),
        ("Alpha peak", "alpha_peak_hz", " Hz"),
        ("Aperiodic slope", "aperiodic_slope", ""),
    )
    return build_dataset_mean_metric_table(runs_df, specs, value_col="Mean")


def build_flag_reason_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    return _build_flag_reason_table(runs_df, reasons_column="qc_flag_reasons", count_label="Records")


def generate_subject_report(
    *,
    record: Mapping[str, object],
    previous_stage_label: str,
    raw_reference_label: str,
    stage_display_name: str,
    figures: Mapping[str, Path],
    run_summary_df: pd.DataFrame,
    output_path: Path,
    channel_diagnostics: Mapping[str, object] | None = None,
    autoreject_figures: Mapping[str, Path] | None = None,
    segment_comparison: pd.DataFrame | None = None,
) -> Path:
    report = Report(title=f"{stage_display_name} QC - {record.get('subject_session_prefix', record.get('subject_id', 'unknown'))}")

    overview = Section("Stage Overview", icon="🧭")
    _add_optional_table(
        overview,
        build_stage_overview_table(record, stage_display_name=stage_display_name, previous_stage_label=previous_stage_label),
        "Overview",
    )
    report.add_section(overview)

    # 1b. Pipeline Warnings
    warnings_raw = str(record.get("pipeline_warnings", "")).strip()
    if warnings_raw:
        formatted_warnings = "- " + warnings_raw.replace("; ", "\n- ")
        warn_section = Section("Pipeline Warnings", icon="⚠️")
        warn_section.add_markdown(
            "The following non-fatal issues were encountered during processing. Specific pipeline steps "
            "may have been skipped to ensure the rest of the subject run could complete:\n\n"
            f"{formatted_warnings}"
        )
        report.add_section(warn_section)

    if previous_stage_label != raw_reference_label:
        effect_prev = Section("Effect Vs Previous Stage", icon="↔️")
        _add_optional_table(effect_prev, build_delta_table(record, suffix="prev", reference_label=previous_stage_label), "Primary Deltas")
        report.add_section(effect_prev)

    effect_raw = Section("Effect Vs Raw", icon="📏")
    _add_optional_table(effect_raw, build_delta_table(record, suffix="raw", reference_label=raw_reference_label), "Raw Reference Deltas")
    report.add_section(effect_raw)

    retention = Section("Retention", icon="🧩")
    _add_optional_table(retention, build_retention_table(record), "Retention")
    report.add_section(retention)

    residual = Section("Residual Artifact Burden", icon="📉")
    _add_optional_table(residual, build_residual_metrics_table(record, channel_diagnostics=channel_diagnostics), "Residual Metrics")
    report.add_section(residual)

    # Channel diagnostics — only top-5 ranking table (flat/noisy names are now in residual metrics)
    if channel_diagnostics:
        ch_diag = Section("Channel Diagnostics", icon="📡")
        rank_df = build_top_channels_table(channel_diagnostics)
        _add_optional_table(ch_diag, rank_df, "Top 5 Problematic Channels")
        report.add_section(ch_diag)

    # Per-condition comparison: pre-base vs cleaned signal quality
    if segment_comparison is not None and not (isinstance(segment_comparison, pd.DataFrame) and segment_comparison.empty):
        cond_section = Section("Per-Condition: Pre vs Post", icon="🔬")
        cond_section.add_markdown(
            "Mean signal-quality metrics per experimental condition block. "
            "Metrics are computed on the full segment window (same basis as pre-base) "
            "and compared against the pre-base stage values from raw_qc_segments.csv."
        )
        _add_optional_table(cond_section, build_condition_comparison_table(segment_comparison), "Condition-Level Comparison")
        report.add_section(cond_section)

    if run_summary_df is not None and not run_summary_df.empty and len(run_summary_df) > 1:
        runs = Section("Per-Run Summary", icon="🗂️")
        _add_optional_table(runs, run_summary_df, "Per-Run Summary")
        report.add_section(runs)

    temporal = Section("Temporal Signal Quality", icon="⏲️")
    temporal.add_markdown(
        "Time-aligned signal quality diagnostics. Horizontal blocks represent the mean metric "
        "value for each experimental segment. Red 'x' markers and shaded backgrounds "
        "indicate segments flagged as bad by the automated diagnostics."
    )
    _add_images(
        temporal,
        figures,
        ("temporal_amplitude", "temporal_line_noise", "temporal_hf_lf_ratio")
    )
    report.add_section(temporal)

    figures_section = Section("Figures", icon="📈")
    # Only topomaps are shown per-subject (single-value histograms moved to dataset report)
    _add_images(
        figures_section,
        figures,
        (
            "amplitude_ptp_uv_topomap",
            "line_noise_ratio_topomap",
            "hf_lf_ratio_topomap",
        ),
    )
    report.add_section(figures_section)

    # AutoReject reject-log plots — one per condition (and chunk if applicable)
    if autoreject_figures:
        ar_section = Section("AutoReject Logs", icon="🔍")
        ar_section.add_markdown(
            "Epoch × channel reject-log heatmaps produced by AutoReject for each experimental "
            "condition. Red cells indicate epochs or channel spans flagged as bad."
        )
        for key in sorted(autoreject_figures):
            path = autoreject_figures[key]
            if path.exists():
                caption = key.split("/", 1)[-1].replace("_autoreject_", " → ").replace("_", " ")
                ar_section.add_element(ImageElement(str(path), caption=caption))
        report.add_section(ar_section)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path


def generate_dataset_report(
    *,
    runs_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    stage_display_name: str,
    previous_stage_label: str,
    raw_reference_label: str,
    figures: Mapping[str, Path],
    output_path: Path,
    condition_summary_df: pd.DataFrame | None = None,
) -> Path:
    report = Report(title=f"{stage_display_name} QC Dataset Report")

    definition = Section("QC Definition", icon="🎯")
    definition.add_markdown(
        f"{stage_display_name} QC summarizes cleaning effect, residual artifact burden, retention, and readiness."
    )
    _add_optional_table(definition, build_dataset_summary_table(runs_df, subjects_df, stage_display_name=stage_display_name), "Dataset Summary")
    report.add_section(definition)

    usability = Section("Usability Summary", icon="🧪")
    _add_optional_table(usability, build_flag_reason_table(runs_df), "Flag Reasons")
    _add_images(usability, figures, ("qc_flag", "qc_flag_reasons"))
    report.add_section(usability)

    if condition_summary_df is not None and not condition_summary_df.empty:
        cond_section = Section("Usability & Signal Quality per Condition", icon="🔬")
        cond_section.add_markdown(
            "Usability counts (number of runs retaining clean data) and mean signal-quality metrics "
            "per experimental condition. Pre = pre-base stage; Post = after cleaning."
        )
        _add_optional_table(cond_section, build_condition_comparison_table(condition_summary_df), "Condition-Level Averages")
        report.add_section(cond_section)

    if previous_stage_label != raw_reference_label:
        effect_prev = Section("Effect Vs Previous Stage", icon="↔️")
        _add_optional_table(effect_prev, build_dataset_effect_table(runs_df, suffix="prev", reference_label=previous_stage_label), "Primary Deltas")
        _add_images(
            effect_prev,
            figures,
            (
                "amplitude_mean_uv_delta_prev",
                "pct_bad_channels_delta_prev",
                "line_noise_ratio_delta_prev",
                "hf_lf_ratio_delta_prev",
                "alpha_peak_hz_delta_prev",
                "aperiodic_slope_delta_prev",
            ),
        )
        report.add_section(effect_prev)

    effect_raw = Section("Effect Vs Raw", icon="📏")
    _add_optional_table(effect_raw, build_dataset_effect_table(runs_df, suffix="raw", reference_label=raw_reference_label), "Raw Reference Deltas")
    _add_images(
        effect_raw,
        figures,
        (
            "amplitude_mean_uv_delta_raw",
            "pct_bad_channels_delta_raw",
            "line_noise_ratio_delta_raw",
            "hf_lf_ratio_delta_raw",
            "alpha_peak_hz_delta_raw",
            "aperiodic_slope_delta_raw",
        ),
    )
    report.add_section(effect_raw)

    retention = Section("Retention", icon="🧩")
    _add_optional_table(retention, build_dataset_retention_table(runs_df), "Retention")
    _add_images(retention, figures, ("duration_retention_pct", "condition_coverage_retention_pct"))
    report.add_section(retention)

    residual = Section("Residual Artifact Burden", icon="📉")
    _add_optional_table(residual, build_dataset_residual_metrics_table(runs_df), "Residual Metrics")
    _add_images(
        residual,
        figures,
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
    report.add_section(residual)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

"""Pre-base raw QC report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from coco_pipe.report.core import Report, Section

from eeg_adhd_epilepsy.reports._common import (
    add_images as _add_images,
    add_optional_table as _add_optional_table,
    build_dataset_mean_metric_table,
    build_flag_reason_table as _build_flag_reason_table,
    build_subject_overview_table,
    format_value as _format_value,
)


def build_usability_table(record: Mapping[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "QC Status", "Value": record.get("subject_flag", "")},
            {"Metric": "Flag Reasons", "Value": record.get("subject_flag_reasons", "") or "None"},
            {"Metric": "Mean amplitude", "Value": _format_value(record.get("amplitude_mean_uv"), suffix=" uV")},
            {"Metric": "Max amplitude", "Value": _format_value(record.get("amplitude_max_uv"), suffix=" uV")},
            {"Metric": "Flat channels", "Value": int(record.get("n_flat_channels", 0) or 0)},
            {"Metric": "Noisy channels", "Value": int(record.get("n_noisy_channels", 0) or 0)},
            {"Metric": "Bad channels", "Value": _format_value(record.get("pct_bad_channels"), suffix="%")},
            {"Metric": "Line-noise ratio", "Value": _format_value(record.get("line_noise_ratio"))},
            {"Metric": "HF/LF ratio", "Value": _format_value(record.get("hf_lf_ratio"))},
            {"Metric": "Alpha peak", "Value": _format_value(record.get("alpha_peak_hz"), suffix=" Hz")},
            {"Metric": "Aperiodic slope", "Value": _format_value(record.get("aperiodic_slope"))},
        ]
    )


def build_channel_diagnostics_tables(channel_diagnostics: Mapping[str, object] | None) -> dict[str, pd.DataFrame]:
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
                "Bad Channels (%)": _format_value(record.get("pct_bad_channels")),
                "Mean Amplitude (uV)": _format_value(record.get("amplitude_mean_uv")),
                "Max Amplitude (uV)": _format_value(record.get("amplitude_max_uv")),
                "Line Noise Ratio": _format_value(record.get("line_noise_ratio")),
                "HF/LF Ratio": _format_value(record.get("hf_lf_ratio")),
            }
        )
    return pd.DataFrame(rows).sort_values("Run") if rows else pd.DataFrame()


def build_dataset_summary_table(runs_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
    status_counts = runs_df.get("subject_flag", pd.Series(dtype=str)).fillna("unknown").astype(str).value_counts()
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
    return _build_flag_reason_table(runs_df, reasons_column="subject_flag_reasons", count_label="Runs")


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
    report = Report(title=f"Raw QC Report - {record.get('subject_session_prefix', record.get('subject_id', 'unknown'))}")

    overview = Section("Signal Overview", icon="🎛️")
    _add_optional_table(overview, build_subject_overview_table(record), "Subject Overview")
    report.add_section(overview)

    usability = Section("Usability", icon="🧪")
    _add_optional_table(usability, build_usability_table(record), "Signal Usability")
    report.add_section(usability)

    if run_summary_df is not None and not run_summary_df.empty:
        runs = Section("Per-Run Summary", icon="🗂️")
        _add_optional_table(runs, run_summary_df, "Run QC Summary")
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

    definition = Section("QC Definition", icon="🎯")
    definition.add_markdown(
        "Pre-base raw QC focuses on broad signal usability, channel quality, line-noise contamination, and high-frequency contamination."
    )
    _add_optional_table(definition, tables.get("dataset_summary_df", pd.DataFrame()), "Dataset Summary")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

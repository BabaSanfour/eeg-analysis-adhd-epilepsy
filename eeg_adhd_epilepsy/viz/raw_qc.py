"""Visualizations for the pre-base raw QC report family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coco_pipe.viz import plot_bar, plot_histogram

from eeg_adhd_epilepsy.viz import qc_plots, topo, utils

matplotlib.use("Agg")

plt.style.use("seaborn-v0_8-whitegrid")

FIGURE_FILENAMES = {
    "amplitude_mean_uv": "amplitude_mean_distribution.png",
    "amplitude_max_uv": "amplitude_max_distribution.png",
    "pct_bad_channels": "pct_bad_channels_distribution.png",
    "line_noise_ratio": "line_noise_ratio_distribution.png",
    "hf_lf_ratio": "hf_lf_ratio_distribution.png",
    "alpha_peak_hz": "alpha_peak_distribution.png",
    "aperiodic_slope": "aperiodic_slope_distribution.png",
    "flag_status": "flag_status_distribution.png",
    "flag_reasons": "flag_reason_counts.png",
    "segment_amplitude_mean_uv": "segment_amplitude_mean_by_type.png",
    "segment_line_noise_ratio": "segment_line_noise_ratio_by_type.png",
    "segment_hf_lf_ratio": "segment_hf_lf_ratio_by_type.png",
    "amplitude_ptp_uv_topomap": "amplitude_ptp_topomap.png",
    "line_noise_ratio_topomap": "line_noise_ratio_topomap.png",
    "hf_lf_ratio_topomap": "hf_lf_ratio_topomap.png",
}

RUN_METRIC_SPECS = (
    ("amplitude_mean_uv", "Run Mean Amplitude", "Mean amplitude (uV)"),
    ("amplitude_max_uv", "Run Max Amplitude", "Max amplitude (uV)"),
    ("pct_bad_channels", "Bad Channels", "Bad channels (%)"),
    ("line_noise_ratio", "Line Noise Ratio", "Line-noise ratio"),
    ("hf_lf_ratio", "HF/LF Ratio", "HF/LF ratio"),
    ("alpha_peak_hz", "Alpha Peak", "Alpha peak (Hz)"),
    ("aperiodic_slope", "Aperiodic Slope", "Aperiodic slope"),
)

SEGMENT_METRIC_SPECS = (
    ("segment_amplitude_mean_uv", "Segment Mean Amplitude", "Mean amplitude (uV)"),
    ("segment_line_noise_ratio", "Segment Line Noise Ratio", "Line-noise ratio"),
    ("segment_hf_lf_ratio", "Segment HF/LF Ratio", "HF/LF ratio"),
)

TOPOMAP_SPECS = (
    ("amplitude_ptp_uv", "Amplitude PTP Topomap", "viridis"),
    ("line_noise_ratio", "Line Noise Ratio Topomap", "RdBu_r"),
    ("hf_lf_ratio", "HF/LF Ratio Topomap", "RdBu_r"),
)


def _plot_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    out_path: Path,
) -> Path | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    bins = min(20, max(5, int(np.sqrt(len(clean)))))
    fig, ax = plot_histogram(
        clean,
        bins=bins,
        color="#4C72B0",
        title=title,
        xlabel=xlabel,
        ylabel="Runs",
        figsize=(7, 4),
    )
    ax.grid(True, axis="y", alpha=0.2)
    return utils.save_fig(fig, out_path)


def _plot_flag_status_distribution(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "subject_flag" not in runs_df:
        return None
    counts = runs_df["subject_flag"].fillna("unknown").astype(str).value_counts()
    if counts.empty:
        return None
    order = [
        status
        for status in ("usable", "borderline", "unusable", "unknown")
        if status in counts.index
    ]
    counts = counts.reindex(order)
    colors = ["#55A868", "#DD8452", "#C44E52", "#8172B2"][: len(counts)]
    fig, ax = plot_bar(
        counts,
        sort=False,
        color=colors,
        title="Run QC Status Distribution",
        ylabel="Runs",
        figsize=(6, 4),
    )
    ax.grid(True, axis="y", alpha=0.2)
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES["flag_status"])


def _plot_flag_reason_counts(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "subject_flag_reasons" not in runs_df:
        return None
    counts: dict[str, int] = {}
    for reasons in runs_df["subject_flag_reasons"].fillna(""):
        for reason in str(reasons).split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    ordered = pd.Series(counts)
    fig, ax = plot_bar(
        ordered,
        ascending=True,
        orientation="horizontal",
        color="#C44E52",
        title="QC Flag Reasons",
        xlabel="Runs",
        figsize=(8, max(3, len(ordered) * 0.4)),
    )
    ax.grid(True, axis="x", alpha=0.2)
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES["flag_reasons"])


def _save_topomap_figures(
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    if not topomap_aggregates:
        return {}
    paths: dict[str, Path] = {}
    for metric_key, title, cmap in TOPOMAP_SPECS:
        payload = topomap_aggregates.get(metric_key)
        if not payload:
            continue
        channels, values = payload
        fig = topo.plot_topomap_from_channel_values(
            channels,
            values,
            title=title,
            cmap=cmap,
            unit=None,
        )
        if fig is None:
            continue
        out_path = output_dir / FIGURE_FILENAMES[f"{metric_key}_topomap"]
        paths[f"{metric_key}_topomap"] = utils.save_fig(fig, out_path)
    return paths


def _save_segment_figures(segment_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    if segment_df is None or segment_df.empty:
        return {}
    paths: dict[str, Path] = {}
    for column, title, xlabel in SEGMENT_METRIC_SPECS:
        path = qc_plots.plot_segment_metric_distribution_by_type(
            segment_df,
            column=column,
            title=title,
            xlabel=xlabel,
            fig_dir=output_dir,
        )
        if path:
            paths[column] = path
    return paths


def save_subject_raw_qc_figures(
    segment_df: pd.DataFrame,
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _save_topomap_figures(topomap_aggregates, output_dir / "topomaps")
    figure_paths.update(_save_segment_figures(segment_df, output_dir / "segments"))
    return figure_paths


def save_dataset_raw_qc_figures(
    runs_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}
    for column, title, xlabel in RUN_METRIC_SPECS:
        if column not in runs_df:
            continue
        path = _plot_histogram(
            runs_df[column],
            title,
            xlabel,
            output_dir / FIGURE_FILENAMES[column],
        )
        if path:
            figure_paths[column] = path
    flag_status_path = _plot_flag_status_distribution(runs_df, output_dir)
    if flag_status_path:
        figure_paths["flag_status"] = flag_status_path
    flag_reason_path = _plot_flag_reason_counts(runs_df, output_dir)
    if flag_reason_path:
        figure_paths["flag_reasons"] = flag_reason_path
    figure_paths.update(_save_topomap_figures(topomap_aggregates, output_dir / "topomaps"))
    figure_paths.update(_save_segment_figures(segment_df, output_dir / "segments"))
    return figure_paths

"""Shared post-preprocessing QC figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eeg_adhd_epilepsy.viz.clean_qc as viz_clean

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")


RUN_METRIC_SPECS = (
    ("amplitude_mean_uv", "Mean Amplitude", "Mean amplitude (uV)"),
    ("amplitude_max_uv", "Max Amplitude", "Max amplitude (uV)"),
    ("pct_bad_channels", "Bad Channels", "Bad channels (%)"),
    ("line_noise_ratio", "Line Noise Ratio", "Line-noise ratio"),
    ("hf_lf_ratio", "HF/LF Ratio", "HF/LF ratio"),
    ("alpha_peak_hz", "Alpha Peak", "Alpha peak (Hz)"),
    ("aperiodic_slope", "Aperiodic Slope", "Aperiodic slope"),
)

DELTA_SPECS = (
    ("amplitude_mean_uv_delta_prev", "Mean Amplitude Change vs Previous", "Delta amplitude (uV)"),
    ("amplitude_mean_uv_delta_raw", "Mean Amplitude Change vs Raw", "Delta amplitude (uV)"),
    ("pct_bad_channels_delta_prev", "Bad Channel Change vs Previous", "Delta bad channels (%)"),
    ("pct_bad_channels_delta_raw", "Bad Channel Change vs Raw", "Delta bad channels (%)"),
    ("line_noise_ratio_delta_prev", "Line Noise Change vs Previous", "Delta line-noise ratio"),
    ("line_noise_ratio_delta_raw", "Line Noise Change vs Raw", "Delta line-noise ratio"),
    ("hf_lf_ratio_delta_prev", "HF/LF Change vs Previous", "Delta HF/LF ratio"),
    ("hf_lf_ratio_delta_raw", "HF/LF Change vs Raw", "Delta HF/LF ratio"),
    ("alpha_peak_hz_delta_prev", "Alpha Peak Change vs Previous", "Delta alpha peak (Hz)"),
    ("alpha_peak_hz_delta_raw", "Alpha Peak Change vs Raw", "Delta alpha peak (Hz)"),
    ("aperiodic_slope_delta_prev", "Aperiodic Slope Change vs Previous", "Delta aperiodic slope"),
    ("aperiodic_slope_delta_raw", "Aperiodic Slope Change vs Raw", "Delta aperiodic slope"),
)

RETENTION_SPECS = (
    ("duration_retention_pct", "Recording Retention", "Retention (%)"),
    ("condition_coverage_retention_pct", "Condition Coverage Retention", "Coverage retention (%)"),
)

TOPOMAP_SPECS = (
    ("amplitude_ptp_uv", "Amplitude PTP Topomap", "viridis"),
    ("line_noise_ratio", "Line Noise Ratio Topomap", "RdBu_r"),
    ("hf_lf_ratio", "HF/LF Ratio Topomap", "RdBu_r"),
)


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_flag_status_distribution(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "qc_flag" not in runs_df:
        return None
    counts = runs_df["qc_flag"].fillna("unknown").astype(str).value_counts()
    if counts.empty:
        return None
    order = [status for status in ("usable", "borderline", "unusable", "unknown") if status in counts.index]
    counts = counts.reindex(order)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(counts))
    colors = ["#55A868", "#DD8452", "#C44E52", "#8172B2"][: len(counts)]
    ax.bar(positions, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(counts.index)
    ax.set_title("QC Status Distribution")
    ax.set_ylabel("Records")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    return _save_fig(fig, output_dir / "qc_flag.png")


def _plot_flag_reason_counts(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    counts: dict[str, int] = {}
    for reasons in runs_df.get("qc_flag_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(reasons).split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    ordered = pd.Series(counts).sort_values(ascending=True)
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, max(3, len(ordered) * 0.4)))
    ax.barh(positions, ordered.values, color="#C44E52")
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered.index)
    ax.set_title("QC Flag Reasons")
    ax.set_xlabel("Records")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    return _save_fig(fig, output_dir / "qc_flag_reasons.png")


def _save_topomap_figures(
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> Dict[str, Path]:
    """Render topomaps composited 2-per-row so they fit compactly side-by-side in the report."""
    if not topomap_aggregates:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    specs = [(k, t, c) for k, t, c in TOPOMAP_SPECS if topomap_aggregates.get(k)]
    # Pair specs into rows of 2
    for i in range(0, len(specs), 2):
        pair = specs[i : i + 2]
        n = len(pair)
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.4))
        if n == 1:
            axes = [axes]
        for ax, (metric_key, title, cmap) in zip(axes, pair):
            channels, values = topomap_aggregates[metric_key]
            sub_fig = viz_clean.plot_topomap_from_channel_values(
                channels, values, title=title, cmap=cmap, unit=None,
            )
            if sub_fig is None:
                ax.set_visible(False)
                continue
            sub_fig.canvas.draw()
            w, h = sub_fig.canvas.get_width_height()
            img = np.frombuffer(sub_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            plt.close(sub_fig)
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        # Save under the first metric key of the pair; also register each key individually
        primary_key = pair[0][0]
        out_path = output_dir / f"{primary_key}_topomap.png"
        _save_fig(fig, out_path)
        for metric_key, _, _ in pair:
            paths[f"{metric_key}_topomap"] = out_path
    return paths


def save_subject_preproc_qc_figures(
    *,
    record: Mapping[str, object],
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths.update(_save_topomap_figures(topomap_aggregates, output_dir / "topomaps"))
    return paths


def save_dataset_preproc_qc_figures(
    *,
    runs_df: pd.DataFrame,
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    
    # Collect clean data for histograms, then pair them 2-per-row
    hist_specs = []
    for column, title, xlabel in RUN_METRIC_SPECS + DELTA_SPECS + RETENTION_SPECS:
        if column in runs_df:
            clean_series = pd.to_numeric(runs_df[column], errors="coerce").dropna()
            if not clean_series.empty:
                hist_specs.append((column, title, xlabel, clean_series))
                
    for i in range(0, len(hist_specs), 2):
        pair = hist_specs[i : i + 2]
        n = len(pair)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 2.8))
        if n == 1:
            axes = [axes]
            
        for ax, (column, title, xlabel, clean_series) in zip(axes, pair):
            bins = min(20, max(5, int(np.sqrt(len(clean_series)))))
            ax.hist(clean_series, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Records", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.2)
            
        plt.tight_layout(pad=0.5)
        primary_key = pair[0][0]
        out_path = output_dir / f"{primary_key}_hist.png"
        _save_fig(fig, out_path)
        for p in pair:
            paths[p[0]] = out_path

    flag_status_path = _plot_flag_status_distribution(runs_df, output_dir)
    if flag_status_path:
        paths["qc_flag"] = flag_status_path
    flag_reasons_path = _plot_flag_reason_counts(runs_df, output_dir)
    if flag_reasons_path:
        paths["qc_flag_reasons"] = flag_reasons_path
    paths.update(_save_topomap_figures(topomap_aggregates, output_dir / "topomaps"))
    return paths

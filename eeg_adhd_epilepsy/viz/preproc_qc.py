"""Shared post-preprocessing QC figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from coco_pipe.viz import plot_bar, plot_histogram

from eeg_adhd_epilepsy.viz import qc_plots, topo, utils

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


def _plot_flag_status_distribution(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "qc_flag" not in runs_df:
        return None
    counts = runs_df["qc_flag"].fillna("unknown").astype(str).value_counts()
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
        title="QC Status Distribution",
        ylabel="Records",
        figsize=(6, 4),
    )
    ax.grid(True, axis="y", alpha=0.2)
    return utils.save_fig(fig, output_dir / "qc_flag.png")


def _plot_flag_reason_counts(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    counts: dict[str, int] = {}
    for reasons in runs_df.get("qc_flag_reasons", pd.Series(dtype=str)).fillna(""):
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
        xlabel="Records",
        figsize=(8, max(3, len(ordered) * 0.4)),
    )
    ax.grid(True, axis="x", alpha=0.2)
    return utils.save_fig(fig, output_dir / "qc_flag_reasons.png")


def _save_topomap_figures(
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
    bad_channels: list[str] | None = None,
) -> dict[str, Path]:
    """Render topomaps composited 2-per-row so they fit compactly side-by-side in the report."""
    if not topomap_aggregates:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
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
            sub_fig = topo.plot_topomap_from_channel_values(
                channels, values, title=title, cmap=cmap, unit=None, bad_channels=bad_channels
            )
            if sub_fig is None:
                ax.set_visible(False)
                continue
            sub_fig.canvas.draw()
            w, h = sub_fig.canvas.get_width_height()
            img = np.frombuffer(sub_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
                :, :, :3
            ]
            plt.close(sub_fig)
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        # Save under the first metric key of the pair; also register each key individually
        primary_key = pair[0][0]
        out_path = output_dir / f"{primary_key}_topomap.png"
        utils.save_fig(fig, out_path)
        for metric_key, _, _ in pair:
            paths[f"{metric_key}_topomap"] = out_path
    return paths


def save_subject_preproc_qc_figures(
    *,
    record: Mapping[str, object],
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    segments_df: pd.DataFrame | None = None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Extract bad channels from diagnostics if available
    diag = record.get("channel_diagnostics", {})
    bad_channels = list(set((diag.get("flat_channels") or []) + (diag.get("noisy_channels") or [])))

    paths.update(
        _save_topomap_figures(
            topomap_aggregates, output_dir / "topomaps", bad_channels=bad_channels
        )
    )

    if segments_df is not None and not segments_df.empty:
        temporal_dir = output_dir / "temporal"
        temporal_paths = save_subject_temporal_qc_figures(segments_df, temporal_dir)
        paths.update(temporal_paths)

    return paths


def save_subject_temporal_qc_figures(
    segments_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Save block-style temporal QC plots for a subject."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Amplitude Blocks
    if "amplitude_mean_uv" in segments_df.columns:
        fig = plot_segment_metric_blocks(
            segments_df, "amplitude_mean_uv", "Mean Amplitude per Segment", "uV"
        )
        if fig:
            out_path = output_dir / "temporal_amplitude.png"
            utils.save_fig(fig, out_path)
            paths["temporal_amplitude"] = out_path

    # 2. Line Noise Blocks
    if "line_noise_ratio" in segments_df.columns:
        fig = plot_segment_metric_blocks(
            segments_df, "line_noise_ratio", "Line Noise Ratio per Segment", "Ratio"
        )
        if fig:
            out_path = output_dir / "temporal_line_noise.png"
            utils.save_fig(fig, out_path)
            paths["temporal_line_noise"] = out_path

    # 3. HF/LF Ratio Blocks
    if "hf_lf_ratio" in segments_df.columns:
        fig = plot_segment_metric_blocks(
            segments_df, "hf_lf_ratio", "HF/LF Ratio per Segment", "Ratio"
        )
        if fig:
            out_path = output_dir / "temporal_hf_lf_ratio.png"
            utils.save_fig(fig, out_path)
            paths["temporal_hf_lf_ratio"] = out_path

    return paths


def save_dataset_preproc_qc_figures(
    *,
    runs_df: pd.DataFrame,
    topomap_aggregates: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

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
            plot_histogram(
                clean_series,
                bins=bins,
                color="#4C72B0",
                title=title,
                xlabel=xlabel,
                ylabel="Records",
                ax=ax,
            )
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.2)

        plt.tight_layout(pad=0.5)
        primary_key = pair[0][0]
        out_path = output_dir / f"{primary_key}_hist.png"
        utils.save_fig(fig, out_path)
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


def truncate_plot_data(
    data: np.ndarray,
    sfreq: float | None,
    max_seconds: float | None,
) -> np.ndarray:
    """Truncate data on the time axis for faster plotting."""
    if sfreq is None or max_seconds is None:
        return data
    n_plot = min(data.shape[-1], int(float(sfreq) * float(max_seconds)))
    if data.ndim == 2:
        return data[:, :n_plot]
    if data.ndim == 3:
        return data[:, :, :n_plot]
    return data


def plot_segment_metric_blocks(
    segments_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
) -> matplotlib.figure.Figure | None:
    """Plot mean metric value per segment type as time-aligned blocks."""
    if column not in segments_df or "t_start" not in segments_df or "t_stop" not in segments_df:
        return None
    df = segments_df.copy()
    df["metric"] = pd.to_numeric(df[column], errors="coerce")
    df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce")
    df["t_stop"] = pd.to_numeric(df["t_stop"], errors="coerce")
    df["segment_type"] = df.get("segment_type", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    val = df.get("segment_flag_bad")
    if val is None:
        df["flag_bad_bool"] = False
    else:
        df["flag_bad_bool"] = pd.to_numeric(val, errors="coerce").fillna(0).astype(bool)
    df = df.dropna(subset=["metric", "t_start", "t_stop"])
    if df.empty:
        return None
    palette = qc_plots.get_segment_palette(df["segment_type"].unique())
    means = df.groupby("segment_type")["metric"].mean()
    boundaries = sorted(set(df["t_start"].dropna().tolist()) | set(df["t_stop"].dropna().tolist()))

    fig, ax = plt.subplots(figsize=(8, 4))
    seen_labels: dict[str, matplotlib.artist.Artist] = {}
    flagged_added = False
    for seg_type, group in df.sort_values("t_start").groupby("segment_type", dropna=False):
        mean_val = means.get(seg_type)
        color = palette.get(seg_type, "#4C72B0")
        for _, row in group.iterrows():
            start = float(row["t_start"])
            stop = float(row["t_stop"])
            if not (np.isfinite(start) and np.isfinite(stop)) or stop <= start:
                continue
            artist = ax.hlines(mean_val, start, stop, colors=color, linewidth=6)
            if seg_type not in seen_labels:
                seen_labels[seg_type] = artist
            if row.get("flag_bad_bool", False):
                midpoint = start + (stop - start) / 2.0
                flag_artist = ax.scatter(
                    midpoint, mean_val, color="red", marker="x", s=70, zorder=5
                )
                if not flagged_added:
                    seen_labels["Flagged"] = flag_artist
                    flagged_added = True
                ax.axvspan(start, stop, color="red", alpha=0.08, linewidth=0)

    for boundary in boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1, alpha=0.3)

    if seen_labels:
        labels, handles = zip(*seen_labels.items())
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def save_eeg_snapshot(
    raw: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    label: str,
    start: float = 30.0,
    duration: float = 60.0,
    n_channels: int = 20,
) -> str:
    """Save a butterfly plot snapshot of EEG channels."""
    raw_eeg = raw.copy().pick_types(eeg=True, exclude="bads")
    max_start = max(0, raw_eeg.times[-1] - duration)
    start = min(start, max_start)
    ch_names = raw_eeg.ch_names[: min(n_channels, len(raw_eeg.ch_names))]
    stop = min(start + duration, raw_eeg.times[-1])
    sfreq = raw_eeg.info["sfreq"]
    data, times = raw_eeg[ch_names, int(start * sfreq) : int(stop * sfreq)]

    fig, ax = plt.subplots(figsize=(16, 5))
    data_uv = data * 1e6
    for i, _ in enumerate(ch_names):
        ax.plot(times[: data.shape[1]], data_uv[i], linewidth=0.4, alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{subject_id} — {label.replace('_', ' ').title()}")
    ax.set_xlim(times[0], times[min(data.shape[1] - 1, len(times) - 1)])
    ax.grid(True, alpha=0.3)

    path = fig_dir / f"{subject_id}_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_artifact_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    artifact_type: str,
    window: float = 10.0,
    n_channels: int = 8,
    search_start: float = 30.0,
) -> str:
    """Save a 3-panel before/after/removed comparison at the largest artifact peak."""
    from scipy.signal import find_peaks

    eeg_before = raw_before.copy().pick_types(eeg=True, exclude="bads")
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude="bads")
    sfreq = eeg_before.info["sfreq"]

    data_before = eeg_before.get_data()
    data_after = eeg_after.get_data()

    n_samples = min(data_before.shape[1], data_after.shape[1])
    data_before = data_before[:, :n_samples]
    data_after = data_after[:, :n_samples]
    removed = data_before - data_after

    ch_names = eeg_before.ch_names
    if artifact_type == "eog":
        frontal = [
            i
            for i, ch in enumerate(ch_names)
            if any(f in ch.upper() for f in ["FP1", "FP2", "F3", "F4", "FZ", "AF"])
        ]
        if not frontal:
            frontal = list(range(min(4, len(ch_names))))
        search_sample = int(search_start * sfreq)
        if search_sample < n_samples:
            envelope = np.abs(removed[frontal, search_sample:]).mean(axis=0)
            offset_samples = search_sample
        else:
            envelope = np.abs(removed[frontal]).mean(axis=0)
            offset_samples = 0
        display_idx = frontal[:n_channels]
    else:
        search_sample = int(search_start * sfreq)
        if search_sample < n_samples:
            ch_power = np.sum(removed[:, search_sample:] ** 2, axis=1)
            top_ch = np.argsort(ch_power)[::-1][:n_channels]
            envelope = np.abs(removed[top_ch[0], search_sample:])
            offset_samples = search_sample
        else:
            ch_power = np.sum(removed**2, axis=1)
            top_ch = np.argsort(ch_power)[::-1][:n_channels]
            envelope = np.abs(removed[top_ch[0]])
            offset_samples = 0
        display_idx = list(top_ch)

    min_dist = int(0.5 * sfreq)
    peaks, properties = find_peaks(envelope, distance=min_dist, height=np.percentile(envelope, 95))

    if len(peaks) == 0:
        peak_idx = int(np.argmax(envelope))
    else:
        peak_idx = peaks[np.argmax(properties["peak_heights"])]

    peak_idx = peak_idx + offset_samples
    peak_time = peak_idx / sfreq
    half_win = window / 2
    t_start = max(0, peak_time - half_win)
    t_end = min(n_samples / sfreq, peak_time + half_win)
    s_start = int(t_start * sfreq)
    s_end = int(t_end * sfreq)
    times = np.arange(s_start, s_end) / sfreq

    display_names = [ch_names[i] for i in display_idx]
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    artifact_label = artifact_type.upper()
    colors = plt.cm.tab10(np.linspace(0, 1, len(display_idx)))

    for panel_idx, (ax, title_sfx, signal) in enumerate(
        [
            (axes[0], "Before (Original)", data_before),
            (axes[1], "After (Cleaned)", data_after),
            (axes[2], f"Removed ({artifact_label} Artifact)", removed),
        ]
    ):
        for j, ch_idx in enumerate(display_idx):
            sig_uv = signal[ch_idx, s_start:s_end] * 1e6
            ax.plot(
                times[: len(sig_uv)],
                sig_uv,
                linewidth=0.6,
                alpha=0.8,
                color=colors[j],
                label=display_names[j] if panel_idx == 0 else None,
            )
        ax.set_ylabel("µV")
        ax.set_title(f"{title_sfx}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(peak_time, color="red", linestyle="--", alpha=0.5, linewidth=1)

    axes[2].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=min(4, len(display_idx)))
    fig.suptitle(
        f"{subject_id} — {artifact_label} Artifact Removal (peak at {peak_time:.2f}s)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = fig_dir / f"{subject_id}_{artifact_type}_artifact_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_removed_variance_topomap(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    title: str = "Removed Variance",
) -> matplotlib.figure.Figure | None:
    """Plot topographic map of variance difference (before - after)."""
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude="bads")
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude="bads")
    n_samples = min(eeg_before.n_times, eeg_after.n_times)
    data_before = eeg_before.get_data()[:, :n_samples]
    data_after = eeg_after.get_data()[:, :n_samples]
    var_before = np.var(data_before, axis=1)
    var_after = np.var(data_after, axis=1)
    var_removed = np.maximum(var_before - var_after, 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(var_removed, eeg_before.info, axes=ax, show=False, cmap="Reds")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label("Variance (V^2)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_channel_variance_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    subject_id: str,
    title: str = "Channel Variance: Before vs After Correction",
) -> matplotlib.figure.Figure:
    """Plot a comparison of per-channel variance before and after correction."""
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude="bads")
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude="bads")
    common_chs = [ch for ch in eeg_before.ch_names if ch in eeg_after.ch_names]
    eeg_before.pick_channels(common_chs)
    eeg_after.pick_channels(common_chs)
    n_samples = min(eeg_before.n_times, eeg_after.n_times)
    data_before = eeg_before.get_data()[:, :n_samples]
    data_after = eeg_after.get_data()[:, :n_samples]
    var_before = np.var(data_before, axis=1) * (1e6**2)
    var_after = np.var(data_after, axis=1) * (1e6**2)
    df = pd.DataFrame({"Channel": common_chs, "Before": var_before, "After": var_after})
    fig, ax = plt.subplots(figsize=(max(10, len(common_chs) * 0.3), 6))
    x = np.arange(len(common_chs))
    width = 0.35
    ax.bar(x - width / 2, df["Before"], width, label="Before", color="indianred", alpha=0.7)
    ax.bar(x + width / 2, df["After"], width, label="After", color="mediumseagreen", alpha=0.7)
    ax.set_ylabel("Variance (uV^2)")
    ax.set_title(f"{subject_id} - {title}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Channel"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def save_dss_pre_plots(
    estimator: Any,
    fit_data: np.ndarray,
    eeg_info: mne.Info,
    fig_dir: Path,
    subject_id: str,
    file_prefix: str,
    *,
    sfreq: float | None = None,
    fit_max_seconds: float | None = None,
    include_score: bool = True,
    include_component_summary: bool = True,
    include_spatial_patterns: bool = True,
    include_component_time_series: bool = True,
    summary_n_components: int = 3,
    spatial_n_components: int = 3,
    time_series_n_components: int = 5,
) -> dict[str, str]:
    """Save shared pre-cleaning DSS diagnostic plots."""
    from mne_denoise.viz import (
        plot_component_summary,
        plot_component_time_series,
        plot_score_curve,
        plot_spatial_patterns,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    fit_data_plot = truncate_plot_data(fit_data, sfreq=sfreq, max_seconds=fit_max_seconds)
    filters = getattr(estimator, "filters_", None)
    n_available = (
        int(filters.shape[0]) if isinstance(filters, np.ndarray) and filters.ndim >= 2 else 0
    )
    if include_score:
        fig_score = plot_score_curve(estimator, show=False)
        if fig_score:
            score_path = fig_dir / f"{subject_id}_{file_prefix}_score.png"
            fig_score.savefig(score_path, dpi=150, bbox_inches="tight")
            plt.close(fig_score)
            plot_paths["score_curve"] = str(score_path)
    if n_available > 0:
        if include_component_summary:
            fig_comp = plot_component_summary(
                estimator,
                fit_data_plot,
                info=eeg_info,
                n_components=min(summary_n_components, n_available),
                show=False,
            )
            if fig_comp:
                comp_path = fig_dir / f"{subject_id}_{file_prefix}_comps.png"
                fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
                plt.close(fig_comp)
                plot_paths["component_summary"] = str(comp_path)
        if include_spatial_patterns:
            fig_topo = plot_spatial_patterns(
                estimator,
                info=eeg_info,
                n_components=min(spatial_n_components, n_available),
                show=False,
            )
            if fig_topo:
                topo_path = fig_dir / f"{subject_id}_{file_prefix}_topo.png"
                fig_topo.savefig(topo_path, dpi=150, bbox_inches="tight")
                plt.close(fig_topo)
                plot_paths["spatial_patterns"] = str(topo_path)
        if include_component_time_series:
            fig_ts = plot_component_time_series(
                estimator,
                fit_data_plot,
                n_components=min(time_series_n_components, n_available),
                show=False,
            )
            if fig_ts:
                ts_path = fig_dir / f"{subject_id}_{file_prefix}_timeseries.png"
                fig_ts.savefig(ts_path, dpi=150, bbox_inches="tight")
                plt.close(fig_ts)
                plot_paths["component_time_series"] = str(ts_path)
    return plot_paths


def save_dss_post_plots(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    file_prefix: str,
    overlay_title: str,
    *,
    fmax: float = 50.0,
    include_time_course: bool = True,
    window_seconds: float = 60.0,
    start_seconds: float = 30.0,
) -> dict[str, str]:
    """Save shared post-cleaning DSS comparison plots."""
    from mne_denoise.viz import (
        plot_overlay_comparison,
        plot_psd_comparison,
        plot_time_course_comparison,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    fig_psd = plot_psd_comparison(raw_before, raw_after, fmax=fmax, show=False)
    if fig_psd:
        psd_path = fig_dir / f"{subject_id}_{file_prefix}_psd.png"
        fig_psd.savefig(psd_path, dpi=150, bbox_inches="tight")
        plt.close(fig_psd)
        plot_paths["psd_comparison"] = str(psd_path)
    fig_ov = plot_overlay_comparison(
        raw_before,
        raw_after,
        start=start_seconds,
        stop=min(raw_before.times[-1], start_seconds + window_seconds),
        title=overlay_title,
        show=False,
    )
    if fig_ov:
        ov_path = fig_dir / f"{subject_id}_{file_prefix}_overlay.png"
        fig_ov.savefig(ov_path, dpi=150, bbox_inches="tight")
        plt.close(fig_ov)
        plot_paths["overlay_comparison"] = str(ov_path)
    if include_time_course:
        sfreq = raw_before.info["sfreq"]
        s_start = int(start_seconds * sfreq)
        s_stop = int(min(raw_before.times[-1], start_seconds + window_seconds) * sfreq)
        fig_tc = plot_time_course_comparison(
            raw_before, raw_after, start=s_start, stop=s_stop, show=False
        )
        if fig_tc:
            tc_path = fig_dir / f"{subject_id}_{file_prefix}_timecourse.png"
            fig_tc.savefig(tc_path, dpi=150, bbox_inches="tight")
            plt.close(fig_tc)
            plot_paths["time_course_comparison"] = str(tc_path)
    return plot_paths


def save_ica_sources_snapshot(
    ica: Any,
    raw: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    picks: list[int],
    label: str,
    start: float = 30.0,
    duration: float = 20.0,
) -> str:
    """Save a clean stacked plot of selected ICA sources."""
    sources_raw = ica.get_sources(raw)
    sfreq = sources_raw.info["sfreq"]
    t_start = min(start, max(0, sources_raw.times[-1] - duration))
    t_stop = min(t_start + duration, sources_raw.times[-1])
    picked_names = [sources_raw.ch_names[i] for i in picks]
    data, times = sources_raw[picks, int(t_start * sfreq) : int(t_stop * sfreq)]
    n_chs = len(picks)
    fig_height = max(2, 1.2 * n_chs)
    fig, axes = plt.subplots(n_chs, 1, figsize=(10, fig_height), sharex=True)
    if n_chs == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(times, data[i], color="black", linewidth=0.6)
        ax.set_ylabel(picked_names[i], rotation=0, labelpad=25, verticalalignment="center")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        if i < n_chs - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} - {label} Excluded ICA Components", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = fig_dir / f"{subject_id}_{label.lower()}_ica_sources.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# -----------------------------------------------------------------------------
# Comparison Suite
# -----------------------------------------------------------------------------


def plot_compare_psd(
    raw_orig: mne.io.BaseRaw,
    raw_dss: mne.io.BaseRaw,
    raw_ica: mne.io.BaseRaw,
    subject_id: str,
    fig_dir: Path,
) -> str:
    """Plot PSD comparison (Side-by-side + Overlay)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    eeg_picks = mne.pick_types(raw_orig.info, eeg=True, exclude="bads")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, raw, label, color in [
        (axes[0], raw_orig, "Original", "#666666"),
        (axes[1], raw_dss, "DSS", "#2196F3"),
        (axes[2], raw_ica, "ICA", "#FF5722"),
    ]:
        try:
            psd = raw.compute_psd(fmin=0.5, fmax=50, picks=eeg_picks, verbose=False)
            pdata = psd.get_data() * 1e12
            ax.semilogy(psd.freqs, pdata.mean(axis=0), color=color, lw=2, label=label)
            ax.fill_between(
                psd.freqs,
                np.percentile(pdata, 5, axis=0),
                np.percentile(pdata, 95, axis=0),
                alpha=0.2,
                color=color,
            )
            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes, ha="center")
    axes[0].set_ylabel("Power (uV^2/Hz)")
    fig.suptitle(f"{subject_id} - PSD Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / f"{subject_id}_psd_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_butterfly(
    raw_orig: mne.io.BaseRaw,
    raw_dss: mne.io.BaseRaw,
    raw_ica: mne.io.BaseRaw,
    subject_id: str,
    fig_dir: Path,
    start: float = 30.0,
    duration: float = 10.0,
) -> str:
    """Plot Butterfly comparison for a specific segment."""
    picks = mne.pick_types(raw_orig.info, eeg=True, exclude="bads")
    t_start = min(start, max(0, raw_orig.times[-1] - duration))
    n_start = int(t_start * raw_orig.info["sfreq"])
    n_dur = int(duration * raw_orig.info["sfreq"])
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, sharey=True)
    times = raw_orig.times[n_start : n_start + n_dur] - raw_orig.times[n_start]
    for ax, raw, label, color in [
        (axes[0], raw_orig, "Original", "#888888"),
        (axes[1], raw_dss, "DSS Corrected", "#2196F3"),
        (axes[2], raw_ica, "ICA Corrected", "#FF5722"),
    ]:
        data = raw.get_data(picks=picks)[:, n_start : n_start + n_dur] * 1e6
        for ch in data:
            ax.plot(times, ch, color=color, alpha=0.3, lw=0.5)
        ax.set_ylabel("uV")
        ax.set_title(label, fontweight="bold", loc="left")
        ax.grid(alpha=0.2)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(
        f"{subject_id} - Signal Butterfly ({t_start:.0f}-{t_start + duration:.0f}s)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = fig_dir / f"{subject_id}_butterfly.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_band_power(
    bp_orig: dict[str, float],
    bp_dss: dict[str, float],
    bp_ica: dict[str, float],
    subject_id: str,
    fig_dir: Path,
) -> str:
    """Plot band power comparison across methods."""
    bands = list(bp_orig.keys())
    x = np.arange(len(bands))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width,
        [bp_orig[b] * 1e12 for b in bands],
        width,
        label="Original",
        color="#888888",
        alpha=0.8,
    )
    ax.bar(x, [bp_dss[b] * 1e12 for b in bands], width, label="DSS", color="#2196F3", alpha=0.8)
    ax.bar(
        x + width, [bp_ica[b] * 1e12 for b in bands], width, label="ICA", color="#FF5722", alpha=0.8
    )
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=12)
    ax.set_ylabel("Power (uV^2/Hz)")
    ax.set_title(f"{subject_id} - Band Power Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = fig_dir / f"{subject_id}_band_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_channel_correlation(
    corr_map: dict[str, float], subject_id: str, fig_dir: Path
) -> str:
    """Plot channel correlation between DSS and ICA versions."""
    channels = list(corr_map.keys())
    values = list(corr_map.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if v > 0.95 else "#FFC107" if v > 0.9 else "#F44336" for v in values]
    ax.bar(range(len(channels)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.95, color="green", ls="--")
    ax.axhline(0.90, color="orange", ls="--")
    ax.set_ylim(min(0.5, min(values) - 0.05 if values else 1.0), 1.02)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"{subject_id} - DSS vs ICA Channel Correlation", fontweight="bold")
    ax.grid(axis="y", opacity=0.3)
    plt.tight_layout()
    path = fig_dir / f"{subject_id}_channel_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_variance_topomaps(
    raw_orig: mne.io.BaseRaw,
    raw_dss: mne.io.BaseRaw,
    raw_ica: mne.io.BaseRaw,
    subject_id: str,
    fig_dir: Path,
) -> str:
    """Compare topographic distribution of variance removed by each method."""
    picks = mne.pick_types(raw_orig.info, eeg=True, exclude="bads")
    n_samples = min(raw_orig.n_times, raw_dss.n_times, raw_ica.n_times)
    var_orig = np.var(raw_orig.get_data(picks=picks)[:, :n_samples], axis=1)
    rem_dss = np.maximum(0, var_orig - np.var(raw_dss.get_data(picks=picks)[:, :n_samples], axis=1))
    rem_ica = np.maximum(0, var_orig - np.var(raw_ica.get_data(picks=picks)[:, :n_samples], axis=1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mne.viz.plot_topomap(var_orig, raw_orig.info, axes=axes[0], show=False)
    axes[0].set_title("Original Variance")
    mne.viz.plot_topomap(rem_dss, raw_orig.info, axes=axes[1], show=False, cmap="Reds")
    axes[1].set_title("Variance Removed (DSS)")
    mne.viz.plot_topomap(rem_ica, raw_orig.info, axes=axes[2], show=False, cmap="Reds")
    axes[2].set_title("Variance Removed (ICA)")
    fig.suptitle(f"{subject_id} - Spatial Distribution of Variance Reduction", fontsize=14)
    plt.tight_layout()
    path = fig_dir / f"{subject_id}_variance_topomap_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_timing(timing_data: list[dict[str, Any]], fig_dir: Path) -> str:
    """Plot processing time comparison across subjects."""
    df = pd.DataFrame(timing_data)
    if df.empty:
        return ""
    subjects = sorted(df["subject"].unique())
    x = np.arange(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.5), 5))
    for i, method in enumerate(["dss", "ica"]):
        mdf = df[df["method"] == method].set_index("subject")
        ax.bar(
            x + (i - 0.5) * width,
            [mdf.loc[s, "duration_sec"] if s in mdf.index else 0 for s in subjects],
            width,
            label=method.upper(),
            color="#2196F3" if method == "dss" else "#FF5722",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Processing Time: DSS vs ICA", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", opacity=0.3)
    plt.tight_layout()
    path = fig_dir / "timing_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_components_removed(comp_data: list[dict[str, Any]], fig_dir: Path) -> str:
    """Plot average components removed per artifact type."""
    df = pd.DataFrame(comp_data)
    if df.empty:
        return ""
    arts = ["eog", "ecg", "emg"]
    x = np.arange(len(arts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(["dss", "ica"]):
        mdf = df[df["method"] == method]
        ax.bar(
            x + (i - 0.5) * width,
            [
                mdf[f"{a}_components"].mean() if f"{a}_components" in mdf.columns else 0
                for a in arts
            ],
            width,
            label=method.upper(),
            color="#2196F3" if method == "dss" else "#FF5722",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in arts])
    ax.set_ylabel("Components Removed")
    ax.set_title("Artifact Components Removed: DSS vs ICA", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", opacity=0.3)
    plt.tight_layout()
    path = fig_dir / "components_removed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_variance_removed(var_data: list[dict[str, Any]], fig_dir: Path) -> str:
    """Plot variance removed percentage across subjects."""
    df = pd.DataFrame(var_data)
    if df.empty:
        return ""
    subjs = sorted(df["subject"].unique())
    x = np.arange(len(subjs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(subjs) * 1.5), 5))
    for i, m in enumerate(["dss", "ica"]):
        mdf = df[df["method"] == m].set_index("subject")
        ax.bar(
            x + (i - 0.5) * width,
            [mdf.loc[s, "variance_removed_pct"] if s in mdf.index else 0 for s in subjs],
            width,
            label=m.upper(),
            color="#2196F3" if m == "dss" else "#FF5722",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(subjs, rotation=30, ha="right")
    ax.set_ylabel("Variance Removed (%)")
    ax.set_title("Total Variance Removed: DSS vs ICA", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", opacity=0.3)
    plt.tight_layout()
    path = fig_dir / "variance_removed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_summary_dashboard(metrics_df: pd.DataFrame, fig_dir: Path) -> str:
    """Plot comprehensive comparison dashboard."""
    if metrics_df.empty:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"dss": "#2196F3", "ica": "#FF5722"}
    for i, (col, title, ylabel) in enumerate(
        [
            ("duration_sec", "Processing Time", "Time (sec)"),
            ("variance_removed_pct", "Variance Removed", "Variance Removed (%)"),
        ]
    ):
        ax = axes[0, i]
        for m in ["dss", "ica"]:
            mdf = metrics_df[metrics_df["method"] == m]
            ax.bar(
                mdf["subject"],
                mdf[col],
                alpha=0.7,
                color=colors[m],
                label=m.upper(),
                width=0.4,
                align="edge" if m == "ica" else "center",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", opacity=0.3)
    ax = axes[1, 0]
    corr_data = metrics_df[metrics_df["method"] == "dss"][["subject", "mean_dss_ica_corr"]].dropna()
    if not corr_data.empty:
        cmap = [
            "#4CAF50" if v > 0.95 else "#FFC107" if v > 0.9 else "#F44336"
            for v in corr_data["mean_dss_ica_corr"]
        ]
        ax.bar(corr_data["subject"], corr_data["mean_dss_ica_corr"], color=cmap, alpha=0.85)
        ax.axhline(0.95, color="green", ls="--")
        ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Pearson r")
    ax.set_title("DSS-ICA Signal Correlation", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", opacity=0.3)
    ax = axes[1, 1]
    for m in ["dss", "ica"]:
        mdf = metrics_df[metrics_df["method"] == m]
        ax.bar(
            mdf["subject"],
            mdf[["eog_components", "ecg_components", "emg_components"]].sum(axis=1),
            alpha=0.7,
            color=colors[m],
            label=m.upper(),
            width=0.4,
            align="edge" if m == "ica" else "center",
        )
    ax.set_ylabel("Total Components")
    ax.set_title("Components Removed", fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", opacity=0.3)
    fig.suptitle("DSS vs ICA - Comparison Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "comparison_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)

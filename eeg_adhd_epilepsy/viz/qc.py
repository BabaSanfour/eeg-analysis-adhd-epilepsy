"""Quality Control Visualization Utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne


def plot_amplitude_histogram(amp_stats: Dict[str, object]) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(6, 4))
    plt.hist(amp_stats["per_channel"], bins=30, alpha=0.85, edgecolor="black")
    plt.axvline(amp_stats["mean"], color="red", linestyle="--", label=f"Mean: {amp_stats['mean']:.1f} uV")
    plt.axvline(
        amp_stats["median"],
        color="green",
        linestyle="--",
        label=f"Median: {amp_stats['median']:.1f} uV",
    )
    plt.xlabel("Peak-to-Peak Amplitude (uV)")
    plt.ylabel("Number of Channels")
    plt.title("Channel Amplitude Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_channel_variance_topomap(raw: mne.io.BaseRaw) -> matplotlib.figure.Figure:
    data = raw.get_data()
    variances = np.var(data, axis=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(variances, raw.info, axes=ax, show=False)
    ax.set_title("Channel Variance Distribution")
    plt.tight_layout()
    return fig


def plot_metric_topomap(
    values: Sequence[float] | np.ndarray,
    raw: mne.io.BaseRaw | mne.Epochs,
    picks: Sequence[str],
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
) -> matplotlib.figure.Figure | None:
    """Generic topomap helper for per-channel metrics."""
    if raw is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    lower_map = {ch.lower(): ch for ch in raw.ch_names}
    pick_names = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]
    if len(pick_names) != len(arr):
        return None
    indices = mne.pick_channels(raw.info["ch_names"], include=pick_names, ordered=True)
    if len(indices) != len(arr):
        return None
    info = mne.pick_info(raw.info, indices)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    if unit:
        cbar.set_label(unit)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_topomap_from_channel_values(
    channel_names: Sequence[str],
    values: Sequence[float],
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
) -> matplotlib.figure.Figure | None:
    """Topomap plotting without a raw object (uses standard montage)."""
    if not channel_names:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or len(channel_names) != arr.size:
        return None
    info = mne.create_info(list(channel_names), sfreq=100.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    try:
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    if unit:
        cbar.set_label(unit)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_topomap_grid(
    metrics_dict: Mapping[str, Sequence[float] | np.ndarray],
    info: mne.Info,
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
    ncols: int = 4,
) -> matplotlib.figure.Figure | None:
    """
    Plot a grid of topomaps (e.g. Band Powers).
    metrics_dict: {SubplotTitle: values_array}
    """
    if not metrics_dict:
        return None
            
    n_plots = len(metrics_dict)
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_arr = np.atleast_1d(axes).flatten()
    
    for idx, (subplot_title, values) in enumerate(metrics_dict.items()):
        ax = axes_arr[idx]
        arr = np.asarray(values, dtype=float)
        
        # Safe check for size match
        if arr.size != len(info.ch_names):
             ax.text(0.5, 0.5, f"Size Mismatch\n{arr.size} vs {len(info.ch_names)}", ha="center", va="center", fontsize=8)
             ax.axis("off")
             continue
        
        try:     
            im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap)
            ax.set_title(subplot_title)
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            if unit:
                cbar.set_label(unit)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot Error:\n{str(e)[:30]}", ha="center", va="center", fontsize=7)
            ax.axis("off")
            
    # Hide empty axes
    for extra_ax in axes_arr[n_plots:]:
        extra_ax.axis("off")
        
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_psd_figures(
    spec: mne.time_frequency.Spectrum, freqs: np.ndarray, psd: np.ndarray, EPS: float
) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    fig_all = spec.plot(average=False, dB=True, show=False)
    fig_avg, ax = plt.subplots(figsize=(6, 4))
    psd_db = 10 * np.log10(psd + np.finfo(float).eps)
    ax.plot(freqs, psd_db.mean(axis=0))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Average PSD Across Channels")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_all, fig_avg


def plot_psd_overlay(
    before_freqs: np.ndarray,
    before_psd: np.ndarray,
    after_freqs: np.ndarray,
    after_psd: np.ndarray,
    EPS: float,
    label_before: str = "Before",
    label_after: str = "After",
) -> matplotlib.figure.Figure:
    """Overlay average PSD curves for before/after comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if before_psd.size:
        ax.plot(before_freqs, 10 * np.log10(before_psd.mean(axis=0) + EPS), label=label_before)
    if after_psd.size:
        ax.plot(after_freqs, 10 * np.log10(after_psd.mean(axis=0) + EPS), label=label_after)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("PSD Overlay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig





def plot_raw_segment(
    raw: mne.io.BaseRaw, start_sec: float, duration_sec: float = 10.0, title: str | None = None
) -> matplotlib.figure.Figure:
    safe_start = max(min(start_sec, raw.times[-1]), 0.0)
    if safe_start >= raw.times[-1]:
        safe_start = max(raw.times[-1] - duration_sec, 0.0)
    end_sec = safe_start + duration_sec
    max_end = min(end_sec, raw.times[-1])
    segment = raw.copy().crop(tmin=safe_start, tmax=max_end)
    fig = segment.plot(
        duration=duration_sec,
        start=0,
        n_channels=20,
        show=False,
        title=title or "Raw segment (10s window)",
    )
    plt.tight_layout()
    return fig


def plot_flagged_percentages(
    series: pd.Series,
    title: str,
    xlabel: str,
    fig_dir: Path,
    filename: str,
) -> Path | None:
    """Save a horizontal bar plot for flagged percentages."""
    if series is None or series.empty:
        return None
    ordered = series.sort_values(ascending=False)
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, max(3, len(ordered) * 0.4)))
    ax.barh(positions, ordered.values, color="#C44E52", alpha=0.85)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(left=0, right=max(100.0, ordered.max() * 1.05))
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _segment_palette(segment_types: Sequence[str]) -> Dict[str, object]:
    cmap = plt.get_cmap("tab20")
    return {seg: cmap(idx % cmap.N) for idx, seg in enumerate(sorted(segment_types))}


def plot_flagged_subject_distribution(
    segments_df: pd.DataFrame,
    fig_dir: Path,
) -> Path | None:
    """Plot distribution of subject counts by number of flagged segments per segment type."""
    if segments_df is None or segments_df.empty or "segment_type" not in segments_df or "subject_id" not in segments_df:
        return None
    df = segments_df.copy()
    df["segment_type"] = df["segment_type"].fillna("Unknown").astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    val = df.get("segment_flag_bad")
    if val is None:
        df["flag_bad_bool"] = False
    else:
        df["flag_bad_bool"] = pd.to_numeric(val, errors="coerce").fillna(0).astype(bool)
    flagged = df[df["flag_bad_bool"]]
    if flagged.empty:
        return None
    flagged_counts = flagged.groupby(["segment_type", "subject_id"]).size()
    if flagged_counts.empty:
        return None
    segment_types = sorted(flagged_counts.index.get_level_values(0).unique())
    n_types = len(segment_types)
    n_cols = 2 if n_types > 1 else 1
    n_rows = int(np.ceil(n_types / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, max(3, 3 * n_rows)))
    axes_arr = np.atleast_1d(axes).flatten()
    palette = _segment_palette(segment_types)
    for idx, seg_type in enumerate(segment_types):
        ax = axes_arr[idx]
        counts = flagged_counts.loc[seg_type]
        hist = counts.value_counts().sort_index()
        ax.bar(hist.index.astype(str), hist.values, color=palette.get(seg_type, "#4C72B0"))
        ax.set_title(seg_type)
        ax.set_xlabel("Flagged segments per subject")
        ax.set_ylabel("Subject count")
        ax.grid(True, axis="y", alpha=0.3)
    for extra_ax in axes_arr[len(segment_types):]:
        extra_ax.axis("off")
    fig.suptitle("Flagged Segment Counts per Subject (by Segment Type)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "flagged_subjects_by_flagged_segments.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
    palette = _segment_palette(df["segment_type"].unique())
    means = df.groupby("segment_type")["metric"].mean()
    boundaries = sorted(
        set(df["t_start"].dropna().tolist()) | set(df["t_stop"].dropna().tolist())
    )

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


def plot_segment_metric_distribution_by_type(
    segments_df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    fig_dir: Path,
) -> Path | None:
    """Histogram subplots per segment type for a given metric."""
    if column not in segments_df:
        return None
    df = segments_df.copy()
    df["metric"] = pd.to_numeric(df[column], errors="coerce")
    df["segment_type"] = df.get("segment_type", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    df = df.dropna(subset=["metric"])
    if df.empty:
        return None
    segment_types = sorted(df["segment_type"].unique())
    n_types = len(segment_types)
    n_cols = 2 if n_types > 1 else 1
    n_rows = int(np.ceil(n_types / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, max(3, 3 * n_rows)))
    axes_arr = np.atleast_1d(axes).flatten()
    palette = _segment_palette(segment_types)
    for idx, seg_type in enumerate(segment_types):
        ax = axes_arr[idx]
        series = df.loc[df["segment_type"] == seg_type, "metric"]
        if series.empty:
            ax.axis("off")
            continue
        ax.hist(series, bins=20, edgecolor="black", alpha=0.8, color=palette.get(seg_type, "#4C72B0"))
        ax.set_title(seg_type)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    for extra_ax in axes_arr[len(segment_types):]:
        extra_ax.axis("off")
    fig.suptitle(f"{title} by Segment Type", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"{column}_by_segment_type.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


    return paths
    
    
def save_subject_figures(
    metrics: Dict[str, object],
    raw: mne.io.BaseRaw,
    output_dir: Path,
) -> Dict[str, str]:
    """Save standard subject-level figures and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    
    pcm = metrics.get("per_channel_metrics", {})
    
    # Create EEG-only info for topomaps (to match per_channel_metrics size)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) > 0:
        eeg_info = mne.pick_info(raw.info, eeg_picks)
        # Apply montage to EEG info for topomap positions
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            eeg_info.set_montage(montage, on_missing="ignore")
        except Exception:
            pass
    else:
        eeg_info = raw.info  # Fallback
    
    # 1. Amplitude Histogram
    if "amplitude_ptp_uv" in pcm:
        amp_stats = {
            "per_channel": pcm["amplitude_ptp_uv"],
            "mean": metrics.get("segment_amplitude_mean_uv", float("nan")),
            "median": metrics.get("segment_amplitude_median_uv", float("nan")),
        }
        fig = plot_amplitude_histogram(amp_stats)
        out_hist = output_dir / "amplitude_hist.png"
        fig.savefig(out_hist, dpi=100)
        plt.close(fig)
        paths["amplitude_hist"] = str(out_hist)
    
    # 2. PSD Plot
    try:
        fig_psd = raw.compute_psd(picks='eeg', fmin=0.5, fmax=60.0, verbose='ERROR').plot(show=False)
        out_psd = output_dir / "psd_all.png"
        fig_psd.savefig(out_psd, dpi=100)
        plt.close(fig_psd)
        paths["psd_all"] = str(out_psd)
    except Exception:
        pass

    # 3. Spectral Grid (Band Powers)
    band_metrics = {}
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
         key = f"band_power_{band}"
         if key in pcm:
             band_metrics[f"{band.capitalize()} Power"] = pcm[key]
             
    if band_metrics:
        fig = plot_topomap_grid(
            band_metrics, 
            eeg_info,
            title="Band Power Topomaps", 
            cmap="viridis", 
            unit="uV^2",
            ncols=3
        )
        if fig:
            out_grid = output_dir / "spectral_topomaps_grid.png"
            fig.savefig(out_grid, dpi=100)
            plt.close(fig)
            paths["spectral_topomaps_grid"] = str(out_grid)
            
    # 4. Quality Metrics Grid
    quality_metrics = {}
    if "variance" in pcm: quality_metrics["Variance"] = pcm["variance"]
    if "line_noise_ratio" in pcm: quality_metrics["Line Noise Ratio"] = pcm["line_noise_ratio"]
    if "hf_lf_ratio" in pcm: quality_metrics["HF/LF Ratio"] = pcm["hf_lf_ratio"]
    if "aperiodic_slope" in pcm: quality_metrics["1/f Slope"] = pcm["aperiodic_slope"]
    
    if quality_metrics:
         fig = plot_topomap_grid(
            quality_metrics,
            eeg_info,  # Use EEG-only info
            title="Signal Quality Topomaps",
            cmap="RdBu_r",
            ncols=2
         )
         if fig:
             out_qual = output_dir / "signal_quality_grid.png"
             fig.savefig(out_qual, dpi=100)
             plt.close(fig)
             paths["signal_quality_grid"] = str(out_qual)

    return paths

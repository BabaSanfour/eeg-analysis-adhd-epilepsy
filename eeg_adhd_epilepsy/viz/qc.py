"""Quality Control Visualization Utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

def save_eeg_snapshot(
    raw: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    label: str,
    start: float = 30.0,
    duration: float = 60.0,
    n_channels: int = 20
) -> str:
    """Save a butterfly plot snapshot of EEG channels.
    
    Captures `duration` seconds of raw EEG data as a static butterfly plot.
    Starts at `start` seconds to skip initial transients.
    """
    raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    # Ensure start is within bounds
    max_start = max(0, raw_eeg.times[-1] - duration)
    start = min(start, max_start)
    ch_names = raw_eeg.ch_names[:min(n_channels, len(raw_eeg.ch_names))]
    stop = min(start + duration, raw_eeg.times[-1])
    sfreq = raw_eeg.info['sfreq']
    data, times = raw_eeg[ch_names, int(start * sfreq):int(stop * sfreq)]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    data_uv = data * 1e6
    for i, _ in enumerate(ch_names):
        ax.plot(times[:data.shape[1]], data_uv[i], linewidth=0.4, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'{subject_id} — {label.replace("_", " ").title()}')
    ax.set_xlim(times[0], times[min(data.shape[1]-1, len(times)-1)])
    ax.grid(True, alpha=0.3)
    
    path = fig_dir / f'{subject_id}_{label}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
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
    """Save a 3-panel before/after/removed comparison at the largest artifact peak.
    
    Finds the time of maximum artifact removal (largest absolute difference),
    then plots a window around that peak showing the Original, Cleaned,
    and what was Removed.
    """
    from scipy.signal import find_peaks
    
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude='bads')
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude='bads')
    sfreq = eeg_before.info['sfreq']
    
    # Compute the removed signal
    data_before = eeg_before.get_data()
    data_after = eeg_after.get_data()
    
    # Ensure same shape
    n_samples = min(data_before.shape[1], data_after.shape[1])
    data_before = data_before[:, :n_samples]
    data_after = data_after[:, :n_samples]
    removed = data_before - data_after
    
    # Choose channels to display and find peaks
    ch_names = eeg_before.ch_names
    if artifact_type == 'eog':
        # Prefer frontal channels for EOG
        frontal = [i for i, ch in enumerate(ch_names)
                   if any(f in ch.upper() for f in ['FP1', 'FP2', 'F3', 'F4', 'FZ', 'AF'])]
        if not frontal:
            frontal = list(range(min(4, len(ch_names))))
        # Find blink peaks using envelope of frontal removed signal
        # Restriction: only search after search_start to skip initial transients
        search_sample = int(search_start * sfreq)
        if search_sample < n_samples:
            envelope = np.abs(removed[frontal, search_sample:]).mean(axis=0)
            offset_samples = search_sample
        else:
            envelope = np.abs(removed[frontal]).mean(axis=0)
            offset_samples = 0
        display_idx = frontal[:n_channels]
    else:
        # For ECG/EMG — find channel with max removed power
        search_sample = int(search_start * sfreq)
        if search_sample < n_samples:
            ch_power = np.sum(removed[:, search_sample:] ** 2, axis=1)
            top_ch = np.argsort(ch_power)[::-1][:n_channels]
            envelope = np.abs(removed[top_ch[0], search_sample:])
            offset_samples = search_sample
        else:
            ch_power = np.sum(removed ** 2, axis=1)
            top_ch = np.argsort(ch_power)[::-1][:n_channels]
            envelope = np.abs(removed[top_ch[0]])
            offset_samples = 0
        display_idx = list(top_ch)
    
    # Find peaks in the envelope
    min_dist = int(0.5 * sfreq)  # at least 0.5s apart
    peaks, properties = find_peaks(envelope, distance=min_dist, height=np.percentile(envelope, 95))
    
    if len(peaks) == 0:
        # Fallback: use the point of maximum absolute difference
        peak_idx = int(np.argmax(envelope))
    else:
        # Use the tallest peak
        peak_idx = peaks[np.argmax(properties['peak_heights'])]
    
    # Convert to time
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
    
    for panel_idx, (ax, title_sfx, signal) in enumerate([
        (axes[0], 'Before (Original)', data_before),
        (axes[1], 'After (Cleaned)', data_after),
        (axes[2], f'Removed ({artifact_label} Artifact)', removed),
    ]):
        for j, ch_idx in enumerate(display_idx):
            sig_uv = signal[ch_idx, s_start:s_end] * 1e6
            ax.plot(times[:len(sig_uv)], sig_uv, linewidth=0.6, alpha=0.8,
                    color=colors[j], label=display_names[j] if panel_idx == 0 else None)
        ax.set_ylabel('µV')
        ax.set_title(f'{title_sfx}', fontsize=11)
        ax.grid(True, alpha=0.3)
        # Mark the peak time
        ax.axvline(peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[2].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right', fontsize=7, ncol=min(4, len(display_idx)))
    
    fig.suptitle(f'{subject_id} — {artifact_label} Artifact Removal (peak at {peak_time:.2f}s)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = fig_dir / f'{subject_id}_{artifact_type}_artifact_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def plot_removed_variance_topomap(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    title: str = "Removed Variance",
) -> matplotlib.figure.Figure | None:
    """Plot topographic map of variance difference (before - after)."""
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude='bads')
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude='bads')
    
    # Ensure same shape
    n_samples = min(eeg_before.n_times, eeg_after.n_times)
    data_before = eeg_before.get_data()[:, :n_samples]
    data_after = eeg_after.get_data()[:, :n_samples]
    
    var_before = np.var(data_before, axis=1)
    var_after = np.var(data_after, axis=1)
    var_removed = var_before - var_after
    
    # Clip negative values (in case of subtle interpolation differences)
    var_removed = np.maximum(var_removed, 0)
    
    # Normalize to % of original variance for better readability?
    # Or just keep raw units. Let's keep raw units but add a good colorbar.
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(var_removed, eeg_before.info, axes=ax, show=False, cmap="Reds")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label('Variance (V^2)')
    ax.set_title(title)
    plt.tight_layout()
    return fig


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
) -> Dict[str, str]:
    """Save shared pre-cleaning DSS diagnostic plots."""
    from mne_denoise.viz import (
        plot_component_summary,
        plot_component_time_series,
        plot_score_curve,
        plot_spatial_patterns,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, str] = {}
    fit_data_plot = truncate_plot_data(fit_data, sfreq=sfreq, max_seconds=fit_max_seconds)

    filters = getattr(estimator, "filters_", None)
    n_available = 0
    if isinstance(filters, np.ndarray) and filters.ndim >= 2:
        n_available = int(filters.shape[0])

    if include_score:
        fig_score = plot_score_curve(estimator, show=False)
        if fig_score is not None:
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
            if fig_comp is not None:
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
            if fig_topo is not None:
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
            if fig_ts is not None:
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
) -> Dict[str, str]:
    """Save shared post-cleaning DSS comparison plots."""
    from mne_denoise.viz import (
        plot_overlay_comparison,
        plot_psd_comparison,
        plot_time_course_comparison,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, str] = {}

    fig_psd = plot_psd_comparison(raw_before, raw_after, fmax=fmax, show=False)
    if fig_psd is not None:
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
    if fig_ov is not None:
        ov_path = fig_dir / f"{subject_id}_{file_prefix}_overlay.png"
        fig_ov.savefig(ov_path, dpi=150, bbox_inches="tight")
        plt.close(fig_ov)
        plot_paths["overlay_comparison"] = str(ov_path)

    if include_time_course:
        s_start = int(start_seconds * raw_before.info["sfreq"])
        s_stop = int(min(raw_before.times[-1], start_seconds + window_seconds) * raw_before.info["sfreq"])
        fig_tc = plot_time_course_comparison(
            raw_before,
            raw_after,
            start=s_start,
            stop=s_stop,
            show=False,
        )
        if fig_tc is not None:
            tc_path = fig_dir / f"{subject_id}_{file_prefix}_timecourse.png"
            fig_tc.savefig(tc_path, dpi=150, bbox_inches="tight")
            plt.close(fig_tc)
            plot_paths["time_course_comparison"] = str(tc_path)

    return plot_paths


def plot_channel_variance_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    subject_id: str,
    title: str = "Channel Variance: Before vs After Correction",
) -> matplotlib.figure.Figure:
    """Plot a comparison of per-channel variance before and after correction."""
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude="bads")
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude="bads")

    # Ensure same channels and samples
    common_chs = [ch for ch in eeg_before.ch_names if ch in eeg_after.ch_names]
    eeg_before.pick_channels(common_chs)
    eeg_after.pick_channels(common_chs)

    n_samples = min(eeg_before.n_times, eeg_after.n_times)
    data_before = eeg_before.get_data()[:, :n_samples]
    data_after = eeg_after.get_data()[:, :n_samples]

    var_before = np.var(data_before, axis=1) * (1e6**2)  # to uV^2
    var_after = np.var(data_after, axis=1) * (1e6**2)

    df = pd.DataFrame({
        "Channel": common_chs,
        "Before": var_before,
        "After": var_after,
    })

    fig, ax = plt.subplots(figsize=(max(10, len(common_chs) * 0.3), 6))
    x = np.arange(len(common_chs))
    width = 0.35

    ax.bar(x - width/2, df["Before"], width, label="Before", color="indianred", alpha=0.7)
    ax.bar(x + width/2, df["After"], width, label="After", color="mediumseagreen", alpha=0.7)

    ax.set_ylabel("Variance (uV^2)")
    ax.set_title(f"{subject_id} - {title}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Channel"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def save_ica_sources_snapshot(
    ica: Any,
    raw: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    picks: List[int],
    label: str,
    start: float = 30.0,
    duration: float = 20.0,
) -> str:
    """Save a clean stacked plot of selected ICA sources.
    
    Provides a more compact and customizable alternative to `ica.plot_sources`.
    """
    sources_raw = ica.get_sources(raw)
    sfreq = sources_raw.info['sfreq']
    
    # Select channels and crop
    t_start = min(start, max(0, sources_raw.times[-1] - duration))
    t_stop = min(t_start + duration, sources_raw.times[-1])
    
    # Get labels for the picked components
    picked_names = [sources_raw.ch_names[i] for i in picks]
    data, times = sources_raw[picks, int(t_start * sfreq):int(t_stop * sfreq)]
    
    n_chs = len(picks)
    # Reasonable standard vertical space: 1.2 inches per channel
    fig_height = max(2, 1.2 * n_chs)
    fig, axes = plt.subplots(n_chs, 1, figsize=(10, fig_height), sharex=True)
    if n_chs == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        trace = data[i]
        # Plots are black (not red) for cleaner look
        ax.plot(times, trace, color="black", linewidth=0.6)
        ax.set_ylabel(picked_names[i], rotation=0, labelpad=25, verticalalignment='center')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        if i < n_chs - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False)
            
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} - {label} Excluded ICA Components", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    path = fig_dir / f"{subject_id}_{label.lower()}_ica_sources.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# -----------------------------------------------------------------------------
# Comparison Plots (DSS vs ICA vs Original)
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
    methods = [
        (axes[0], raw_orig, "Original", "#666666"),
        (axes[1], raw_dss, "DSS", "#2196F3"),
        (axes[2], raw_ica, "ICA", "#FF5722"),
    ]

    for ax, raw, label, color in methods:
        try:
            psd = raw.compute_psd(fmin=0.5, fmax=50, picks=eeg_picks, verbose=False)
            psd_data = psd.get_data() * 1e12  # to uV^2/Hz
            freqs = psd.freqs
            mean_psd = np.mean(psd_data, axis=0)
            ax.semilogy(freqs, mean_psd, color=color, lw=2, label=label)
            ax.fill_between(
                freqs,
                np.percentile(psd_data, 5, axis=0),
                np.percentile(psd_data, 95, axis=0),
                alpha=0.2,
                color=color,
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error: {exc}", transform=ax.transAxes, ha="center")

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
    eeg_picks = mne.pick_types(raw_orig.info, eeg=True, exclude="bads")
    max_start = raw_orig.times[-1] - duration
    t_start = min(start, max(0, max_start))
    n_start = int(t_start * raw_orig.info["sfreq"])
    n_dur = int(duration * raw_orig.info["sfreq"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, sharey=True)
    times = raw_orig.times[n_start : n_start + n_dur] - raw_orig.times[n_start]

    for ax, raw, label, color in [
        (axes[0], raw_orig, "Original", "#888888"),
        (axes[1], raw_dss, "DSS Corrected", "#2196F3"),
        (axes[2], raw_ica, "ICA Corrected", "#FF5722"),
    ]:
        data = raw.get_data(picks=eeg_picks)[:, n_start : n_start + n_dur] * 1e6
        for ch in data:
            ax.plot(times, ch, color=color, alpha=0.3, lw=0.5)
        ax.set_ylabel("uV")
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left")
        ax.grid(alpha=0.2)

    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} - Signal Butterfly ({t_start:.0f}-{t_start + duration:.0f}s)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    path = fig_dir / f"{subject_id}_butterfly.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_band_power(
    bp_orig: Dict[str, float],
    bp_dss: Dict[str, float],
    bp_ica: Dict[str, float],
    subject_id: str,
    fig_dir: Path,
) -> str:
    """Plot band power comparison across methods."""
    bands = list(bp_orig.keys())
    x = np.arange(len(bands))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, [bp_orig[b] * 1e12 for b in bands], width, label="Original", color="#888888", alpha=0.8)
    ax.bar(x, [bp_dss[b] * 1e12 for b in bands], width, label="DSS", color="#2196F3", alpha=0.8)
    ax.bar(x + width, [bp_ica[b] * 1e12 for b in bands], width, label="ICA", color="#FF5722", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=12)
    ax.set_ylabel("Power (uV^2/Hz)")
    ax.set_title(f"{subject_id} - Band Power Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    path = fig_dir / f"{subject_id}_band_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_channel_correlation(
    corr_map: Dict[str, float],
    subject_id: str,
    fig_dir: Path,
) -> str:
    """Plot channel correlation between DSS and ICA versions."""
    channels = list(corr_map.keys())
    values = list(corr_map.values())
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if v > 0.95 else "#FFC107" if v > 0.9 else "#F44336" for v in values]
    ax.bar(range(len(channels)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.95, color="green", ls="--", alpha=0.5)
    ax.axhline(0.90, color="orange", ls="--", alpha=0.5)
    ax.set_ylim(min(0.5, min(values) - 0.05 if values else 1.0), 1.02)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(f"{subject_id} - DSS vs ICA Channel Correlation", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
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
    
    # Actually, let's implement it directly to avoid complexity
    picks = mne.pick_types(raw_orig.info, eeg=True, exclude="bads")
    n_samples = min(raw_orig.n_times, raw_dss.n_times, raw_ica.n_times)
    
    data_orig = raw_orig.get_data(picks=picks)[:, :n_samples]
    data_dss = raw_dss.get_data(picks=picks)[:, :n_samples]
    data_ica = raw_ica.get_data(picks=picks)[:, :n_samples]
    
    var_orig = np.var(data_orig, axis=1)
    var_dss = np.var(data_dss, axis=1)
    var_ica = np.var(data_ica, axis=1)
    
    # Removed variance
    rem_dss = np.maximum(0, var_orig - var_dss)
    rem_ica = np.maximum(0, var_orig - var_ica)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Original Variance
    mne.viz.plot_topomap(var_orig, raw_orig.info, axes=axes[0], show=False)
    axes[0].set_title("Original Variance")
    
    # 2. Variance Removed by DSS
    mne.viz.plot_topomap(rem_dss, raw_orig.info, axes=axes[1], show=False, cmap="Reds")
    axes[1].set_title("Variance Removed (DSS)")
    
    # 3. Variance Removed by ICA
    mne.viz.plot_topomap(rem_ica, raw_orig.info, axes=axes[2], show=False, cmap="Reds")
    axes[2].set_title("Variance Removed (ICA)")
    
    fig.suptitle(f"{subject_id} - Spatial Distribution of Variance Reduction", fontsize=14)
    plt.tight_layout()
    
    path = fig_dir / f"{subject_id}_variance_topomap_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_timing(timing_data: List[Dict[str, Any]], fig_dir: Path) -> str:
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
        vals = [mdf.loc[s, "duration_sec"] if s in mdf.index else 0 for s in subjects]
        color = "#2196F3" if method == "dss" else "#FF5722"
        ax.bar(x + (i - 0.5) * width, vals, width, label=method.upper(), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Processing Time (sec)", fontsize=12)
    ax.set_title("Processing Time: DSS vs ICA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    path = fig_dir / "timing_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_components_removed(comp_data: List[Dict[str, Any]], fig_dir: Path) -> str:
    """Plot average components removed per artifact type."""
    df = pd.DataFrame(comp_data)
    if df.empty:
        return ""

    artifact_types = ["eog", "ecg", "emg"]
    methods = ["dss", "ica"]
    x = np.arange(len(artifact_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        mdf = df[df["method"] == method]
        vals = []
        for art in artifact_types:
            col = f"{art}_components"
            vals.append(mdf[col].mean() if col in mdf.columns else 0)
        color = "#2196F3" if method == "dss" else "#FF5722"
        ax.bar(x + (i - 0.5) * width, vals, width, label=method.upper(), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in artifact_types], fontsize=12)
    ax.set_ylabel("Components Removed (avg)", fontsize=12)
    ax.set_title("Artifact Components Removed: DSS vs ICA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    path = fig_dir / "components_removed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_compare_variance_removed(var_data: List[Dict[str, Any]], fig_dir: Path) -> str:
    """Plot variance removed percentage across subjects."""
    df = pd.DataFrame(var_data)
    if df.empty:
        return ""

    subjects = sorted(df["subject"].unique())
    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.5), 5))
    for i, method in enumerate(["dss", "ica"]):
        mdf = df[df["method"] == method].set_index("subject")
        vals = [mdf.loc[s, "variance_removed_pct"] if s in mdf.index else 0 for s in subjects]
        color = "#2196F3" if method == "dss" else "#FF5722"
        ax.bar(x + (i - 0.5) * width, vals, width, label=method.upper(), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Variance Removed (%)", fontsize=12)
    ax.set_title("Total Variance Removed: DSS vs ICA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
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

    ax = axes[0, 0]
    for method in ["dss", "ica"]:
        mdf = metrics_df[metrics_df["method"] == method]
        ax.bar(
            mdf["subject"],
            mdf["duration_sec"],
            alpha=0.7,
            color=colors[method],
            label=method.upper(),
            width=0.4,
            align="edge" if method == "ica" else "center",
        )
    ax.set_ylabel("Time (sec)")
    ax.set_title("Processing Time", fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    for method in ["dss", "ica"]:
        mdf = metrics_df[metrics_df["method"] == method]
        ax.bar(
            mdf["subject"],
            mdf["variance_removed_pct"],
            alpha=0.7,
            color=colors[method],
            label=method.upper(),
            width=0.4,
            align="edge" if method == "ica" else "center",
        )
    ax.set_ylabel("Variance Removed (%)")
    ax.set_title("Variance Removed", fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    corr_data = metrics_df[metrics_df["method"] == "dss"][["subject", "mean_dss_ica_corr"]].dropna()
    if not corr_data.empty:
        color_map = ["#4CAF50" if v > 0.95 else "#FFC107" if v > 0.9 else "#F44336" for v in corr_data["mean_dss_ica_corr"]]
        ax.bar(corr_data["subject"], corr_data["mean_dss_ica_corr"], color=color_map, alpha=0.85)
        ax.axhline(0.95, color="green", ls="--", alpha=0.5)
        ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Pearson r")
    ax.set_title("DSS-ICA Signal Correlation", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    for method in ["dss", "ica"]:
        mdf = metrics_df[metrics_df["method"] == method]
        total = mdf[["eog_components", "ecg_components", "emg_components"]].sum(axis=1)
        ax.bar(
            mdf["subject"],
            total.values,
            alpha=0.7,
            color=colors[method],
            label=method.upper(),
            width=0.4,
            align="edge" if method == "ica" else "center",
        )
    ax.set_ylabel("Total Components")
    ax.set_title("Components Removed", fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DSS vs ICA - Comparison Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    path = fig_dir / "comparison_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)

"""Visualization utilities for condition segment analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

FIGURE_FILENAMES = {
    "segment_duration": "segment_total_duration.png",
    "eye_state_breakdown": "segment_eye_state_breakdown.png",
    "photo_frequency": "photo_frequency_duration.png",
    "hv_blocks": "hv_block_eye_states.png",
    "post_hv_blocks": "post_hv_block_eye_states.png",
    "timeline": "segment_timeline.png",
}


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_total_duration_by_segment(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = (
        df.groupby("segment_type", dropna=False)["duration"]
        .sum()
        .sort_values(ascending=True)
    )
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, len(group) * 0.35)))
    ax.barh(group.index.astype(str), group.values, color="#4C72B0")
    ax.set_xlabel("Total Duration (s)")
    ax.set_title("Total Duration by Segment Type")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["segment_duration"])


def plot_eye_state_breakdown(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = df.groupby("segment_type")[["eyes_open_duration", "eyes_closed_duration"]].sum()
    if group.empty:
        return None
    group = group.sort_values(by="eyes_open_duration", ascending=False)
    labels = group.index.astype(str)
    open_vals = group["eyes_open_duration"].to_numpy(dtype=float)
    closed_vals = group["eyes_closed_duration"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(3, len(group) * 0.35)))
    positions = np.arange(len(labels))
    ax.barh(positions, open_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, closed_vals, left=open_vals, label="Eyes Closed", color="#C44E52")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Eyes-Open vs Eyes-Closed Duration per Segment Type")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["eye_state_breakdown"])


def plot_photo_frequency_durations(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    segment_types = df["segment_type"].fillna("").astype(str)
    photo = df[segment_types.str.startswith("PHOTO_") | (segment_types == "PHOTO_block")]
    if photo.empty or "freq_hz" not in photo:
        return None
    group = (
        photo.dropna(subset=["freq_hz"])
        .groupby("freq_hz")["duration"]
        .sum()
        .sort_index()
    )
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(freq) for freq in group.index]
    ax.bar(labels, group.values, color="#8172B2")
    ax.set_xlabel("PHOTO Frequency (Hz)")
    ax.set_ylabel("Total Duration (s)")
    ax.set_title("PHOTO Block Duration by Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["photo_frequency"])


def plot_block_eye_states(df: pd.DataFrame, block_type: str, fig_dir: Path) -> Path | None:
    if block_type not in {"HV", "PostHV"}:
        raise ValueError("block_type must be 'HV' or 'PostHV'")
    column = "hv_index" if block_type == "HV" else "post_hv_index"
    segment_types = df["segment_type"].fillna("").astype(str)
    legacy_label = f"{block_type}_block"
    mask = segment_types.str.startswith(f"{block_type}_") | (segment_types == legacy_label)
    block_df = df.loc[mask].copy()
    if block_df.empty or column not in block_df:
        return None

    block_df[column] = pd.to_numeric(block_df[column], errors="coerce")
    fallback = pd.Series(np.arange(1, len(block_df) + 1, dtype=int), index=block_df.index)
    block_df[column] = block_df[column].fillna(fallback).astype(int)

    def _label_eye_state(seg_type: str) -> str:
        if "_EO" in seg_type:
            return "EO"
        if "_EC" in seg_type:
            return "EC"
        return "Unknown"

    block_df["eye_state"] = [_label_eye_state(val) for val in block_df["segment_type"]]

    grouped = block_df.groupby([column, "eye_state"])["duration"].sum().unstack(fill_value=0.0)

    legacy_mask = segment_types[mask] == legacy_label
    if legacy_mask.any():
        legacy_df = block_df[legacy_mask]
        legacy_grouped = legacy_df.groupby(column)[["eyes_open_duration", "eyes_closed_duration"]].sum()
        grouped["EO"] = grouped.get("EO", 0.0) + legacy_grouped.get("eyes_open_duration", 0.0)
        grouped["EC"] = grouped.get("EC", 0.0) + legacy_grouped.get("eyes_closed_duration", 0.0)

    if grouped.empty:
        return None

    labels = [f"{block_type} #{idx}" for idx in grouped.index]
    fig, ax = plt.subplots(figsize=(7, max(3, len(grouped) * 0.4)))
    positions = np.arange(len(labels))
    eo_vals = grouped.get("EO", pd.Series(0.0, index=grouped.index))
    ec_vals = grouped.get("EC", pd.Series(0.0, index=grouped.index))
    unknown_vals = grouped.drop(columns=[c for c in grouped.columns if c in {"EO", "EC"}], errors="ignore").sum(axis=1)

    ax.barh(positions, eo_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, ec_vals, left=eo_vals, label="Eyes Closed", color="#C44E52")
    if not (unknown_vals == 0).all():
        ax.barh(
            positions,
            unknown_vals,
            left=eo_vals + ec_vals,
            label="Unknown",
            color="#8172B2",
            alpha=0.7,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title(f"{block_type} Eyes-Open vs Eyes-Closed")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    key = "hv_blocks" if block_type == "HV" else "post_hv_blocks"
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES[key])


def plot_segment_timeline(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    if df.empty:
        return None
    ordered = df.sort_values("t_start")
    segment_types = list(dict.fromkeys(ordered["segment_type"]))
    if not segment_types:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, len(segment_types) * 0.6)))
    cmap = plt.get_cmap("tab20")
    type_to_y = {seg: idx for idx, seg in enumerate(segment_types)}
    for idx, row in ordered.iterrows():
        seg_type = row["segment_type"]
        y = type_to_y.get(seg_type)
        if y is None:
            continue
        start = float(row["t_start"])
        stop = float(row["t_stop"])
        if not np.isfinite(start) or not np.isfinite(stop):
            continue
        color = cmap(y % cmap.N)
        ax.plot([start, stop], [y, y], linewidth=8, solid_capstyle="butt", color=color)
    ax.set_yticks(list(type_to_y.values()))
    ax.set_yticklabels(segment_types)
    ax.set_xlabel("Time (s)")
    ax.set_title("Condition Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["timeline"])


def save_condition_segment_figures(df: pd.DataFrame, fig_dir: Path) -> Dict[str, Path]:
    """Create bar plots and other figures summarizing segments."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}
    for key, func in [
        ("segment_duration", plot_total_duration_by_segment),
        ("eye_state_breakdown", plot_eye_state_breakdown),
        ("photo_frequency", plot_photo_frequency_durations),
    ]:
        path = func(df, fig_dir)
        if path:
            figure_paths[key] = path
    for block_type in ("HV", "PostHV"):
        path = plot_block_eye_states(df, block_type, fig_dir)
        if path:
            figure_paths["hv_blocks" if block_type == "HV" else "post_hv_blocks"] = path
    timeline_path = plot_segment_timeline(df, fig_dir)
    if timeline_path:
        figure_paths["timeline"] = timeline_path
    return figure_paths


# --- Dataset Level Visualizations ---

def _plot_custom_bins(
    values_series: pd.Series, 
    bins: List[int],
    color: str, 
    ax: plt.Axes,
    label: str | None = None,
    alpha: float = 0.8
) -> None:
    counts = []
    labels = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        count = ((values_series >= low) & (values_series < high)).sum()
        counts.append(count)
        labels.append(f"{low}-{high}")
    last_val = bins[-1]
    overflow_count = (values_series >= last_val).sum()
    counts.append(overflow_count)
    labels.append(f">{last_val}")
    
    x = np.arange(len(labels))
    ax.bar(x, counts, color=color, alpha=alpha, label=label, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")


def plot_dataset_durations(summary_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Plot histograms of total durations with custom binning."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if "total_duration" in summary_df:
        durations_min = summary_df["total_duration"] / 60.0
        bins_total = list(range(0, 65, 5)) 
        bins_total = list(range(0, 65, 5)) 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_custom_bins(durations_min, bins_total[:-1], "#4C72B0", ax)
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Number of Subjects")
        ax.set_title("Distribution of Total Analysis Duration")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = output_dir / "dataset_duration_hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["duration_hist"] = path

    if "total_eyes_open_duration" in summary_df and "total_eyes_closed_duration" in summary_df:
        bins_eye = list(range(0, 60, 5)) 
        
        eo_min = summary_df["total_eyes_open_duration"] / 60.0
        ec_min = summary_df["total_eyes_closed_duration"] / 60.0
        
        def _get_counts(values):
            c = []
            l = []
            for i in range(len(bins_eye) - 1):
                low, high = bins_eye[i], bins_eye[i+1]
                c.append(((values >= low) & (values < high)).sum())
                l.append(f"{low}-{high}")
            c.append((values >= bins_eye[-1]).sum())
            l.append(f">{bins_eye[-1]}")
            return c, l
            
        eo_counts, labels = _get_counts(eo_min)
        ec_counts, _ = _get_counts(ec_min)
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, eo_counts, width, label="Eyes Open", color="#55A868", edgecolor="black", alpha=0.8)
        ax.bar(x + width/2, ec_counts, width, label="Eyes Closed", color="#C44E52", edgecolor="black", alpha=0.8)
        
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Subjects")
        ax.set_title("Distribution of Eye State Durations")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        
        path = output_dir / "dataset_eye_states_hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["eye_states_hist"] = path

    return paths


def plot_events_distribution(event_counts: Dict[str, object]) -> matplotlib.figure.Figure | None:
    """Plot event counts with custom binning for high-count events (e.g. Movement)."""
    if not event_counts:
        return None

    def _is_sequence(value: object) -> bool:
        return isinstance(value, (list, tuple, np.ndarray, pd.Series))

    if any(_is_sequence(v) for v in event_counts.values()):
        labels: List[str] = []
        sequences: List[np.ndarray] = []
        for label, values in event_counts.items():
            if not _is_sequence(values):
                continue
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            labels.append(label)
            sequences.append(arr)
            
        if not labels:
            return None
            
        cols = 2 if len(labels) > 1 else 1
        rows = int(np.ceil(len(labels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        for ax_idx, (label, data) in enumerate(zip(labels, sequences)):
            if data.size == 0:
                continue
            
            is_high_count_category = any(k in label.lower() for k in ["bad", "mvt", "movement", "artefact", "clinical", "seizure", "spike"])
            
            if is_high_count_category:
                data_nonzero = data[data > 0]
                if data_nonzero.size == 0:
                    axes[ax_idx].text(0.5, 0.5, "No nonzero counts", ha='center', va='center')
                    axes[ax_idx].set_title(label)
                    continue
                
                data_to_plot = data_nonzero
                data_max = int(np.ceil(data_to_plot.max()))
                                
                if data_max > 30:
                    clipped_data = data_to_plot.copy()
                    clipped_data[clipped_data > 30] = 30
                    bin_edges = np.arange(0.5, 31.5, 1.0)
                    
                    axes[ax_idx].hist(clipped_data, bins=bin_edges, color="#4C72B0", alpha=0.85, edgecolor="black")
                    ticks = [1, 10, 20, 30]
                    axes[ax_idx].set_xticks(ticks)
                    axes[ax_idx].set_xticklabels(["1", "10", "20", "30+"])
                    axes[ax_idx].set_xlim(0.5, 30.5)
                else:
                    data_min = int(np.floor(data_to_plot.min()))
                    start_edge = max(0.5, data_min - 0.5) 
                    end_edge = data_max + 0.5
                    bin_edges = np.arange(start_edge, end_edge + 1.0, 1.0)
                    axes[ax_idx].hist(data_to_plot, bins=bin_edges, color="#4C72B0", alpha=0.85, edgecolor="black")
                    axes[ax_idx].set_xticks(np.arange(data_min, data_max + 1, 1))

            else:
                data_min = int(np.floor(data.min()))
                data_max = int(np.ceil(data.max()))
                
                if data_max == data_min:
                    bin_edges = np.array([data_min - 0.5, data_min + 0.5])
                else:
                    start_edge = data_min - 0.5
                    end_edge = data_max + 0.5
                    if (data_max - data_min) > 50:
                        bin_edges = 30
                    else:
                        bin_edges = np.arange(start_edge, end_edge + 1.0, 1.0)
                        
                axes[ax_idx].hist(data, bins=bin_edges, color="#4C72B0", alpha=0.85, edgecolor="black")
            
            axes[ax_idx].set_title(label)
            axes[ax_idx].set_xlabel("Count per Subject")
            axes[ax_idx].set_ylabel("Subjects")
            axes[ax_idx].grid(True, axis="y", alpha=0.3)
            
        for extra_ax in axes[len(labels):]:
            extra_ax.axis("off")
            
        fig.suptitle("Event Count Distributions", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # Single Subject Mode
    sorted_items = sorted(event_counts.items(), key=lambda item: item[1], reverse=True)
    labels, counts = zip(*sorted_items)
    height = max(4, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(8, height))
    positions = np.arange(len(labels))
    ax.barh(positions, counts, color="#4C72B0")
    ax.set_xlabel("Count")
    ax.set_title("Annotation Counts")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def save_dataset_events_distribution(event_counts_list: List[Dict[str, int]], output_dir: Path) -> Path | None:
    """Prepare data and save two sets of event count distributions: Conditions and Clinical/Bad."""
    if not event_counts_list:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_keys = set().union(*[c.keys() for c in event_counts_list])
    n_subjects = len(event_counts_list)
    
    agg_data = {k: np.zeros(n_subjects) for k in all_keys}
    for idx, counts in enumerate(event_counts_list):
        for k, v in counts.items():
            agg_data[k][idx] = v

    cond_keywords = ["eyes", "yeux", "photo"]
    
    bad_keywords = ["bad", "clinical", "artefact", "seizure", "spike", "slow", "mvt", "movement"]
    
    cond_keys = [k for k in all_keys if any(cw in k.lower() for cw in cond_keywords)]
    bad_keys = [k for k in all_keys if any(bw in k.lower() for bw in bad_keywords) and k not in cond_keys]
    
    if cond_keys:
        sorted_cond = sorted(cond_keys, key=lambda k: agg_data[k].mean(), reverse=True)[:12]
        cond_data = {k: agg_data[k] for k in sorted_cond}
        fig_cond = plot_events_distribution(cond_data)
        if fig_cond:
             p = output_dir / "dataset_event_distributions_conditions.png"
             fig_cond.suptitle("Condition Event Counts (EO/EC/HV/Photo)")
             fig_cond.savefig(p, dpi=150)
             plt.close(fig_cond)
             pass
    if bad_keys:
        sorted_bad = sorted(bad_keys, key=lambda k: agg_data[k].mean(), reverse=True)[:16]
        bad_data = {k: agg_data[k] for k in sorted_bad}
        fig_bad = plot_events_distribution(bad_data)
        if fig_bad:
             p = output_dir / "dataset_event_distributions_clinical.png"
             fig_bad.suptitle("Clinical & Artifact Event Counts (Non-zero)")
             fig_bad.savefig(p, dpi=150)
             plt.close(fig_bad)
             return p

    if cond_keys and not bad_keys:
        return output_dir / "dataset_event_distributions_conditions.png"
    return None

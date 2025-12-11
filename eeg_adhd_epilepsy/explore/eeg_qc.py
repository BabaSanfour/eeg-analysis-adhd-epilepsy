"""Automated EEG quality control (QC) using MNE-Python.

This script scans resting-state EEG recordings in BIDS-style directories,
computes QC metrics, flags problematic recordings, and exports per-subject and
dataset-level reports. It is intentionally light on preprocessing: the goal is
to quantify quality, not to clean data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel
from tqdm import tqdm
from mne_bids import BIDSPath, read_raw_bids

from eeg_adhd_epilepsy.utils.qc_config import (
    BAND_LIMITS,
    BASIC_1020_CHANNELS,
    KNOWN_EVENT_LABELS,
)
from eeg_adhd_epilepsy.utils.qc_annotations import (
    compute_special_event_counts,
    crop_raw_after_reference_event,
    summarize_annotations,
)

EVENT_LABELS_ALWAYS_KEEP_ZERO = {"Eyes Open", "Eyes Closed", "HV", "PHOTO"}

# Headless-friendly backend for figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated EEG QC (no preprocessing).")
    parser.add_argument(
        "--input_dir", required=True, type=Path, help="BIDS root directory with raw EEG files."
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Directory to store QC outputs."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Jobs for parallel processing (-1 for all cores)."
    )
    parser.add_argument(
        "--generate_subject_reports", action="store_true", help="Create per-subject HTML reports."
    )
    parser.add_argument(
        "--save_json", action="store_true", help="Also save metrics to qc_report.json."
    )
    parser.add_argument(
        "--skip_figures", action="store_true", help="Skip all figure generation (CSV/JSON only)."
    )
    parser.add_argument(
        "--subjects_list", type=Path, help="File with subject IDs to include (one per line)."
    )
    parser.add_argument(
        "--amplitude_threshold", type=float, default=500.0, help="Max amplitude threshold in uV."
    )
    parser.add_argument("--min_duration", type=float, default=5.0, help="Minimum duration in minutes.")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Maximum duration in minutes.")
    parser.add_argument("--bids_session", default=None, help="BIDS session entity, e.g., '01'.")
    parser.add_argument("--bids_task", default="RESTING", help="BIDS task entity, e.g., 'RESTING'.")
    parser.add_argument("--bids_run", default=None, help="BIDS run entity, e.g., '01'.")
    parser.add_argument("--bids_acq", default=None, help="BIDS acquisition entity if any.")
    parser.add_argument("--bids_proc", default=None, help="BIDS processing label if any.")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    return parser.parse_args()


def setup_logging(log_file: Path, level: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("eeg_qc")


def prepare_channel_selection(
    raw: mne.io.BaseRaw,
    standard_names: set[str],
    logger: logging.Logger | None = None,
) -> Tuple[mne.io.BaseRaw | None, List[str], Dict[str, object]]:
    """Return (raw restricted to 10-20 channels, picks, montage stats)."""
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        canonical = {name.lower(): name for name in montage.ch_names}
        mapping: Dict[str, str] = {}
        for ch_name in raw.ch_names:
            canon = canonical.get(ch_name.lower())
            if canon and ch_name != canon:
                mapping[ch_name] = canon
        if mapping:
            raw.rename_channels(mapping)
        raw.set_montage(montage, on_missing="ignore")
        if logger:
            filenames = getattr(raw, "filenames", None)
            logger.debug(
                "Applied standard_1020 montage to %s", filenames[0] if filenames else "recording"
            )
    except Exception as exc:  # pragma: no cover - defensive branch
        if logger:
            logger.warning("Unable to set default montage: %s", exc)

    lower_map = {name.lower(): name for name in raw.ch_names}
    basic_picks = [lower_map[ch.lower()] for ch in BASIC_1020_CHANNELS if ch.lower() in lower_map]
    eeg_chs = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)).ch_names
    matched = [ch for ch in eeg_chs if ch.lower() in standard_names]
    non_standard = [ch for ch in eeg_chs if ch.lower() not in standard_names]
    pct_missing = (1.0 - (len(matched) / max(len(BASIC_1020_CHANNELS), 1))) * 100.0
    montage_info = {
        "n_channels_1020_match": len(matched),
        "non_standard_channels": non_standard,
        "pct_missing_1020": pct_missing,
    }
    analysis_raw = raw.copy().pick(basic_picks) if basic_picks else None
    return analysis_raw, basic_picks, montage_info


def discover_bids_files(
    bids_root: Path, args: argparse.Namespace, subjects_filter: set[str] | None = None
) -> List[Path]:
    """Use BIDSPath matching to find EEG BrainVision files under a BIDS root."""
    template = BIDSPath(
        root=bids_root,
        subject=None,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    matches = template.match()
    files: List[Path] = []
    for match in matches:
        subj = match.subject or ""
        subj_tag = f"sub-{subj}" if subj else ""
        if subjects_filter:
            if subj_tag not in subjects_filter and subj not in subjects_filter:
                continue
        if match.fpath is not None and match.fpath.exists():
            files.append(match.fpath)
    return sorted(files)


def read_subjects_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def parse_subject_id(filepath: Path) -> str:
    match = re.search(r"sub-([A-Za-z0-9]+)", filepath.name)
    if match:
        return f"sub-{match.group(1)}"
    return filepath.stem


def load_raw(filepath: Path, bids_root: Path, args: argparse.Namespace) -> mne.io.BaseRaw:
    if filepath.suffix.lower() != ".vhdr":
        raise ValueError(f"Unsupported file extension (only .vhdr supported): {filepath.suffix}")
    subject_clean = parse_subject_id(filepath).replace("sub-", "")
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject_clean,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    return read_raw_bids(bids_path)


def load_meas_datetimes(bids_root: Path) -> pd.Series:
    tsv_path = bids_root / "participants.tsv"
    if not tsv_path.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_csv(tsv_path, sep="\t")
    if "meas" not in df:
        return pd.Series(dtype="datetime64[ns]")
    meas_series = pd.to_datetime(df["meas"], errors="coerce", utc=True)
    meas_series = meas_series.dropna()
    if meas_series.empty:
        return pd.Series(dtype="datetime64[ns]")
    try:
        meas_series = meas_series.dt.tz_convert(None)
    except TypeError:
        meas_series = meas_series.dt.tz_localize(None)
    return meas_series

def extract_metadata(raw: mne.io.BaseRaw) -> Dict[str, object]:
    duration_sec = raw.n_times / float(raw.info["sfreq"])
    meas_date = raw.info.get("meas_date")
    meas_date_iso = meas_date.isoformat() if meas_date else ""
    ch_names = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)).ch_names
    return {
        "duration_min": duration_sec / 60.0,
        "meas_date": meas_date_iso,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(ch_names),
        "channel_names": ch_names,
    }




def compute_channel_amplitude_stats(raw: mne.io.BaseRaw | None, picks: List[str]) -> Dict[str, object]:
    if raw is None or not picks:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "per_channel": np.array([]),
        }
    data = raw.get_data(picks=picks) * 1e6  # uV
    ptp = np.ptp(data, axis=1)
    return {
        "mean": float(ptp.mean()),
        "median": float(np.median(ptp)),
        "std": float(ptp.std()),
        "min": float(ptp.min()),
        "max": float(ptp.max()),
        "per_channel": ptp,
    }


def detect_flat_and_noisy_channels(raw: mne.io.BaseRaw | None, picks: List[str]) -> Dict[str, object]:
    if raw is None or not picks:
        return {
            "flat_channels": [],
            "noisy_channels": [],
            "n_flat_channels": 0,
            "n_noisy_channels": 0,
            "pct_bad_channels": float("nan"),
            "variances": np.array([]),
        }
    data = raw.get_data(picks=picks) * 1e6  # uV
    variances = np.var(data, axis=1)
    low_thresh = np.percentile(variances, 1)
    high_thresh = np.percentile(variances, 99)
    flat_idx = np.where(variances < low_thresh)[0]
    noisy_idx = np.where(variances > high_thresh)[0]
    eeg_chs = picks
    n_channels = len(eeg_chs)
    return {
        "flat_channels": [eeg_chs[i] for i in flat_idx],
        "noisy_channels": [eeg_chs[i] for i in noisy_idx],
        "n_flat_channels": int(len(flat_idx)),
        "n_noisy_channels": int(len(noisy_idx)),
        "% bad_channels": float((len(flat_idx) + len(noisy_idx)) / max(n_channels, 1) * 100.0),
        "variances": variances,
    }


def compute_psd_metrics(
    raw: mne.io.BaseRaw | None, picks: List[str], fmin: float = 1.0, fmax: float = 60.0
) -> Tuple[mne.time_frequency.Spectrum | None, np.ndarray, np.ndarray, float, Dict[str, float]]:
    if raw is None or not picks:
        empty = np.array([])
        return None, empty, empty, float("nan"), {k: float("nan") for k in BAND_LIMITS}
    spec = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose="ERROR")
    psd, freqs = spec.get_data(return_freqs=True)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    if alpha_mask.any():
        alpha_idx = np.argmax(psd[:, alpha_mask].mean(axis=0))
        alpha_peak = float(freqs[alpha_mask][alpha_idx])
    else:
        alpha_peak = float("nan")

    band_powers: Dict[str, float] = {}
    for band, (low, high) in BAND_LIMITS.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if band_mask.any():
            band_power = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1).mean()
            band_powers[band] = float(band_power * 1e12)  # convert V^2 to uV^2
        else:
            band_powers[band] = float("nan")

    return spec, psd, freqs, alpha_peak, band_powers


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


def plot_psd_figures(
    spec: mne.time_frequency.Spectrum, freqs: np.ndarray, psd: np.ndarray
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


def plot_events_distribution(event_counts: Dict[str, object]) -> matplotlib.figure.Figure | None:
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
            data_min = int(np.floor(data.min()))
            data_max = int(np.ceil(data.max()))
            if data_max == data_min:
                bin_edges = np.array([data_min - 0.5, data_min + 0.5])
            else:
                start_edge = data_min - 0.5
                end_edge = data_max + 0.5
                bin_edges = np.arange(start_edge, end_edge + 1.0, 1.0)
            axes[ax_idx].hist(data, bins=bin_edges, color="#4C72B0", alpha=0.85, edgecolor="black")
            xticks = np.arange(data_min, data_max + 1, 1)
            if xticks.size == 0:
                xticks = np.array([data_min])
            display_ticks = xticks
            if xticks.size > 20:
                display_ticks = xticks[::5]
                if xticks[-1] not in display_ticks:
                    display_ticks = np.append(display_ticks, xticks[-1])
            axes[ax_idx].set_xticks(display_ticks)
            axes[ax_idx].set_xlim(bin_edges[0], bin_edges[-1])
            axes[ax_idx].set_title(label)
            axes[ax_idx].set_xlabel("Count")
            axes[ax_idx].set_ylabel("Subjects")
            axes[ax_idx].grid(True, axis="y", alpha=0.3)
        for extra_ax in axes[len(labels):]:
            extra_ax.axis("off")
        fig.suptitle("Event Count Distributions", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

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


def save_meas_distribution_figures(meas_datetimes: pd.Series, fig_dir: Path) -> Dict[str, Path]:
    meas_datetimes = meas_datetimes.dropna()
    if meas_datetimes.empty:
        return {}

    paths: Dict[str, Path] = {}

    def _save_hist(values: np.ndarray, bins: np.ndarray, title: str, xlabel: str, filename: str,
                   xticks: np.ndarray | None = None, xlabels: List[str] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        if xticks is not None:
            ax.set_xticks(xticks)
            if xlabels is not None:
                ax.set_xticklabels(xlabels)
        plt.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths[filename.replace(".png", "")] = out_path

    hour_values = meas_datetimes.dt.hour + (meas_datetimes.dt.minute / 60.0)
    hour_bins = np.arange(0.0, 24.5, 0.5)
    _save_hist(
        hour_values.to_numpy(dtype=float),
        hour_bins,
        "Recording Start Hour",
        "Hour (30 min bins)",
        "meas_hour_distribution.png",
        xticks=np.arange(0, 25, 2),
    )

    day_values = meas_datetimes.dt.day
    day_bins = np.arange(0.5, 32.5, 1.0)
    _save_hist(
        day_values.to_numpy(dtype=float),
        day_bins,
        "Recording Day of Month",
        "Day of Month",
        "meas_day_distribution.png",
        xticks=np.arange(1, 32, 2),
    )

    dow_values = meas_datetimes.dt.dayofweek
    dow_bins = np.arange(-0.5, 7.5, 1.0)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _save_hist(
        dow_values.to_numpy(dtype=float),
        dow_bins,
        "Recording Day of Week",
        "Day of Week",
        "meas_dayofweek_distribution.png",
        xticks=np.arange(0, 7, 1),
        xlabels=dow_labels,
    )

    month_values = meas_datetimes.dt.month
    month_bins = np.arange(0.5, 12.5 + 1, 1.0)
    _save_hist(
        month_values.to_numpy(dtype=float),
        month_bins,
        "Recording Month",
        "Month",
        "meas_month_distribution.png",
        xticks=np.arange(1, 13, 1),
    )

    year_values = meas_datetimes.dt.year
    year_min = int(year_values.min())
    year_max = int(year_values.max())
    year_bins = np.arange(year_min - 0.5, year_max + 1.5, 1.0)
    _save_hist(
        year_values.to_numpy(dtype=float),
        year_bins,
        "Recording Year",
        "Year",
        "meas_year_distribution.png",
        xticks=np.arange(year_min, year_max + 1, 1),
    )

    return paths


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


def create_subject_report(
    raw: mne.io.BaseRaw,
    metrics: Dict[str, object],
    subject_id: str,
    output_path: Path,
    fig_psd_all: matplotlib.figure.Figure | None,
    fig_psd_avg: matplotlib.figure.Figure | None,
    fig_amp_hist: matplotlib.figure.Figure | None,
    fig_var_topo: matplotlib.figure.Figure | None,
    fig_raw_segment_start: matplotlib.figure.Figure | None,
    fig_raw_segment_end: matplotlib.figure.Figure | None,
    fig_events: matplotlib.figure.Figure | None,
) -> None:
    report = mne.Report(title=f"EEG QC Report - {subject_id}")
    try:
        report.add_raw(raw, title="Raw Data (with PSD)", psd=True)
    except Exception:
        pass

    if fig_psd_all is not None:
        report.add_figure(fig_psd_all, title="PSD - All Channels", section="Power Spectral Density")
    if fig_psd_avg is not None:
        report.add_figure(fig_psd_avg, title="PSD - Average", section="Power Spectral Density")
    if fig_amp_hist is not None:
        report.add_figure(fig_amp_hist, title="Amplitude Distribution", section="Signal Quality")
    if fig_var_topo is not None:
        report.add_figure(fig_var_topo, title="Channel Variance Topomap", section="Signal Quality")
    if fig_raw_segment_start is not None:
        report.add_figure(fig_raw_segment_start, title="Raw Segment - Start", section="Signal Quality")
    if fig_raw_segment_end is not None:
        report.add_figure(fig_raw_segment_end, title="Raw Segment - End", section="Signal Quality")
    if fig_events is not None:
        report.add_figure(fig_events, title="Annotation Counts", section="Events")

    duration_min = metrics.get("duration_min", float("nan"))
    sfreq = metrics.get("sfreq", float("nan"))
    n_channels = metrics.get("n_channels", 0)
    n_1020 = metrics.get("n_channels_1020_match", 0)
    pct_bad = metrics.get("pct_bad_channels", float("nan"))
    amp_mean = metrics.get("amplitude_mean_uv", float("nan"))
    amp_median = metrics.get("amplitude_median_uv", float("nan"))
    amp_max = metrics.get("amplitude_max_uv", float("nan"))
    alpha_peak = metrics.get("alpha_peak_hz", float("nan"))
    start_sec = metrics.get("actual_signal_start_sec", float("nan"))
    end_sec = metrics.get("actual_signal_end_sec", float("nan"))
    empty_start = metrics.get("empty_start_sec", float("nan"))
    empty_end = metrics.get("empty_end_sec", float("nan"))
    n_flat = metrics.get("n_flat_channels", 0)
    n_noisy = metrics.get("n_noisy_channels", 0)
    eyes_open_count = metrics.get("eyes_open_event_count", 0)
    eyes_closed_count = metrics.get("eyes_closed_event_count", 0)
    movement_count = metrics.get("movement_event_count", 0)
    artefact_count = metrics.get("artefact_event_count", 0)
    effort_count = metrics.get("effort_event_count", 0)
    pat_count = metrics.get("pat_montage_event_count", 0)
    hv_count = metrics.get("hv_event_count", 0)
    post_hv_count = metrics.get("post_hv_event_count", 0)
    photo_count = metrics.get("photo_event_count", 0)
    yawn_cough_count = metrics.get("yawning_coughing_event_count", 0)
    jaw_face_count = metrics.get("jaw_face_tension_event_count", 0)
    sleepy_count = metrics.get("sleepy_event_count", 0)
    sleep_count = metrics.get("sleep_event_count", 0)
    collab_count = metrics.get("collaboration_event_count", 0)
    emotion_count = metrics.get("emotion_behavior_event_count", 0)
    oral_count = metrics.get("oral_activity_event_count", 0)
    eye_move_count = metrics.get("eye_movement_event_count", 0)
    wake_count = metrics.get("wakefulness_event_count", 0)
    resp_count = metrics.get("respiration_event_count", 0)
    sensor_action_kw_count = metrics.get("sensor_action_keyword_event_count", 0)
    eye_keyword_count = metrics.get("eye_movement_keyword_event_count", 0)
    clinical_comment_count = metrics.get("clinical_comment_event_count", 0)
    band_power_items = []
    for band in BAND_LIMITS:
        value = metrics.get(f"band_power_{band}", float("nan"))
        if np.isnan(value):
            continue
        band_power_items.append(f"{band.title()}: {value:.2e} uV^2")
    band_str = ", ".join(band_power_items) if band_power_items else "Unavailable"

    qc_summary_html = "<ul>"
    qc_summary_html += f"<li>Duration: {duration_min:.2f} min @ {sfreq:.1f} Hz</li>"
    qc_summary_html += f"<li>Channels: {n_channels} total / {n_1020} (10-20 match)</li>"
    qc_summary_html += f"<li>Bad channels: {pct_bad:.1f}% (flat={n_flat}, noisy={n_noisy})</li>"
    qc_summary_html += (
        f"<li>Signal activity: start {start_sec:.1f}s (empty {empty_start:.1f}s), "
        f"end {end_sec:.1f}s (empty tail {empty_end:.1f}s)</li>"
    )
    qc_summary_html += (
        f"<li>Amplitude (uV): mean {amp_mean:.1f}, median {amp_median:.1f}, max {amp_max:.1f}</li>"
    )
    qc_summary_html += f"<li>Alpha peak: {alpha_peak:.2f} Hz</li>"
    qc_summary_html += f"<li>Band powers: {band_str}</li>"
    if movement_count:
        qc_summary_html += f"<li>Movement events: {movement_count}</li>"
    if artefact_count:
        qc_summary_html += f"<li>Artefact events: {artefact_count}</li>"
    if effort_count:
        qc_summary_html += f"<li>Effort events: {effort_count}</li>"
    if pat_count:
        qc_summary_html += f"<li>PAT montage events: {pat_count}</li>"
    if hv_count:
        qc_summary_html += f"<li>HV events: {hv_count}</li>"
    if post_hv_count:
        qc_summary_html += f"<li>Post-HV events: {post_hv_count}</li>"
    if photo_count:
        qc_summary_html += f"<li>PHOTO events: {photo_count}</li>"
    if yawn_cough_count:
        qc_summary_html += f"<li>Yawning/Coughing events: {yawn_cough_count}</li>"
    if jaw_face_count:
        qc_summary_html += f"<li>Jaw/Face tension events: {jaw_face_count}</li>"
    if sleepy_count:
        qc_summary_html += f"<li>Sleepy events: {sleepy_count}</li>"
    if sleep_count:
        qc_summary_html += f"<li>Sleep events: {sleep_count}</li>"
    if collab_count:
        qc_summary_html += f"<li>Collaboration comments: {collab_count}</li>"
    if emotion_count:
        qc_summary_html += f"<li>Emotion/Behavior comments: {emotion_count}</li>"
    if oral_count:
        qc_summary_html += f"<li>Oral activity events: {oral_count}</li>"
    if eye_move_count:
        qc_summary_html += f"<li>Eye movement events: {eye_move_count}</li>"
    if wake_count:
        qc_summary_html += f"<li>Wakefulness events: {wake_count}</li>"
    if resp_count:
        qc_summary_html += f"<li>Respiration events: {resp_count}</li>"
    if sensor_action_kw_count:
        qc_summary_html += f"<li>Sensor action keyword mentions: {sensor_action_kw_count}</li>"
    if eye_keyword_count:
        qc_summary_html += f"<li>Eye-movement keyword mentions: {eye_keyword_count}</li>"
    if clinical_comment_count:
        qc_summary_html += f"<li>Clinical comment labels: {clinical_comment_count}</li>"
    if metrics.get("flag_reasons"):
        qc_summary_html += f"<li>Flag reasons: {metrics.get('flag_reasons')}</li>"
    if metrics.get("event_counts"):
        qc_summary_html += f"<li>Events: {metrics.get('event_counts')}</li>"
    qc_summary_html += "</ul>"
    report.add_html(qc_summary_html, title="QC Summary", section="Quality Control")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallBack(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def process_file(
    filepath: Path,
    args: argparse.Namespace,
    standard_names: set[str],
    output_dirs: Dict[str, Path],
    logger: logging.Logger,
) -> Dict[str, object]:
    subject_id = parse_subject_id(filepath)
    metrics: Dict[str, object] = {
        "filepath": str(filepath),
        "subject_id": subject_id,
        "duration_min": float("nan"),
        "actual_signal_start_sec": float("nan"),
        "empty_start_sec": float("nan"),
        "actual_signal_end_sec": float("nan"),
        "empty_end_sec": float("nan"),
        "meas_date": "",
        "sfreq": float("nan"),
        "n_channels": 0,
        "channel_names": "",
        "n_channels_1020_match": 0,
        "non_standard_channels": "",
        "n_flat_channels": 0,
        "n_noisy_channels": 0,
        "pct_bad_channels": float("nan"),
        "amplitude_mean_uv": float("nan"),
        "amplitude_median_uv": float("nan"),
        "amplitude_std_uv": float("nan"),
        "amplitude_min_uv": float("nan"),
        "amplitude_max_uv": float("nan"),
        "alpha_peak_hz": float("nan"),
        "eyes_open_event_count": 0,
        "eyes_closed_event_count": 0,
        "movement_event_count": 0,
        "artefact_event_count": 0,
        "effort_event_count": 0,
        "pat_montage_event_count": 0,
        "hv_event_count": 0,
        "post_hv_event_count": 0,
        "photo_event_count": 0,
        "yawning_coughing_event_count": 0,
        "jaw_face_tension_event_count": 0,
        "sleepy_event_count": 0,
        "sleep_event_count": 0,
        "collaboration_event_count": 0,
        "emotion_behavior_event_count": 0,
        "oral_activity_event_count": 0,
        "eye_movement_event_count": 0,
        "wakefulness_event_count": 0,
        "respiration_event_count": 0,
        "sensor_action_keyword_event_count": 0,
        "eye_movement_keyword_event_count": 0,
        "clinical_comment_event_count": 0,
        "event_counts": "",
        "flag_bad": False,
        "flag_reasons": "",
        "error": "",
    }
    band_power_fields = {f"band_power_{band}": float("nan") for band in BAND_LIMITS}
    metrics.update(band_power_fields)
    analysis_raw: mne.io.BaseRaw | None = None
    basic_picks: List[str] = []
    montage_info: Dict[str, object] = {}

    analysis_start_offset = 0.0
    original_duration = float("nan")
    try:
        raw = load_raw(filepath, args.input_dir, args)
        raw.load_data()
        raw.filter(1, None, fir_design="firwin", verbose="ERROR")
        original_duration = raw.times[-1]
        analysis_start_offset = crop_raw_after_reference_event(raw, raw.annotations, logger)
        analysis_raw, basic_picks, montage_info = prepare_channel_selection(raw, standard_names, logger)
    except Exception as exc:  # pragma: no cover - defensive branch
        err_msg = f"Failed to read {filepath.name}: {exc}"
        logger.error(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True, "flag_reasons": "load_error"})
        return metrics

    try:
        meta = extract_metadata(raw)
        metrics.update(meta)

        metrics["n_channels_1020_match"] = montage_info["n_channels_1020_match"]
        metrics["non_standard_channels"] = ",".join(montage_info["non_standard_channels"])
        metrics["channel_names"] = ",".join(basic_picks) if basic_picks else ",".join(meta.get("channel_names", []))
        pct_missing_1020 = montage_info["pct_missing_1020"]

        amp_stats = compute_channel_amplitude_stats(analysis_raw, basic_picks)
        metrics.update(
            {
                "amplitude_mean_uv": amp_stats["mean"],
                "amplitude_median_uv": amp_stats["median"],
                "amplitude_std_uv": amp_stats["std"],
                "amplitude_min_uv": amp_stats["min"],
                "amplitude_max_uv": amp_stats["max"],
            }
        )


        noise_info = detect_flat_and_noisy_channels(analysis_raw, basic_picks)
        metrics["n_flat_channels"] = noise_info["n_flat_channels"]
        metrics["n_noisy_channels"] = noise_info["n_noisy_channels"]
        metrics["pct_bad_channels"] = noise_info["% bad_channels"]
        metrics["% bad_channels"] = noise_info["% bad_channels"]

        analysis_duration = raw.times[-1] if raw is not None else float("nan")
        onset_sec = analysis_start_offset
        offset_sec = analysis_start_offset + (analysis_duration if np.isfinite(analysis_duration) else 0.0)
        if np.isfinite(original_duration):
            offset_sec = min(offset_sec, original_duration)
            empty_end = max(original_duration - offset_sec, 0.0)
        else:
            empty_end = 0.0
        metrics["empty_start_sec"] = onset_sec
        metrics["actual_signal_start_sec"] = onset_sec
        metrics["empty_end_sec"] = empty_end
        metrics["actual_signal_end_sec"] = offset_sec

        spec, psd, freqs, alpha_peak, band_powers = compute_psd_metrics(analysis_raw, basic_picks)
        metrics["alpha_peak_hz"] = alpha_peak
        metrics.update({f"band_power_{k}": v for k, v in band_powers.items()})

        annotation_counts = summarize_annotations(raw.annotations)
        metrics["event_counts"] = (
            json.dumps(annotation_counts, ensure_ascii=False) if annotation_counts else ""
        )
        metrics["eyes_open_event_count"] = int(annotation_counts.get("Eyes Open", 0))
        metrics["eyes_closed_event_count"] = int(annotation_counts.get("Eyes Closed", 0))
        metrics["movement_event_count"] = int(annotation_counts.get("Movement", 0))
        metrics["artefact_event_count"] = int(annotation_counts.get("Artefact", 0))
        metrics["effort_event_count"] = int(annotation_counts.get("Effort", 0))
        metrics["pat_montage_event_count"] = int(annotation_counts.get("PAT Montage", 0))
        metrics["hv_event_count"] = int(annotation_counts.get("HV", 0))
        metrics["photo_event_count"] = int(annotation_counts.get("PHOTO", 0))
        metrics["yawning_coughing_event_count"] = int(annotation_counts.get("Yawning/Coughing", 0))
        metrics["jaw_face_tension_event_count"] = int(annotation_counts.get("Jaw/Face Tension", 0))
        metrics["sleepy_event_count"] = int(annotation_counts.get("Sleepy", 0))
        metrics["sleep_event_count"] = int(annotation_counts.get("Sleep", 0))
        metrics["collaboration_event_count"] = int(annotation_counts.get("Collaboration", 0))
        metrics["emotion_behavior_event_count"] = int(annotation_counts.get("Emotion/Behavior", 0))
        metrics["oral_activity_event_count"] = int(annotation_counts.get("Oral Activity", 0))
        metrics["eye_movement_event_count"] = int(annotation_counts.get("Eye Movement", 0))
        metrics["wakefulness_event_count"] = int(annotation_counts.get("Wakefulness", 0))
        metrics["respiration_event_count"] = int(annotation_counts.get("Respiration", 0))
        special_counts = compute_special_event_counts(raw.annotations)
        metrics["sensor_action_keyword_event_count"] = special_counts["sensor_action_keyword_events"]
        metrics["eye_movement_keyword_event_count"] = special_counts["eye_movement_keyword_events"]
        metrics["clinical_comment_event_count"] = special_counts["clinical_comment_events"]

        # Base flagging (dataset-level outlier flags added later)
        reasons: List[str] = []
        if metrics["duration_min"] < args.min_duration:
            reasons.append("short_duration")
        if metrics["duration_min"] > args.max_duration:
            reasons.append("long_duration")
        if metrics["pct_bad_channels"] > 20:
            reasons.append("too_many_bad_channels")
        if pct_missing_1020 > 20:
            reasons.append("missing_1020_channels")
        if metrics["amplitude_max_uv"] > args.amplitude_threshold:
            reasons.append("amplitude_above_threshold")
        if metrics["empty_start_sec"] > 120:
            reasons.append("long_empty_start")
        if metrics["empty_end_sec"] > 120:
            reasons.append("long_empty_end")
        if np.isnan(metrics["alpha_peak_hz"]):
            reasons.append("no_alpha_peak")
        if not basic_picks:
            reasons.append("no_basic_1020_channels")

        metrics["flag_bad"] = bool(reasons)
        metrics["flag_reasons"] = ";".join(reasons)

        if args.generate_subject_reports:
            fig_psd_all = fig_psd_avg = fig_amp_hist = fig_var_topo = None
            fig_raw_segment_start = fig_raw_segment_end = fig_events = None

            if not args.skip_figures:
                if spec is not None and psd.size > 0:
                    try:
                        fig_psd_all, fig_psd_avg = plot_psd_figures(spec, freqs, psd)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("PSD plotting failed for %s: %s", filepath.name, exc)
                if amp_stats["per_channel"].size > 0:
                    try:
                        fig_amp_hist = plot_amplitude_histogram(amp_stats)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Amplitude histogram failed for %s: %s", filepath.name, exc)
                if analysis_raw is not None:
                    try:
                        fig_var_topo = plot_channel_variance_topomap(analysis_raw)
                        safe_onset = onset_sec if np.isfinite(onset_sec) else 0.0
                        fig_raw_segment_start = plot_raw_segment(
                            analysis_raw, max(safe_onset, 0.0), title="Raw Segment - Start (10s)"
                        )
                        if np.isfinite(offset_sec):
                            last_start = max(offset_sec - 10.0, 0.0)
                        else:
                            last_start = max(analysis_raw.times[-1] - 10.0, 0.0)
                        fig_raw_segment_end = plot_raw_segment(
                            analysis_raw, last_start, title="Raw Segment - End (10s)"
                        )
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Raw/variance plotting failed for %s: %s", filepath.name, exc)
                if annotation_counts:
                    try:
                        fig_events = plot_events_distribution(annotation_counts)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Event count plotting failed for %s: %s", filepath.name, exc)

            report_path = output_dirs["subject_reports"] / f"{subject_id}_qc_report.html"
            try:
                create_subject_report(
                    analysis_raw if analysis_raw is not None else raw,
                    metrics,
                    subject_id,
                    report_path,
                    fig_psd_all,
                    fig_psd_avg,
                    fig_amp_hist,
                    fig_var_topo,
                    fig_raw_segment_start,
                    fig_raw_segment_end,
                    fig_events,
                )
            except Exception as fig_exc:  # pragma: no cover - defensive branch
                logger.warning("Report generation failed for %s: %s", filepath.name, fig_exc)

    except Exception as exc:  # pragma: no cover - defensive branch
        err_msg = f"Processing failed for {filepath.name}: {exc}"
        logger.exception(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True})
    finally:
        plt.close("all")
        try:
            raw.close()
        except Exception:
            pass

    return metrics


def compute_dataset_stats(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(records)
    metric_cols = [
        "amplitude_mean_uv",
        "amplitude_max_uv",
        "pct_bad_channels",
        "duration_min",
        "alpha_peak_hz",
        "band_power_delta",
        "band_power_theta",
        "band_power_alpha",
        "band_power_beta",
        "band_power_gamma",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {"mean": float(series.mean(skipna=True)), "std": float(series.std(skipna=True))}
    return stats


def apply_dataset_outlier_flags(
    records: List[Dict[str, object]], dataset_stats: Dict[str, Dict[str, float]]
) -> None:
    for rec in records:
        if rec.get("error"):
            continue
        reasons = rec.get("flag_reasons", "").split(";") if rec.get("flag_reasons") else []
        for col, stats in dataset_stats.items():
            value = rec.get(col)
            if value is None or np.isnan(value):
                continue
            std = stats["std"]
            mean = stats["mean"]
            if std > 0 and abs(value - mean) > 3 * std:
                reasons.append(f"{col}_outlier")
        if reasons:
            rec["flag_bad"] = True
            rec["flag_reasons"] = ";".join(sorted(set(filter(None, reasons))))
        else:
            rec["flag_bad"] = False
            rec["flag_reasons"] = ""


def summarize_flags(records: List[Dict[str, object]]) -> Counter:
    counter: Counter = Counter()
    for rec in records:
        reasons = rec.get("flag_reasons", "")
        if not reasons:
            continue
        for reason in reasons.split(";"):
            if reason:
                counter[reason] += 1
    return counter


def collect_unknown_events(
    records: List[Dict[str, object]], known_labels: set[str]
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, object]] = {}
    for rec in records:
        payload = rec.get("event_counts")
        if not payload:
            continue
        try:
            counts = json.loads(payload)
        except Exception:
            continue
        subject_id = rec.get("subject_id", "unknown")
        for label, value in counts.items():
            if label in known_labels:
                continue
            try:
                occurrences = int(value)
            except Exception:
                continue
            if occurrences <= 0:
                continue
            entry = summary.setdefault(label, {"occurrences": 0, "subjects": set()})
            entry["occurrences"] += occurrences
            entry["subjects"].add(subject_id)
    formatted: Dict[str, Dict[str, int]] = {}
    for label, data in summary.items():
        formatted[label] = {
            "occurrences": int(data["occurrences"]),
            "n_subjects": len(data["subjects"]),
        }
    return formatted


def save_figures(
    df: pd.DataFrame,
    flags_counter: Counter,
    fig_dir: Path,
    meas_datetimes: pd.Series | None = None,
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def _save_hist(column: str, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(series, bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths[column] = out_path

    _save_hist("duration_min", "Duration Distribution (min)", "dataset_duration_distribution.png")
    _save_hist("amplitude_mean_uv", "Mean Amplitude Distribution (uV)", "dataset_amplitude_distribution.png")
    _save_hist("alpha_peak_hz", "Alpha Peak Distribution (Hz)", "dataset_alpha_peak_distribution.png")

    # Flag reasons bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    if flags_counter:
        labels, values = zip(*flags_counter.most_common())
        ax.bar(labels, values)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Flagged Subjects by Reason")
    plt.tight_layout()
    flag_path = fig_dir / "flagged_subjects_summary.png"
    fig.savefig(flag_path, dpi=150)
    plt.close(fig)
    paths["flag_reasons"] = flag_path

    event_specs = [
        ("Eyes Open", "eyes_open_event_count"),
        ("Eyes Closed", "eyes_closed_event_count"),
        ("Movement", "movement_event_count"),
        ("Artefact", "artefact_event_count"),
        ("PAT Montage", "pat_montage_event_count"),
        ("HV", "hv_event_count"),
        ("PHOTO", "photo_event_count"),
        ("Yawning/Coughing", "yawning_coughing_event_count"),
        ("Jaw/Face Tension", "jaw_face_tension_event_count"),
        ("Sleepy", "sleepy_event_count"),
        ("Sleep", "sleep_event_count"),
        ("Collaboration", "collaboration_event_count"),
        ("Emotion/Behavior", "emotion_behavior_event_count"),
        ("Oral Activity", "oral_activity_event_count"),
        ("Eye Movement", "eye_movement_event_count"),
        ("Wakefulness", "wakefulness_event_count"),
        ("Respiration", "respiration_event_count"),
        ("Sensor Actions", "sensor_action_keyword_event_count"),
        ("Eye Movement Keywords", "eye_movement_keyword_event_count"),
        ("Clinical Comments", "clinical_comment_event_count"),
    ]
    events_distribution: Dict[str, np.ndarray] = {}
    for label, count_col in event_specs:
        if count_col not in df:
            continue
        series = pd.to_numeric(df[count_col], errors="coerce").dropna()
        if label not in EVENT_LABELS_ALWAYS_KEEP_ZERO:
            series = series[series > 0]
        if series.empty:
            continue
        events_distribution[label] = series.to_numpy(dtype=float)
    if events_distribution:
        fig = plot_events_distribution(events_distribution)
        if fig is not None:
            event_path = fig_dir / "event_count_distributions.png"
            fig.savefig(event_path, dpi=150)
            plt.close(fig)
            paths["event_stats"] = event_path

    if meas_datetimes is not None and not meas_datetimes.empty:
        meas_paths = save_meas_distribution_figures(meas_datetimes, fig_dir)
        paths.update(meas_paths)

    return paths


def create_summary_report(
    df: pd.DataFrame,
    fig_paths: Dict[str, Path],
    output_path: Path,
    total_files: int,
    flags_counter: Counter,
    unknown_events: Dict[str, Dict[str, int]] | None = None,
) -> None:
    report = mne.Report(title="EEG QC Dataset Summary")
    valid_records = int((df["error"] == "").sum()) if "error" in df else len(df)
    flagged_count = int(df["flag_bad"].sum()) if "flag_bad" in df else 0
    summary_html = f"""
    <h3>Dataset Summary</h3>
    <ul>
        <li>Total files processed: {total_files}</li>
        <li>Valid records: {valid_records}</li>
        <li>Flagged bad: {flagged_count}</li>
    </ul>
    """
    if flags_counter:
        summary_html += "<p>Most common flag reasons:</p><ul>"
        for reason, count in flags_counter.most_common():
            summary_html += f"<li>{reason}: {count}</li>"
        summary_html += "</ul>"

    def _describe_series(column: str) -> Dict[str, float] | None:
        if column not in df:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return None
        return {
            "mean": float(series.mean()),
            "max": float(series.max()),
            "min": float(series.min()),
            "std": float(series.std()),
        }

    def _sum_counts(column: str) -> int | None:
        if column not in df:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return None
        return int(series.sum())

    event_fields = [
        ("Eyes Open", "eyes_open_event_count"),
        ("Eyes Closed", "eyes_closed_event_count"),
        ("Movement", "movement_event_count"),
        ("Artefact", "artefact_event_count"),
        ("PAT Montage", "pat_montage_event_count"),
        ("HV", "hv_event_count"),
        ("PHOTO", "photo_event_count"),
        ("Yawning/Coughing", "yawning_coughing_event_count"),
        ("Jaw/Face Tension", "jaw_face_tension_event_count"),
        ("Sleepy", "sleepy_event_count"),
        ("Sleep", "sleep_event_count"),
        ("Collaboration", "collaboration_event_count"),
        ("Emotion/Behavior", "emotion_behavior_event_count"),
        ("Oral Activity", "oral_activity_event_count"),
        ("Eye Movement", "eye_movement_event_count"),
        ("Wakefulness", "wakefulness_event_count"),
        ("Respiration", "respiration_event_count"),
        ("Sensor Actions", "sensor_action_keyword_event_count"),
        ("Eye Movement Keywords", "eye_movement_keyword_event_count"),
        ("Clinical Comments", "clinical_comment_event_count"),
    ]
    event_items: List[str] = []
    for label, column in event_fields:
        total = _sum_counts(column)
        if total:
            event_items.append(f"<li>{label}: {total}</li>")
    if event_items:
        summary_html += "<p>Annotation totals:</p><ul>" + "".join(event_items) + "</ul>"

    report.add_html(summary_html, title="Summary", section="Overview")

    for title, path in [
        ("Duration Distribution", fig_paths.get("duration_min")),
        ("Mean Amplitude Distribution", fig_paths.get("amplitude_mean_uv")),
        ("Alpha Peak Distribution", fig_paths.get("alpha_peak_hz")),
        ("Flag Reasons", fig_paths.get("flag_reasons")),
        ("Event Count Distributions", fig_paths.get("event_stats")),
        ("Recording Start Hour", fig_paths.get("meas_hour_distribution")),
        ("Recording Day of Month", fig_paths.get("meas_day_distribution")),
        ("Recording Day of Week", fig_paths.get("meas_dayofweek_distribution")),
        ("Recording Month", fig_paths.get("meas_month_distribution")),
        ("Recording Year", fig_paths.get("meas_year_distribution")),
    ]:
        if path and path.exists():
            report.add_image(path, title=title, section="Figures")

    if unknown_events:
        unknown_html = "<p>Unrecognized annotation labels:</p><ul>"
        for label, stats in sorted(unknown_events.items(), key=lambda item: item[1]["occurrences"], reverse=True):
            unknown_html += (
                f"<li>{label}: {stats['occurrences']} occurrences; {stats['n_subjects']} subjects</li>"
            )
        unknown_html += "</ul>"
        report.add_html(unknown_html, title="Unrecognized Annotation Labels", section="Unrecognized Annotations")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    subject_reports_dir = output_dir / "subject_reports"
    fig_dir = output_dir / "figures"
    log_dir = output_dir / "logs"

    logger = setup_logging(log_dir / "qc_processing.log", args.log_level)
    logger.info("Starting EEG QC")

    subjects_filter = read_subjects_list(args.subjects_list) if args.subjects_list else None
    files = discover_bids_files(args.input_dir, args, subjects_filter)
    if not files:
        logger.error("No BIDS EEG (.vhdr) files found in %s with specified filters", args.input_dir)
        sys.exit(1)

    standard_names = {ch.lower() for ch in BASIC_1020_CHANNELS}
    logger.info("Found %d files to process", len(files))

    output_dirs = {"subject_reports": subject_reports_dir, "figures": fig_dir, "logs": log_dir}
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm(total=len(files), desc="Processing EEG files")):
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(process_file)(
                filepath=f,
                args=args,
                standard_names=standard_names,
                output_dirs=output_dirs,
                logger=logger,
            )
            for f in files
        )

    dataset_stats = compute_dataset_stats(results)
    apply_dataset_outlier_flags(results, dataset_stats)

    df = pd.DataFrame(results)
    csv_path = output_dir / "qc_report.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV report to %s", csv_path)

    if args.save_json:
        json_path = output_dir / "qc_report.json"
        json_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved JSON report to %s", json_path)

    flags_counter = summarize_flags(results)
    unknown_events = collect_unknown_events(results, KNOWN_EVENT_LABELS)
    meas_datetimes = load_meas_datetimes(args.input_dir)
    fig_paths = {}
    if not args.skip_figures:
        fig_paths = save_figures(df, flags_counter, fig_dir, meas_datetimes)
        summary_report_path = output_dir / "qc_summary_report.html"
        create_summary_report(
            df,
            fig_paths,
            summary_report_path,
            len(files),
            flags_counter,
            unknown_events,
        )
        logger.info("Saved summary HTML report to %s", summary_report_path)

    logger.info("QC finished. Total files: %d, flagged bad: %d", len(files), int(df["flag_bad"].sum()))


if __name__ == "__main__":
    main()

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
import unicodedata
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

# Headless-friendly backend for figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BAND_LIMITS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}

BASIC_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

EYES_OPEN_LABELS = (
    "eyes open",
    "eye open",
    "eyes-open",
    "eo",
    "yeux ouverts",
    "yeux ouvert",
)
EYES_CLOSED_LABELS = (
    "eyes closed",
    "eye closed",
    "eyes-closed",
    "ec",
    "yeux fermes",
    "yeux ferme",
)
MOVEMENT_LABELS = ("bouge", "movement", "mouvement")
ARTEFACT_LABELS = ("artefact", "artefacts", "artifact", "artifacts")
EFFORT_LABELS = ("effort",)
PAT_MONTAGE_LABELS = ("pat montage",)
HV_LABELS = ("hv",)
PHOTO_LABELS = ("photo",)


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


def get_basic_1020_picks(raw: mne.io.BaseRaw) -> List[str]:
    """Return channel names present in raw that belong to the 19-channel 10-20 set."""
    lower_map = {name.lower(): name for name in raw.ch_names}
    picks: List[str] = []
    for ch in BASIC_1020_CHANNELS:
        name = lower_map.get(ch.lower())
        if name:
            picks.append(name)
    return picks


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


def validate_montage(raw: mne.io.BaseRaw, standard_names: set[str]) -> Dict[str, object]:
    eeg_chs = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)).ch_names
    matched = [ch for ch in eeg_chs if ch.lower() in standard_names]
    non_standard = [ch for ch in eeg_chs if ch.lower() not in standard_names]
    pct_missing = (1.0 - (len(matched) / max(len(BASIC_1020_CHANNELS), 1))) * 100.0
    return {
        "n_channels_1020_match": len(matched),
        "non_standard_channels": non_standard,
        "pct_missing_1020": pct_missing,
    }

def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def normalize_annotation_label(desc: str) -> str:
    clean = _strip_accents(desc).lower().strip()
    interest_map = {
        "Eyes Open": EYES_OPEN_LABELS,
        "Eyes Closed": EYES_CLOSED_LABELS,
        "Movement": MOVEMENT_LABELS,
        "Artefact": ARTEFACT_LABELS,
        "Effort": EFFORT_LABELS,
        "PAT Montage": PAT_MONTAGE_LABELS,
        "HV": HV_LABELS,
        "PHOTO": PHOTO_LABELS,
    }
    for canonical, patterns in interest_map.items():
        if any(pattern in clean for pattern in patterns):
            return canonical
    return desc.strip()


def summarize_annotations(annotations: mne.Annotations) -> Counter:
    counts: Counter = Counter()
    if annotations is None or len(annotations) == 0:
        return counts
    for desc in annotations.description:
        if desc is None:
            continue
        label = normalize_annotation_label(desc)
        if not label:
            continue
        counts[label] += 1
    return counts


def _normalize_channel_names_for_montage(raw: mne.io.BaseRaw, montage: mne.channels.DigMontage) -> None:
    """Rename channels to match montage labels (case-insensitive) when possible."""
    canonical = {name.lower(): name for name in montage.ch_names}
    mapping: Dict[str, str] = {}
    for ch_name in raw.ch_names:
        canon = canonical.get(ch_name.lower())
        if canon and ch_name != canon:
            mapping[ch_name] = canon
    if mapping:
        raw.rename_channels(mapping)


def ensure_default_montage(raw: mne.io.BaseRaw, logger: logging.Logger | None = None) -> None:
    """Attach a standard montage when EEG locations are missing."""
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        _normalize_channel_names_for_montage(raw, montage)
        raw.set_montage(montage, on_missing="ignore")
        if logger:
            filenames = getattr(raw, "filenames", None)
            logger.debug(
                "Applied standard_1020 montage to %s", filenames[0] if filenames else "recording"
            )
    except Exception as exc:  # pragma: no cover - defensive branch
        if logger:
            logger.warning("Unable to set default montage: %s", exc)


def compute_channel_amplitude_stats(raw: mne.io.BaseRaw, picks: List[str]) -> Dict[str, object]:
    if not picks:
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


def detect_flat_and_noisy_channels(raw: mne.io.BaseRaw, picks: List[str]) -> Dict[str, object]:
    if not picks:
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
    raw: mne.io.BaseRaw, picks: List[str], fmin: float = 1.0, fmax: float = 60.0
) -> Tuple[mne.time_frequency.Spectrum | None, np.ndarray, np.ndarray, float, Dict[str, float]]:
    if not picks:
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


def detect_signal_activity_bounds(
    raw: mne.io.BaseRaw, picks: List[str], threshold_var_uv2: float = 5.0
) -> Tuple[float, float, float, float]:
    """Return empty duration at start/end and signal onset/offset timestamps."""
    if not picks:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    data = raw.get_data(picks=picks) * 1e6  # uV
    sfreq = float(raw.info["sfreq"])
    window_samples = max(int(round(sfreq)), 1)
    if window_samples <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    n_windows = data.shape[1] // window_samples
    if n_windows == 0:
        return (0.0, 0.0, 0.0, 0.0)
    reshaped = data[:, : n_windows * window_samples].reshape(data.shape[0], n_windows, window_samples)
    window_vars = np.var(reshaped, axis=2).mean(axis=0)
    if not np.any(np.isfinite(window_vars)):
        total_duration = float(data.shape[1]) / sfreq
        return (total_duration, float("nan"), total_duration, float("nan"))

    perc_lo = float(np.percentile(window_vars, 10))
    perc_hi = float(np.percentile(window_vars, 90))
    span = max(perc_hi - perc_lo, 0.0)
    adaptive_thresh = perc_lo + 0.25 * span
    effective_thresh = max(threshold_var_uv2, adaptive_thresh)

    active_windows = window_vars > effective_thresh
    window_duration = window_samples / sfreq
    total_duration = float(data.shape[1]) / sfreq
    if not np.any(active_windows):
        return (total_duration, float("nan"), total_duration, float("nan"))

    active_indices = np.where(active_windows)[0]
    first_idx = int(active_indices[0])
    last_idx = int(active_indices[-1])
    empty_start_sec = float(first_idx) * window_duration
    signal_start_sec = empty_start_sec
    signal_end_sec = min(float(last_idx + 1) * window_duration, total_duration)
    empty_end_sec = max(total_duration - signal_end_sec, 0.0)
    return empty_start_sec, signal_start_sec, empty_end_sec, signal_end_sec


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


def plot_event_counts(event_counts: Dict[str, int]) -> matplotlib.figure.Figure | None:
    if not event_counts:
        return None
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
        "event_counts": "",
        "flag_bad": False,
        "flag_reasons": "",
        "error": "",
    }
    band_power_fields = {f"band_power_{band}": float("nan") for band in BAND_LIMITS}
    metrics.update(band_power_fields)

    try:
        raw = load_raw(filepath, args.input_dir, args)
        raw.load_data()
        ensure_default_montage(raw, logger)
        raw.filter(1, None, fir_design="firwin", verbose="ERROR")
    except Exception as exc:  # pragma: no cover - defensive branch
        err_msg = f"Failed to read {filepath.name}: {exc}"
        logger.error(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True, "flag_reasons": "load_error"})
        return metrics

    try:
        meta = extract_metadata(raw)
        metrics.update(meta)

        basic_picks = get_basic_1020_picks(raw)
        raw_basic = raw.copy().pick(basic_picks) if basic_picks else None
        analysis_raw = raw_basic if raw_basic is not None else raw

        montage_info = validate_montage(raw, standard_names)
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

        empty_start_sec, onset_sec, empty_end_sec, offset_sec = detect_signal_activity_bounds(
            analysis_raw, basic_picks
        )
        metrics["empty_start_sec"] = empty_start_sec
        metrics["actual_signal_start_sec"] = onset_sec
        metrics["empty_end_sec"] = empty_end_sec
        metrics["actual_signal_end_sec"] = offset_sec

        spec, psd, freqs, alpha_peak, band_powers = compute_psd_metrics(analysis_raw, basic_picks)
        metrics["alpha_peak_hz"] = alpha_peak
        metrics.update({f"band_power_{k}": v for k, v in band_powers.items()})

        annotation_counts, annotation_durations = summarize_annotations(raw.annotations)
        annotation_counts_dict = dict(annotation_counts)
        metrics["event_counts"] = (
            json.dumps(annotation_counts_dict, ensure_ascii=False) if annotation_counts_dict else ""
        )
        metrics["eyes_open_event_count"] = int(annotation_counts.get("Eyes Open", 0))
        metrics["eyes_closed_event_count"] = int(annotation_counts.get("Eyes Closed", 0))
        metrics["movement_event_count"] = int(annotation_counts.get("Movement", 0))
        metrics["artefact_event_count"] = int(annotation_counts.get("Artefact", 0))
        metrics["effort_event_count"] = int(annotation_counts.get("Effort", 0))
        metrics["pat_montage_event_count"] = int(annotation_counts.get("PAT Montage", 0))
        metrics["hv_event_count"] = int(annotation_counts.get("HV", 0))
        metrics["photo_event_count"] = int(annotation_counts.get("PHOTO", 0))

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
                if raw_basic is not None:
                    try:
                        fig_var_topo = plot_channel_variance_topomap(raw_basic)
                        safe_onset = onset_sec if np.isfinite(onset_sec) else 0.0
                        fig_raw_segment_start = plot_raw_segment(
                            raw_basic, max(safe_onset, 0.0), title="Raw Segment - Start (10s)"
                        )
                        if np.isfinite(offset_sec):
                            last_start = max(offset_sec - 10.0, 0.0)
                        else:
                            last_start = max(raw_basic.times[-1] - 10.0, 0.0)
                        fig_raw_segment_end = plot_raw_segment(
                            raw_basic, last_start, title="Raw Segment - End (10s)"
                        )
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Raw/variance plotting failed for %s: %s", filepath.name, exc)
                if annotation_counts_dict:
                    try:
                        fig_events = plot_event_counts(annotation_counts_dict)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Event count plotting failed for %s: %s", filepath.name, exc)

            report_path = output_dirs["subject_reports"] / f"{subject_id}_qc_report.html"
            try:
                create_subject_report(
                    raw_basic if raw_basic is not None else raw,
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


def save_figures(df: pd.DataFrame, flags_counter: Counter, fig_dir: Path) -> Dict[str, Path]:
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

    def _event_count_stats(column: str) -> Dict[str, float] | None:
        if column not in df:
            return None
        series = pd.to_numeric(df[column], errors="coerce").fillna(0)
        if series.empty:
            return None
        values = series.to_numpy(dtype=float)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    event_specs = [
        ("Eyes Open", "eyes_open_event_count"),
        ("Eyes Closed", "eyes_closed_event_count"),
        ("Movement", "movement_event_count"),
        ("Artefact", "artefact_event_count"),
        ("PAT Montage", "pat_montage_event_count"),
        ("HV", "hv_event_count"),
        ("PHOTO", "photo_event_count"),
    ]
    event_labels: List[str] = []
    event_means: List[float] = []
    event_stds: List[float] = []
    event_ranges: List[Tuple[float, float]] = []
    for label, count_col in event_specs:
        stats = _event_count_stats(count_col)
        if not stats:
            continue
        event_labels.append(label)
        event_means.append(stats["mean"])
        event_stds.append(stats["std"])
        event_ranges.append((stats["min"], stats["max"]))
    if event_labels:
        height = max(3.5, 0.5 * len(event_labels))
        fig, ax = plt.subplots(figsize=(8, height))
        positions = np.arange(len(event_labels))
        ax.barh(positions, event_means, xerr=event_stds, color="#55A868", alpha=0.85, capsize=5)
        ax.set_xlabel("Average Event Count per Subject")
        ax.set_title("Event Count Averages (with Std)")
        ax.set_yticks(positions)
        ax.set_yticklabels(event_labels)
        ax.invert_yaxis()
        max_mean = max(event_means) if event_means else 0.0
        for idx, (mean_val, (min_val, max_val)) in enumerate(zip(event_means, event_ranges)):
            offset = max_mean * 0.03 if max_mean else 0.1
            ax.text(
                mean_val + offset,
                positions[idx],
                f"min {min_val:.1f} / max {max_val:.1f}",
                va="center",
            )
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        event_path = fig_dir / "event_count_average.png"
        fig.savefig(event_path, dpi=150)
        plt.close(fig)
        paths["event_stats"] = event_path

    return paths


def create_summary_report(
    df: pd.DataFrame,
    fig_paths: Dict[str, Path],
    output_path: Path,
    total_files: int,
    flags_counter: Counter,
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

    report.add_html(summary_html, title="Summary", section="Overview")

    for title, path in [
        ("Duration Distribution", fig_paths.get("duration_min")),
        ("Mean Amplitude Distribution", fig_paths.get("amplitude_mean_uv")),
        ("Alpha Peak Distribution", fig_paths.get("alpha_peak_hz")),
        ("Flag Reasons", fig_paths.get("flag_reasons")),
    ]:
        if path and path.exists():
            report.add_image(path, title=title, section="Figures")

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
    fig_paths = {}
    if not args.skip_figures:
        fig_paths = save_figures(df, flags_counter, fig_dir)
        summary_report_path = output_dir / "qc_summary_report.html"
        create_summary_report(df, fig_paths, summary_report_path, len(files), flags_counter)
        logger.info("Saved summary HTML report to %s", summary_report_path)

    logger.info("QC finished. Total files: %d, flagged bad: %d", len(files), int(df["flag_bad"].sum()))


if __name__ == "__main__":
    main()

"""Core EEG quality control utilities shared by pre- and post-preprocessing QC scripts.

The module keeps the legacy metrics and figures but exposes them as reusable
functions so both CLI wrappers can orchestrate QC workflows without duplicating
logic. Everything here is functional and intentionally simple to stay readable
for large-scale runs.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib
import mne
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from joblib import Parallel, delayed, parallel
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm

from eeg_adhd_epilepsy.utils.qc_config import BAND_LIMITS, BASIC_1020_CHANNELS

from fooof import FOOOF

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# QC threshold constants (tune as needed)
# ---------------------------------------------------------------------------
DATA_RETENTION_GOOD = 0.50
DATA_RETENTION_BORDERLINE = 0.30
BAD_CHANNEL_GOOD = 3
BAD_CHANNEL_BORDERLINE = 6
HF_RATIO_FLAG = 0.50
APERIODIC_SLOPE_MIN = 0.50
APERIODIC_SLOPE_MAX = 3.0
LINE_NOISE_RATIO_FLAG = 5.0
ARTIFACT_Z_OK = 2.0
ARTIFACT_Z_BAD = 3.0
MAX_BAD_CHANNEL_PCT = 30.0

CONDITIONS = ("EO", "EC", "HV", "POST_HV", "PHOTO", "UNKNOWN")
EPS = np.finfo(float).eps

# ---------------------------------------------------------------------------
# Logging / misc helpers
# ---------------------------------------------------------------------------
def setup_logging(log_file: Path | None, level: str) -> logging.Logger:
    """Configure a logger that writes to both file and stdout."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("eeg_qc")


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


# ---------------------------------------------------------------------------
# BIDS / file helpers
# ---------------------------------------------------------------------------
def discover_bids_files(
    bids_root: Path,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    acquisition: str | None = None,
    processing: str | None = None,
    suffix: str = "eeg",
    extension: str = ".vhdr",
    subjects_filter: set[str] | None = None,
) -> List[Path]:
    """Use BIDSPath matching to find EEG files under a BIDS root."""
    template = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        processing=processing,
        datatype="eeg",
        suffix=suffix,
        extension=extension,
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


def read_subjects_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def parse_subject_id(filepath: Path) -> str:
    match = re.search(r"sub-([A-Za-z0-9]+)", filepath.name)
    if match:
        return f"sub-{match.group(1)}"
    return filepath.stem


def load_raw(
    filepath: Path,
    bids_root: Path | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    acquisition: str | None = None,
    processing: str | None = None,
) -> mne.io.BaseRaw:
    """Load a raw BrainVision file."""
    if filepath.suffix.lower() != ".vhdr":
        raise ValueError(f"Unsupported file extension (only .vhdr supported): {filepath.suffix}")
    if bids_root is None:
        return mne.io.read_raw_brainvision(filepath, preload=False)
    subject_clean = parse_subject_id(filepath).replace("sub-", "")
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject_clean,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        processing=processing,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    return read_raw_bids(bids_path)


def load_meas_datetimes(bids_root: Path) -> pd.Series:
    """Return measurement datetimes from participants.tsv if present."""
    tsv_path = bids_root / "participants.tsv"
    if not tsv_path.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_csv(tsv_path, sep="\t")
    if "meas" not in df:
        return pd.Series(dtype="datetime64[ns]")
    meas_series = pd.to_datetime(df["meas"], errors="coerce", utc=True).dropna()
    if meas_series.empty:
        return pd.Series(dtype="datetime64[ns]")
    try:
        meas_series = meas_series.dt.tz_convert(None)
    except TypeError:
        meas_series = meas_series.dt.tz_localize(None)
    return meas_series


# ---------------------------------------------------------------------------
# Channel selection / metadata
# ---------------------------------------------------------------------------
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


def extract_metadata(raw: mne.io.BaseRaw) -> Dict[str, object]:
    """Basic metadata shared by pre- and post-preproc QC."""
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


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_channel_amplitude_stats(raw: mne.io.BaseRaw | None, picks: List[str]) -> Dict[str, object]:
    """Peak-to-peak amplitude per channel."""
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
    """Detect flat/noisy channels using variance percentiles."""
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
    n_channels = len(picks)
    return {
        "flat_channels": [picks[i] for i in flat_idx],
        "noisy_channels": [picks[i] for i in noisy_idx],
        "n_flat_channels": int(len(flat_idx)),
        "n_noisy_channels": int(len(noisy_idx)),
        "pct_bad_channels": float((len(flat_idx) + len(noisy_idx)) / max(n_channels, 1) * 100.0),
        "variances": variances,
    }


def compute_psd_metrics(
    data: mne.io.BaseRaw | mne.Epochs | None,
    picks: List[str],
    fmin: float = 1.0,
    fmax: float = 60.0,
    band_limits: Mapping[str, Tuple[float, float]] | None = None,
) -> Tuple[mne.time_frequency.Spectrum | None, np.ndarray, np.ndarray, float, Dict[str, float]]:
    """PSD summary: spectrum object, PSD array, freqs, alpha peak, band powers."""
    if data is None or not picks:
        empty = np.array([])
        band_limits = band_limits or BAND_LIMITS
        return None, empty, empty, float("nan"), {k: float("nan") for k in band_limits}
    band_limits = band_limits or BAND_LIMITS
    spec = data.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose="ERROR")
    psd, freqs = spec.get_data(return_freqs=True)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    if alpha_mask.any():
        alpha_idx = np.argmax(psd[:, alpha_mask].mean(axis=0))
        alpha_peak = float(freqs[alpha_mask][alpha_idx])
    else:
        alpha_peak = float("nan")

    band_powers: Dict[str, float] = {}
    for band, (low, high) in band_limits.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if band_mask.any():
            band_power = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1).mean()
            band_powers[band] = float(band_power * 1e12)  # convert V^2 to uV^2
        else:
            band_powers[band] = float("nan")

    return spec, psd, freqs, alpha_peak, band_powers


def compute_line_noise_index(
    psd: np.ndarray,
    freqs: np.ndarray,
    line_freq: float = 60.0,
    band_width: float = 1.0,
    neighbor_width: float = 2.0,
) -> Tuple[float, np.ndarray]:
    """Residual line noise ratio comparing the target bin to nearby bins."""
    if psd.size == 0 or freqs.size == 0:
        return float("nan"), np.array([])
    center_mask = (freqs >= line_freq - band_width) & (freqs <= line_freq + band_width)
    neighbor_mask = (
        ((freqs >= line_freq - band_width - neighbor_width) & (freqs < line_freq - band_width))
        | ((freqs > line_freq + band_width) & (freqs <= line_freq + band_width + neighbor_width))
    )
    if not center_mask.any() or not neighbor_mask.any():
        return float("nan"), np.array([])
    center_power = psd[:, center_mask].mean(axis=1)
    neighbor_power = psd[:, neighbor_mask].mean(axis=1) + EPS
    ratios = center_power / neighbor_power
    return float(np.nanmean(ratios)), ratios


def compute_hf_lf_ratio(
    psd: np.ndarray,
    freqs: np.ndarray,
    hf_band: Tuple[float, float] = (30.0, 100.0),
    lf_band: Tuple[float, float] = (1.0, 30.0),
) -> Tuple[float, float]:
    """High-frequency / low-frequency power ratio."""
    if psd.size == 0 or freqs.size == 0:
        return float("nan"), float("nan")
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_power = np.trapezoid(psd[:, hf_mask], freqs[hf_mask], axis=1)
    lf_power = np.trapezoid(psd[:, lf_mask], freqs[lf_mask], axis=1) + EPS
    ratios = hf_power / lf_power
    return float(np.nanmean(ratios)), float(np.nanmax(ratios))


def compute_aperiodic_slope(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 30.0,
) -> Tuple[float, float, float]:
    """Fit 1/f slope using FOOOF (fallback to polyfit if FOOOF unavailable)."""
    if psd.size == 0 or freqs.size == 0:
        return float("nan"), float("nan"), float("nan")
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan"), float("nan"), float("nan")
    slopes: List[float] = []
    intercepts: List[float] = []

    for row in psd:
        try:
            fm = FOOOF(
                peak_width_limits=(1.0, 12.0),
                max_n_peaks=6,
                min_peak_height=0.1,
                verbose=False,
                aperiodic_mode="fixed",
            )
            fm.fit(freqs[mask], row[mask])
            offset, exponent = fm.get_params("aperiodic_params")
            slopes.append(float(exponent))
            intercepts.append(float(offset))
        except Exception:
            slopes.append(float("nan"))
            intercepts.append(float("nan"))

    slopes_arr = np.asarray(slopes)
    intercepts_arr = np.asarray(intercepts)
    return float(np.nanmean(slopes_arr)), float(np.nanstd(slopes_arr)), float(np.nanmean(intercepts_arr))



def compute_epoch_amplitude_stats(
    epochs: mne.Epochs | None,
    picks: List[str] | None = None,
) -> Dict[str, float]:
    """Peak-to-peak amplitude statistics across epochs."""
    if epochs is None:
        return {"mean_ptp_uv": float("nan"), "max_ptp_uv": float("nan")}
    data = epochs.get_data(picks=picks) * 1e6
    ptp = np.ptp(data, axis=2)  # shape (n_epochs, n_channels)
    ptp_epoch = ptp.mean(axis=1)
    return {"mean_ptp_uv": float(np.nanmean(ptp_epoch)), "max_ptp_uv": float(np.nanmax(ptp_epoch))}


# ---------------------------------------------------------------------------
# Segment-level helpers
# ---------------------------------------------------------------------------
def crop_segment(
    raw: mne.io.BaseRaw,
    t_start: float,
    t_stop: float,
    picks: List[str] | None = None,
) -> mne.io.BaseRaw | None:
    """Return a cropped copy of raw between t_start and t_stop (seconds)."""
    if raw is None:
        return None
    start = max(float(t_start), 0.0)
    end = min(float(t_stop), raw.times[-1])
    if end <= start:
        return None
    segment = raw.copy().crop(tmin=start, tmax=end)
    if picks:
        lower_map = {ch.lower(): ch for ch in segment.ch_names}
        pick_names = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]
        if not pick_names:
            return None
        segment = segment.copy().pick(pick_names)
    return segment


def _evaluate_segment_flags(metrics: Mapping[str, object]) -> Tuple[bool, str]:
    reasons: List[str] = []
    hf_ratio = metrics.get("segment_hf_lf_ratio", float("nan"))
    if np.isfinite(hf_ratio) and hf_ratio > HF_RATIO_FLAG:
        reasons.append("high_hf_lf_ratio")
    slope = metrics.get("segment_aperiodic_slope", float("nan"))
    if np.isfinite(slope) and (slope < APERIODIC_SLOPE_MIN or slope > APERIODIC_SLOPE_MAX):
        reasons.append("aperiodic_slope_out_of_range")
    bad_pct = metrics.get("segment_pct_bad_channels", float("nan"))
    if np.isfinite(bad_pct) and bad_pct > MAX_BAD_CHANNEL_PCT:
        reasons.append("too_many_bad_channels")
    return bool(reasons), ";".join(reasons)


def compute_segment_qc(
    raw_segment: mne.io.BaseRaw | None,
    picks: List[str] | None = None,
    logger: logging.Logger | None = None,
    line_freq: float = 60.0,
) -> Dict[str, object]:
    """Compute QC metrics for a single raw segment."""
    if raw_segment is None:
        return {}
    available = raw_segment.ch_names
    if picks:
        lower_map = {ch.lower(): ch for ch in available}
        picks = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]
    else:
        picks = available

    amp_stats = compute_channel_amplitude_stats(raw_segment, picks)
    noise_info = detect_flat_and_noisy_channels(raw_segment, picks)
    duration_sec = float(raw_segment.times[-1]) if raw_segment.times.size else float("nan")

    _, psd, freqs, alpha_peak, band_powers = compute_psd_metrics(
        raw_segment, picks, fmin=1.0, fmax=100.0
    )
    line_noise_mean, _ = compute_line_noise_index(psd, freqs, line_freq=line_freq)
    hf_ratio_mean, _ = compute_hf_lf_ratio(psd, freqs, hf_band=(30.0, 100.0), lf_band=(1.0, 30.0))
    slope_mean, _, _ = compute_aperiodic_slope(psd, freqs, fmin=1.0, fmax=30.0)

    metrics: Dict[str, object] = {
        "segment_duration_sec": duration_sec,
        "segment_n_channels": len(picks),
        "segment_amplitude_mean_uv": amp_stats["mean"],
        "segment_amplitude_median_uv": amp_stats["median"],
        "segment_amplitude_std_uv": amp_stats["std"],
        "segment_amplitude_min_uv": amp_stats["min"],
        "segment_amplitude_max_uv": amp_stats["max"],
        "segment_n_flat_channels": noise_info["n_flat_channels"],
        "segment_n_noisy_channels": noise_info["n_noisy_channels"],
        "segment_pct_bad_channels": noise_info["pct_bad_channels"],
        "segment_band_power_delta": band_powers.get("delta", float("nan")),
        "segment_band_power_theta": band_powers.get("theta", float("nan")),
        "segment_band_power_alpha": band_powers.get("alpha", float("nan")),
        "segment_band_power_beta": band_powers.get("beta", float("nan")),
        "segment_band_power_gamma": band_powers.get("gamma", float("nan")),
        "segment_alpha_peak_hz": alpha_peak,
        "segment_hf_lf_ratio": hf_ratio_mean,
        "segment_line_noise_ratio": line_noise_mean,
        "segment_aperiodic_slope": slope_mean,
    }

    flag_bad, reasons = _evaluate_segment_flags(metrics)
    metrics["segment_flag_bad"] = flag_bad
    metrics["segment_flag_reasons"] = reasons
    return metrics


def aggregate_segment_qc(
    segment_qc_rows: List[Mapping[str, object]], group_cols: List[str] | None = None
) -> pd.DataFrame:
    """Aggregate segment QC metrics per subject/segment_type."""
    if not segment_qc_rows:
        return pd.DataFrame()
    df = pd.DataFrame(segment_qc_rows)
    group_cols = group_cols or ["subject_id", "segment_type"]
    for col in group_cols:
        if col not in df.columns:
            return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return df[group_cols]
    grouped = df.groupby(group_cols, dropna=False)[numeric_cols]
    agg_df = grouped.agg(["mean", "median"])
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    return agg_df.reset_index()


# ---------------------------------------------------------------------------
# Condition-level helpers
# ---------------------------------------------------------------------------
def _label_condition(
    event_name: str | None,
    condition_map: Mapping[str, str] | None = None,
) -> str:
    if event_name and condition_map and event_name in condition_map:
        return condition_map[event_name]
    if not event_name:
        return "UNKNOWN"
    name = event_name.lower()
    if "eyes_open" in name or "eo" in name or "eyes-open" in name or "eyesopen" in name:
        return "EO"
    if "eyes_closed" in name or "ec" in name or "eyes-closed" in name or "eyesclosed" in name:
        return "EC"
    if "hv" in name:
        return "HV"
    if "post" in name and "hv" in name:
        return "POST_HV"
    if "photo" in name or "photic" in name:
        return "PHOTO"
    return "UNKNOWN"


def compute_condition_retention(
    epochs: mne.Epochs,
    condition_map: Mapping[str, str] | None = None,
    condition_key: str = "condition",
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Return retention stats per condition and labels for kept epochs."""
    id_to_name = {v: k for k, v in (epochs.event_id or {}).items()}
    epoch_duration = float(epochs.tmax - epochs.tmin)

    metadata_labels: Dict[int, str] = {}
    if epochs.metadata is not None and condition_key in epochs.metadata.columns:
        for meta_idx, event_idx in enumerate(epochs.selection):
            metadata_labels[int(event_idx)] = str(epochs.metadata.iloc[meta_idx][condition_key])

    stats: Dict[str, Dict[str, float]] = {}
    reasons_per_condition: Dict[str, Counter] = defaultdict(Counter)
    kept_conditions: List[str] = []

    drop_log = epochs.drop_log
    for idx, (log, event) in enumerate(zip(drop_log, epochs.events)):
        event_id = int(event[2])
        event_name = id_to_name.get(event_id)
        cond = metadata_labels.get(idx) or _label_condition(event_name, condition_map)
        cond_stats = stats.setdefault(
            cond,
            {"total": 0, "kept": 0, "rejected": 0, "pct_retained": float("nan"), "usable_minutes": float("nan")},
        )
        cond_stats["total"] += 1
        if len(log) == 0:
            cond_stats["kept"] += 1
            kept_conditions.append(cond)
        else:
            cond_stats["rejected"] += 1
            for reason in log:
                reasons_per_condition[cond][reason] += 1

    for cond, cond_stats in stats.items():
        cond_total = cond_stats["total"]
        cond_stats["pct_retained"] = cond_stats["kept"] / max(cond_total, 1)
        cond_stats["usable_minutes"] = cond_stats["kept"] * epoch_duration / 60.0
        cond_stats["rejection_reasons"] = dict(reasons_per_condition.get(cond, {}))

    return stats, kept_conditions


def compute_condition_amplitude_metrics(
    epochs: mne.Epochs,
    kept_conditions: Sequence[str],
    picks: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Mean/max peak-to-peak amplitude per condition."""
    if epochs is None or not kept_conditions:
        return {}
    data = epochs.get_data(picks=picks) * 1e6
    ptp = np.ptp(data, axis=2).mean(axis=1)  # (n_epochs,)
    per_condition: Dict[str, Dict[str, float]] = {}
    for cond in set(kept_conditions):
        mask = [c == cond for c in kept_conditions]
        if not any(mask):
            continue
        values = ptp[mask]
        per_condition[cond] = {
            "mean_ptp_uv": float(np.nanmean(values)),
            "max_ptp_uv": float(np.nanmax(values)),
        }
    return per_condition


def compute_epoch_rejection_breakdown(epochs: mne.Epochs) -> Dict[str, float]:
    """Percentage of epochs rejected by each criterion (if drop_log is populated)."""
    if epochs is None:
        return {}
    breakdown: Counter = Counter()
    total = len(epochs.drop_log)
    for log in epochs.drop_log:
        for reason in log:
            breakdown[reason] += 1
    return {reason: count / max(total, 1) for reason, count in breakdown.items()}


# ---------------------------------------------------------------------------
# Flagging helpers
# ---------------------------------------------------------------------------
def _flag_retention(pct: float) -> str:
    if not np.isfinite(pct):
        return "unknown"
    if pct >= DATA_RETENTION_GOOD:
        return "good"
    if pct >= DATA_RETENTION_BORDERLINE:
        return "borderline"
    return "unusable"


def _flag_bad_channels(n_bad: float) -> str:
    if not np.isfinite(n_bad):
        return "unknown"
    if n_bad <= BAD_CHANNEL_GOOD:
        return "good"
    if n_bad <= BAD_CHANNEL_BORDERLINE:
        return "borderline"
    return "unusable"


def _flag_artifact_metric(zscore: float) -> str:
    if not np.isfinite(zscore):
        return "unknown"
    if abs(zscore) <= ARTIFACT_Z_OK:
        return "good"
    if abs(zscore) <= ARTIFACT_Z_BAD:
        return "borderline"
    return "unusable"


def evaluate_condition_flags(
    condition_retention: Dict[str, Dict[str, float]],
    condition_amp: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Assign flags per condition using simple thresholds."""
    flags: Dict[str, Dict[str, object]] = {}
    for cond, stats in condition_retention.items():
        reasons: List[str] = []
        status = _flag_retention(stats.get("pct_retained", float("nan")))
        if status == "borderline":
            reasons.append("low_retention")
        elif status == "unusable":
            reasons.append("very_low_retention")
        if condition_amp and cond in condition_amp:
            max_ptp = condition_amp[cond].get("max_ptp_uv", float("nan"))
            if np.isfinite(max_ptp) and max_ptp > 500.0:
                status = "unusable"
                reasons.append("extreme_amplitude")
        flags[cond] = {"flag": status, "flag_reasons": ";".join(reasons)}
    return flags


def evaluate_subject_flag(
    metrics: MutableMapping[str, object],
    dataset_stats: Mapping[str, Dict[str, float]] | None = None,
) -> Tuple[str, List[str]]:
    """Aggregate subject-level usability flag."""
    reasons: List[str] = []
    # Basic duration / amplitude / bad channels
    duration = float(metrics.get("duration_min", float("nan")))
    if np.isfinite(duration) and duration < 5:
        reasons.append("short_duration")
    if np.isfinite(duration) and duration > 60:
        reasons.append("long_duration")
    n_bad = float(metrics.get("n_flat_channels", 0)) + float(metrics.get("n_noisy_channels", 0))
    bad_flag = _flag_bad_channels(n_bad)
    if bad_flag == "borderline":
        reasons.append("many_bad_channels")
    elif bad_flag == "unusable":
        reasons.append("too_many_bad_channels")

    if np.isfinite(metrics.get("amplitude_max_uv", float("nan"))) and metrics["amplitude_max_uv"] > 800:
        reasons.append("amplitude_above_threshold")
    hf_ratio = metrics.get("hf_lf_ratio_mean", float("nan"))
    if np.isfinite(hf_ratio) and hf_ratio > HF_RATIO_FLAG:
        reasons.append("high_hf_ratio")
    slope = metrics.get("aperiodic_slope_mean", float("nan"))
    if np.isfinite(slope) and (slope < APERIODIC_SLOPE_MIN or slope > APERIODIC_SLOPE_MAX):
        reasons.append("extreme_aperiodic_slope")
    line_noise = metrics.get("line_noise_ratio_mean", float("nan"))
    if np.isfinite(line_noise) and line_noise > LINE_NOISE_RATIO_FLAG:
        reasons.append("line_noise_residual")

    if dataset_stats:
        for col, stats in dataset_stats.items():
            value = metrics.get(col)
            if value is None or not np.isfinite(value):
                continue
            std = stats.get("std", 0.0)
            mean = stats.get("mean", 0.0)
            if std > 0 and abs(value - mean) > 3 * std:
                reasons.append(f"{col}_outlier")

    condition_flags = metrics.get("condition_flags")
    if isinstance(condition_flags, dict):
        worst = "good"
        order = {"good": 0, "borderline": 1, "unusable": 2, "unknown": 1}
        for cond_info in condition_flags.values():
            flag = cond_info.get("flag", "good")
            if order.get(flag, 0) > order.get(worst, 0):
                worst = flag
        if worst == "borderline":
            reasons.append("borderline_condition")
        elif worst == "unusable":
            reasons.append("unusable_condition")

    if not reasons:
        return "usable", []
    if any(reason in {"unusable_condition", "too_many_bad_channels"} for reason in reasons):
        return "unusable", reasons
    if len(reasons) >= 2:
        return "borderline", reasons
    return "borderline", reasons


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
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


def plot_psd_overlay(
    before_freqs: np.ndarray,
    before_psd: np.ndarray,
    after_freqs: np.ndarray,
    after_psd: np.ndarray,
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


SEGMENT_REPORT_METRICS = (
    {"column": "segment_duration_sec", "title": "Segment Duration", "ylabel": "Duration (s)", "kind": "bar"},
    {"column": "segment_amplitude_mean_uv", "title": "Mean Amplitude", "ylabel": "Mean amplitude (uV)", "kind": "line"},
    {"column": "segment_pct_bad_channels", "title": "Percent Bad Channels", "ylabel": "Bad channels (%)", "kind": "line"},
    {"column": "segment_line_noise_ratio", "title": "Line Noise Ratio", "ylabel": "Line-noise ratio", "kind": "line"},
    {"column": "segment_hf_lf_ratio", "title": "HF/LF Ratio", "ylabel": "HF/LF ratio", "kind": "line"},
    {"column": "segment_aperiodic_slope", "title": "Aperiodic Slope", "ylabel": "Aperiodic slope", "kind": "line"},
    {"column": "segment_band_power_alpha", "title": "Alpha Band Power", "ylabel": "Alpha power (uV^2)", "kind": "line"},
)

SUBJECT_SEGMENT_REPORT_METRICS = tuple(
    spec for spec in SEGMENT_REPORT_METRICS if spec["column"] != "segment_duration_sec"
)


def _segment_palette(segment_types: Sequence[str]) -> Dict[str, object]:
    cmap = plt.get_cmap("tab20")
    return {seg: cmap(idx % cmap.N) for idx, seg in enumerate(sorted(segment_types))}


def format_seconds_hms(seconds: float | None) -> str:
    """Return a human-readable H:M:S string for a seconds value."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "0s"
    if not math.isfinite(value):
        return "0s"
    value = max(0.0, value)
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    value -= minutes * 60
    seconds_str = f"{value:.2f}".rstrip("0").rstrip(".")
    if not seconds_str:
        seconds_str = "0"
    sec_component = f"{seconds_str}s"
    if hours > 0:
        return f"{hours}h {minutes}m {sec_component}"
    if minutes > 0:
        return f"{minutes}m {sec_component}"
    return sec_component


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
    df["flag_bad_bool"] = pd.to_numeric(df.get("segment_flag_bad"), errors="coerce").fillna(0).astype(bool)
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


def create_segment_subject_report(
    segments_df: pd.DataFrame,
    subject_id: str,
    output_path: Path,
    metric_specs: Sequence[Mapping[str, str]] = SUBJECT_SEGMENT_REPORT_METRICS,
) -> None:
    """Subject-level HTML report for segment QC metrics."""
    if segments_df is None or segments_df.empty:
        return
    report = mne.Report(title=f"Segment QC Report - {subject_id}")
    durations = pd.to_numeric(segments_df.get("duration"), errors="coerce").dropna()
    total_duration = float(durations.sum()) if not durations.empty else float("nan")
    total_duration_readable = format_seconds_hms(total_duration)
    total_duration_readable = format_seconds_hms(total_duration)
    flagged = segments_df.get("segment_flag_bad")
    flagged_count = int(pd.to_numeric(flagged, errors="coerce").fillna(0).astype(bool).sum()) if flagged is not None else 0
    type_counts = (
        segments_df.get("segment_type", pd.Series(dtype=str)).fillna("Unknown").value_counts().to_dict()
        if not segments_df.empty
        else {}
    )
    summary_html = "<ul>"
    summary_html += f"<li>Segments: {len(segments_df)}</li>"
    summary_html += f"<li>Total duration: {total_duration:.1f} s ({total_duration_readable})</li>"
    summary_html += f"<li>Flagged segments: {flagged_count}</li>"
    if type_counts:
        summary_html += "<li>Segment types:<ul>"
        for seg_type, count in type_counts.items():
            summary_html += f"<li>{seg_type}: {count}</li>"
        summary_html += "</ul></li>"
    summary_html += "</ul>"
    report.add_html(summary_html, title="Segment Summary", section="Overview")

    for spec in metric_specs:
        block_fig = plot_segment_metric_blocks(
            segments_df,
            column=spec["column"],
            title=f"{spec['title']} (Mean per Segment Type)",
            ylabel=spec["ylabel"],
        )
        if block_fig is not None:
            report.add_figure(block_fig, title=f"{spec['title']} Blocks", section="Segment Metrics")

    flag_mask = pd.Series(False, index=segments_df.index)
    flag_col = segments_df.get("segment_flag_bad")
    if flag_col is not None:
        flag_mask = pd.to_numeric(flag_col, errors="coerce").fillna(0).astype(bool)
    flagged_table = segments_df[flag_mask]
    if not flagged_table.empty:
        preview_cols = [
            "segment_type",
            "t_start",
            "t_stop",
            "duration",
            "segment_hf_lf_ratio",
            "segment_line_noise_ratio",
            "segment_aperiodic_slope",
            "segment_flag_reasons",
        ]
        available_cols = [col for col in preview_cols if col in flagged_table.columns]
        html_rows = flagged_table[available_cols].head(25).to_html(index=False)
        report.add_html(html_rows, title="Flagged Segment Details (first 25)", section="Quality Flags")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


def _compute_flagged_percentages_by_segment(
    segments_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (% flagged segments, % flagged subjects, subject counts, flagged subject counts) per segment_type."""
    if segments_df is None or segments_df.empty or "segment_type" not in segments_df:
        empty = pd.Series(dtype=float)
        empty_int = pd.Series(dtype=int)
        return empty, empty, empty_int, empty_int
    df = segments_df.copy()
    df["segment_type"] = df["segment_type"].fillna("Unknown").astype(str)
    df["flag_bad_bool"] = pd.to_numeric(df.get("segment_flag_bad"), errors="coerce").fillna(0).astype(bool)

    seg_pct = df.groupby("segment_type")["flag_bad_bool"].mean() * 100.0
    subj_pct = pd.Series(dtype=float)
    subject_counts = pd.Series(dtype=int)
    flagged_subject_counts = pd.Series(dtype=int)
    if "subject_id" in df:
        subject_counts = df.groupby("segment_type")["subject_id"].nunique()
        flagged_subject_counts = (
            df[df["flag_bad_bool"]]
            .groupby("segment_type")["subject_id"]
            .nunique()
        )
        subj_pct = (flagged_subject_counts / subject_counts.replace(0, np.nan) * 100.0).fillna(0.0)

    subject_counts = subject_counts.sort_index()
    flagged_subject_counts = flagged_subject_counts.reindex(subject_counts.index).fillna(0).astype(int)
    return seg_pct.sort_index(), subj_pct.sort_index(), subject_counts, flagged_subject_counts


def _plot_flagged_percentages(
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


def _plot_flagged_subject_distribution(
    segments_df: pd.DataFrame,
    fig_dir: Path,
) -> Path | None:
    """Plot distribution of subject counts by number of flagged segments per segment type."""
    if segments_df is None or segments_df.empty or "segment_type" not in segments_df or "subject_id" not in segments_df:
        return None
    df = segments_df.copy()
    df["segment_type"] = df["segment_type"].fillna("Unknown").astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["flag_bad_bool"] = pd.to_numeric(df.get("segment_flag_bad"), errors="coerce").fillna(0).astype(bool)
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


def save_segment_dataset_figures(
    segments_df: pd.DataFrame, fig_dir: Path, metric_specs: Sequence[Mapping[str, str]] = SEGMENT_REPORT_METRICS
) -> Dict[str, Path]:
    """Save dataset-level histograms for segment metrics."""
    paths: Dict[str, Path] = {}
    for spec in metric_specs:
        path = plot_segment_metric_distribution_by_type(
            segments_df,
            column=spec["column"],
            title=spec["title"],
            xlabel=spec["ylabel"],
            fig_dir=fig_dir,
        )
        if path:
            paths[spec["column"]] = path
    flagged_seg_pct, flagged_subj_pct, _subj_counts, _flagged_subj_counts = _compute_flagged_percentages_by_segment(segments_df)
    flagged_segments_path = _plot_flagged_percentages(
        flagged_seg_pct,
        title="Flagged Segments by Type (%)",
        xlabel="Flagged segments (%)",
        fig_dir=fig_dir,
        filename="flagged_segments_pct.png",
    )
    if flagged_segments_path:
        paths["flagged_segments_pct"] = flagged_segments_path
    flagged_subjects_path = _plot_flagged_percentages(
        flagged_subj_pct,
        title="Flagged Subjects by Type (%)",
        xlabel="Flagged subjects (%)",
        fig_dir=fig_dir,
        filename="flagged_subjects_pct.png",
    )
    if flagged_subjects_path:
        paths["flagged_subjects_pct"] = flagged_subjects_path
    flagged_subject_dist_path = _plot_flagged_subject_distribution(segments_df, fig_dir)
    if flagged_subject_dist_path:
        paths["flagged_subjects_distribution"] = flagged_subject_dist_path
    return paths


def create_segment_dataset_report(
    segments_df: pd.DataFrame,
    fig_paths: Mapping[str, Path],
    output_path: Path,
) -> None:
    """Dataset-level HTML report for segment QC."""
    if segments_df is None or segments_df.empty:
        return
    report = mne.Report(title="Segment QC Dataset Summary")
    subject_count = int(segments_df.get("subject_id", pd.Series(dtype=str)).nunique())
    flagged = segments_df.get("segment_flag_bad")
    flagged_count = int(pd.to_numeric(flagged, errors="coerce").fillna(0).astype(bool).sum()) if flagged is not None else 0
    durations = pd.to_numeric(segments_df.get("duration"), errors="coerce").dropna()
    total_duration = float(durations.sum()) if not durations.empty else float("nan")
    total_duration_readable = format_seconds_hms(total_duration)
    type_counts = (
        segments_df.get("segment_type", pd.Series(dtype=str)).fillna("Unknown").value_counts().to_dict()
        if not segments_df.empty
        else {}
    )
    summary_html = f"""
    <h3>Dataset Summary</h3>
    <ul>
        <li>Subjects: {subject_count}</li>
        <li>Total segments: {len(segments_df)}</li>
        <li>Total duration: {total_duration:.1f} s ({total_duration_readable})</li>
        <li>Flagged segments: {flagged_count}</li>
    </ul>
    """
    if type_counts:
        summary_html += "<p>Segment type counts:</p><ul>"
        for seg_type, count in type_counts.items():
            summary_html += f"<li>{seg_type}: {count}</li>"
        summary_html += "</ul>"

    flagged_seg_pct, flagged_subj_pct, subject_counts, flagged_subject_counts = _compute_flagged_percentages_by_segment(segments_df)
    if not flagged_seg_pct.empty:
        summary_html += "<p>Flagged segments (% of segments) by type:</p><ul>"
        for seg_type, pct in flagged_seg_pct.sort_values(ascending=False).items():
            summary_html += f"<li>{seg_type}: {pct:.1f}%</li>"
        summary_html += "</ul>"
    if not flagged_subj_pct.empty:
        summary_html += "<p>Flagged subjects (% of subjects with that segment type):</p><ul>"
        for seg_type, pct in flagged_subj_pct.sort_values(ascending=False).items():
            total = int(subject_counts.get(seg_type, 0))
            flagged_n = int(flagged_subject_counts.get(seg_type, 0))
            summary_html += f"<li>{seg_type}: {pct:.1f}% ({flagged_n}/{total} subjects)</li>"
        summary_html += "</ul>"
    report.add_html(summary_html, title="Summary", section="Overview")

    for spec in SEGMENT_REPORT_METRICS:
        path = fig_paths.get(spec["column"])
        if path and path.exists():
            report.add_image(path, title=f"{spec['title']} Distribution by Type", section="Metric Distributions")

    for title, key in [
        ("Flagged Segments by Type (%)", "flagged_segments_pct"),
        ("Flagged Subjects by Type (%)", "flagged_subjects_pct"),
        ("Flagged Segments per Subject Distribution", "flagged_subjects_distribution"),
    ]:
        fig_path = fig_paths.get(key)
        if fig_path and fig_path.exists():
            report.add_image(fig_path, title=title, section="Flagged Rates")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def create_subject_report(
    raw: mne.io.BaseRaw | mne.Epochs,
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
    fig_psd_overlay_before_after: matplotlib.figure.Figure | None = None,
) -> None:
    """Reusable subject HTML report."""
    report = mne.Report(title=f"EEG QC Report - {subject_id}")
    try:
        report.add_raw(raw, title="Data (with PSD)", psd=True)
    except Exception:
        pass

    if fig_psd_all is not None:
        report.add_figure(fig_psd_all, title="PSD - All Channels", section="Power Spectral Density")
    if fig_psd_avg is not None:
        report.add_figure(fig_psd_avg, title="PSD - Average", section="Power Spectral Density")
    if fig_psd_overlay_before_after is not None:
        report.add_figure(
            fig_psd_overlay_before_after,
            title="PSD Overlay (Before vs After)",
            section="Power Spectral Density",
        )
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
    line_noise_ratio = metrics.get("line_noise_ratio_mean", float("nan"))
    hf_ratio = metrics.get("hf_lf_ratio_mean", float("nan"))
    slope = metrics.get("aperiodic_slope_mean", float("nan"))

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
    qc_summary_html += f"<li>Line-noise ratio: {line_noise_ratio:.2f}</li>"
    qc_summary_html += f"<li>HF/LF ratio: {hf_ratio:.2f}</li>"
    qc_summary_html += f"<li>Aperiodic slope: {slope:.2f}</li>"
    if metrics.get("condition_flags"):
        qc_summary_html += f"<li>Condition flags: {metrics['condition_flags']}</li>"
    if metrics.get("flag_reasons"):
        qc_summary_html += f"<li>Flag reasons: {metrics.get('flag_reasons')}</li>"
    if metrics.get("event_counts"):
        qc_summary_html += f"<li>Events: {metrics.get('event_counts')}</li>"
    qc_summary_html += "</ul>"
    report.add_html(qc_summary_html, title="QC Summary", section="Quality Control")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------
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
        "hf_lf_ratio_mean",
        "line_noise_ratio_mean",
        "aperiodic_slope_mean",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)
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
            if value is None or not np.isfinite(value):
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


def save_figures(
    df: pd.DataFrame,
    flags_counter: Counter,
    fig_dir: Path,
    meas_datetimes: pd.Series | None = None,
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def _save_hist(column: str, title: str, filename: str):
        if column not in df:
            return
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
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
    _save_hist("hf_lf_ratio_mean", "HF/LF Ratio Distribution", "dataset_hf_ratio_distribution.png")
    _save_hist("aperiodic_slope_mean", "Aperiodic Slope Distribution", "dataset_slope_distribution.png")
    _save_hist("line_noise_ratio_mean", "Line Noise Ratio Distribution", "dataset_line_noise_distribution.png")

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
        if label not in {"Eyes Open", "Eyes Closed", "HV", "PHOTO"}:
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

    report.add_html(summary_html, title="Summary", section="Overview")

    for title, path in [
        ("Duration Distribution", fig_paths.get("duration_min")),
        ("Mean Amplitude Distribution", fig_paths.get("amplitude_mean_uv")),
        ("Alpha Peak Distribution", fig_paths.get("alpha_peak_hz")),
        ("HF Ratio Distribution", fig_paths.get("hf_lf_ratio_mean")),
        ("Aperiodic Slope Distribution", fig_paths.get("aperiodic_slope_mean")),
        ("Line Noise Distribution", fig_paths.get("line_noise_ratio_mean")),
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


__all__ = [
    "apply_dataset_outlier_flags",
    "aggregate_segment_qc",
    "BAND_LIMITS",
    "BASIC_1020_CHANNELS",
    "collect_unknown_events",
    "compute_segment_qc",
    "compute_aperiodic_slope",
    "compute_channel_amplitude_stats",
    "compute_condition_amplitude_metrics",
    "compute_condition_retention",
    "compute_dataset_stats",
    "compute_epoch_amplitude_stats",
    "compute_epoch_rejection_breakdown",
    "compute_hf_lf_ratio",
    "compute_line_noise_index",
    "compute_psd_metrics",
    "create_segment_dataset_report",
    "create_segment_subject_report",
    "create_subject_report",
    "create_summary_report",
    "detect_flat_and_noisy_channels",
    "discover_bids_files",
    "evaluate_condition_flags",
    "evaluate_subject_flag",
    "extract_metadata",
    "parse_subject_id",
    "crop_segment",
    "load_meas_datetimes",
    "load_raw",
    "plot_amplitude_histogram",
    "plot_channel_variance_topomap",
    "plot_events_distribution",
    "plot_psd_figures",
    "plot_psd_overlay",
    "plot_raw_segment",
    "plot_segment_metric_blocks",
    "plot_segment_metric_distribution_by_type",
    "format_seconds_hms",
    "prepare_channel_selection",
    "read_subjects_list",
    "save_segment_dataset_figures",
    "save_figures",
    "SEGMENT_REPORT_METRICS",
    "SUBJECT_SEGMENT_REPORT_METRICS",
    "setup_logging",
    "summarize_flags",
    "tqdm_joblib",
]

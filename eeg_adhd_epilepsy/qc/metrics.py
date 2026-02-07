"""Segment-level quality control orchestration."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Tuple, Optional

import numpy as np
import mne

from eeg_adhd_epilepsy.features.time import (
    compute_channel_amplitude_stats,
    detect_flat_and_noisy_channels
)
from eeg_adhd_epilepsy.features.spectral import (
    compute_spectral_metrics,
    compute_line_noise_index,
    compute_hf_lf_ratio,
    compute_aperiodic_slope,
    compute_lsd
)

LOGGER = logging.getLogger("qc_metrics")


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
    if picks is not None and len(picks) > 0 and hasattr(picks[0], "lower"):
        lower_map = {ch.lower(): ch for ch in segment.ch_names}
        picks = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]
        if not picks:
            return None
        segment = segment.copy().pick(picks)
    elif picks is not None and len(picks) > 0:
         segment = segment.copy().pick(picks)
    return segment


def _evaluate_segment_flags(metrics: Mapping[str, object]) -> Tuple[bool, str]:
    reasons: List[str] = []
    HF_RATIO_FLAG = 0.50
    APERIODIC_SLOPE_MIN = 0.50
    APERIODIC_SLOPE_MAX = 3.0
    MAX_BAD_CHANNEL_PCT = 30.0
    
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
    line_freq: float = 60.0,
    include_channel_metrics: bool = False,
) -> Dict[str, object]:
    """Compute QC metrics for a single raw segment."""
    if raw_segment is None:
        return {}
    available = raw_segment.ch_names
    if picks is None or len(picks) == 0:
        picks = available
    elif hasattr(picks[0], "lower"):
        lower_map = {ch.lower(): ch for ch in available}
        picks = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]

    amp_stats = compute_channel_amplitude_stats(raw_segment, picks)
    noise_info = detect_flat_and_noisy_channels(raw_segment, picks)
    duration_sec = float(raw_segment.times[-1]) if raw_segment.times.size else float("nan")

    # Compute spectral metrics in one go
    spec, psd, freqs, alpha_peak, band_powers, band_powers_per_channel = compute_spectral_metrics(
        raw_segment, picks, fmin=0.5, fmax=99.5
    )
    
    line_noise_mean, line_noise_ratios = compute_line_noise_index(psd, freqs, line_freq=line_freq)
    hf_ratio_mean, _hf_ratio_max, hf_ratios = compute_hf_lf_ratio(psd, freqs, hf_band=(30.0, 100.0), lf_band=(1.0, 30.0))
    slope_mean, _, _, slope_per_channel = compute_aperiodic_slope(psd, freqs, fmin=1.0, fmax=30.0)

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

    if include_channel_metrics:
        channel_metrics: Dict[str, np.ndarray] = {
            "amplitude_ptp_uv": amp_stats["per_channel"],
            "variance": noise_info.get("variances", np.array([])),
            "line_noise_ratio": line_noise_ratios,
            "hf_lf_ratio": hf_ratios,
            "aperiodic_slope": slope_per_channel,
        }
        for band, values in band_powers_per_channel.items():
            channel_metrics[f"band_power_{band}"] = values
        metrics["per_channel_metrics"] = channel_metrics

    flag_bad, reasons = _evaluate_segment_flags(metrics)
    metrics["segment_flag_bad"] = flag_bad
    metrics["segment_flag_reasons"] = reasons
    return metrics


def compute_spectral_fidelity(
    raw_clean: mne.io.BaseRaw, 
    raw_orig: mne.io.BaseRaw, 
    bands: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    Compute Log-Spectral Distance (LSD) per band between clean and original data.
    
    Args:
        raw_clean: Preprocessed Raw object.
        raw_orig: Original Raw object.
        bands: Dictionary of frequency bands {name: (fmin, fmax)}. 
               If None, defaults to standard bands + Line(58-62).
               
    Returns:
        Dictionary mapping "LSD_{band}" to the float score.
    """    
    metrics: Dict[str, float] = {}
    
    # Compute PSDs (Welch) - quick check on up to 60s for speed
    tmax = min(raw_clean.times[-1], 60.0) 
    
    # Compute PSD for both (returns Spectrum)
    spec_clean = raw_clean.compute_psd(tmax=tmax, fmax=120, n_jobs=1, verbose="ERROR")
    spec_orig = raw_orig.compute_psd(tmax=tmax, fmax=120, n_jobs=1, verbose="ERROR")
    
    psd_clean, freqs = spec_clean.get_data(return_freqs=True)
    psd_orig, _ = spec_orig.get_data(return_freqs=True)
    
    # Average over channels
    avg_clean = np.mean(psd_clean, axis=0)
    avg_orig = np.mean(psd_orig, axis=0)

    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        if np.any(idx):
            lsd_val = compute_lsd(avg_clean[idx], avg_orig[idx])
            metrics[f"LSD_{band}"] = float(lsd_val)
            
    return metrics

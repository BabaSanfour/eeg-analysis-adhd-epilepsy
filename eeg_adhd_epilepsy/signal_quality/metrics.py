"""Signal quality control metrics."""

from __future__ import annotations

import logging

import mne
import numpy as np

from eeg_adhd_epilepsy.signal_quality.spectral import (
    compute_aperiodic_slope,
    compute_hf_lf_ratio,
    compute_line_noise_index,
    compute_lsd,
    compute_spectral_metrics,
)
from eeg_adhd_epilepsy.signal_quality.time import (
    compute_channel_amplitude_stats,
    detect_flat_and_noisy_channels,
)

LOGGER = logging.getLogger("qc_metrics")


def crop_segment(
    raw: mne.io.BaseRaw,
    t_start: float,
    t_stop: float,
    picks: list[str] | None = None,
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


def compute_signal_qc_metrics(
    signal: mne.io.BaseRaw | None,
    picks: list[str] | None = None,
    line_freq: float = 60.0,
    include_channel_metrics: bool = False,
) -> dict[str, object]:
    """Compute broad signal QC metrics for a raw recording or segment."""
    if signal is None:
        return {}
    available = signal.ch_names
    if picks is None or len(picks) == 0:
        picks = available
    elif hasattr(picks[0], "lower"):
        lower_map = {ch.lower(): ch for ch in available}
        picks = [lower_map[p.lower()] for p in picks if p.lower() in lower_map]

    amp_stats = compute_channel_amplitude_stats(signal, picks)
    noise_info = detect_flat_and_noisy_channels(signal, picks)
    duration_sec = float(signal.times[-1]) if signal.times.size else float("nan")

    psd, freqs, alpha_peak, _band_powers = compute_spectral_metrics(
        signal, picks, fmin=0.5, fmax=99.5
    )

    line_noise_mean, line_noise_ratios = compute_line_noise_index(psd, freqs, line_freq=line_freq)
    hf_ratio_mean, _hf_ratio_max, hf_ratios = compute_hf_lf_ratio(
        psd, freqs, hf_band=(30.0, 100.0), lf_band=(1.0, 30.0)
    )
    slope_mean, _, _, slope_per_channel = compute_aperiodic_slope(psd, freqs, fmin=1.0, fmax=30.0)

    metrics: dict[str, object] = {
        "duration_sec": duration_sec,
        "n_channels": len(picks),
        "amplitude_mean_uv": amp_stats["mean"],
        "amplitude_median_uv": amp_stats["median"],
        "amplitude_std_uv": amp_stats["std"],
        "amplitude_min_uv": amp_stats["min"],
        "amplitude_max_uv": amp_stats["max"],
        "flat_channels": noise_info["flat_channels"],
        "noisy_channels": noise_info["noisy_channels"],
        "n_flat_channels": noise_info["n_flat_channels"],
        "n_noisy_channels": noise_info["n_noisy_channels"],
        "pct_bad_channels": noise_info["pct_bad_channels"],
        "alpha_peak_hz": alpha_peak,
        "hf_lf_ratio": hf_ratio_mean,
        "line_noise_ratio": line_noise_mean,
        "aperiodic_slope": slope_mean,
    }

    if include_channel_metrics:
        channel_metrics: dict[str, np.ndarray] = {
            "amplitude_ptp_uv": amp_stats["per_channel"],
            "line_noise_ratio": line_noise_ratios,
            "hf_lf_ratio": hf_ratios,
            "aperiodic_slope": slope_per_channel,
        }
        metrics["per_channel_metrics"] = channel_metrics

    reasons: list[str] = []
    hf_ratio = metrics.get("hf_lf_ratio", float("nan"))
    if np.isfinite(hf_ratio) and hf_ratio > 0.50:
        reasons.append("high_hf_lf_ratio")
    slope = metrics.get("aperiodic_slope", float("nan"))
    if np.isfinite(slope) and (slope < 0.50 or slope > 3.0):
        reasons.append("aperiodic_slope_out_of_range")
    bad_pct = metrics.get("pct_bad_channels", float("nan"))
    if np.isfinite(bad_pct) and bad_pct > 30.0:
        reasons.append("too_many_bad_channels")
    metrics["flag_bad"] = bool(reasons)
    metrics["flag_reasons"] = ";".join(reasons)
    return metrics


def compute_spectral_fidelity(
    raw_clean: mne.io.BaseRaw, raw_orig: mne.io.BaseRaw, bands: dict[str, tuple[float, float]]
) -> dict[str, float]:
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
    metrics: dict[str, float] = {}

    # Compute PSDs (Welch) - quick check on up to 60s for speed
    tmax = min(raw_clean.times[-1], 60.0)

    # Compute PSD-derived summaries for both signals
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

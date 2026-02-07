"""Spectral QC metrics (PSD, band powers, line noise, 1/f slope)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import mne
from mne.time_frequency import Spectrum

from fooof import FOOOF, FOOOFGroup

from eeg_adhd_epilepsy.utils.config import BAND_LIMITS

EPS = np.finfo(float).eps


def _integrate_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    band_limits: Mapping[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """Calculate band powers using trapezoidal integration (vectorized)."""
    powers: Dict[str, np.ndarray] = {}
    for band, (low, high) in band_limits.items():
        mask = (freqs >= low) & (freqs <= high)
        if not mask.any():
            powers[band] = np.full(psd.shape[0], np.nan)
            continue
        band_power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        powers[band] = band_power * 1e12  # convert V^2 to uV^2
    return powers


def compute_spectral_metrics(
    data: mne.io.BaseRaw | mne.Epochs | None,
    picks: List[str] | None,
    fmin: float = 1.0,
    fmax: float = 60.0,
    band_limits: Mapping[str, Tuple[float, float]] | None = None,
) -> Tuple[Spectrum | None, np.ndarray, np.ndarray, float, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Compute comprehensive spectral metrics.
    
    Returns:
        spec: MNE Spectrum object
        psd: PSD array (n_channels, n_freqs)
        freqs: Frequency array
        alpha_peak: Alpha peak frequency (Hz)
        band_powers_mean: Dictionary of mean band powers (across channels)
        band_powers_per_channel: Dictionary of band powers per channel
    """
    band_limits = band_limits or BAND_LIMITS
    if data is None:
        empty = np.array([])
        return None, empty, empty, float("nan"), {k: float("nan") for k in band_limits}, {k: empty for k in band_limits}

    # Auto-pick EEG channels if picks is None or empty
    if picks is None or len(picks) == 0:
        picks = mne.pick_types(data.info, eeg=True, meg=False, exclude='bads')
        if len(picks) == 0:
            empty = np.array([])
            return None, empty, empty, float("nan"), {k: float("nan") for k in band_limits}, {k: empty for k in band_limits}

    spec = data.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose="ERROR")
    psd, freqs = spec.get_data(return_freqs=True)
    
    # Calculate alpha peak using Alpha band limits (default or overrides)
    alpha_band = band_limits.get("alpha", (8.0, 13.0))
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    
    if alpha_mask.any():
        # Alpha peak from mean PSD across channels
        mean_psd_alpha = np.nanmean(psd[:, alpha_mask], axis=0)
        alpha_idx = np.argmax(mean_psd_alpha)
        alpha_peak = float(freqs[alpha_mask][alpha_idx])
    else:
        alpha_peak = float("nan")

    per_channel_powers = _integrate_power(psd, freqs, band_limits)
    band_powers_mean: Dict[str, float] = {
        band: float(np.nanmean(values)) for band, values in per_channel_powers.items()
    }

    return spec, psd, freqs, alpha_peak, band_powers_mean, per_channel_powers

def get_spectral_metrics_per_channel(
    psd: np.ndarray,
    freqs: np.ndarray,
    band_limits: Mapping[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    """Return band power per channel for each band (in uV^2)."""
    if psd.size == 0 or freqs.size == 0:
        band_limits = band_limits or BAND_LIMITS
        return {band: np.array([]) for band in band_limits}
    
    return _integrate_power(psd, freqs, band_limits or BAND_LIMITS)


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
    center_power = np.nanmean(psd[:, center_mask], axis=1)
    neighbor_power = np.nanmean(psd[:, neighbor_mask], axis=1) + EPS
    ratios = center_power / neighbor_power
    return float(np.nanmean(ratios)), ratios


def compute_hf_lf_ratio(
    psd: np.ndarray,
    freqs: np.ndarray,
    hf_band: Tuple[float, float] = (30.0, 100.0),
    lf_band: Tuple[float, float] = (1.0, 30.0),
) -> Tuple[float, float, np.ndarray]:
    """High-frequency / low-frequency power ratio."""
    if psd.size == 0 or freqs.size == 0:
        return float("nan"), float("nan"), np.array([])
    
    bands = {"hf": hf_band, "lf": lf_band}
    powers = _integrate_power(psd, freqs, bands)
    
    hf_power = powers["hf"]
    lf_power = powers["lf"] + EPS
    
    ratios = hf_power / lf_power 
    return float(np.nanmean(ratios)), float(np.nanmax(ratios)), ratios


def compute_aperiodic_slope(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 30.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Fit 1/f slope using FOOOF."""
    if psd.size == 0 or freqs.size == 0:
        return float("nan"), float("nan"), float("nan"), np.array([])
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan"), float("nan"), float("nan"), np.array([])
    
    try:
        fg = FOOOFGroup(
            peak_width_limits=(1.0, 12.0),
            max_n_peaks=6,
            min_peak_height=0.1,
            verbose=False,
            aperiodic_mode="fixed",
        )
        # Let's simple check:
        valid_ch = ~np.isnan(psd[:, mask]).any(axis=1)

        fg.fit(freqs[mask], psd[valid_ch][:, mask], n_jobs=1)
        
        res = fg.get_params("aperiodic_params")
        intercepts_valid = res[:, 0]
        slopes_valid = res[:, 1]
        
        # Map back to full array
        n_ch = psd.shape[0]
        slopes_arr = np.full(n_ch, np.nan)
        intercepts_arr = np.full(n_ch, np.nan)
        
        slopes_arr[valid_ch] = slopes_valid
        intercepts_arr[valid_ch] = intercepts_valid
        
    except Exception:
        # Fallback to empty/nan if group fit fails completely
        n_ch = psd.shape[0]
        slopes_arr = np.full(n_ch, np.nan)
        intercepts_arr = np.full(n_ch, np.nan)

    return (
        float(np.nanmean(slopes_arr)),
        float(np.nanstd(slopes_arr)),
        float(np.nanmean(intercepts_arr)),
        slopes_arr,
    )


def compute_lsd(psd_clean: np.ndarray, psd_raw: np.ndarray, eps: float = 1e-20) -> float:
    """
    Compute Log-Spectral Distance (LSD) between two PSDs in dB.
    LSD = sqrt( mean( (10*log10(P_clean) - 10*log10(P_raw))^2 ) )
    """
    if psd_clean.shape != psd_raw.shape:
        return float("nan")
        
    log_clean = 10 * np.log10(psd_clean + eps)
    log_raw = 10 * np.log10(psd_raw + eps)
    
    diff_sq = (log_clean - log_raw) ** 2
    return float(np.sqrt(np.nanmean(diff_sq)))

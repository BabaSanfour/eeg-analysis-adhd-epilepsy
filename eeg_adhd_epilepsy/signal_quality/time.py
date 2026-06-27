"""Time-domain QC metrics."""

from __future__ import annotations

import mne
import numpy as np


def compute_channel_amplitude_stats(
    raw: mne.io.BaseRaw | None, picks: list[str], units: str = "uV"
) -> dict[str, object]:
    """Peak-to-peak amplitude per channel."""
    if raw is None or picks is None or len(picks) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "per_channel": np.array([]),
        }
    data = raw.get_data(picks=picks, units=units)
    ptp = np.ptp(data, axis=1)
    return {
        "mean": float(ptp.mean()),
        "median": float(np.median(ptp)),
        "std": float(ptp.std()),
        "min": float(ptp.min()),
        "max": float(ptp.max()),
        "per_channel": ptp,
    }


def detect_flat_and_noisy_channels(
    raw: mne.io.BaseRaw | None, picks: list[str], units: str = "uV"
) -> dict[str, object]:
    """Detect flat/noisy channels using variance percentiles."""
    if raw is None or picks is None or len(picks) == 0:
        return {
            "flat_channels": [],
            "noisy_channels": [],
            "n_flat_channels": 0,
            "n_noisy_channels": 0,
            "pct_bad_channels": float("nan"),
            "variances": np.array([]),
        }
    data = raw.get_data(picks=picks, units=units)
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


def compute_epoch_amplitude_stats(
    epochs: mne.Epochs | None,
    picks: list[str] | None = None,
    units: str = "uV",
) -> dict[str, float]:
    """Peak-to-peak amplitude statistics across epochs."""
    if epochs is None:
        return {"mean_ptp_uv": float("nan"), "max_ptp_uv": float("nan")}
    data = epochs.get_data(picks=picks, units=units)
    ptp = np.ptp(data, axis=2)  # shape (n_epochs, n_channels)
    ptp_epoch = ptp.mean(axis=1)
    return {"mean_ptp_uv": float(np.nanmean(ptp_epoch)), "max_ptp_uv": float(np.nanmax(ptp_epoch))}

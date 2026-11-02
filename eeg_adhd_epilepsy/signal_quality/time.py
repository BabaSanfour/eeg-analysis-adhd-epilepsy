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

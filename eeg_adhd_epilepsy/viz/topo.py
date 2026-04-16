"""Topographic visualization utilities for EEG analysis."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import mne

from eeg_adhd_epilepsy.viz.utils import save_fig

LOGGER = logging.getLogger(__name__)


def plot_topomap_from_channel_values(
    channel_names: Sequence[str],
    values: Sequence[float],
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
    bad_channels: List[str] | None = None,
) -> plt.Figure | None:
    """Topomap plotting without a raw object (uses standard montage).
    
    Args:
        channel_names: List of channel names.
        values: Values to plot (one per channel).
        title: Figure title.
        cmap: Colormap name.
        unit: Optional unit for colorbar label.
        bad_channels: Optional list of channels to mark with an 'X'.
    """
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
    
    # 1. Plot the topomap
    im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap)
    
    # 2. Mark Bad Channels with X (if provided)
    if bad_channels:
        picks = mne.pick_types(info, eeg=True, exclude=[])
        pos = mne.channels.layout._find_topomap_coords(info, picks)
        for i, ch in enumerate(channel_names):
            if ch in bad_channels:
                ax.plot(pos[i, 0], pos[i, 1], "kx", markersize=8, markeredgewidth=1.5)

    # 3. Add Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    if unit:
        cbar.set_label(unit)
        
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_bad_channels_topo(
    raw: mne.io.BaseRaw,
    global_bads: List[str],
    artifact_stats: Dict | None = None,
    title: str = "Bad Channels & Artifact Frequency",
    show: bool = False
) -> plt.Figure:
    """Generate a topographic map showing global bads and local artifact frequency.
    
    This function leverages the new plot_topomap_from_channel_values but with 
    additional raw-specific annotations logic.
    """
    info = raw.info
    picks = mne.pick_types(info, eeg=True, exclude=[])
    ch_names = [info["ch_names"][i] for i in picks]
    
    freqs = np.zeros(len(ch_names))
    ch_to_idx = {name: i for i, name in enumerate(ch_names)}
    for annot in raw.annotations:
        if annot["description"].startswith("BAD_") and annot["ch_names"]:
            for ch in annot["ch_names"]:
                if ch in ch_to_idx:
                    freqs[ch_to_idx[ch]] += 1

    fig = plot_topomap_from_channel_values(
        channel_names=ch_names,
        values=freqs,
        title=title,
        cmap="YlOrRd",
        unit="Artifact Count",
        bad_channels=global_bads
    )
    
    if fig and show:
        plt.show()
    return fig

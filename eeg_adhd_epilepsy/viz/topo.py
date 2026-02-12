"""Topographic visualization utilities for EEG analysis."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import mne

LOGGER = logging.getLogger(__name__)

def plot_bad_channels_topo(
    raw: mne.io.BaseRaw,
    global_bads: List[str],
    artifact_stats: Dict,
    title: str = "Bad Channels & Artifact Frequency",
    show: bool = False
) -> plt.Figure:
    """Generate a topographic map showing global bads and local artifact frequency.

    Args:
        raw: The raw MNE object (used for info/montage).
        global_bads: List of channels marked as globally bad (e.g. by RANSAC).
        artifact_stats: Dictionary containing 'autoreject_log' or similar 
                        to compute per-channel artifact frequency.
        title: Title for the figure.
        show: Whether to show the plot.

    Returns:
        The matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    
    info = raw.info
    picks = mne.pick_types(info, eeg=True, exclude=[])
    ch_names = [info['ch_names'][i] for i in picks]
    
    # 1. Compute Artifact Frequency per Channel (Local Bads)
    # We expect artifact_stats to have aggregated info from AutoReject
    # If not present, we just plot zeros
    freqs = np.zeros(len(ch_names))
    
    # Attempt to extract from artifact_stats (AutoReject style)
    # The 'bad_channel_spans' count isn't enough, we need per-channel counts
    # base.py doesn't currently store per-channel artifact counts in provenance, 
    # so we might need to compute it from raw.annotations if available with ch_names.
    
    ch_to_idx = {name: i for i, name in enumerate(ch_names)}
    for annot in raw.annotations:
        if annot['description'].startswith('BAD_') and annot['ch_names']:
            for ch in annot['ch_names']:
                if ch in ch_to_idx:
                    freqs[ch_to_idx[ch]] += 1

    # Normalize to a 0-1 range (or percentage) if needed, 
    # but raw counts are also fine for a heatmap if we set a reasonable max.
    
    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. Plot Topomap (The heatmap component)
    im, _ = mne.viz.plot_topomap(
        freqs,
        info,
        axes=ax,
        show=False,
        cmap="YlOrRd", # Yellow to Red for artifact frequency
        contours=0,
        outlines="head",
        sphere=None
    )
    
    # 4. Mark Global Bads
    # We want to put an 'X' or similar on sensors marked globally bad
    pos = mne.channels.layout._find_topomap_coords(info, picks)
    for i, ch in enumerate(ch_names):
        if ch in global_bads:
            ax.plot(pos[i, 0], pos[i, 1], 'kx', markersize=10, markeredgewidth=2)
            # ax.text(pos[i, 0], pos[i, 1], ch, fontsize=8, ha='center', va='bottom')

    # 5. Colorbar and Labels
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Artifact Count (Local/AutoReject)")
    
    ax.set_title(title)
    
    # Add Legend/Note
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='black', label='Global Bad (RANSAC)', 
               markerfacecolor='black', markersize=10, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.0))

    if show:
        plt.show()
    
    return fig

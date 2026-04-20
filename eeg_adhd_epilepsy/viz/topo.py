"""Topographic visualization utilities for EEG analysis."""

from __future__ import annotations

import base64
import io
import logging
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import mne
import plotly.graph_objects as go

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
        
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_names = set(montage.ch_names)
    kept = [
        (str(channel), float(value))
        for channel, value in zip(channel_names, arr.tolist())
        if str(channel) in montage_names
    ]
    if len(kept) < 3:
        return None

    kept_names = [channel for channel, _ in kept]
    kept_values = np.asarray([value for _, value in kept], dtype=float)
    info = mne.create_info(kept_names, sfreq=100.0, ch_types="eeg")
    try:
        info.set_montage(montage, on_missing="raise")
    except Exception:
        return None
        
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    # 1. Plot the topomap
    try:
        im, _ = mne.viz.plot_topomap(kept_values, info, axes=ax, show=False, cmap=cmap)
    except Exception:
        plt.close(fig)
        return None
    
    # 2. Mark Bad Channels with X (if provided)
    if bad_channels:
        picks = mne.pick_types(info, eeg=True, exclude=[])
        pos = mne.channels.layout._find_topomap_coords(info, picks)
        for i, ch in enumerate(kept_names):
            if ch in bad_channels:
                ax.plot(pos[i, 0], pos[i, 1], "kx", markersize=8, markeredgewidth=1.5)

    # 3. Add Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    if unit:
        cbar.set_label(unit)
        
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_topomap_selector(
    value_maps: Dict[str, tuple[Sequence[str], Sequence[float]]],
    title: str,
    unit: str | None = None,
) -> go.Figure | None:
    """Interactive selector over MNE-rendered topomaps."""
    if not value_maps:
        return None

    rendered_maps: list[tuple[str, str]] = []
    for label, payload in value_maps.items():
        channel_names, values = payload
        topo_fig = plot_topomap_from_channel_values(
            channel_names=channel_names,
            values=values,
            title=f"{title} - {label}",
            unit=unit,
        )
        if topo_fig is None:
            continue
        buf = io.BytesIO()
        topo_fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(topo_fig)
        buf.seek(0)
        rendered_maps.append(
            (str(label), f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}")
        )

    if not rendered_maps:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker={"opacity": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    first_label, first_image = rendered_maps[0]

    fig.update_layout(
        title=f"{title} - {first_label}",
        width=520,
        height=380,
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1], "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        images=[
            {
                "source": first_image,
                "xref": "x",
                "yref": "y",
                "x": 0,
                "y": 1,
                "sizex": 1,
                "sizey": 1,
                "sizing": "contain",
                "layer": "below",
            }
        ],
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 1.0,
                "y": 1.16,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": label,
                        "method": "relayout",
                        "args": [
                            {
                                "title": f"{title} - {label}",
                                "images": [
                                    {
                                        "source": image_src,
                                        "xref": "x",
                                        "yref": "y",
                                        "x": 0,
                                        "y": 1,
                                        "sizex": 1,
                                        "sizey": 1,
                                        "sizing": "contain",
                                        "layer": "below",
                                    }
                                ],
                            }
                        ],
                    }
                    for label, image_src in rendered_maps
                ],
            }
        ],
    )
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

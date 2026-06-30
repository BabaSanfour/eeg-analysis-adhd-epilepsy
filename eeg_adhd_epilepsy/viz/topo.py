"""Topographic visualization utilities for EEG analysis."""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from coco_pipe.viz import info_from_montage
from coco_pipe.viz import plot_topomap as coco_plot_topomap

from eeg_adhd_epilepsy.utils.constants import BASIC_1020_CHANNELS

LOGGER = logging.getLogger(__name__)


def feature_names_are_sensors(feature_names: Sequence[str] | None) -> bool:
    """Return ``True`` when every feature name is a standard 10-20 channel.

    Used to gate scalp topomaps of component loadings: a topomap is only
    meaningful when the feature axis *is* the sensor montage (e.g. raw
    sensor-space input), not when features are ``band×channel`` descriptors
    flattened into composite names.
    """
    names = [str(name) for name in (feature_names or [])]
    if len(names) < 3:
        return False

    known = {ch.lower() for ch in BASIC_1020_CHANNELS}
    return all(name.lower() in known for name in names)


def plot_topomap_from_channel_values(
    channel_names: Sequence[str],
    values: Sequence[float],
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
    bad_channels: list[str] | None = None,
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

    names = [str(channel) for channel in channel_names]
    info = info_from_montage(names, montage="standard_1020")
    if len(info.ch_names) < 3:
        return None

    fig, ax = coco_plot_topomap(
        values=arr,
        index=names,
        info=info,
        cmap=cmap,
        symmetric=False,
        title=title,
        cbar=True,
        cbar_label=unit,
        figsize=(4.0, 3.2),
    )

    # Mark bad channels with an 'X', as the generic helper has no concept of
    # this project's "bad channel" annotation.
    if bad_channels:
        picks = mne.pick_types(info, eeg=True, exclude=[])
        pos = mne.channels.layout._find_topomap_coords(info, picks)
        for i, ch in enumerate(info.ch_names):
            if ch in bad_channels:
                ax.plot(pos[i, 0], pos[i, 1], "kx", markersize=8, markeredgewidth=1.5)

    return fig


def plot_topomap_selector(
    value_maps: dict[str, tuple[Sequence[str], Sequence[float]]],
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

"""Shared visualizations for QC reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coco_pipe.viz import plot_histogram

from eeg_adhd_epilepsy.viz.utils import save_fig


def get_segment_palette(segment_types: Sequence[str]) -> dict[str, Any]:
    """Uniform discrete color palette for segment types."""
    cmap = plt.get_cmap("tab20")
    return {seg: cmap(idx % cmap.N) for idx, seg in enumerate(sorted(segment_types))}


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
    palette = get_segment_palette(segment_types)

    for idx, seg_type in enumerate(segment_types):
        ax = axes_arr[idx]
        series = df.loc[df["segment_type"] == seg_type, "metric"]
        if series.empty:
            ax.axis("off")
            continue
        plot_histogram(
            series,
            bins=20,
            color=palette.get(seg_type, "#4C72B0"),
            title=seg_type,
            xlabel=xlabel,
            ax=ax,
        )
        ax.grid(True, alpha=0.3)

    for extra_ax in axes_arr[len(segment_types) :]:
        extra_ax.axis("off")

    fig.suptitle(f"{title} by Segment Type", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = fig_dir / f"{column}_by_segment_type.png"
    return save_fig(fig, out_path)

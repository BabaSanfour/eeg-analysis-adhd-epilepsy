"""Generic visualization utilities and plotting helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from coco_pipe.viz import save_figure

LOGGER = logging.getLogger(__name__)

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 150) -> Path:
    """Standardized figure saving helper.

    Thin wrapper around :func:`coco_pipe.viz.save_figure` that also creates
    the parent directory and closes the figure afterwards.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path), dpi=dpi)
    plt.close(fig)
    return out_path

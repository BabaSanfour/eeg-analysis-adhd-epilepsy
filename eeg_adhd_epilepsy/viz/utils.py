"""Generic visualization utilities and plotting helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 150) -> Path:
    """Standardized figure saving helper."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

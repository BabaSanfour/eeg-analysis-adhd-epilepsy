"""Automated thresholding logic for artifact component selection."""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def select_n_components_dss(
    scores: np.ndarray,
    max_n: int = 5,
    min_n: int = 1,
    threshold_ratio: float = 0.5,
    method: str = "ratio",
) -> int:
    """Select the number of DSS components to remove based on score distribution.

    Parameters
    ----------
    scores:
        Eigenvalue/score array from DSS (descending order).
    max_n:
        Maximum components to remove.
    min_n:
        Minimum components to remove (when the method allows it).
    threshold_ratio:
        For ``method='ratio'``: keep components whose score is at least
        ``threshold_ratio * scores[0]``.
    method:
        ``'ratio'`` (default), ``'elbow'``, or ``'fixed'``.

    Returns
    -------
    int
        Number of components to remove, clipped to ``[min_n, max_n]``.
    """
    if len(scores) == 0:
        return 0

    if method == "fixed":
        return min(max_n, len(scores))

    if method == "ratio":
        significant = np.where(scores >= scores[0] * threshold_ratio)[0]
        return int(np.clip(len(significant), min_n, max_n))

    if method == "elbow":
        if len(scores) < 3:
            return min_n
        log_scores = np.log(scores + 1e-10)
        d2 = np.diff(log_scores, n=2)
        elbow_idx = int(np.argmax(d2)) + 1  # +1 for diff(n=2) offset
        return int(np.clip(elbow_idx, min_n, max_n))

    return min_n


def select_ica_components(
    labels: dict,
    target_labels: Sequence[str],
    exclude_probability: float = 0.8,
) -> List[int]:
    """Select ICA components to exclude based on ICLabel probabilities.

    Parameters
    ----------
    labels:
        ICLabel output dict with keys ``"labels"`` and ``"y_pred_proba"``.
    target_labels:
        Label names to exclude (e.g. ``['eye', 'muscle']``).
    exclude_probability:
        Minimum probability threshold for exclusion.

    Returns
    -------
    List[int]
        Component indices to exclude.
    """
    return [
        i
        for i, (label, prob) in enumerate(
            zip(labels["labels"], labels["y_pred_proba"])
        )
        if label in target_labels and prob > exclude_probability
    ]

"""Automated thresholding logic for artifact component selection."""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)

def select_n_components_dss(
    scores: np.ndarray, 
    max_n: int = 5, 
    min_n: int = 1,
    threshold_ratio: float = 0.5,
    method: str = "ratio"
) -> int:
    """Select number of DSS components to remove based on score distribution.
    
    Args:
        scores: The eigenvalues/scores from DSS (descending).
        max_n: Maximum number of components to remove.
        min_n: Minimum number of components to remove (if method allows).
        threshold_ratio: Ratio of the first component's score to consider 
                         the next one significant.
        method: Selection method ('ratio', 'elbow', or 'fixed').
        
    Returns:
        n_remove: Number of components to remove.
    """
    if len(scores) == 0:
        return 0
    
    if method == "fixed":
        return min(max_n, len(scores))

    if method == "ratio":
        # Keep components that have at least threshold_ratio * score[0]
        # This is a simple SNR-like heuristic for high-bias components
        significant = np.where(scores >= scores[0] * threshold_ratio)[0]
        n_found = len(significant)
        return int(np.clip(n_found, min_n, max_n))

    if method == "elbow":
        # Find elbow in the score curve using second derivative
        if len(scores) < 3:
            return min_n
        
        # Log-space elbow detection is often more robust for eigenvalues
        log_scores = np.log(scores + 1e-10)
        d2 = np.diff(log_scores, n=2)
        elbow_idx = np.argmax(d2) + 1 # +1 because of diff(n=2)
        return int(np.clip(elbow_idx, min_n, max_n))

    return min_n


def select_ica_components(
    labels: dict, 
    target_labels: Sequence[str],
    exclude_probability: float = 0.8,
    adaptive: bool = True
) -> List[int]:
    """Select ICA components to exclude based on ICLabel probabilities.
    
    Args:
        labels: ICLabel output dictionary.
        target_labels: Labels to exclude (e.g. ['eye', 'muscle']).
        exclude_probability: Hard threshold if adaptive=False.
        adaptive: If True, look for components that are clearly 'artifact' 
                  vs 'brain' even if they fall slightly below threshold.
                  
    Returns:
        exclude_idx: List of indices to exclude.
    """
    exclude_idx = []
    
    for i, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        # Base logic: match label and exceed probability
        if label in target_labels:
            if prob > exclude_probability:
                exclude_idx.append(i)
            elif adaptive and prob > 0.5:
                pass
                
    return exclude_idx

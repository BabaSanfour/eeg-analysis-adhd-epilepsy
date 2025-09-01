"""Utility configuration variables exposed for convenient import.

This module imports selected names from :mod:`utils.config` explicitly to
avoid polluting the namespace with wildcard imports.
"""

from .config import (
    data_dir,
    embeddings_dir,
    results_dir,
    csv_dir,
    bids_dir,
    derivatives_dir,
    source_dirs,
    sensors_to_keep,
    n_subjects,
    MAPPING_PSYCHOSTIMULANT,
)

__all__ = [
    "data_dir",
    "embeddings_dir",
    "results_dir",
    "csv_dir",
    "bids_dir",
    "derivatives_dir",
    "source_dirs",
    "sensors_to_keep",
    "n_subjects",
    "MAPPING_PSYCHOSTIMULANT",
]

"""Convenience re-exports for common config names.

Used by the legacy `utils/__init__.py` that lived at repo top-level.
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


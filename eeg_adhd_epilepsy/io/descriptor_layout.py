"""Shared file-layout contract for descriptor extraction shards.

Both ``analysis/extract_descriptors.py`` (producer) and
``analysis/merge_descriptors.py`` (consumer) need to agree on which files
constitute a "complete" descriptor shard. Centralizing the lists here means
adding/renaming an output file is a one-place change.
"""

from __future__ import annotations

SENSOR_TABLE_FILES: tuple[str, ...] = (
    "_SUCCESS",
    "sensor_descriptor_bundle.npz",
    "sensor_epoch_features.csv",
    "sensor_epoch_features.parquet",
    "sensor_epoch_features_feature_columns.json",
    "sensor_subject_features.csv",
    "sensor_subject_features.parquet",
    "sensor_subject_features_feature_columns.json",
    "failures.csv",
)
"""Sensor-level descriptor table files written for every shard."""

POOLED_TABLE_FILES: tuple[str, ...] = (
    "pooled_epoch_features.csv",
    "pooled_epoch_features.parquet",
    "pooled_epoch_features_feature_columns.json",
    "pooled_subject_features.csv",
    "pooled_subject_features.parquet",
    "pooled_subject_features_feature_columns.json",
)
"""Pooled (channel-group) descriptor table files, written when channel-group
pooling is enabled."""

QC_FILES: tuple[str, ...] = (
    "qc/summary_row.csv",
    "qc/summary_metrics.csv",
    "qc/flags.csv",
    "qc/failure_summary.csv",
    "qc/feature_missingness.csv",
    "qc/family_summary.csv",
)
"""Per-shard QC artifacts written by :func:`run_descriptor_subject_qc`."""

# Feature-column sidecars consulted when merging shards, keyed by the table
# they describe.
FEATURE_COLUMN_FILES: dict[str, str] = {
    "sensor_epoch": "sensor_epoch_features_feature_columns.json",
    "sensor_subject": "sensor_subject_features_feature_columns.json",
    "pooled_epoch": "pooled_epoch_features_feature_columns.json",
    "pooled_subject": "pooled_subject_features_feature_columns.json",
}


def required_descriptor_files(
    include_pooled: bool, *, include_qc: bool = False
) -> tuple[str, ...]:
    """Return the tuple of relative paths a complete shard must contain.

    Parameters
    ----------
    include_pooled
        Whether channel-group pooling is enabled for this run.
    include_qc
        Whether to additionally require the per-shard QC artifacts.
    """
    files = SENSOR_TABLE_FILES
    if include_pooled:
        files = files + POOLED_TABLE_FILES
    if include_qc:
        files = files + QC_FILES
    return files

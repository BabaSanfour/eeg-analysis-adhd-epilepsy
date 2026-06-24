"""Shared file-layout contract for descriptor extraction shards.

Both ``analysis/extract_descriptors.py`` (producer) and
``analysis/merge_descriptors.py`` (consumer) must agree on which files make up a
"complete" descriptor shard. Centralizing it here means adding or renaming an
output file is a one-place change.

Per-table filenames are derived from each table *stem* plus the suffix
convention of coco-pipe's ``save_descriptor_table`` (``{stem}.parquet`` /
``{stem}.csv`` / ``{stem}_feature_columns.json``). Those suffixes are kept as
local constants here; only the markers and QC artifacts the producer writes
directly are listed literally.
"""

from __future__ import annotations

# Suffix convention written by coco-pipe's ``save_descriptor_table``.
_TABLE_FORMATS = ("parquet", "csv")
_FEATURE_COLUMNS_SUFFIX = "_feature_columns.json"

# Descriptor-table stems written via ``save_descriptor_table``, by tier.
SENSOR_TABLE_STEMS: tuple[str, ...] = ("sensor_epoch_features", "sensor_subject_features")
POOLED_TABLE_STEMS: tuple[str, ...] = ("pooled_epoch_features", "pooled_subject_features")

# Non-table artifacts the producer writes directly (no save_descriptor_table).
SENSOR_MARKER_FILES: tuple[str, ...] = (
    "_SUCCESS",
    "sensor_descriptor_bundle.npz",
    "failures.csv",
)
QC_FILES: tuple[str, ...] = (
    "qc/summary_row.csv",
    "qc/summary_metrics.csv",
    "qc/flags.csv",
    "qc/failure_summary.csv",
    "qc/feature_missingness.csv",
    "qc/family_summary.csv",
)
"""Per-shard QC artifacts written by :func:`run_descriptor_subject_qc`."""


def _table_files(stems: tuple[str, ...]) -> tuple[str, ...]:
    files: list[str] = []
    for stem in stems:
        files.extend(f"{stem}.{fmt}" for fmt in _TABLE_FORMATS)
        files.append(f"{stem}{_FEATURE_COLUMNS_SUFFIX}")
    return tuple(files)


# Sensor-level files written for every shard (markers + derived table files).
SENSOR_TABLE_FILES: tuple[str, ...] = SENSOR_MARKER_FILES + _table_files(SENSOR_TABLE_STEMS)
"""Sensor-level descriptor files written for every shard."""

# Pooled (channel-group) table files, written when channel-group pooling is on.
POOLED_TABLE_FILES: tuple[str, ...] = _table_files(POOLED_TABLE_STEMS)
"""Pooled (channel-group) descriptor table files."""

# Feature-column sidecars consulted when merging shards, keyed by table tier.
FEATURE_COLUMN_FILES: dict[str, str] = {
    stem.removesuffix("_features"): f"{stem}{_FEATURE_COLUMNS_SUFFIX}"
    for stem in SENSOR_TABLE_STEMS + POOLED_TABLE_STEMS
}


def required_descriptor_files(include_pooled: bool, *, include_qc: bool = False) -> tuple[str, ...]:
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

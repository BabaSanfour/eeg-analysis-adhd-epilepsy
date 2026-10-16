"""File layout of a descriptor shard, shared by stages 05 (writer), 06 and 07 (readers).

A shard is ``<derivative_root>/sub-<subject>/ses-<session>/eeg/<condition>/`` with two files:

- ``sensor_epoch_features.parquet``: one row per epoch; the table carries its feature
  columns in its own metadata (coco-pipe ``save_descriptor_table``).
- ``shard.json``: the shard's QC summary, QC flags and extraction failures. It is written
  last, so a shard without it is incomplete.

The recording table is not kept per shard: stage 07 rebuilds it from the merged epoch
table with the aggregation settings in ``config_used.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.descriptors import DescriptorConfig, build_descriptor_tables
from coco_pipe.io import DataContainer, read_json

EPOCH_TABLE = "sensor_epoch_features"
SHARD_RECORD = "shard.json"
SHARD_FILES: tuple[str, ...] = (f"{EPOCH_TABLE}.parquet", SHARD_RECORD)


def find_shards(derivative_root: Path) -> list[Path]:
    """Every shard folder under *derivative_root*, complete or not."""
    return sorted(path for path in Path(derivative_root).glob("sub-*/ses-*/eeg/*") if path.is_dir())


def read_shard_record(shard_root: Path) -> dict[str, Any]:
    """The shard's ``shard.json``: ``qc_summary``, ``qc_flags`` and ``failures``."""
    return read_json(Path(shard_root) / SHARD_RECORD)


def recording_tables(
    metadata_df: pd.DataFrame,
    container: DataContainer,
    config_used: dict[str, Any],
    *,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Epoch and recording tables from one condition's epochs, as configured.

    Stage 05 calls it on a shard's epochs for its QC, stage 07 on the merged epochs of
    each condition: both get the recording values from the same settings.
    """
    bands = DescriptorConfig.model_validate(
        {
            key: value
            for key, value in config_used.items()
            if key not in {"aggregation", "qc", "desc"}
        }
    ).families.bands
    return build_descriptor_tables(
        container,
        metadata_df,
        group_by="recording_id",
        id_col="obs_id",
        target_col=target_col,
        aggregation_groups=list(config_used["aggregation"]["descriptors"]),
        ratio_pairs=list(bands.ratio_pairs) if bands.enabled else [],
        ratio_floor=float(bands.min_denominator_power) if bands.enabled else 0.0,
        on_insufficient="warn",
    )

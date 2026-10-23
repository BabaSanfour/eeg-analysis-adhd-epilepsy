"""File layout of a descriptor shard: ``extract_descriptors`` writes it, ``verify_descriptors``
and ``merge_descriptors`` read it.

A shard is ``<derivative_root>/sub-<subject>/ses-<session>/eeg/<condition>/`` with two files:

- ``sensor_epoch_features.parquet``: one row per epoch; the table carries its feature
  columns in its own metadata (coco-pipe ``save_descriptor_table``).
- ``shard.json``: the shard's QC summary, QC flags and extraction failures. It is written
  last, so a shard without it is incomplete.

The recording table is not kept per shard: the merge rebuilds it from the merged epoch
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


def descriptor_dataset_description(derivative_root: Path) -> dict[str, Any]:
    """The ``dataset_description.json`` of a descriptor derivative.

    ``SourceDatasets`` points at the raw dataset relative to the derivative, which sits
    under ``<BIDS>/derivatives/`` on scratch and on the project alike, so the link
    survives a copy between clusters.
    """
    depth = Path(derivative_root).parts[::-1].index("derivatives") + 1
    return {
        "Name": "Signal Features",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "eeg-descriptors",
                "Description": "Checkpointed coco-pipe descriptor feature extraction",
            }
        ],
        "SourceDatasets": [{"URL": "../" * depth}],
    }


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
) -> dict[str, Any]:
    """Epoch and recording tables from one condition's epochs, as configured.

    The extraction calls it on a shard's epochs for its QC, the merge on the merged epochs of
    each condition: both get the recording values from the same settings. A recording
    row's ``obs_id`` is ``<recording_id>_cond-<condition>``, as coco-pipe names the
    recording level of raw epochs and embeddings.
    """
    bands = DescriptorConfig.model_validate(
        {
            key: value
            for key, value in config_used.items()
            if key not in {"aggregation", "qc", "desc"}
        }
    ).families.bands
    tables = build_descriptor_tables(
        container,
        metadata_df,
        group_by="recording_id",
        id_col="obs_id",
        aggregation_groups=list(config_used["aggregation"]["descriptors"]),
        ratio_pairs=list(bands.ratio_pairs) if bands.enabled else [],
        ratio_floor=float(bands.min_denominator_power) if bands.enabled else 0.0,
        on_insufficient="warn",
    )
    recordings = tables["subject_df"]
    recordings.insert(
        0,
        "obs_id",
        recordings["recording_id"].astype(str) + "_cond-" + recordings["condition"].astype(str),
    )
    return tables

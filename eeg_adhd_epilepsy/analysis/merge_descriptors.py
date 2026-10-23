"""Merge the descriptor shards of stage 05 into combined tables.

- ``combined/sensor_epoch_features.parquet``: the shards' epoch tables, 1 row / epoch
- ``combined/sensor_recording_features.parquet``: 1 row / recording and condition, built
  from the merged epochs with the aggregation settings of ``config_used.yaml``

Each table carries its feature columns in its own metadata. ``combined/merge_manifest.json``
records the inputs and counts. Stage 06 runs first: a shard missing a file, or with other
feature columns, stops the merge.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from coco_pipe.descriptors import merge_descriptor_tables, save_descriptor_table
from coco_pipe.io import DataContainer, write_json

from eeg_adhd_epilepsy.analysis.utils.descriptor_shards import (
    EPOCH_TABLE,
    SHARD_FILES,
    descriptor_dataset_description,
    find_shards,
    read_shard_record,
    recording_tables,
)
from eeg_adhd_epilepsy.qc.descriptor_qc import run_descriptor_dataset_qc
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def _nunique_or_none(df: pd.DataFrame, column: str) -> int | None:
    """Distinct-value count for *column*, or ``None`` when it is absent."""
    return int(df[column].nunique()) if column in df.columns else None


def _recording_table(
    epoch_df: pd.DataFrame,
    feature_columns: list[str],
    config_used: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """The recording table of the merged epochs, one condition at a time.

    A recording holds several conditions, so its epochs are grouped within each condition,
    as stage 05 does within a shard.
    """
    metadata_columns = [column for column in epoch_df.columns if column not in set(feature_columns)]
    frames: list[pd.DataFrame] = []
    recording_feature_columns: list[str] = []
    for _condition, epochs in epoch_df.groupby("condition", sort=False):
        tables = recording_tables(
            epochs[metadata_columns],
            DataContainer(
                X=epochs[feature_columns].to_numpy(),
                dims=("obs", "feature"),
                coords={"feature": np.asarray(feature_columns, dtype=object)},
            ),
            config_used,
        )
        frames.append(tables["subject_df"])
        recording_feature_columns = tables["subject_feature_columns"]
    return pd.concat(frames, ignore_index=True), recording_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge checkpointed descriptor shards into combined tables."
    )
    parser.add_argument(
        "--derivative_root",
        required=True,
        help="Descriptor derivative with the per-subject shards (stage 05, on scratch).",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help=(
            "Derivative to write the merged tables to (combined/), with its "
            "dataset_description.json and the shards' .bidsignore and config_used.yaml."
        ),
    )
    parser.add_argument("--reports_root", required=True, help="Root directory for reports.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    derivative_root = Path(args.derivative_root)
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)

    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config used not found: {config_path}")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_used = load_yaml_config(config_path)

    shard_roots = find_shards(derivative_root)
    if not shard_roots:
        raise ValueError(f"No descriptor shards found under {derivative_root}.")
    # A shard without its shard.json (written last) crashed halfway; stage 06 lists it.
    incomplete = [
        str(shard_root.relative_to(derivative_root))
        for shard_root in shard_roots
        if not all((shard_root / name).exists() for name in SHARD_FILES)
    ]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} incomplete shard(s) under {derivative_root}, e.g. "
            f"{incomplete[:5]}; rerun stage 05 for them (stage 06 lists them)."
        )
    LOGGER.info("Merging %d shards from %s", len(shard_roots), derivative_root)

    records = [read_shard_record(shard_root) for shard_root in shard_roots]
    failure_tables = [pd.DataFrame(record["failures"]) for record in records]
    failures_df = pd.concat(
        [df for df in failure_tables if not df.empty] or [pd.DataFrame()], ignore_index=True
    )

    combined_root = output_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "dataset_description.json", descriptor_dataset_description(output_root)
    )
    for name in (".bidsignore", "config_used.yaml"):
        shutil.copy2(derivative_root / name, output_root / name)

    # The merge refuses a shard whose feature columns differ from the first one's.
    LOGGER.info("Combining the epoch tables...")
    combined_sensor_epoch_df, epoch_feature_columns = merge_descriptor_tables(
        [shard_root / f"{EPOCH_TABLE}.parquet" for shard_root in shard_roots],
        out_base_path=combined_root / EPOCH_TABLE,
    )
    LOGGER.info("Building the recording table...")
    combined_sensor_recording_df, recording_feature_columns = _recording_table(
        combined_sensor_epoch_df, epoch_feature_columns, config_used
    )
    save_descriptor_table(
        combined_sensor_recording_df,
        combined_root / "sensor_recording_features",
        feature_columns=recording_feature_columns,
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "derivative_root": str(derivative_root),
        "output_root": str(output_root),
        "config_used_path": str(config_path),
        "config_used_sha256": config_hash,
        "n_shards": len(shard_roots),
        "n_subjects": _nunique_or_none(combined_sensor_recording_df, "subject"),
        "n_sessions": _nunique_or_none(combined_sensor_recording_df, "session"),
        "n_conditions": _nunique_or_none(combined_sensor_recording_df, "condition"),
        "n_epoch_rows": int(len(combined_sensor_epoch_df)),
        "n_recording_rows": int(len(combined_sensor_recording_df)),
    }
    # QC runs on the recording-level table — the granularity decoding/dim-reduction
    # actually consume (the keyword args keep their historical `_subject` names).
    dataset_qc = run_descriptor_dataset_qc(
        reports_root=reports_root,
        merged_sensor_epoch_df=combined_sensor_epoch_df,
        merged_sensor_subject_df=combined_sensor_recording_df,
        merged_sensor_epoch_feature_columns=epoch_feature_columns,
        merged_sensor_subject_feature_columns=recording_feature_columns,
        shard_qc_rows_df=pd.DataFrame([record["qc_summary"] for record in records]),
        merged_failures_df=failures_df,
        config_snapshot=config_used,
        manifest=manifest,
    )
    for key in ("n_failures_total", "n_epochs_rejected", "qc_status", "report_path"):
        manifest[key] = dataset_qc[key]
    manifest_path = combined_root / "merge_manifest.json"
    write_json(manifest_path, manifest)

    LOGGER.info(
        "Merged %d shards into %s; %d failures, %d rejected epochs, dataset qc=%s, report=%s",
        len(shard_roots),
        combined_root,
        dataset_qc["n_failures_total"],
        dataset_qc["n_epochs_rejected"],
        dataset_qc["qc_status"],
        dataset_qc["report_path"],
    )


if __name__ == "__main__":
    main()

"""Merge the descriptor shards of stage 05 into combined tables.

- ``combined/sensor_epoch_features.parquet``: the shards' epoch tables, 1 row / epoch
- ``combined/sensor_recording_features.parquet``: 1 row / recording and condition, built
  from the merged epochs with the aggregation settings of ``config_used.yaml``

Each table carries its feature columns in its own metadata. ``combined/failures.csv``
holds the extraction failures of every shard.
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
from coco_pipe.descriptors import (
    check_feature_column_consistency,
    feature_columns_of,
    merge_descriptor_tables,
    save_descriptor_table,
)
from coco_pipe.io import DataContainer, write_json

from eeg_adhd_epilepsy.analysis.utils.descriptor_shards import (
    EPOCH_TABLE,
    SHARD_FILES,
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


def _check_shard_feature_columns(
    shard_root: Path,
    feature_cols: dict[str, list[str] | None],
) -> list[str]:
    """Check the feature columns of *shard_root*'s epoch table against the first shard's.

    Returns a list of human-readable mismatch descriptions (empty if the
    shard is consistent with previously-seen shards).
    """
    table_path = shard_root / f"{EPOCH_TABLE}.parquet"
    # Reading no column still restores the table's metadata, the feature columns with it.
    columns = feature_columns_of(pd.read_parquet(table_path, columns=[]), table_path)
    try:
        check_feature_column_consistency(columns, feature_cols, "sensor_epoch", table_path)
    except ValueError as exc:
        return [str(exc)]
    return []


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
            "Derivative to write the merged tables to (combined/), with the shards' "
            "dataset_description.json, .bidsignore and config_used.yaml."
        ),
    )
    parser.add_argument(
        "--skip_inconsistent",
        action="store_true",
        help=(
            "If a shard's feature columns differ from the first shard, exclude it "
            "from the merge (recorded in merge_manifest.json) instead of aborting "
            "the whole merge."
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

    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config_used = load_yaml_config(config_path)

    # A shard without its shard.json (written last) crashed halfway.
    LOGGER.info("Searching for descriptor shards in %s", derivative_root)
    candidate_roots: list[Path] = []
    excluded_shards: list[dict[str, object]] = []
    found_roots = find_shards(derivative_root)
    n_shards_discovered = len(found_roots)
    for shard_root in found_roots:
        missing = [f for f in SHARD_FILES if not (shard_root / f).exists()]
        if missing:
            excluded_shards.append(
                {
                    "shard": str(shard_root.relative_to(derivative_root)),
                    "reasons": [f"missing required file(s): {', '.join(missing)}"],
                }
            )
            continue
        candidate_roots.append(shard_root)

    if not candidate_roots:
        raise ValueError(f"No completed descriptor shards found under {derivative_root}.")

    LOGGER.info(
        "Found %d candidate shards (%d excluded for missing files).",
        len(candidate_roots),
        len(excluded_shards),
    )

    feature_cols: dict[str, list[str] | None] = {"sensor_epoch": None}

    shard_roots: list[Path] = []
    for shard_root in candidate_roots:
        mismatches = _check_shard_feature_columns(shard_root, feature_cols)
        if mismatches:
            if args.skip_inconsistent:
                excluded_shards.append(
                    {
                        "shard": str(shard_root.relative_to(derivative_root)),
                        "reasons": mismatches,
                    }
                )
                LOGGER.warning(
                    "Excluding shard %s due to feature-column mismatch: %s",
                    shard_root,
                    "; ".join(mismatches),
                )
                continue
            raise ValueError(
                f"Feature column mismatch detected in shard {shard_root!r}:\n"
                + "\n".join(mismatches)
                + "\nRe-run with --skip_inconsistent to exclude inconsistent shards "
                "instead of aborting, or clear the derivative root and re-run "
                "extraction with a single config."
            )
        shard_roots.append(shard_root)

    if not shard_roots:
        raise ValueError(
            f"All {len(candidate_roots)} candidate shards under {derivative_root} "
            "were excluded due to feature-column mismatches."
        )

    LOGGER.info("Merging %d shards (%d excluded total).", len(shard_roots), len(excluded_shards))

    # Per-shard failures and QC summaries. The epoch tables are combined below by
    # coco-pipe's merge primitive, which reads each shard once.
    failure_tables: list[pd.DataFrame] = []
    shard_qc_rows: list[dict[str, object]] = []
    shard_manifest_rows: list[dict[str, object]] = []

    n_shards = len(shard_roots)
    for index, shard_root in enumerate(shard_roots, start=1):
        LOGGER.info(
            "Loading shard %d/%d: %s",
            index,
            n_shards,
            shard_root.relative_to(derivative_root),
        )
        record = read_shard_record(shard_root)
        failure_df = pd.DataFrame(record["failures"])
        failure_tables.append(failure_df)
        shard_qc_rows.append(record["qc_summary"])

        shard_manifest_rows.append(
            {
                "shard": str(shard_root.relative_to(derivative_root)),
                "n_failures": len(failure_df),
            }
        )

    combined_root = output_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    for name in ("dataset_description.json", ".bidsignore", "config_used.yaml"):
        shutil.copy2(derivative_root / name, output_root / name)

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

    # Some shards may have zero-row failures.csv; concatenating only the
    # non-empty frames avoids dtype-mismatch warnings from pd.concat while
    # still falling back to a well-formed empty frame when nothing failed.
    nonempty_failure_tables = [df for df in failure_tables if not df.empty]
    if nonempty_failure_tables:
        combined_failures_df = pd.concat(nonempty_failure_tables, ignore_index=True)
    else:
        combined_failures_df = failure_tables[0].copy() if failure_tables else pd.DataFrame()
    combined_failures_df.to_csv(combined_root / "failures.csv", index=False)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "derivative_root": str(derivative_root),
        "output_root": str(output_root),
        "config_used_path": str(config_path),
        "config_used_sha256": config_hash,
        "n_shards_discovered": n_shards_discovered,
        "n_shards_merged": len(shard_roots),
        "n_shards_excluded": len(excluded_shards),
        "skip_inconsistent": bool(args.skip_inconsistent),
        "n_subjects": _nunique_or_none(combined_sensor_recording_df, "subject"),
        "n_sessions": _nunique_or_none(combined_sensor_recording_df, "session"),
        "n_conditions": _nunique_or_none(combined_sensor_recording_df, "condition"),
        "n_sensor_epoch_rows": int(len(combined_sensor_epoch_df)),
        "n_sensor_recording_rows": int(len(combined_sensor_recording_df)),
        "n_failures_total": int(len(combined_failures_df)),
        "excluded_shards": excluded_shards,
        "merged_shards": shard_manifest_rows,
    }

    # QC runs on the recording-level table — the granularity decoding/dim-reduction
    # actually consume (the keyword args keep their historical `_subject` names).
    dataset_qc = run_descriptor_dataset_qc(
        reports_root=reports_root,
        merged_sensor_epoch_df=combined_sensor_epoch_df,
        merged_sensor_subject_df=combined_sensor_recording_df,
        merged_sensor_epoch_feature_columns=epoch_feature_columns,
        merged_sensor_subject_feature_columns=recording_feature_columns,
        shard_qc_rows_df=pd.DataFrame(shard_qc_rows),
        merged_failures_df=combined_failures_df,
        config_snapshot=config_used,
        manifest=manifest,
    )

    manifest["qc_status"] = dataset_qc["qc_status"]
    manifest["report_path"] = dataset_qc["report_path"]
    manifest_path = combined_root / "merge_manifest.json"
    write_json(manifest_path, manifest)

    LOGGER.info(
        "Merged %d shards (%d excluded) into %s; dataset qc=%s, report=%s, manifest=%s",
        len(shard_roots),
        len(excluded_shards),
        combined_root,
        dataset_qc["qc_status"],
        dataset_qc["report_path"],
        manifest_path,
    )


if __name__ == "__main__":
    main()

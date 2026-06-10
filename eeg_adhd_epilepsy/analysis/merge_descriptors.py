"""
Merge checkpointed descriptor shards into combined tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from coco_pipe.io.descriptors import check_feature_column_consistency
from eeg_adhd_epilepsy.io.bids import get_reports_root
from eeg_adhd_epilepsy.io.descriptor_layout import (
    FEATURE_COLUMN_FILES,
    required_descriptor_files,
)
from eeg_adhd_epilepsy.io.table import save
from eeg_adhd_epilepsy.qc.descriptor_qc import run_descriptor_dataset_qc

LOGGER = logging.getLogger(__name__)


def _check_shard_feature_columns(
    shard_root: Path,
    include_pooled: bool,
    feature_cols: dict[str, list[str] | None],
) -> list[str]:
    """Check *shard_root*'s feature-column sidecars against the accumulated reference.

    Returns a list of human-readable mismatch descriptions (empty if the
    shard is consistent with previously-seen shards).
    """
    keys = ["sensor_epoch", "sensor_subject"]
    if include_pooled:
        keys += ["pooled_epoch", "pooled_subject"]
    mismatches: list[str] = []
    for key in keys:
        try:
            check_feature_column_consistency(shard_root, FEATURE_COLUMN_FILES[key], feature_cols, key)
        except ValueError as exc:
            mismatches.append(str(exc))
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge checkpointed descriptor shards into combined tables."
    )
    parser.add_argument(
        "--bids_root",
        required=True,
        help="Path to the BIDS dataset root.",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    bids_root = Path(args.bids_root)
    reports_root = get_reports_root(bids_root)
    derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"

    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config used not found: {config_path}")

    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    with config_path.open("r", encoding="utf-8") as handle:
        config_used = yaml.safe_load(handle) or {}
    include_pooled = bool((config_used.get("pooling") or {}).get("channel_groups"))

    # Discover completed shards across sessions
    LOGGER.info("Searching for descriptor shards in %s", derivative_root)
    candidate_roots: list[Path] = []
    excluded_shards: list[dict[str, object]] = []
    success_paths = sorted(derivative_root.glob("sub-*/ses-*/eeg/*/_SUCCESS"))
    n_shards_discovered = len(success_paths)
    for success_path in success_paths:
        shard_root = success_path.parent
        missing = [
            f
            for f in required_descriptor_files(include_pooled)
            if not (shard_root / f).exists()
        ]
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

    feature_cols: dict[str, list[str] | None] = {
        "sensor_epoch": None,
        "sensor_subject": None,
        "pooled_epoch": None,
        "pooled_subject": None,
    }

    shard_roots: list[Path] = []
    for shard_root in candidate_roots:
        mismatches = _check_shard_feature_columns(shard_root, include_pooled, feature_cols)
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

    sensor_epoch_tables: list[pd.DataFrame] = []
    sensor_agg_tables: list[pd.DataFrame] = []
    pooled_epoch_tables: list[pd.DataFrame] = []
    pooled_agg_tables: list[pd.DataFrame] = []
    failure_tables: list[pd.DataFrame] = []
    shard_qc_rows: list[pd.DataFrame] = []
    shard_manifest_rows: list[dict[str, object]] = []

    n_shards = len(shard_roots)
    for index, shard_root in enumerate(shard_roots, start=1):
        LOGGER.info(
            "Loading shard %d/%d: %s",
            index,
            n_shards,
            shard_root.relative_to(derivative_root),
        )
        sensor_epoch_df = pd.read_parquet(shard_root / "sensor_epoch_features.parquet")
        sensor_subject_df = pd.read_parquet(shard_root / "sensor_subject_features.parquet")
        sensor_epoch_tables.append(sensor_epoch_df)
        sensor_agg_tables.append(sensor_subject_df)

        pooled_epoch_rows = 0
        pooled_subject_rows = 0
        if include_pooled:
            pooled_epoch_df = pd.read_parquet(shard_root / "pooled_epoch_features.parquet")
            pooled_subject_df = pd.read_parquet(shard_root / "pooled_subject_features.parquet")
            pooled_epoch_tables.append(pooled_epoch_df)
            pooled_agg_tables.append(pooled_subject_df)
            pooled_epoch_rows = len(pooled_epoch_df)
            pooled_subject_rows = len(pooled_subject_df)

        failure_df = pd.read_csv(shard_root / "failures.csv")
        failure_tables.append(failure_df)

        shard_qc_path = shard_root / "qc" / "summary_row.csv"
        if shard_qc_path.exists():
            shard_qc_rows.append(pd.read_csv(shard_qc_path))

        shard_manifest_rows.append(
            {
                "shard": str(shard_root.relative_to(derivative_root)),
                "n_sensor_epoch_rows": len(sensor_epoch_df),
                "n_sensor_subject_rows": len(sensor_subject_df),
                "n_pooled_epoch_rows": pooled_epoch_rows,
                "n_pooled_subject_rows": pooled_subject_rows,
                "n_failures": len(failure_df),
            }
        )

    combined_root = derivative_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Combining sensor families...")
    combined_sensor_epoch_df = pd.concat(sensor_epoch_tables, ignore_index=True)
    combined_sensor_subject_df = pd.concat(sensor_agg_tables, ignore_index=True)
    save(
        combined_sensor_epoch_df,
        combined_root / "sensor_epoch_features",
        feature_columns=feature_cols["sensor_epoch"],
    )
    save(
        combined_sensor_subject_df,
        combined_root / "sensor_subject_features",
        feature_columns=feature_cols["sensor_subject"],
    )

    combined_pooled_epoch_df = None
    combined_pooled_subject_df = None
    if include_pooled:
        LOGGER.info("Combining pooled families...")
        combined_pooled_epoch_df = pd.concat(pooled_epoch_tables, ignore_index=True)
        combined_pooled_subject_df = pd.concat(pooled_agg_tables, ignore_index=True)
        save(
            combined_pooled_epoch_df,
            combined_root / "pooled_epoch_features",
            feature_columns=feature_cols["pooled_epoch"],
        )
        save(
            combined_pooled_subject_df,
            combined_root / "pooled_subject_features",
            feature_columns=feature_cols["pooled_subject"],
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bids_root": str(bids_root),
        "derivative_root": str(derivative_root),
        "config_used_path": str(config_path),
        "config_used_sha256": config_hash,
        "n_shards_discovered": n_shards_discovered,
        "n_shards_merged": len(shard_roots),
        "n_shards_excluded": len(excluded_shards),
        "skip_inconsistent": bool(args.skip_inconsistent),
        "n_subjects": int(combined_sensor_subject_df["subject"].nunique()) if "subject" in combined_sensor_subject_df.columns else None,
        "n_sessions": int(combined_sensor_subject_df["session"].nunique()) if "session" in combined_sensor_subject_df.columns else None,
        "n_conditions": int(combined_sensor_subject_df["condition"].nunique()) if "condition" in combined_sensor_subject_df.columns else None,
        "n_sensor_epoch_rows": int(len(combined_sensor_epoch_df)),
        "n_sensor_subject_rows": int(len(combined_sensor_subject_df)),
        "n_failures_total": int(len(combined_failures_df)),
        "excluded_shards": excluded_shards,
        "merged_shards": shard_manifest_rows,
    }

    combined_sensor_subject_df.attrs["qc_dir"] = str(combined_root / "qc")
    dataset_qc = run_descriptor_dataset_qc(
        derivative_root=derivative_root,
        reports_root=reports_root,
        merged_sensor_epoch_df=combined_sensor_epoch_df,
        merged_sensor_subject_df=combined_sensor_subject_df,
        merged_sensor_epoch_feature_columns_path=combined_root / "sensor_epoch_features_feature_columns.json",
        merged_sensor_subject_feature_columns_path=combined_root / "sensor_subject_features_feature_columns.json",
        merged_pooled_epoch_df=combined_pooled_epoch_df,
        merged_pooled_subject_df=combined_pooled_subject_df,
        merged_pooled_epoch_feature_columns_path=None if not include_pooled else combined_root / "pooled_epoch_features_feature_columns.json",
        merged_pooled_subject_feature_columns_path=None if not include_pooled else combined_root / "pooled_subject_features_feature_columns.json",
        shard_qc_rows_df=pd.concat(shard_qc_rows, ignore_index=True) if shard_qc_rows else None,
        merged_failures_df=combined_failures_df,
        config_snapshot=config_used,
        manifest=manifest,
    )

    manifest["qc_status"] = dataset_qc["qc_status"]
    manifest["report_path"] = dataset_qc["report_path"]
    manifest_path = combined_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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

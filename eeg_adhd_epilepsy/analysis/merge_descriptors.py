"""
Merge checkpointed descriptor shards into combined tables.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from eeg_adhd_epilepsy.analysis.utils import (
    required_descriptor_files,
    save_table,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge checkpointed descriptor shards into combined tables."
    )
    parser.add_argument(
        "--bids_root",
        required=True,
        help="Path to the BIDS dataset root.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    bids_root = Path(args.bids_root)
    derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"

    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config used not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_used = yaml.safe_load(handle) or {}
    include_pooled = bool((config_used.get("pooling") or {}).get("channel_groups"))

    # Discover completed shards across sessions
    LOGGER.info("Searching for descriptor shards in %s", derivative_root)
    shard_roots: list[Path] = []
    for success_path in sorted(derivative_root.glob("sub-*/ses-*/eeg/*/_SUCCESS")):
        shard_root = success_path.parent
        if all((shard_root / f).exists() for f in required_descriptor_files(include_pooled)):
            shard_roots.append(shard_root)

    if not shard_roots:
        raise ValueError(f"No completed descriptor shards found under {derivative_root}.")

    LOGGER.info("Found %d completed shards.", len(shard_roots))

    sensor_epoch_tables: list[pd.DataFrame] = []
    sensor_agg_tables: list[pd.DataFrame] = []
    pooled_epoch_tables: list[pd.DataFrame] = []
    pooled_agg_tables: list[pd.DataFrame] = []
    failure_tables: list[pd.DataFrame] = []
    
    feature_cols = {
        "sensor_epoch": None,
        "sensor_subject": None,
        "pooled_epoch": None,
        "pooled_subject": None,
    }

    for shard_root in shard_roots:
        sensor_epoch_tables.append(pd.read_parquet(shard_root / "sensor_epoch_features.parquet"))
        sensor_agg_tables.append(pd.read_parquet(shard_root / "sensor_subject_features.parquet"))
        
        if include_pooled:
            pooled_epoch_tables.append(pd.read_parquet(shard_root / "pooled_epoch_features.parquet"))
            pooled_agg_tables.append(pd.read_parquet(shard_root / "pooled_subject_features.parquet"))
            
        failure_tables.append(pd.read_csv(shard_root / "failures.csv"))

        # Harvest feature column metadata from the first shard
        if feature_cols["sensor_epoch"] is None:
            feature_cols["sensor_epoch"] = json.loads((shard_root / "sensor_epoch_features_feature_columns.json").read_text(encoding="utf-8"))
        if feature_cols["sensor_subject"] is None:
            feature_cols["sensor_subject"] = json.loads((shard_root / "sensor_subject_features_feature_columns.json").read_text(encoding="utf-8"))
        
        if include_pooled:
            if feature_cols["pooled_epoch"] is None:
                feature_cols["pooled_epoch"] = json.loads((shard_root / "pooled_epoch_features_feature_columns.json").read_text(encoding="utf-8"))
            if feature_cols["pooled_subject"] is None:
                feature_cols["pooled_subject"] = json.loads((shard_root / "pooled_subject_features_feature_columns.json").read_text(encoding="utf-8"))

    combined_root = derivative_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Combining sensor families...")
    save_table(
        pd.concat(sensor_epoch_tables, ignore_index=True),
        combined_root / "sensor_epoch_features",
        feature_columns=feature_cols["sensor_epoch"],
    )
    save_table(
        pd.concat(sensor_agg_tables, ignore_index=True),
        combined_root / "sensor_subject_features",
        feature_columns=feature_cols["sensor_subject"],
    )

    if include_pooled:
        LOGGER.info("Combining pooled families...")
        save_table(
            pd.concat(pooled_epoch_tables, ignore_index=True),
            combined_root / "pooled_epoch_features",
            feature_columns=feature_cols["pooled_epoch"],
        )
        save_table(
            pd.concat(pooled_agg_tables, ignore_index=True),
            combined_root / "pooled_subject_features",
            feature_columns=feature_cols["pooled_subject"],
        )

    pd.concat(failure_tables, ignore_index=True).to_csv(combined_root / "failures.csv", index=False)
    
    LOGGER.info("Merged %d shards into %s", len(shard_roots), combined_root)


if __name__ == "__main__":
    main()

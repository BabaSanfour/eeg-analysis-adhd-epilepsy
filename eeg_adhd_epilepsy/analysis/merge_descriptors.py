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
    parser.add_argument(
        "--derivative_root",
        default=None,
        help=(
            "Optional descriptor derivative root. Defaults to "
            "<bids_root>/derivatives/signal_features/descriptors."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    bids_root = Path(args.bids_root)
    if args.derivative_root is not None:
        derivative_root = Path(args.derivative_root)
    else:
        derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"

    with (derivative_root / "config_used.yaml").open("r", encoding="utf-8") as handle:
        config_used = yaml.safe_load(handle) or {}
    include_pooled = bool((config_used.get("pooling") or {}).get("channel_groups"))

    shard_roots: list[Path] = []
    for success_path in sorted(derivative_root.glob("sub-*/eeg/*/_SUCCESS")):
        shard_root = success_path.parent
        if all(
            (shard_root / filename).exists()
            for filename in required_descriptor_files(include_pooled)
        ):
            shard_roots.append(shard_root)

    if not shard_roots:
        raise ValueError(f"No completed descriptor shards found under {derivative_root}.")

    sensor_epoch_tables: list[pd.DataFrame] = []
    sensor_agg_tables: list[pd.DataFrame] = []
    pooled_epoch_tables: list[pd.DataFrame] = []
    pooled_agg_tables: list[pd.DataFrame] = []
    failure_tables: list[pd.DataFrame] = []
    sensor_epoch_feature_columns: list[str] | None = None
    sensor_agg_feature_columns: list[str] | None = None
    pooled_epoch_feature_columns: list[str] | None = None
    pooled_agg_feature_columns: list[str] | None = None

    for shard_root in shard_roots:
        sensor_epoch_tables.append(
            pd.read_parquet(shard_root / "sensor_epoch_features.parquet")
        )
        sensor_agg_tables.append(
            pd.read_parquet(shard_root / "sensor_subject_features.parquet")
        )
        if include_pooled:
            pooled_epoch_tables.append(
                pd.read_parquet(shard_root / "pooled_epoch_features.parquet")
            )
            pooled_agg_tables.append(
                pd.read_parquet(shard_root / "pooled_subject_features.parquet")
            )
        failure_tables.append(pd.read_csv(shard_root / "failures.csv"))
        if sensor_epoch_feature_columns is None:
            sensor_epoch_feature_columns = json.loads(
                (shard_root / "sensor_epoch_features_feature_columns.json").read_text(
                    encoding="utf-8"
                )
            )
        if sensor_agg_feature_columns is None:
            sensor_agg_feature_columns = json.loads(
                (shard_root / "sensor_subject_features_feature_columns.json").read_text(
                    encoding="utf-8"
                )
            )
        if include_pooled and pooled_epoch_feature_columns is None:
            pooled_epoch_feature_columns = json.loads(
                (shard_root / "pooled_epoch_features_feature_columns.json").read_text(
                    encoding="utf-8"
                )
            )
        if include_pooled and pooled_agg_feature_columns is None:
            pooled_agg_feature_columns = json.loads(
                (shard_root / "pooled_subject_features_feature_columns.json").read_text(
                    encoding="utf-8"
                )
            )

    combined_root = derivative_root / "combined"
    save_table(
        pd.concat(sensor_epoch_tables, ignore_index=True),
        combined_root / "sensor_epoch_features",
        feature_columns=sensor_epoch_feature_columns,
    )
    save_table(
        pd.concat(sensor_agg_tables, ignore_index=True),
        combined_root / "sensor_subject_features",
        feature_columns=sensor_agg_feature_columns,
    )
    if include_pooled:
        save_table(
            pd.concat(pooled_epoch_tables, ignore_index=True),
            combined_root / "pooled_epoch_features",
            feature_columns=pooled_epoch_feature_columns,
        )
        save_table(
            pd.concat(pooled_agg_tables, ignore_index=True),
            combined_root / "pooled_subject_features",
            feature_columns=pooled_agg_feature_columns,
        )
    combined_root.mkdir(parents=True, exist_ok=True)
    pd.concat(failure_tables, ignore_index=True).to_csv(
        combined_root / "failures.csv",
        index=False,
    )
    LOGGER.info("Merged %d descriptor shards into %s", len(shard_roots), combined_root)


if __name__ == "__main__":
    main()

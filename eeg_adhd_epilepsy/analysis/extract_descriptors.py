"""
End-to-end feature extraction from EEG epochs.

This script loads saved epoched derivatives, extracts broad EEG descriptors,
and writes checkpointed per-subject outputs under a derivative-style root.

Outputs
-------
For each processed subject-condition pair, the script writes a shard under
``<derivative_root>/sub-<subject>/eeg/<condition>/`` containing:

- a sensor descriptor bundle (`.npz`)
- a sensor epoch-level feature table (`.parquet` and `.csv`)
- a sensor subject-level aggregated feature table (`.parquet` and `.csv`)
- optionally, pooled epoch-level and pooled subject-level feature tables
- a failures table (`.csv`)
- a `_SUCCESS` marker for resume-safe checkpointing

Combined outputs are built separately by ``merge_descriptors.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from coco_pipe.descriptors import DescriptorConfig, DescriptorPipeline
from coco_pipe.io import DataContainer
from eeg_adhd_epilepsy.analysis.utils import (
    required_descriptor_files,
    save_table,
)

from eeg_adhd_epilepsy.io.bids import load_eeg_data
from eeg_adhd_epilepsy.io.patients import (
    clean_patients_df,
    load_raw_patients_df,
    validate_bids_coverage,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONDITIONS = [
    "EO_baseline",
    "EC_baseline",
    "HV_EO",
    "HV_EC",
    "PostHV_EO",
    "PostHV_EC",
    "PHOTO_EO",
    "PHOTO_EC",
]
def _build_feature_outputs(
    result: dict[str, Any],
    metadata_df: pd.DataFrame,
    condition: str,
    target_col: str | None,
    aggregation_descriptors: list[dict[str, Any]],
    aggregated_ratio_pairs: list[tuple[str, str]],
    aggregated_ratio_floor: float,
) -> dict[str, Any]:
    epoch_feature_df = pd.DataFrame(result["X"], columns=result["descriptor_names"])
    epoch_df = pd.concat(
        [metadata_df.reset_index(drop=True), epoch_feature_df.reset_index(drop=True)],
        axis=1,
    )

    coords: dict[str, Any] = {
        "obs": metadata_df["obs_id"].to_numpy(dtype=object),
        "feature": np.asarray(result["descriptor_names"], dtype=object),
    }
    for column in metadata_df.columns:
        if column != "obs_id":
            coords[column] = metadata_df[column].to_numpy(dtype=object)
    y = (
        metadata_df[target_col].to_numpy(dtype=object)
        if target_col and target_col in metadata_df.columns
        else None
    )
    feature_container = DataContainer(
        X=result["X"],
        y=y,
        ids=metadata_df["obs_id"].to_numpy(dtype=object),
        dims=("obs", "feature"),
        coords=coords,
        meta={},
    )

    grouped_mean = feature_container.aggregate(
        by="subject",
        stats="mean",
        min_count=1,
        on_insufficient="raise",
    )
    agg_metadata_df = grouped_mean.obs_table(
        include_y=bool(target_col),
        y_col=target_col or "y",
    )
    if "subject" not in agg_metadata_df.columns:
        raise ValueError(
            "Aggregated feature container must expose a 'subject' coordinate."
        )
    agg_metadata_df["condition"] = condition
    agg_front = ["subject", "condition", "epoch_count"]
    agg_metadata_df = agg_metadata_df[
        agg_front + [column for column in agg_metadata_df.columns if column not in agg_front]
    ]

    base_agg_feature_df = pd.DataFrame(
        grouped_mean.X,
        columns=list(np.asarray(grouped_mean.coords["feature"], dtype=object)),
    )
    grouped_features = feature_container.aggregate_groups(
        by="subject",
        groups=aggregation_descriptors,
        min_count=1,
        on_insufficient="raise",
    )
    agg_feature_frames = [
        pd.DataFrame(
            grouped_features.X,
            columns=list(np.asarray(grouped_features.coords["feature"], dtype=object)),
        )
    ]
    agg_ratio_columns: dict[str, np.ndarray] = {}
    for numerator, denominator in aggregated_ratio_pairs:
        raw_numerator_prefix = f"band_abs_{numerator}_"
        corr_numerator_prefix = f"band_corr_abs_{numerator}_"
        for numerator_column in base_agg_feature_df.columns:
            if numerator_column.startswith(raw_numerator_prefix):
                suffix = numerator_column.removeprefix(raw_numerator_prefix)
                denominator_column = f"band_abs_{denominator}_{suffix}"
                output_name = f"agg_band_ratio_{numerator}_{denominator}_{suffix}"
            elif numerator_column.startswith(corr_numerator_prefix):
                suffix = numerator_column.removeprefix(corr_numerator_prefix)
                denominator_column = f"band_corr_abs_{denominator}_{suffix}"
                output_name = f"agg_band_corr_ratio_{numerator}_{denominator}_{suffix}"
            else:
                continue
            if denominator_column not in base_agg_feature_df.columns:
                continue
            numerator_values = base_agg_feature_df[numerator_column].to_numpy(dtype=float)
            denominator_values = base_agg_feature_df[denominator_column].to_numpy(dtype=float)
            agg_ratio_columns[output_name] = np.divide(
                numerator_values,
                denominator_values,
                out=np.full_like(numerator_values, np.nan, dtype=float),
                where=denominator_values > aggregated_ratio_floor,
            )
    if agg_ratio_columns:
        agg_feature_frames.append(
            pd.DataFrame(agg_ratio_columns, index=base_agg_feature_df.index)
        )
    agg_feature_df = pd.concat(agg_feature_frames, axis=1)
    agg_df = pd.concat(
        [agg_metadata_df.reset_index(drop=True), agg_feature_df.reset_index(drop=True)],
        axis=1,
    )
    return {
        "epoch_df": epoch_df,
        "subject_df": agg_df,
        "epoch_feature_columns": list(result["descriptor_names"]),
        "subject_feature_columns": list(agg_feature_df.columns),
    }


def _build_failure_df(
    result: dict[str, Any],
    metadata_df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    if not result["failures"]:
        return pd.DataFrame(
            columns=[
                "condition",
                "subject",
                "obs_id",
                "obs_index",
                "channel_index",
                "channel_name",
                "family",
                "exception_type",
                "message",
            ]
        )

    failure_df = pd.DataFrame(result["failures"])
    failure_df["obs_id"] = failure_df["obs_id"].astype(str)
    failure_df = failure_df.merge(
        metadata_df[["obs_id", "subject"]].drop_duplicates(),
        how="left",
        on="obs_id",
    )
    failure_df["condition"] = condition
    failure_front = [
        "condition",
        "subject",
        "obs_id",
        "obs_index",
        "channel_index",
        "channel_name",
        "family",
        "exception_type",
        "message",
    ]
    return failure_df[
        failure_front
        + [column for column in failure_df.columns if column not in failure_front]
    ]


def _save_subject_shard(
    shard_root: Path,
    sensor_result: dict[str, Any],
    sensor_outputs: dict[str, Any],
    pooled_outputs: dict[str, Any] | None,
    failure_df: pd.DataFrame,
) -> None:
    shard_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_root / "sensor_descriptor_bundle.npz",
        X=sensor_result["X"],
        descriptor_names=np.asarray(sensor_result["descriptor_names"], dtype=object),
        failures=np.asarray(sensor_result["failures"], dtype=object),
    )
    save_table(
        sensor_outputs["epoch_df"],
        shard_root / "sensor_epoch_features",
        feature_columns=sensor_outputs["epoch_feature_columns"],
    )
    save_table(
        sensor_outputs["subject_df"],
        shard_root / "sensor_subject_features",
        feature_columns=sensor_outputs["subject_feature_columns"],
    )
    if pooled_outputs is not None:
        save_table(
            pooled_outputs["epoch_df"],
            shard_root / "pooled_epoch_features",
            feature_columns=pooled_outputs["epoch_feature_columns"],
        )
        save_table(
            pooled_outputs["subject_df"],
            shard_root / "pooled_subject_features",
            feature_columns=pooled_outputs["subject_feature_columns"],
        )
    failure_df.to_csv(shard_root / "failures.csv", index=False)
    (shard_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract coco-pipe EEG descriptors from saved epoch derivatives."
    )
    parser.add_argument(
        "--bids_root",
        required=True,
        help="Path to the BIDS dataset root.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to the patient metadata CSV.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "descriptors.yaml"),
        help="Path to descriptor YAML config.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
        help="Conditions to extract.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific BIDS subjects to process.",
    )
    parser.add_argument(
        "--subject_col",
        default="Study ID",
        help="Subject identifier column in metadata.",
    )
    parser.add_argument(
        "--target_col",
        default=None,
        help="Optional label column to also expose as container y during aggregation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    bids_root = Path(args.bids_root)
    metadata_path = Path(args.metadata)
    config_path = Path(args.config)
    derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    aggregation_config = raw_config.pop("aggregation", None) or {}
    pooling_config = raw_config.pop("pooling", None) or {}
    aggregation_descriptors = aggregation_config.get("descriptors")
    if not aggregation_descriptors:
        raise ValueError("Config must define aggregation.descriptors.")
    channel_groups = pooling_config.get("channel_groups")
    include_pooled = bool(channel_groups)
    config_snapshot = deepcopy(raw_config)
    config_snapshot["aggregation"] = deepcopy(aggregation_config)
    if pooling_config:
        config_snapshot["pooling"] = deepcopy(pooling_config)
    derivative_root.mkdir(parents=True, exist_ok=True)
    config_used_path = derivative_root / "config_used.yaml"
    config_text = yaml.safe_dump(config_snapshot, sort_keys=True)
    if config_used_path.exists() and config_used_path.read_text(encoding="utf-8") != config_text:
        raise ValueError(
            "Existing descriptor derivative root was generated with a different "
            "configuration. Clear the derivative root."
        )
    config_used_path.write_text(config_text, encoding="utf-8")

    dataset_description = {
        "Name": "Signal Features",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "eeg-descriptors",
                "Description": "Checkpointed coco-pipe descriptor feature extraction",
            }
        ],
        "SourceDatasets": [{"URL": bids_root.resolve().as_uri()}],
    }
    (derivative_root / "dataset_description.json").write_text(
        json.dumps(dataset_description, indent=2),
        encoding="utf-8",
    )

    descriptor_config = DescriptorConfig.model_validate(raw_config)
    pipeline = DescriptorPipeline(descriptor_config)
    if descriptor_config.families.bands.enabled:
        aggregated_ratio_pairs = list(descriptor_config.families.bands.ratio_pairs)
        aggregated_ratio_floor = float(descriptor_config.families.bands.min_denominator_power)
    else:
        aggregated_ratio_pairs = []
        aggregated_ratio_floor = 0.0

    coverage_root = bids_root / "derivatives" / "preproc"
    raw_meta_df = load_raw_patients_df(metadata_path)
    coverage = validate_bids_coverage(
        raw_meta_df,
        coverage_root,
        desc=None,
        suffix="epo",
        subject_col=args.subject_col,
    )
    available_subjects = list(coverage["present_subjects"])
    filtered_meta = raw_meta_df[
        raw_meta_df[args.subject_col]
        .map(lambda value: f"{int(value):04d}")
        .isin(available_subjects)
    ].copy()
    meta_df, meta_stats = clean_patients_df(filtered_meta)
    LOGGER.info(
        "Metadata cleaning stats: potential_dropped=%s, mismatches_dropped=%s, duplicates_dropped=%s",
        meta_stats.get("n_potential_dropped", 0),
        meta_stats.get("n_mismatches_dropped", 0),
        meta_stats.get("n_duplicates_dropped", 0),
    )
    valid_subjects = set(
        meta_df[args.subject_col].map(lambda value: f"{int(value):04d}")
    )
    available_subjects = [
        subject for subject in available_subjects if subject in valid_subjects
    ]

    available_subject_set = set(available_subjects)
    if args.subjects:
        requested_subjects = [f"{int(subject):04d}" for subject in args.subjects]
        subjects = [
            subject for subject in requested_subjects if subject in available_subject_set
        ]
        missing_subjects = [
            subject for subject in requested_subjects if subject not in available_subject_set
        ]
        if missing_subjects:
            LOGGER.warning(
                "Skipping %d requested subjects with no saved derivatives: %s",
                len(missing_subjects),
                ", ".join(missing_subjects),
            )
    else:
        subjects = available_subjects

    if not subjects:
        raise ValueError(
            "No matching saved-derivative subjects were found for descriptor extraction."
        )
    LOGGER.info("Using %d subjects from saved derivatives.", len(subjects))

    for subject in subjects:
        for condition in args.conditions:
            shard_root = derivative_root / f"sub-{subject}" / "eeg" / condition
            if all(
                (shard_root / filename).exists()
                for filename in required_descriptor_files(include_pooled)
            ):
                LOGGER.info(
                    "Skipping %s / %s: checkpoint shard already complete.",
                    condition,
                    subject,
                )
                continue

            subject_meta_df = meta_df[
                meta_df[args.subject_col].map(lambda value: f"{int(value):04d}")
                == subject
            ].copy()

            LOGGER.info("Loading condition %s for subject %s", condition, subject)
            try:
                dc_loaded = load_eeg_data(
                    bids_root=bids_root,
                    use_derivatives=True,
                    subjects=[subject],
                    metadata_df=subject_meta_df,
                    subject_col=args.subject_col,
                    target_col=args.target_col,
                    desc="base",
                    condition=condition,
                )
            except RuntimeError as error:
                if str(error).startswith("No valid data found in "):
                    LOGGER.info(
                        "Skipping %s / %s: no saved epochs found for this condition.",
                        condition,
                        subject,
                    )
                    continue
                raise
            channel_names = [
                str(name) for name in np.asarray(dc_loaded.coords["channel"], dtype=object)
            ]
            sfreq = float(dc_loaded.meta["sfreq"])
            if dc_loaded.ids is None:
                raise ValueError(
                    "Loaded EEG container must expose observation ids in `container.ids`."
                )
            ids = np.asarray(dc_loaded.ids, dtype=object)
            if ids.size == 0:
                raise ValueError("Loaded EEG container did not include any epochs.")

            metadata_df = dc_loaded.obs_table(
                include_ids=True,
                include_y=bool(args.target_col),
                y_col=args.target_col or "y",
            )
            metadata_df["obs_id"] = metadata_df["obs_id"].astype(str)
            metadata_df["condition"] = condition
            metadata_df["subject"] = subject
            metadata_front = ["obs_id", "subject", "condition"]
            metadata_df = metadata_df[
                metadata_front
                + [column for column in metadata_df.columns if column not in metadata_front]
            ]
            sensor_result = pipeline.extract(
                X=dc_loaded.X,
                ids=ids,
                sfreq=sfreq,
                channel_names=channel_names,
            )
            sensor_outputs = _build_feature_outputs(
                sensor_result,
                metadata_df,
                condition,
                args.target_col,
                aggregation_descriptors,
                aggregated_ratio_pairs,
                aggregated_ratio_floor,
            )
            pooled_outputs = None
            if include_pooled:
                pooled_result = pipeline.pool_channels(sensor_result, channel_groups)
                pooled_outputs = _build_feature_outputs(
                    pooled_result,
                    metadata_df,
                    condition,
                    args.target_col,
                    aggregation_descriptors,
                    aggregated_ratio_pairs,
                    aggregated_ratio_floor,
                )
            failure_df = _build_failure_df(sensor_result, metadata_df, condition)
            _save_subject_shard(
                shard_root,
                sensor_result,
                sensor_outputs,
                pooled_outputs,
                failure_df,
            )
            LOGGER.info(
                "%s / %s: saved %d epoch rows, %d subject rows, %d failures",
                condition,
                subject,
                len(sensor_outputs["epoch_df"]),
                len(sensor_outputs["subject_df"]),
                len(failure_df),
            )

    LOGGER.info("Derivative feature root: %s", derivative_root)


if __name__ == "__main__":
    main()

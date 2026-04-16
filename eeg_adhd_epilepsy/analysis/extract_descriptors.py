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
import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml

from coco_pipe.descriptors import DescriptorConfig, DescriptorPipeline
from coco_pipe.io import DataContainer
from eeg_adhd_epilepsy.analysis.utils import (
    required_descriptor_files,
    save_table,
)
from eeg_adhd_epilepsy.io.bids import (
    load_eeg_data,
    normalize_subject_id,
    parse_bids_components,
    validate_bids_coverage,
)
from eeg_adhd_epilepsy.io.csv import load as load_csv
from eeg_adhd_epilepsy.utils.config import DEFAULT_ANALYSIS_CONDITIONS

LOGGER = logging.getLogger(__name__)
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)
def _build_feature_outputs(
    result: dict[str, Any],
    metadata_df: pd.DataFrame,
    condition: str,
    target_col: str | None,
    aggregation_descriptors: list[dict[str, Any]],
    aggregated_ratio_pairs: list[tuple[str, str]],
    aggregated_ratio_floor: float,
) -> dict[str, Any]:
    # Extract feature columns and merge with metadata for epoch-level DF
    epoch_feature_df = pd.DataFrame(result["X"], columns=result["descriptor_names"])
    epoch_df = pd.concat([metadata_df.reset_index(drop=True), epoch_feature_df], axis=1)

    # Wrap in DataContainer for aggregation utilities
    coords = {
        col: metadata_df[col].to_numpy(dtype=object)
        for col in metadata_df.columns if col != "obs_id"
    }
    coords["feature"] = np.asarray(result["descriptor_names"], dtype=object)
    
    y = metadata_df[target_col].to_numpy() if target_col and target_col in metadata_df.columns else None
    
    container = DataContainer(
        X=result["X"],
        y=y,
        ids=metadata_df["obs_id"].to_numpy(dtype=object),
        dims=("obs", "feature"),
        coords=coords,
    )

    # Calculate subject-level means
    grouped_mean = container.aggregate(by="subject", stats="mean")
    agg_df = grouped_mean.obs_table(include_y=bool(target_col), y_col=target_col or "y")
    agg_df["condition"] = condition
    
    # Calculate additional grouped descriptors (e.g., medians)
    grouped_features = container.aggregate_groups(by="subject", groups=aggregation_descriptors)
    base_agg_features = pd.DataFrame(grouped_mean.X, columns=grouped_mean.coords["feature"])
    agg_features = pd.DataFrame(grouped_features.X, columns=grouped_features.coords["feature"])

    # Calculate ratios on the aggregated features
    agg_ratio_columns = {}
    for num, den in aggregated_ratio_pairs:
        for p in ["band_abs_", "band_corr_abs_"]:
            out_p = "agg_band_ratio_" if p == "band_abs_" else "agg_band_corr_ratio_"
            for col in base_agg_features.columns:
                if col.startswith(f"{p}{num}_"):
                    suffix = col.removeprefix(f"{p}{num}_")
                    den_col = f"{p}{den}_{suffix}"
                    if den_col in base_agg_features.columns:
                        n_vals = base_agg_features[col].to_numpy(dtype=float)
                        d_vals = base_agg_features[den_col].to_numpy(dtype=float)
                        agg_ratio_columns[f"{out_p}{num}_{den}_{suffix}"] = np.divide(
                            n_vals, d_vals, out=np.full_like(n_vals, np.nan), where=d_vals > aggregated_ratio_floor
                        )

    if agg_ratio_columns:
        agg_features = pd.concat([agg_features, pd.DataFrame(agg_ratio_columns)], axis=1)

    # Final subject DF assembly
    subject_df = pd.concat([agg_df.reset_index(drop=True), agg_features], axis=1)
    
    return {
        "epoch_df": epoch_df,
        "subject_df": subject_df,
        "epoch_feature_columns": list(result["descriptor_names"]),
        "subject_feature_columns": list(agg_features.columns),
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
        help="Path to the canonical metadata CSV.",
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
        default="study_id",
        help="Subject identifier column in cleaned metadata.",
    )
    parser.add_argument(
        "--target_col",
        default=None,
        help="Optional canonical metadata column to also expose as container y during aggregation.",
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
    meta_df = load_csv(str(metadata_path), sep=None)
    coverage = validate_bids_coverage(
        meta_df,
        coverage_root,
        desc=None,
        suffix="epo",
        subject_col=args.subject_col,
    )
    available_subjects = list(coverage["present_subjects"])
    meta_df = meta_df[
        meta_df[args.subject_col]
        .map(lambda value: f"{int(value):04d}")
        .isin(available_subjects)
    ].copy()
    valid_subjects = set(
        meta_df[args.subject_col].map(lambda value: f"{int(value):04d}")
    )
    available_subjects = [
        subject for subject in available_subjects if subject in valid_subjects
    ]

    available_subject_set = set(available_subjects)
    if args.subjects:
        requested_subjects = [
            normalize_subject_id(subject).replace("sub-", "")
            for subject in args.subjects
        ]
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
        epochs_root = bids_root / "derivatives" / "preproc"
        files = list(epochs_root.rglob(f"sub-{subject}*_desc-base_epo.fif"))
        
        sessions = sorted(list({parse_bids_components(f).get("session", "01") for f in files}))
        if not sessions:
            sessions = ["01"]

        if args.conditions == ["all"]:
            subject_conditions = set()
            for f in files:
                try:
                    subject_conditions.update(mne.read_epochs(f, verbose="ERROR").event_id.keys())
                except Exception as e:
                    LOGGER.debug("Failed to read conditions from %s: %s", f.name, e)
            subject_conditions = sorted(list(subject_conditions))
            if not subject_conditions:
                LOGGER.warning("No saved conditions found for subject %s", subject)
                continue
        else:
            subject_conditions = args.conditions

        for session, condition in itertools.product(sessions, subject_conditions):
            shard_root = derivative_root / f"sub-{subject}" / f"ses-{session}" / "eeg" / condition
            if (shard_root / "_SUCCESS").exists():
                LOGGER.info("Skipping %s / %s (ses %s): already complete.", condition, subject, session)
                continue

            subject_meta_df = meta_df[
                meta_df[args.subject_col].map(lambda v: f"{int(v):04d}") == subject
            ].copy()

            LOGGER.info("Loading %s for %s (ses %s)", condition, subject, session)
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
                    session=session,
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

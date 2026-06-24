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
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml
from coco_pipe.descriptors import (
    DescriptorConfig,
    DescriptorPipeline,
    build_descriptor_tables,
    mad_failures_from_qc,
    save_descriptor_table,
)
from coco_pipe.io import DataContainer, read_table, save_npz, write_json
from coco_pipe.io.quality import drop_epoch_outliers

from eeg_adhd_epilepsy.analysis.utils.descriptor_shards import required_descriptor_files
from eeg_adhd_epilepsy.io.bids import (
    add_recording_id,
    bids_session_label,
    bids_subject_label,
    parse_bids_components,
    study_id_to_bids_subject,
)
from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    subject_report_dir,
)
from eeg_adhd_epilepsy.qc.descriptor_qc import run_descriptor_subject_qc
from eeg_adhd_epilepsy.utils.constants import DEFAULT_ANALYSIS_CONDITIONS
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)


def _shard_complete(
    shard_root: Path,
    include_pooled: bool,
    reports_root: Path,
    subject: str,
    session: str,
    condition: str,
) -> bool:
    required_paths = [
        shard_root / relative_path
        for relative_path in required_descriptor_files(include_pooled, include_qc=True)
    ]
    required_paths.append(
        subject_report_dir(
            reports_root=reports_root,
            subject=subject,
            session=session,
            stage=ReportStage.DESCRIPTOR_QC,
        )
        / (
            f"{bids_subject_label(subject)}_{bids_session_label(session)}_"
            f"{condition}_descriptor_qc_report.html"
        )
    )
    return all(path.exists() for path in required_paths)


def _apply_mad_rejection(
    container: DataContainer,
    metadata_df: pd.DataFrame,
    condition: str,
    subject: str,
    mad_threshold: float = 10.0,
    fraction_thresh: float = 0.05,
    min_epochs: int = 5,
    group_by: str | None = None,
) -> tuple[DataContainer, pd.DataFrame]:
    """Drop epochs where too large a fraction of features are MAD outliers.

    Delegates MAD scoring/masking to
    :func:`coco_pipe.io.quality.drop_epoch_outliers` and records the dropped
    epochs via :func:`coco_pipe.descriptors.mad_failures_from_qc`. A
    ``RuntimeError`` is raised if fewer than ``min_epochs`` would survive.
    """
    if container.X.shape[0] == 0:
        return container, metadata_df

    clean_or_masks, qc_result = drop_epoch_outliers(
        container,
        z_threshold=mad_threshold,
        outlier_fraction_threshold=fraction_thresh,
        group_by=group_by,
        min_obs=min_epochs,
    )
    if isinstance(clean_or_masks, dict):
        keep_mask = np.logical_and.reduce(list(clean_or_masks.values()))
        clean = container.isel(obs=np.flatnonzero(keep_mask).tolist())
    else:
        clean = clean_or_masks
        clean_ids = set(np.asarray(clean.ids, dtype=object).astype(str))
        keep_mask = np.isin(
            metadata_df["obs_id"].astype(str).to_numpy(),
            list(clean_ids),
        )
    remaining_count = int(keep_mask.sum())
    if remaining_count < min_epochs:
        raise RuntimeError(
            f"Only {remaining_count} epoch(s) remain after MAD rejection "
            f"(minimum required: {min_epochs})."
        )

    bad_count = int((~keep_mask).sum())
    if bad_count > 0:
        clean.meta["failures"] = list(clean.meta.get("failures", [])) + (
            mad_failures_from_qc(qc_result)
        )
        LOGGER.info(
            "MAD Rejection: dropped %d / %d epochs for %s / %s "
            "(> %.1f%% of features exceeded MAD=%.1f)",
            bad_count,
            container.X.shape[0],
            condition,
            subject,
            fraction_thresh * 100,
            mad_threshold,
        )

    return clean, metadata_df[keep_mask].reset_index(drop=True)


def _build_failure_df(
    failures: list[dict[str, Any]],
    metadata_df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    if not failures:
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

    failure_df = pd.DataFrame(failures)
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
        failure_front + [column for column in failure_df.columns if column not in failure_front]
    ]


def _save_subject_shard(
    shard_root: Path,
    sensor_container: DataContainer,
    sensor_outputs: dict[str, Any],
    pooled_outputs: dict[str, Any] | None,
    failure_df: pd.DataFrame,
) -> None:
    shard_root.mkdir(parents=True, exist_ok=True)
    save_npz(
        shard_root / "sensor_descriptor_bundle.npz",
        X=sensor_container.X,
        descriptor_names=np.asarray(sensor_container.coords["feature"], dtype=object),
        failures=np.asarray(sensor_container.meta.get("failures", []), dtype=object),
    )
    save_descriptor_table(
        sensor_outputs["epoch_df"],
        shard_root / "sensor_epoch_features",
        feature_columns=sensor_outputs["epoch_feature_columns"],
        formats=("parquet", "csv"),
    )
    save_descriptor_table(
        sensor_outputs["subject_df"],
        shard_root / "sensor_subject_features",
        feature_columns=sensor_outputs["subject_feature_columns"],
        formats=("parquet", "csv"),
    )
    if pooled_outputs is not None:
        save_descriptor_table(
            pooled_outputs["epoch_df"],
            shard_root / "pooled_epoch_features",
            feature_columns=pooled_outputs["epoch_feature_columns"],
            formats=("parquet", "csv"),
        )
        save_descriptor_table(
            pooled_outputs["subject_df"],
            shard_root / "pooled_subject_features",
            feature_columns=pooled_outputs["subject_feature_columns"],
            formats=("parquet", "csv"),
        )
    failure_df.to_csv(shard_root / "failures.csv", index=False)


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
        "--metadata_row",
        type=int,
        default=None,
        help=(
            "One-based row number in the metadata CSV to process. "
            "Useful for Slurm job arrays where SLURM_ARRAY_TASK_ID selects a metadata row."
        ),
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
    parser.add_argument(
        "--derivative_root",
        default=None,
        help=(
            "Custom root directory for descriptor derivatives "
            "(defaults to bids_root/derivatives/signal_features/descriptors)."
        ),
    )
    parser.add_argument(
        "--reports_root",
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root).",
    )
    args = parser.parse_args()
    if args.metadata_row is not None and args.subjects:
        raise ValueError("--metadata_row and --subjects are mutually exclusive.")
    if args.metadata_row is not None and args.metadata_row < 1:
        raise ValueError("--metadata_row is one-based and must be >= 1.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    bids_root = Path(args.bids_root).expanduser()
    reports_root = (
        Path(args.reports_root).expanduser()
        if args.reports_root
        else default_reports_root(bids_root)
    )
    reports_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata).expanduser()
    config_path = Path(args.config).expanduser()
    if args.derivative_root:
        derivative_root = Path(args.derivative_root).expanduser()
    else:
        derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"

    raw_config = load_yaml_config(config_path)
    aggregation_config = raw_config.pop("aggregation", None) or {}
    pooling_config = raw_config.pop("pooling", None) or {}
    qc_config = raw_config.pop("qc", None) or {}
    aggregation_descriptors = aggregation_config.get("descriptors")
    if not aggregation_descriptors:
        raise ValueError("Config must define aggregation.descriptors.")
    channel_groups = pooling_config.get("channel_groups")
    include_pooled = bool(channel_groups)
    config_snapshot = deepcopy(raw_config)
    config_snapshot["aggregation"] = deepcopy(aggregation_config)
    if pooling_config:
        config_snapshot["pooling"] = deepcopy(pooling_config)
    if qc_config:
        config_snapshot["qc"] = deepcopy(qc_config)
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
    write_json(derivative_root / "dataset_description.json", dataset_description)

    descriptor_config = DescriptorConfig.model_validate(raw_config)
    pipeline = DescriptorPipeline(descriptor_config)
    if descriptor_config.families.bands.enabled:
        aggregated_ratio_pairs = list(descriptor_config.families.bands.ratio_pairs)
        aggregated_ratio_floor = float(descriptor_config.families.bands.min_denominator_power)
    else:
        aggregated_ratio_pairs = []
        aggregated_ratio_floor = 0.0

    meta_df = read_table(metadata_path, sep=None)
    valid_subjects = set(meta_df[args.subject_col].map(lambda value: f"{int(value):04d}"))
    row_requested_subjects: list[str] | None = None
    if args.metadata_row is not None:
        row_position = args.metadata_row - 1
        if row_position >= len(meta_df):
            LOGGER.warning(
                "Metadata row %d is outside the metadata table with %d rows; nothing to do.",
                args.metadata_row,
                len(meta_df),
            )
            return
        row_value = meta_df.iloc[row_position][args.subject_col]
        row_requested_subjects = [f"{int(row_value):04d}"]
        LOGGER.info(
            "Resolved metadata row %d to %s=%s.",
            args.metadata_row,
            args.subject_col,
            row_requested_subjects[0],
        )

    if row_requested_subjects is not None:
        subjects = [subject for subject in row_requested_subjects if subject in valid_subjects]
    elif args.subjects:
        requested_subjects = [study_id_to_bids_subject(subject) for subject in args.subjects]
        subjects = [subject for subject in requested_subjects if subject in valid_subjects]
    else:
        subjects = sorted(list(valid_subjects))

    if not subjects:
        raise ValueError(
            "No matching saved-derivative subjects were found for descriptor extraction."
        )
    LOGGER.info("Using %d subjects from saved derivatives.", len(subjects))

    for subject in subjects:
        epochs_root = bids_root / "derivatives" / "preproc"
        files = list(epochs_root.rglob(f"{bids_subject_label(subject)}*_desc-base_epo.fif"))

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
            shard_root = (
                derivative_root
                / bids_subject_label(subject)
                / bids_session_label(session)
                / "eeg"
                / condition
            )
            if _shard_complete(
                shard_root, include_pooled, reports_root, subject, session, condition
            ):
                LOGGER.info(
                    "Skipping %s / %s (ses %s): already complete.", condition, subject, session
                )
                continue

            subject_meta_df = meta_df[
                meta_df[args.subject_col].map(lambda v: f"{int(v):04d}") == subject
            ].copy()

            LOGGER.info("Loading %s for %s (ses %s)", condition, subject, session)
            try:
                dc_loaded = build_container(
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

            dc_loaded = add_recording_id(dc_loaded)
            metadata_df = dc_loaded.obs_table(
                include_ids=True,
                include_y=bool(args.target_col),
                y_col=args.target_col or "y",
            )
            metadata_df["obs_id"] = metadata_df["obs_id"].astype(str)
            metadata_df["condition"] = condition
            metadata_df["subject"] = subject
            metadata_front = ["obs_id", "subject", "session", "run", "recording_id", "condition"]
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

            outlier_config = qc_config.get("outlier", {})
            if qc_config:
                mad_threshold = float(outlier_config.get("z_threshold", 5.0))
                fraction_threshold = float(outlier_config.get("epoch_outlier_fraction", 0.30))
                min_epochs = int(qc_config.get("min_obs", 5))
                group_by = outlier_config.get("group_by", "family")
            else:
                mad_threshold = 10.0
                fraction_threshold = 0.05
                min_epochs = 5
                group_by = None
            failure_metadata_df = metadata_df.copy()
            try:
                sensor_result, metadata_df = _apply_mad_rejection(
                    sensor_result,
                    metadata_df,
                    condition,
                    subject,
                    mad_threshold=mad_threshold,
                    fraction_thresh=fraction_threshold,
                    min_epochs=min_epochs,
                    group_by=group_by,
                )
            except RuntimeError as error:
                LOGGER.warning("Skipping %s / %s: %s", condition, subject, str(error))
                continue

            sensor_outputs = build_descriptor_tables(
                sensor_result,
                metadata_df,
                group_by="recording_id",
                id_col="obs_id",
                target_col=args.target_col,
                aggregation_groups=aggregation_descriptors,
                ratio_pairs=aggregated_ratio_pairs,
                ratio_floor=aggregated_ratio_floor,
            )
            pooled_outputs = None
            if include_pooled:
                pooled_result = pipeline.pool_channels(sensor_result, channel_groups)
                pooled_outputs = build_descriptor_tables(
                    pooled_result,
                    metadata_df,
                    group_by="recording_id",
                    id_col="obs_id",
                    target_col=args.target_col,
                    aggregation_groups=aggregation_descriptors,
                    ratio_pairs=aggregated_ratio_pairs,
                    ratio_floor=aggregated_ratio_floor,
                )
            failure_df = _build_failure_df(
                sensor_result.meta.get("failures", []),
                failure_metadata_df,
                condition,
            )
            _save_subject_shard(
                shard_root,
                sensor_result,
                sensor_outputs,
                pooled_outputs,
                failure_df,
            )
            qc_summary = run_descriptor_subject_qc(
                shard_root=shard_root,
                reports_root=reports_root,
                subject=subject,
                session=session,
                condition=condition,
                sensor_epoch_df=sensor_outputs["epoch_df"],
                sensor_subject_df=sensor_outputs["subject_df"],
                sensor_epoch_feature_columns_path=shard_root
                / "sensor_epoch_features_feature_columns.json",
                sensor_subject_feature_columns_path=shard_root
                / "sensor_subject_features_feature_columns.json",
                pooled_epoch_df=None if pooled_outputs is None else pooled_outputs["epoch_df"],
                pooled_subject_df=None if pooled_outputs is None else pooled_outputs["subject_df"],
                pooled_epoch_feature_columns_path=None
                if pooled_outputs is None
                else shard_root / "pooled_epoch_features_feature_columns.json",
                pooled_subject_feature_columns_path=None
                if pooled_outputs is None
                else shard_root / "pooled_subject_features_feature_columns.json",
                failure_df=failure_df,
                config_snapshot=config_snapshot,
            )
            (shard_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            LOGGER.info(
                "%s / %s: saved %d epoch rows, %d subject rows, %d failures, qc=%s, report=%s",
                condition,
                subject,
                len(sensor_outputs["epoch_df"]),
                len(sensor_outputs["subject_df"]),
                len(failure_df),
                qc_summary["qc_status"],
                qc_summary["report_path"],
            )

    LOGGER.info("Derivative feature root: %s", derivative_root)


if __name__ == "__main__":
    main()

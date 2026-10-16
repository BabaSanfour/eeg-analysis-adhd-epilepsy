"""
End-to-end feature extraction from EEG epochs.

This script loads saved epoched derivatives, extracts broad EEG descriptors,
and writes checkpointed per-subject outputs under a derivative-style root.

Outputs
-------
For each processed subject-condition pair, the script writes a shard under
``<derivative_root>/sub-<subject>/ses-<session>/eeg/<condition>/`` containing:

- ``sensor_epoch_features.parquet``: one row per epoch, the feature columns in the
  table's metadata
- ``shard.json``: the shard's QC summary, QC flags and extraction failures, written
  last, so a crashed shard is never merged

and one QC report per subject and session, with every condition. The recording table
is built by ``merge_descriptors.py`` from the merged epoch table.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from coco_pipe.descriptors import (
    DescriptorConfig,
    DescriptorPipeline,
    mad_failures_from_qc,
    save_descriptor_table,
)
from coco_pipe.io import DataContainer
from coco_pipe.io.quality import drop_epoch_outliers
from mne_bids import get_entities_from_fname

from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.analysis.utils.common import require_config
from eeg_adhd_epilepsy.analysis.utils.descriptor_shards import (
    EPOCH_TABLE,
    SHARD_RECORD,
    recording_tables,
)
from eeg_adhd_epilepsy.analysis.utils.subject_resolution import (
    resolve_cohort_subjects,
    resolve_metadata_row,
)
from eeg_adhd_epilepsy.io.bids import (
    DESCRIPTION_ONLY_BIDSIGNORE,
    DerivativeStage,
    bids_session_label,
    bids_subject_label,
    get_derivative_root,
    read_participants,
    write_bidsignore,
)
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    descriptor_qc_report_name,
    subject_report_dir,
)
from eeg_adhd_epilepsy.qc.descriptor_qc import (
    run_descriptor_shard_qc,
    write_descriptor_subject_report,
)
from eeg_adhd_epilepsy.utils.artifacts import freeze_config_used, write_text_atomic
from eeg_adhd_epilepsy.utils.constants import DEFAULT_ANALYSIS_CONDITIONS
from eeg_adhd_epilepsy.utils.formatting import reorder_columns_front
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)


def _apply_mad_rejection(
    container: DataContainer,
    metadata_df: pd.DataFrame,
    condition: str,
    subject: str,
    mad_threshold: float = 10.0,
    fraction_thresh: float = 0.05,
    group_by: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[DataContainer, pd.DataFrame]:
    """Drop epochs where too large a fraction of features are MAD outliers.

    Delegates MAD scoring/masking to
    :func:`coco_pipe.io.quality.drop_epoch_outliers` and records the dropped
    epochs via :func:`coco_pipe.descriptors.mad_failures_from_qc`. Features whose
    names start with ``exclude_prefixes`` are not scored.
    """
    if container.X.shape[0] == 0:
        return container, metadata_df

    features = [str(name) for name in np.asarray(container.coords["feature"])]
    clean_or_masks, qc_result = drop_epoch_outliers(
        container,
        feature_cols=[name for name in features if not name.startswith(exclude_prefixes)],
        z_threshold=mad_threshold,
        outlier_fraction_threshold=fraction_thresh,
        group_by=group_by,
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
    return reorder_columns_front(failure_df, failure_front)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _save_descriptor_shard(
    shard_root: Path,
    sensor_outputs: dict[str, Any],
    failure_df: pd.DataFrame,
    shard_qc: dict[str, Any],
) -> None:
    """Write the epoch table, then ``shard.json``, which marks the shard complete."""
    shard_root.mkdir(parents=True, exist_ok=True)
    save_descriptor_table(
        sensor_outputs["epoch_df"],
        shard_root / EPOCH_TABLE,
        feature_columns=sensor_outputs["epoch_feature_columns"],
    )
    record = {
        "qc_summary": shard_qc["summary"],
        "qc_flags": shard_qc["flags"],
        "failures": failure_df.astype(object).where(failure_df.notna(), None).to_dict("records"),
    }
    write_text_atomic(
        shard_root / SHARD_RECORD, json.dumps(record, indent=2, default=_json_scalar) + "\n"
    )


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
        help="Path to the BIDS participants.tsv.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "descriptors.yaml"),
        help="Path to descriptor YAML config.",
    )
    parser.add_argument(
        "--desc",
        default="base",
        help="Desc of the saved epochs to read: base (every epoch) or baserejected.",
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
            "One-based row number in participants.tsv to process. "
            "Useful for Slurm job arrays where SLURM_ARRAY_TASK_ID selects a metadata row."
        ),
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
        derivative_root = get_derivative_root(bids_root, DerivativeStage.DESCRIPTORS)

    raw_config = load_yaml_config(config_path)
    aggregation_config = raw_config.pop("aggregation", None) or {}
    qc_config = raw_config.pop("qc", None) or {}
    require_config(aggregation_config, "descriptors", expected_type=list)
    config_snapshot = deepcopy(raw_config)
    config_snapshot["aggregation"] = deepcopy(aggregation_config)
    # The epoch variant: a root made from one variant refuses the other.
    config_snapshot["desc"] = args.desc
    if qc_config:
        config_snapshot["qc"] = deepcopy(qc_config)
    freeze_config_used(
        config_snapshot,
        derivative_root,
        mismatch_message=(
            "Existing descriptor derivative root was generated with a different "
            "configuration. Clear the derivative root."
        ),
    )

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
    dataset_description_path = derivative_root / "dataset_description.json"
    if not dataset_description_path.exists():
        write_text_atomic(
            dataset_description_path,
            json.dumps(dataset_description, indent=2) + "\n",
        )
    write_bidsignore(derivative_root, DESCRIPTION_ONLY_BIDSIGNORE)

    pipeline = DescriptorPipeline(DescriptorConfig.model_validate(raw_config))

    meta_df = read_participants(metadata_path)
    requested: list[Any] | None = list(args.subjects) if args.subjects else None
    if args.metadata_row is not None:
        subject = resolve_metadata_row(meta_df, args.metadata_row, "subject")
        if subject is None:
            return
        requested = [subject]
        LOGGER.info("Resolved metadata row %d to %s.", args.metadata_row, subject)

    subjects = resolve_cohort_subjects(meta_df, "subject", requested)
    LOGGER.info("Using %d subjects from saved derivatives.", len(subjects))

    for subject in subjects:
        label = bids_subject_label(subject)
        subject_root = get_derivative_root(bids_root, DerivativeStage.PREPROC) / label
        files = list(subject_root.glob(f"**/{label}*_desc-{args.desc}_epo.fif"))

        # The session entity of each file; a file without one is labelled ses-01 in the
        # outputs and loaded without a session filter.
        session_entities = sorted(
            {get_entities_from_fname(f)["session"] for f in files}, key=lambda s: s or ""
        ) or [None]

        if args.conditions == ["all"]:
            # A file that cannot be read stops the task, as in the loading below.
            subject_conditions = set()
            for f in files:
                subject_conditions.update(mne.read_epochs(f, verbose="ERROR").event_id.keys())
            subject_conditions = sorted(list(subject_conditions))
            if not subject_conditions:
                LOGGER.warning("No saved conditions found for subject %s", subject)
                continue
        else:
            subject_conditions = args.conditions

        for session_entity in session_entities:
            session = session_entity or "01"
            # Every run recomputes: a shard or report left from an earlier run (an earlier
            # stage 03/04, another config, a crash halfway) never survives it.
            (
                subject_report_dir(
                    reports_root=reports_root,
                    subject=subject,
                    session=session,
                    stage=ReportStage.DESCRIPTOR_QC,
                )
                / descriptor_qc_report_name(subject, session)
            ).unlink(missing_ok=True)
            shard_qcs: list[dict[str, Any]] = []
            for condition in subject_conditions:
                shard_root = (
                    derivative_root
                    / bids_subject_label(subject)
                    / bids_session_label(session)
                    / "eeg"
                    / condition
                )
                shutil.rmtree(shard_root, ignore_errors=True)

                LOGGER.info("Loading %s for %s (ses %s)", condition, subject, session)
                try:
                    dc_loaded = build_container(
                        bids_root=bids_root,
                        use_derivatives=True,
                        subjects=[subject],
                        desc=args.desc,
                        condition=condition,
                        session=session_entity,
                    )
                except RuntimeError as error:
                    # Skipped only when no recording has this condition; the loader's
                    # message then names no recording that could not be loaded.
                    message = str(error)
                    if message.startswith("No valid data found in ") and (
                        "could not be loaded" not in message
                    ):
                        LOGGER.info(
                            "Skipping %s / %s: no saved epochs found for this condition.",
                            condition,
                            subject,
                        )
                        continue
                    raise
                # A recording that could not be loaded stops the task: the shard would
                # silently lack it.
                load_failures = dc_loaded.meta.get("load_failures", [])
                if load_failures:
                    raise RuntimeError(
                        f"{condition} / {subject}: {len(load_failures)} recording(s) could "
                        f"not be loaded: {load_failures}"
                    )
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

                metadata_df = dc_loaded.obs_table(include_ids=True)
                metadata_df["obs_id"] = metadata_df["obs_id"].astype(str)
                metadata_df["condition"] = condition
                metadata_df["subject"] = subject
                metadata_front = [
                    "obs_id",
                    "subject",
                    "session",
                    "run",
                    "recording_id",
                    "condition",
                ]
                metadata_df = reorder_columns_front(metadata_df, metadata_front)
                sensor_result = pipeline.extract(
                    X=dc_loaded.X,
                    ids=ids,
                    sfreq=sfreq,
                    channel_names=channel_names,
                )

                outlier_config = qc_config.get("outlier", {})
                mad_threshold = float(outlier_config.get("z_threshold", 5.0))
                fraction_threshold = float(outlier_config.get("epoch_outlier_fraction", 0.30))
                min_epochs = int(qc_config.get("min_obs", 3))
                group_by = outlier_config.get("group_by", "family")
                failure_metadata_df = metadata_df.copy()
                if outlier_config.get("enabled", True):
                    sensor_result, metadata_df = _apply_mad_rejection(
                        sensor_result,
                        metadata_df,
                        condition,
                        subject,
                        mad_threshold=mad_threshold,
                        fraction_thresh=fraction_threshold,
                        group_by=group_by,
                        exclude_prefixes=tuple(outlier_config.get("exclude_prefixes", ())),
                    )
                # Every recording value rests on at least min_epochs epochs: a short recording
                # is left out even when the subject's other recordings fill the shard.
                n_per_recording = metadata_df["recording_id"].value_counts()
                short = n_per_recording.index[n_per_recording < min_epochs]
                if len(short):
                    LOGGER.warning(
                        "%s / %s: leaving out %d recording(s) with fewer than %d epochs: %s",
                        condition,
                        subject,
                        len(short),
                        min_epochs,
                        {str(k): int(v) for k, v in n_per_recording[short].items()},
                    )
                    keep = ~metadata_df["recording_id"].isin(short).to_numpy()
                    sensor_result = sensor_result.isel(obs=np.flatnonzero(keep).tolist())
                    metadata_df = metadata_df[keep].reset_index(drop=True)
                if sensor_result.X.shape[0] == 0:
                    LOGGER.warning(
                        "Skipping %s / %s: no recording with %d epochs.",
                        condition,
                        subject,
                        min_epochs,
                    )
                    continue

                sensor_outputs = recording_tables(metadata_df, sensor_result, config_snapshot)
                failure_df = _build_failure_df(
                    sensor_result.meta.get("failures", []),
                    failure_metadata_df,
                    condition,
                )
                shard_qc = run_descriptor_shard_qc(
                    subject=subject,
                    session=session,
                    condition=condition,
                    sensor_epoch_df=sensor_outputs["epoch_df"],
                    sensor_subject_df=sensor_outputs["subject_df"],
                    sensor_epoch_feature_columns=sensor_outputs["epoch_feature_columns"],
                    sensor_subject_feature_columns=sensor_outputs["subject_feature_columns"],
                    failure_df=failure_df,
                    config_snapshot=config_snapshot,
                )
                _save_descriptor_shard(shard_root, sensor_outputs, failure_df, shard_qc)
                shard_qcs.append(shard_qc)
                LOGGER.info(
                    "%s / %s: saved %d epoch rows (%d recordings), %d failures, "
                    "%d rejected epochs, qc=%s",
                    condition,
                    subject,
                    len(sensor_outputs["epoch_df"]),
                    len(sensor_outputs["subject_df"]),
                    shard_qc["summary"]["n_failures_total"],
                    shard_qc["summary"]["n_epochs_rejected"],
                    shard_qc["summary"]["qc_status"],
                )
            if shard_qcs:
                report_path = write_descriptor_subject_report(
                    reports_root, subject, session, shard_qcs, config_snapshot
                )
                LOGGER.info("%s (ses %s): QC report %s", subject, session, report_path)

    LOGGER.info("Derivative feature root: %s", derivative_root)


if __name__ == "__main__":
    main()

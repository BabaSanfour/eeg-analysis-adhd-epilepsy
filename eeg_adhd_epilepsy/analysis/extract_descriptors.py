"""
End-to-end feature extraction from EEG epochs.

This script loads saved epoched derivatives, extracts broad EEG descriptors,
and writes checkpointed per-subject outputs under a derivative-style root.

Outputs
-------
For each processed subject-condition pair, the script writes a shard under
``<derivative_root>/sub-<subject>/ses-<session>/eeg/<condition>/`` containing:

- a sensor epoch-level and a recording-level feature table (`.parquet`, each with
  its `_feature_columns.json`)
- a failures table (`.csv`) and the shard's QC summary and flags (`qc/`)
- a `_SUCCESS` marker, written last, so a crashed shard is never merged

Combined outputs are built separately by ``merge_descriptors.py``.
"""

from __future__ import annotations

import argparse
import itertools
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
    build_descriptor_tables,
    mad_failures_from_qc,
    save_descriptor_table,
)
from coco_pipe.io import DataContainer
from coco_pipe.io.quality import drop_epoch_outliers
from mne_bids import get_entities_from_fname

from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.analysis.utils.common import require_config
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
from eeg_adhd_epilepsy.qc.descriptor_qc import run_descriptor_subject_qc
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


def _save_descriptor_shard(
    shard_root: Path,
    sensor_container: DataContainer,
    sensor_outputs: dict[str, Any],
    failure_df: pd.DataFrame,
) -> None:
    shard_root.mkdir(parents=True, exist_ok=True)
    save_descriptor_table(
        sensor_outputs["epoch_df"],
        shard_root / "sensor_epoch_features",
        feature_columns=sensor_outputs["epoch_feature_columns"],
        formats=("parquet",),
    )
    save_descriptor_table(
        sensor_outputs["subject_df"],
        shard_root / "sensor_recording_features",
        feature_columns=sensor_outputs["subject_feature_columns"],
        formats=("parquet",),
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
        "--subject_col",
        default="subject",
        help="Subject identifier column in cleaned metadata.",
    )
    parser.add_argument(
        "--target_col",
        default=None,
        help="Optional canonical metadata column to also expose as container y during aggregation.",
    )
    parser.add_argument(
        "--units",
        default="V",
        help="Units to load the EEG data in (e.g. 'V' or 'uV').",
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
    aggregation_descriptors = require_config(aggregation_config, "descriptors", expected_type=list)
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

    descriptor_config = DescriptorConfig.model_validate(raw_config)
    pipeline = DescriptorPipeline(descriptor_config)
    if descriptor_config.families.bands.enabled:
        aggregated_ratio_pairs = list(descriptor_config.families.bands.ratio_pairs)
        aggregated_ratio_floor = float(descriptor_config.families.bands.min_denominator_power)
    else:
        aggregated_ratio_pairs = []
        aggregated_ratio_floor = 0.0

    meta_df = read_participants(metadata_path)
    requested: list[Any] | None = list(args.subjects) if args.subjects else None
    if args.metadata_row is not None:
        subject = resolve_metadata_row(meta_df, args.metadata_row, args.subject_col)
        if subject is None:
            return
        requested = [subject]
        LOGGER.info("Resolved metadata row %d to %s.", args.metadata_row, subject)

    subjects = resolve_cohort_subjects(meta_df, args.subject_col, requested)
    LOGGER.info("Using %d subjects from saved derivatives.", len(subjects))

    for subject in subjects:
        epochs_root = get_derivative_root(bids_root, DerivativeStage.PREPROC)
        files = list(epochs_root.rglob(f"{bids_subject_label(subject)}*_desc-{args.desc}_epo.fif"))

        sessions = sorted(list({(get_entities_from_fname(f)["session"] or "01") for f in files}))
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
            # Every run recomputes: a shard or report left from an earlier run (an earlier
            # stage 03/04, another config, a crash halfway) never survives it.
            shutil.rmtree(shard_root, ignore_errors=True)
            (
                subject_report_dir(
                    reports_root=reports_root,
                    subject=subject,
                    session=session,
                    stage=ReportStage.DESCRIPTOR_QC,
                )
                / descriptor_qc_report_name(subject, session, condition)
            ).unlink(missing_ok=True)

            LOGGER.info("Loading %s for %s (ses %s)", condition, subject, session)
            try:
                dc_loaded = build_container(
                    bids_root=bids_root,
                    use_derivatives=True,
                    subjects=[subject],
                    target_col=args.target_col,
                    desc=args.desc,
                    condition=condition,
                    session=session,
                    units=args.units,
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
            metadata_front = ["obs_id", "subject", "session", "run", "recording_id", "condition"]
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
                    "Skipping %s / %s: no recording with %d epochs.", condition, subject, min_epochs
                )
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
                on_insufficient="warn",
            )
            failure_df = _build_failure_df(
                sensor_result.meta.get("failures", []),
                failure_metadata_df,
                condition,
            )
            _save_descriptor_shard(
                shard_root,
                sensor_result,
                sensor_outputs,
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
                / "sensor_recording_features_feature_columns.json",
                failure_df=failure_df,
                config_snapshot=config_snapshot,
            )
            (shard_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            LOGGER.info(
                "%s / %s: saved %d epoch rows, %d recording rows, %d failures, qc=%s, report=%s",
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

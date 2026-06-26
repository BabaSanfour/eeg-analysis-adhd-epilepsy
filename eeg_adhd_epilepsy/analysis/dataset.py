"""High-level dataset builders for EEG analysis."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.descriptors import KNOWN_FAMILY_TOKENS, load_descriptor_table
from coco_pipe.descriptors.qc import aggregate_family_qc
from coco_pipe.io import BIDSConfig, DataContainer, load_data
from coco_pipe.io.embeddings import load_embedding_derivatives
from coco_pipe.io.quality import GROUP_BY_COLUMN, run_qc

from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    add_recording_id,
    bids_session_label,
    bids_subject_label,
    get_derivative_root,
)
from eeg_adhd_epilepsy.io.readers import read_preproc_stage
from eeg_adhd_epilepsy.preproc.epochs import make_epochs_from_preproc_raw
from eeg_adhd_epilepsy.qc.feature_qc import compute_grouped_qc_masks

logger = logging.getLogger(__name__)


def build_container(
    bids_root: Path,
    use_derivatives: bool = False,
    subjects: list[str] | None = None,
    task: str = "clinical",
    session: str | list[str] | None = None,
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    metadata_df: pd.DataFrame | None = None,
    subject_col: str = "study_id",
    target_col: str | None = None,
    desc: str = "base",
    condition: str | None = None,
    window_source: str = "auto",
) -> DataContainer:
    """Load raw BIDS data or saved epoch derivatives into a DataContainer."""
    subjects = subjects or []
    if window_source == "re_epoch":
        return reepoch_eeg(
            bids_root=bids_root,
            subjects=list(subjects),
            task=task,
            segment_duration=segment_duration,
            overlap=overlap,
            condition=condition,
            metadata_df=metadata_df,
            subject_col=subject_col,
            desc=desc,
            session=session if isinstance(session, str) else "01",
        )
    external_metadata_df = metadata_df.copy() if metadata_df is not None else None
    if external_metadata_df is not None:
        external_metadata_df[subject_col] = (
            external_metadata_df[subject_col].astype(int).map(lambda value: f"{value:04d}")
        )

    if use_derivatives:
        epochs_root = get_derivative_root(bids_root, DerivativeStage.PREPROC)
        logger.info(f"Loading saved epochs from {epochs_root}")
        config = BIDSConfig(
            path=epochs_root,
            datatype="eeg",
            suffix="epo",
            loading_mode="load_existing",
            subjects=subjects if subjects else None,
            event_id=condition if condition else None,
            target_col=target_col,
        )
        return load_data(
            config=config,
            subject_metadata_df=external_metadata_df,
            subject_key=subject_col if external_metadata_df is not None else None,
        )

    logger.info(f"Loading data for {len(subjects)} subjects. Task: {task}")
    config = BIDSConfig(
        path=bids_root,
        task=task,
        session=session,
        loading_mode="epochs",
        window_length=segment_duration,
        stride=segment_duration - overlap,
        subjects=subjects if subjects else None,
        datatype="eeg",
        suffix="eeg",
        target_col=target_col,
    )
    container = load_data(
        config=config,
        subject_metadata_df=external_metadata_df,
        subject_key=subject_col if external_metadata_df is not None else None,
    )
    logger.info(f"Initial Container Shape: {container.X.shape}, Dims: {container.dims}")
    return container


def reepoch_eeg(
    bids_root: Path,
    subjects: list[str],
    task: str,
    segment_duration: float,
    overlap: float,
    condition: str | None,
    metadata_df: pd.DataFrame | None = None,
    subject_col: str = "study_id",
    desc: str = "base",
    session: str = "01",
) -> DataContainer:
    """Re-epoch the cleaned continuous ``desc`` derivative at ``segment_duration``."""
    preproc_root = get_derivative_root(bids_root, DerivativeStage.PREPROC)
    meta_lookup: pd.DataFrame | None = None
    if metadata_df is not None:
        meta_lookup = metadata_df.copy()
        meta_lookup[subject_col] = (
            meta_lookup[subject_col].astype(int).map(lambda value: f"{value:04d}")
        )
        meta_lookup = meta_lookup.set_index(subject_col)

    x_chunks: list[np.ndarray] = []
    obs_rows: list[dict[str, Any]] = []
    ids: list[str] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None
    times: np.ndarray | None = None
    issues: list[str] = []

    for raw_sid in subjects or []:
        study_id = f"{int(raw_sid):04d}" if str(raw_sid).isdigit() else str(raw_sid)
        raw, _prov, load_issues = read_preproc_stage(study_id, preproc_root, desc=desc, task=task)
        if raw is None:
            issues.extend(load_issues)
            continue
        try:
            epochs = make_epochs_from_preproc_raw(
                raw, segment_duration=segment_duration, overlap=overlap
            )
        except ValueError as exc:
            issues.append(f"no_epochs:{bids_subject_label(study_id)}:{exc}")
            continue
        if condition is None or condition not in epochs.event_id:
            issues.append(f"missing_condition:{bids_subject_label(study_id)}:{condition}")
            continue
        cond_epochs = epochs[condition]
        data = np.asarray(cond_epochs.get_data(), dtype=np.float32)
        if data.shape[0] == 0:
            continue
        if ch_names is None:
            ch_names = list(cond_epochs.ch_names)
            sfreq = float(cond_epochs.info["sfreq"])
            times = np.asarray(cond_epochs.times)
        recording_id = f"{bids_subject_label(study_id)}_{bids_session_label(session)}_run-01"
        meta_row: dict[str, Any] = {}
        if meta_lookup is not None and study_id in meta_lookup.index:
            meta_row = {
                str(column): meta_lookup.loc[study_id, column] for column in meta_lookup.columns
            }
        for epoch_idx in range(data.shape[0]):
            obs_rows.append(
                {
                    subject_col: study_id,
                    "subject": bids_subject_label(study_id),
                    "session": session,
                    "run": "01",
                    "condition": condition,
                    "recording_id": recording_id,
                    **meta_row,
                }
            )
            ids.append(f"{recording_id}_epoch-{epoch_idx:03d}")
        x_chunks.append(data)

    if not x_chunks:
        raise RuntimeError(
            f"No cleaned-continuous epochs for condition {condition!r} at "
            f"{segment_duration:g}s. First issues: {issues[:5]}"
        )

    X = np.concatenate(x_chunks, axis=0)
    obs_df = pd.DataFrame(obs_rows)
    coords: dict[str, Any] = {
        "channel": np.asarray(ch_names, dtype=object),
        "time": np.asarray(times),
    }
    for column in obs_df.columns:
        coords[column] = obs_df[column].to_numpy()

    return DataContainer(
        X=X,
        dims=("obs", "channel", "time"),
        coords=coords,
        ids=np.asarray(ids, dtype=object),
        meta={
            "sfreq": sfreq,
            "window_source": "re_epoch_cleaned_continuous",
            "autoreject_applied": False,
            "segment_duration": float(segment_duration),
            "load_issues": issues,
        },
    )


def build_dataset(
    args,
    subjects: Sequence[str] | None,
    meta_df: pd.DataFrame | None,
    condition: str,
    target_col: str | None = None,
) -> DataContainer:
    """Top-level dataset orchestrator."""
    input_mode = getattr(args, "input_mode", "raw")
    effective_input_mode = (
        getattr(args, "reduced_source_input_mode", "descriptors")
        if input_mode == "reduced_dimensions"
        else input_mode
    )
    analysis_mode = getattr(args, "analysis_mode", "flat")
    if effective_input_mode == "raw":
        container = build_container(
            bids_root=Path(args.bids_root),
            use_derivatives=args.use_derivatives,
            subjects=list(subjects) if subjects is not None else None,
            task=args.task,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            metadata_df=meta_df,
            subject_col=args.subject_col,
            target_col=target_col,
            desc=args.desc,
            condition=condition,
            window_source=getattr(args, "window_source", "auto"),
        )
    elif effective_input_mode == "descriptors":
        qc_config = getattr(args, "qc", None) or {}
        column_prune = qc_config.get("column_prune", {})
        container = load_descriptor_table(
            table_path=Path(args.descriptor_table_path),
            feature_columns_path=Path(args.descriptor_feature_columns_path),
            known_families=KNOWN_FAMILY_TOKENS,
            condition=condition,
            target_col=target_col,
            subjects=subjects,
            subject_col=args.subject_col,
            analysis_mode=analysis_mode,
            descriptor_families=getattr(args, "descriptor_families", None),
            descriptor_max_abs_value=getattr(args, "descriptor_max_abs_value", None),
            drop_degenerate_columns=bool(column_prune.get("enabled", False)),
            max_missing_rate=float(column_prune.get("max_missing_rate", 0.20)),
            drop_constant_columns=bool(column_prune.get("drop_constant", True)),
            max_row_drop_rate=column_prune.get("max_row_drop_rate"),
            location_statistic=getattr(args, "location_statistic", None),
        )
        scoring_container = (
            container if container.dims == ("obs", "feature") else container.flatten(preserve="obs")
        )
        feature_names = (
            np.asarray(scoring_container.coords.get("feature", []), dtype=object)
            .astype(str)
            .tolist()
        )
        outlier_config = qc_config.get("outlier", {})
        group_by = outlier_config.get("group_by")
        if group_by is not None and group_by not in GROUP_BY_COLUMN:
            raise ValueError("qc.outlier.group_by must be 'family', 'measure', or 'feature'.")
        if qc_config and group_by is not None:
            family_masks, qc_result = compute_grouped_qc_masks(
                scoring_container, subject_col=args.subject_col, qc_config=qc_config
            )
            ids = np.asarray(scoring_container.ids, dtype=object).astype(str)
            bad_ids = {family: ids[~mask].tolist() for family, mask in family_masks.items()}
            container.meta = {
                **dict(container.meta),
                "family_qc_keep_masks": family_masks,
                "family_qc_bad_ids": bad_ids,
                "family_qc_group_by": group_by,
            }
        else:
            outlier = qc_config.get("outlier", {})
            clean_scoring, qc_result = run_qc(
                scoring_container,
                epoch_z_threshold=float(outlier.get("z_threshold", 5.0)),
                epoch_outlier_fraction_threshold=float(outlier.get("epoch_outlier_fraction", 0.30)),
                subject_z_threshold=float(outlier.get("z_threshold", 5.0)),
                subject_outlier_fraction_threshold=float(
                    outlier.get("subject_outlier_fraction", 0.20)
                ),
                subject_col=args.subject_col,
            )
            qc_result.feature_columns_dropped = container.meta.get("dropped_feature_columns")
            qc_result.family_qc = aggregate_family_qc(
                pd.DataFrame(clean_scoring.X, columns=feature_names),
                feature_names,
                known_families=KNOWN_FAMILY_TOKENS,
                feature_schema=clean_scoring.feature_schema(),
            )
            if clean_scoring.ids is not None and container.ids is not None:
                keep_ids = set(np.asarray(clean_scoring.ids).astype(str))
                keep_indices = np.flatnonzero(
                    np.isin(np.asarray(container.ids).astype(str), list(keep_ids))
                )
                container = container.isel(obs=keep_indices)
        container.meta = {**dict(container.meta), "qc_result": qc_result}
    elif effective_input_mode == "foundation_embeddings":
        container = load_embedding_derivatives(
            Path(args.embedding_derivative_root),
            representation=getattr(args, "embedding_representation", "recording"),
            aggregate_by=getattr(args, "embedding_aggregate_by", None),
            model_key=getattr(args, "embedding_model_key", None),
        )
        if condition and "condition" in container.coords:
            values = np.asarray(container.coords["condition"]).astype(str)
            container = container.isel(obs=np.flatnonzero(values == str(condition)))
        if subjects is not None:
            subject_key = args.subject_col if args.subject_col in container.coords else "subject"
            if subject_key in container.coords:
                values = np.asarray(container.coords[subject_key]).astype(str)
                wanted = {str(subject) for subject in subjects}
                container = container.isel(obs=np.flatnonzero(np.isin(values, list(wanted))))
    else:
        raise ValueError(f"Unsupported input mode '{input_mode}'.")
    group_filters = getattr(args, "group_filters", None)
    if group_filters:
        n_obs = container.X.shape[0]
        final_mask = np.zeros(n_obs, dtype=bool)
        for group_def in group_filters:
            group_mask = np.ones(n_obs, dtype=bool)
            for col, vals in group_def.items():
                if col in container.coords:
                    group_mask &= np.isin(
                        np.asarray(container.coords[col]).astype(str), [str(v) for v in vals]
                    )
                else:
                    group_mask[:] = False
            final_mask |= group_mask
        container = container.isel(obs=np.flatnonzero(final_mask))

    for column, values in zip(args.filter_col, args.filter_val):
        if not values:
            continue
        container = container.select(**{column: list(values)})
    if args.balance_target:
        container = container.balance(
            target=args.balance_target,
            strategy=args.balance_strategy,
        )

    represented = container
    if effective_input_mode == "raw":
        if args.representation in {
            "subject_flat",
            "subject_time_as_sample",
            "subject_native",
            "recording_flat",
            "recording_time_as_sample",
            "recording_native",
        }:
            aggregation_unit = getattr(args, "aggregation_unit", "recording")
            if aggregation_unit == "recording":
                container = add_recording_id(container, args.subject_col)
                container = container.aggregate(by="recording_id", stats="mean")
            elif aggregation_unit == "subject":
                container = container.aggregate(by=args.subject_col, stats="mean")
            else:
                raise ValueError(f"Unsupported aggregation_unit '{aggregation_unit}'.")

        if analysis_mode == "flat":
            if args.representation in {"epoch_flat", "subject_flat", "recording_flat"}:
                represented = container.flatten(preserve="obs")
            elif args.representation in {
                "epoch_time_as_sample",
                "subject_time_as_sample",
                "recording_time_as_sample",
            }:
                represented = container.stack(dims=("obs", "time"), new_dim="obs")
            else:
                raise ValueError(f"Unsupported raw flat representation '{args.representation}'.")
        elif analysis_mode == "sensor" and args.representation in {
            "epoch_native",
            "subject_native",
            "recording_native",
        }:
            represented = container
        else:
            raise ValueError(
                f"Unsupported raw representation '{args.representation}' "
                f"for analysis_mode='{analysis_mode}'."
            )
    represented.meta = dict(represented.meta)
    represented.meta.update(
        {
            "input_mode": input_mode,
            "source_input_mode": effective_input_mode,
            "condition": condition,
            "analysis_mode": analysis_mode,
            "aggregation_unit": getattr(args, "aggregation_unit", None),
            "loaded_obs": int(container.X.shape[0]),
            "loaded_subjects": int(
                pd.Index(np.asarray(container.coords.get(args.subject_col, []))).nunique()
            )
            if args.subject_col in container.coords
            else None,
            "samples_used": int(represented.X.shape[0]),
        }
    )
    return represented

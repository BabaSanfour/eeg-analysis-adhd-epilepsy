"""Shared analysis input loading and shaping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from coco_pipe.io.structures import DataContainer
from coco_pipe.io.quality import run_qc
from coco_pipe.descriptors._constants import KNOWN_FAMILY_TOKENS
from coco_pipe.descriptors.qc import aggregate_family_qc

from eeg_adhd_epilepsy.io.bids import load_eeg_data
from coco_pipe.io.descriptors import load_descriptor_table


def _infer_bids_entity_from_text(value: object, entity: str) -> str | None:
    match = re.search(rf"(?:^|[_/]){entity}-([^_/]+)", str(value))
    if match:
        return match.group(1)
    return None


def _clean_group_value(value: object, *, missing: str) -> str:
    if pd.isna(value):
        return missing
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return missing
    return text


def _ensure_recording_id(container: DataContainer, subject_col: str) -> DataContainer:
    """Add a run-aware observation coordinate for aggregating repeated recordings."""
    obs_len = container.X.shape[0]
    coords = dict(container.coords)

    if "recording_id" in coords:
        return container

    subject_values = np.asarray(coords.get(subject_col, coords.get("subject", container.ids)), dtype=object)
    if subject_values.ndim != 1 or len(subject_values) != obs_len:
        subject_values = np.asarray(container.ids, dtype=object) if container.ids is not None else np.arange(obs_len).astype(str)
    subject_values = np.asarray([_clean_group_value(value, missing=str(idx)) for idx, value in enumerate(subject_values)], dtype=object)

    id_values = np.asarray(container.ids, dtype=object) if container.ids is not None else np.asarray([""] * obs_len, dtype=object)

    session_values = np.asarray(coords.get("session", [None] * obs_len), dtype=object)
    if session_values.ndim != 1 or len(session_values) != obs_len:
        session_values = np.asarray([None] * obs_len, dtype=object)
    session_values = np.asarray(
        [
            _clean_group_value(value, missing=_infer_bids_entity_from_text(obs_id, "ses") or "01")
            for value, obs_id in zip(session_values, id_values)
        ],
        dtype=object,
    )

    run_values = np.asarray(coords.get("run", [None] * obs_len), dtype=object)
    if run_values.ndim != 1 or len(run_values) != obs_len:
        run_values = np.asarray([None] * obs_len, dtype=object)
    run_values = np.asarray(
        [
            _clean_group_value(value, missing=_infer_bids_entity_from_text(obs_id, "run") or "none")
            for value, obs_id in zip(run_values, id_values)
        ],
        dtype=object,
    )

    coords["session"] = session_values
    coords["run"] = run_values
    coords["recording_id"] = np.asarray(
        [
            f"{subject}_ses-{session}_run-{run}"
            for subject, session, run in zip(subject_values, session_values, run_values)
        ],
        dtype=object,
    )

    return DataContainer(
        X=container.X,
        dims=container.dims,
        coords=coords,
        y=container.y,
        ids=container.ids,
        meta=dict(container.meta),
    )


def _to_epoch_scalar_mean(container: DataContainer) -> DataContainer:
    required_dims = {"obs", "channel", "time"}
    if not required_dims.issubset(set(container.dims)):
        raise ValueError(
            f"epoch_scalar_mean requires dims {required_dims}, got {container.dims}"
        )

    scalar_series = container.stack(dims=("obs", "channel", "time"), new_dim="obs")
    epoch_ids = np.array([str(i).rsplit("_", 2)[0] for i in scalar_series.ids])
    epoch_means = scalar_series.aggregate(by=epoch_ids, stats="mean")

    X_epoch = np.asarray(epoch_means.X)
    if X_epoch.ndim == 1:
        X_epoch = X_epoch[:, np.newaxis]

    n_obs = X_epoch.shape[0]
    coords = {}
    for key, values in epoch_means.coords.items():
        try:
            if len(values) == n_obs:
                coords[key] = np.asarray(values)
        except TypeError:
            continue
    coords["feature"] = np.array(["epoch_scalar_mean"])

    return DataContainer(
        X=X_epoch,
        dims=("obs", "feature"),
        coords=coords,
        y=epoch_means.y,
        ids=np.asarray(epoch_means.ids) if epoch_means.ids is not None else None,
        meta=dict(epoch_means.meta),
    )


def load_container(
    args,
    subjects: Sequence[str] | None,
    meta_df: Optional[pd.DataFrame],
    condition: str,
    target_col: str | None = None,
) -> DataContainer:
    input_mode = getattr(args, "input_mode", "raw")
    analysis_mode = getattr(args, "analysis_mode", "flat")
    if input_mode == "raw":
        container = load_eeg_data(
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
        )
    elif input_mode == "descriptors":
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
        )
        scoring_container = (
            container
            if container.dims == ("obs", "feature")
            else container.flatten(preserve="obs")
        )
        clean_scoring, qc_result = run_qc(
            scoring_container,
            subject_col=args.subject_col,
        )
        if container.dims == ("obs", "feature"):
            feature_names = np.asarray(
                clean_scoring.coords.get("feature", []),
                dtype=object,
            ).astype(str).tolist()
        else:
            sensors = np.asarray(container.coords["sensor"], dtype=object)
            features = np.asarray(container.coords["feature"], dtype=object)
            families = np.asarray(
                container.coords["feature_family"],
                dtype=object,
            )
            feature_names = [
                f"{family}_{feature}_ch-{sensor}"
                for sensor in sensors
                for family, feature in zip(families, features)
            ]
        qc_result.family_qc = aggregate_family_qc(
            pd.DataFrame(clean_scoring.X, columns=feature_names),
            feature_names,
            known_families=KNOWN_FAMILY_TOKENS,
        )
        if clean_scoring.ids is not None and container.ids is not None:
            keep_ids = set(np.asarray(clean_scoring.ids).astype(str))
            keep_indices = np.flatnonzero(
                np.isin(np.asarray(container.ids).astype(str), list(keep_ids))
            )
            container = container.isel(obs=keep_indices)
        container.meta = {**dict(container.meta), "qc_result": qc_result}
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
                    group_mask &= np.isin(np.asarray(container.coords[col]).astype(str), [str(v) for v in vals])
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
    if input_mode == "raw":
        if args.representation in {
            "subject_flat",
            "subject_time_as_sample",
            "subject_scalar_mean",
            "subject_native",
            "recording_flat",
            "recording_time_as_sample",
            "recording_scalar_mean",
            "recording_native",
        }:
            aggregation_unit = getattr(args, "aggregation_unit", "recording")
            if aggregation_unit == "recording":
                container = _ensure_recording_id(container, args.subject_col)
                container = container.aggregate(by="recording_id", stats="mean")
            elif aggregation_unit == "subject":
                container = container.aggregate(by=args.subject_col, stats="mean")
            else:
                raise ValueError(f"Unsupported aggregation_unit '{aggregation_unit}'.")

        if analysis_mode == "flat":
            if args.representation in {"epoch_flat", "subject_flat", "recording_flat"}:
                represented = container.flatten(preserve="obs")
            elif args.representation in {"epoch_time_as_sample", "subject_time_as_sample", "recording_time_as_sample"}:
                represented = container.stack(dims=("obs", "time"), new_dim="obs")
            elif args.representation in {"epoch_scalar_mean", "subject_scalar_mean", "recording_scalar_mean"}:
                represented = _to_epoch_scalar_mean(container)
            else:
                raise ValueError(f"Unsupported raw flat representation '{args.representation}'.")
        elif analysis_mode == "sensor" and args.representation in {"epoch_native", "subject_native", "recording_native"}:
            represented = container
        else:
            raise ValueError(
                f"Unsupported raw representation '{args.representation}' for analysis_mode='{analysis_mode}'."
            )
    represented.meta = dict(represented.meta)
    represented.meta.update(
        {
            "input_mode": input_mode,
            "condition": condition,
            "analysis_mode": analysis_mode,
            "aggregation_unit": getattr(args, "aggregation_unit", None),
            "loaded_obs": int(container.X.shape[0]),
            "loaded_subjects": int(pd.Index(np.asarray(container.coords.get(args.subject_col, []))).nunique())
            if args.subject_col in container.coords
            else None,
            "samples_used": int(represented.X.shape[0]),
        }
    )
    return represented

"""Shared analysis input loading and shaping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from coco_pipe.io.structures import DataContainer

from eeg_adhd_epilepsy.dl import load_temp_dl_data
from eeg_adhd_epilepsy.io.bids import load_eeg_data
from eeg_adhd_epilepsy.io.table import load_tabular_data


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


def concat_containers(containers: Sequence[DataContainer]) -> DataContainer:
    if not containers:
        raise ValueError("Need at least one container to concatenate.")

    base = containers[0]
    if any(container.dims != base.dims for container in containers[1:]):
        raise ValueError("All pooled containers must have matching dims.")
    if any(container.X.shape[1:] != base.X.shape[1:] for container in containers[1:]):
        raise ValueError("All pooled containers must have matching non-observation dimensions.")

    coords: dict[str, np.ndarray] = {}
    for dim_name in base.dims:
        if dim_name == "obs":
            continue
        if dim_name in base.coords:
            coords[dim_name] = np.asarray(base.coords[dim_name])
    if "feature_family" in base.coords:
        coords["feature_family"] = np.asarray(base.coords["feature_family"])

    obs_keys = set()
    for container in containers:
        obs_len = container.X.shape[0]
        for key, values in container.coords.items():
            arr = np.asarray(values)
            if arr.ndim == 1 and len(arr) == obs_len and key != "feature":
                obs_keys.add(key)

    for key in sorted(obs_keys):
        coords[key] = np.concatenate(
            [np.asarray(container.coords[key]) for container in containers],
            axis=0,
        )

    if "condition" not in coords:
        coords["condition"] = np.concatenate(
            [
                np.full(container.X.shape[0], container.meta.get("condition"), dtype=object)
                for container in containers
            ],
            axis=0,
        )

    y_values = [container.y for container in containers if container.y is not None]
    y = np.concatenate(y_values, axis=0) if len(y_values) == len(containers) else None
    ids_values = [container.ids for container in containers if container.ids is not None]
    ids = np.concatenate(ids_values, axis=0) if len(ids_values) == len(containers) else None

    return DataContainer(
        X=np.concatenate([np.asarray(container.X) for container in containers], axis=0),
        dims=base.dims,
        coords=coords,
        y=y,
        ids=ids,
        meta={
            "source": "pooled",
            "conditions": [container.meta.get("condition") for container in containers],
        },
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
        container = load_tabular_data(
            table_path=Path(args.descriptor_table_path),
            feature_columns_path=Path(args.descriptor_feature_columns_path),
            condition=condition,
            target_col=target_col,
            subjects=subjects,
            subject_col=args.subject_col,
            analysis_mode=analysis_mode,
            descriptor_families=getattr(args, "descriptor_families", None),
            descriptor_max_abs_value=getattr(args, "descriptor_max_abs_value", None),
        )
    elif input_mode == "embeddings":
        container = load_temp_dl_data(
            embeddings_root=Path(args.embeddings_root),
            segments_root=Path(args.segments_root or args.bids_root),
            model=args.embedding_model,
            desc=args.embedding_desc,
            subjects=subjects,
            metadata_df=meta_df,
            subject_col=args.subject_col,
            target_col=target_col,
            conditions=[condition],
            min_overlap_fraction=float(args.embedding_min_overlap_fraction),
            reve_segment_duration=float(args.reve_segment_duration),
            cbramod_sampling_rate=float(args.cbramod_sampling_rate),
            drop_unassigned=True,
        )
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
        if args.representation in {"subject_flat", "subject_time_as_sample", "subject_scalar_mean", "subject_native"}:
            aggregation_unit = getattr(args, "aggregation_unit", "recording")
            if aggregation_unit == "recording":
                container = _ensure_recording_id(container, args.subject_col)
                container = container.aggregate(by="recording_id", stats="mean")
            elif aggregation_unit == "subject":
                container = container.aggregate(by=args.subject_col, stats="mean")
            else:
                raise ValueError(f"Unsupported aggregation_unit '{aggregation_unit}'.")

        if analysis_mode == "flat":
            if args.representation in {"epoch_flat", "subject_flat"}:
                represented = container.flatten(preserve="obs")
            elif args.representation in {"epoch_time_as_sample", "subject_time_as_sample"}:
                represented = container.stack(dims=("obs", "time"), new_dim="obs")
            elif args.representation in {"epoch_scalar_mean", "subject_scalar_mean"}:
                represented = _to_epoch_scalar_mean(container)
            else:
                raise ValueError(f"Unsupported raw flat representation '{args.representation}'.")
        elif analysis_mode == "sensor" and args.representation in {"epoch_native", "subject_native"}:
            represented = container
        else:
            raise ValueError(
                f"Unsupported raw representation '{args.representation}' for analysis_mode='{analysis_mode}'."
            )
    elif input_mode == "embeddings":
        if container.X.ndim != 2:
            container = container.flatten(preserve="obs")
        if args.representation == "subject_flat":
            represented = container.aggregate(by=args.subject_col, stats="mean")
        elif args.representation == "epoch_flat":
            represented = container
        else:
            raise ValueError(
                "Embeddings mode currently supports only 'epoch_flat' and 'subject_flat'."
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

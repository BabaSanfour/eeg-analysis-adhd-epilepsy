"""Shared analysis input loading and shaping."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from coco_pipe.descriptors import KNOWN_FAMILY_TOKENS, load_descriptor_table
from coco_pipe.descriptors.qc import (
    aggregate_family_qc,
    classify_descriptor_columns,
)
from coco_pipe.io import DataContainer
from coco_pipe.io.embeddings import load_embedding_derivatives
from coco_pipe.io.quality import (
    GROUP_BY_COLUMN,
    QCResult,
    compute_feature_missingness,
    drop_epoch_outliers,
    drop_subject_outliers,
    run_qc,
)

from eeg_adhd_epilepsy.io.bids import load_eeg_data
from eeg_adhd_epilepsy.io.recording import ensure_recording_id

LOGGER = logging.getLogger(__name__)


def _to_epoch_scalar_mean(container: DataContainer) -> DataContainer:
    required_dims = {"obs", "channel", "time"}
    if not required_dims.issubset(set(container.dims)):
        raise ValueError(f"epoch_scalar_mean requires dims {required_dims}, got {container.dims}")

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


def _descriptor_feature_names(container: DataContainer) -> list[str]:
    if container.dims == ("obs", "feature"):
        return np.asarray(container.coords.get("feature", []), dtype=object).astype(str).tolist()
    sensors = np.asarray(container.coords["sensor"], dtype=object)
    features = np.asarray(container.coords["feature"], dtype=object)
    families = np.asarray(container.coords["feature_family"], dtype=object)
    return [
        f"{family}_{feature}_ch-{sensor}"
        for sensor in sensors
        for family, feature in zip(families, features)
    ]


def _unit_group_labels(source: DataContainer, unit: dict, group_by: str) -> list[str]:
    """Return the grouping labels one analysis unit consumes.

    Analysis-unit containers are reshaped and no longer carry canonical
    descriptor names, so we classify the *source* canonical names and filter by
    the unit's semantic selector (family / measure / channel) instead.
    """
    names = (source.meta or {}).get("family_qc_descriptor_names")
    if not names:
        names = np.asarray(source.coords.get("feature", []), dtype=object).astype(str).tolist()
    if not names:
        return []
    classification = classify_descriptor_columns([str(name) for name in names])
    label_column = GROUP_BY_COLUMN.get(group_by, "family")
    unit_type = unit.get("unit_type")
    unit_name = unit.get("unit_name")
    family = unit.get("family")
    subfamily = unit.get("subfamily")
    if subfamily is not None:
        selector = classification["subfamily"].astype(str) == str(subfamily)
    elif unit_type == "descriptor":
        selector = classification["descriptor"].astype(str) == str(unit_name)
    elif family or unit_type == "family":
        selector = classification["family"].astype(str) == str(family or unit_name)
    elif unit_type == "feature":
        selector = classification["measure"].astype(str) == str(unit_name)
    elif unit_type == "sensor":
        selector = classification["channel"].astype(str) == str(unit_name)
    else:  # global / flat → the unit consumes every column
        selector = pd.Series(True, index=classification.index)
    subset = classification[selector]
    if subset.empty:
        subset = classification
    return list(dict.fromkeys(subset[label_column].fillna("unknown").astype(str).tolist()))


def families_for_analysis_unit(
    source: DataContainer,
    unit: dict,
    descriptor_families: Sequence[str] | None = None,
) -> list[str]:
    """Return the QC grouping labels represented by one analysis unit.

    Labels are at the granularity recorded in ``source.meta`` during loading
    (``family`` by default, or ``measure`` / ``feature``), matching the keys of
    ``family_qc_bad_ids`` so ``apply_family_qc_mask`` can look them up.
    """
    group_by = (source.meta or {}).get("family_qc_group_by", "family")
    if group_by != "family":
        return _unit_group_labels(source, unit, group_by)
    if unit.get("family"):
        return [str(unit["family"])]
    available = np.asarray(source.coords.get("feature_family", []), dtype=object).astype(str)
    features = np.asarray(source.coords.get("feature", []), dtype=object).astype(str)
    if available.size == 0 and features.size:
        classification = classify_descriptor_columns(features.tolist())
        available = classification["family"].fillna("unknown").astype(str).to_numpy()
    if unit.get("unit_type") == "feature" and len(features) == len(available):
        matched = available[features == str(unit.get("unit_name"))]
        if matched.size:
            return list(dict.fromkeys(matched.tolist()))
    wanted = set(str(value) for value in (descriptor_families or available.tolist()))
    return [family for family in dict.fromkeys(available.tolist()) if family in wanted]


def apply_family_qc_mask(
    container: DataContainer,
    families: Sequence[str],
) -> tuple[DataContainer, np.ndarray]:
    """Filter observations using only the requested families' QC decisions."""
    bad_ids_by_family = dict(container.meta.get("family_qc_bad_ids", {}))
    if not bad_ids_by_family:
        indices = np.arange(container.X.shape[0], dtype=int)
        return container, indices
    if container.ids is None:
        raise ValueError("Family-scoped QC requires observation IDs.")
    bad_ids: set[str] = set()
    for family in families:
        bad_ids.update(str(value) for value in bad_ids_by_family.get(str(family), []))
    ids = np.asarray(container.ids, dtype=object).astype(str)
    keep_indices = np.flatnonzero(~np.isin(ids, list(bad_ids)))
    return container.isel(obs=keep_indices), keep_indices


def _family_scoped_family_qc(
    scoring_container: DataContainer,
    descriptor_names: list[str],
    family_masks: dict[str, np.ndarray],
    group_by: str,
) -> pd.DataFrame:
    """Family-level QC summary, each family on the rows kept for that family.

    ``aggregate_family_qc`` reports per family; when ``group_by`` is finer than
    ``family``, a family's "kept rows" are the rows clean across *all* of that
    family's groups (logical AND of the relevant masks).
    """
    classification = classify_descriptor_columns(list(descriptor_names))
    column_family = classification["family"].fillna("unknown").astype(str).to_numpy()
    column_group = (
        classification[GROUP_BY_COLUMN.get(group_by, "family")]
        .fillna("unknown")
        .astype(str)
        .to_numpy()
    )
    matrix = np.asarray(scoring_container.X)
    n_rows = matrix.shape[0]
    summaries = []
    for family in dict.fromkeys(column_family.tolist()):
        column_indices = np.flatnonzero(column_family == family)
        if column_indices.size == 0:
            continue
        groups_in_family = dict.fromkeys(column_group[column_indices].tolist())
        masks = [family_masks[group] for group in groups_in_family if group in family_masks]
        keep = np.logical_and.reduce(masks) if masks else np.ones(n_rows, dtype=bool)
        columns = [str(descriptor_names[index]) for index in column_indices]
        kept_rows = matrix[np.flatnonzero(keep)][:, column_indices]
        summaries.append(
            aggregate_family_qc(
                pd.DataFrame(kept_rows, columns=columns),
                columns,
                known_families=KNOWN_FAMILY_TOKENS,
            )
        )
    if not summaries:
        return aggregate_family_qc(
            pd.DataFrame(matrix, columns=list(descriptor_names)),
            list(descriptor_names),
            known_families=KNOWN_FAMILY_TOKENS,
        )
    return pd.concat(summaries, ignore_index=True)


def _run_family_scoped_qc(
    scoring_container: DataContainer,
    *,
    subject_col: str,
    descriptor_names: list[str],
    qc_config: dict,
) -> tuple[dict[str, np.ndarray], QCResult]:
    outlier = qc_config.get("outlier", {})
    z_threshold = float(outlier.get("z_threshold", 5.0))
    epoch_fraction = float(outlier.get("epoch_outlier_fraction", 0.30))
    subject_fraction = float(outlier.get("subject_outlier_fraction", 0.20))
    group_by = str(outlier.get("group_by", "family"))
    min_obs = qc_config.get("min_obs")

    subject_values = np.asarray(
        scoring_container.coords.get(subject_col, scoring_container.ids),
        dtype=object,
    )
    # One row per subject ⇒ the epoch-level (L2) MAD pass is redundant with the
    # subject-level (L3) pass; run only L3 in that case.
    subject_aggregated = pd.Index(subject_values.astype(str)).nunique() == len(subject_values)

    subject_masks, subject_result = drop_subject_outliers(
        scoring_container,
        z_threshold=z_threshold,
        outlier_fraction_threshold=subject_fraction,
        subject_col=subject_col,
        descriptor_names=descriptor_names,
        group_by=group_by,
    )
    assert isinstance(subject_masks, dict)

    epoch_dropped: dict[str, list] = {}
    epoch_threshold: float | None = None
    epoch_fraction_used: float | None = None
    if subject_aggregated:
        family_masks = dict(subject_masks)
    else:
        # ``min_obs`` is an extraction-time gate; at decode we degrade gracefully
        # (warn below) instead of letting the library raise and abort the load.
        epoch_masks, epoch_result = drop_epoch_outliers(
            scoring_container,
            z_threshold=z_threshold,
            outlier_fraction_threshold=epoch_fraction,
            subject_col=subject_col,
            descriptor_names=descriptor_names,
            group_by=group_by,
            min_obs=None,
        )
        assert isinstance(epoch_masks, dict)
        family_masks = {
            family: epoch_masks[family]
            & subject_masks.get(family, np.ones_like(epoch_masks[family]))
            for family in epoch_masks
        }
        epoch_dropped = epoch_result.per_family_dropped
        epoch_threshold = z_threshold
        epoch_fraction_used = epoch_fraction

    # Drops are recorded per group at the configured ``group_by`` granularity:
    # each entry says which subjects/observations are bad *for that group only*.
    per_family_dropped = {
        group: [
            *epoch_dropped.get(group, []),
            *subject_result.per_family_dropped.get(group, []),
        ]
        for group in family_masks
    }
    if min_obs is not None:
        for group, mask in family_masks.items():
            kept = int(mask.sum())
            if kept < int(min_obs):
                LOGGER.warning(
                    "Family-scoped QC left %d observation(s) for %s, below "
                    "min_obs=%d; keeping them (min_obs is enforced at "
                    "extraction, not decode).",
                    kept,
                    group,
                    int(min_obs),
                )

    n_obs = scoring_container.X.shape[0]
    n_subjects = pd.Index(subject_values.astype(str)).nunique()
    meta = scoring_container.meta or {}
    # Deferred masking: the scope container keeps every row — per-group drops are
    # applied later, per analysis unit, by ``apply_family_qc_mask``. So the scope
    # QCResult reports *no global drop* (n_obs_out == n_obs_in, subjects_dropped
    # empty); the conditional per-group drops live in ``per_family_dropped``.
    result = QCResult(
        n_rows_entering_qc=meta.get("n_rows_entering_qc"),
        n_dropped_nan_inf=int(meta.get("n_dropped_nan_inf", 0)),
        n_dropped_extreme=int(meta.get("dropped_extreme_rows", 0)),
        n_obs_in=n_obs,
        n_obs_out=n_obs,
        n_subjects_in=n_subjects,
        n_subjects_out=n_subjects,
        epoch_drop_threshold=epoch_threshold,
        epoch_outlier_fraction_threshold=epoch_fraction_used,
        subject_drop_threshold=z_threshold,
        subject_outlier_fraction_threshold=subject_fraction,
        per_family_dropped=per_family_dropped,
        subject_outlier_burden=subject_result.subject_outlier_burden,
        feature_missingness=compute_feature_missingness(
            pd.DataFrame(scoring_container.X, columns=descriptor_names),
            descriptor_names,
        ),
        feature_columns_dropped=meta.get("dropped_feature_columns"),
        thresholds={
            "epoch_z_threshold": epoch_threshold,
            "epoch_outlier_fraction_threshold": epoch_fraction_used,
            "subject_z_threshold": z_threshold,
            "subject_outlier_fraction_threshold": subject_fraction,
            "group_by": group_by,
            "subject_aggregated": bool(subject_aggregated),
        },
    )
    result.family_qc = _family_scoped_family_qc(
        scoring_container, descriptor_names, family_masks, group_by
    )
    return family_masks, result


def load_container(
    args,
    subjects: Sequence[str] | None,
    meta_df: pd.DataFrame | None,
    condition: str,
    target_col: str | None = None,
) -> DataContainer:
    input_mode = getattr(args, "input_mode", "raw")
    effective_input_mode = (
        getattr(args, "reduced_source_input_mode", "descriptors")
        if input_mode == "reduced_dimensions"
        else input_mode
    )
    analysis_mode = getattr(args, "analysis_mode", "flat")
    if effective_input_mode == "raw":
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
        feature_names = _descriptor_feature_names(container)
        outlier_config = qc_config.get("outlier", {})
        # group_by is the single grouping knob: None/absent → one global drop
        # decision; 'family' | 'measure' | 'feature' → per-group decisions.
        group_by = outlier_config.get("group_by")
        if group_by is not None and group_by not in GROUP_BY_COLUMN:
            raise ValueError("qc.outlier.group_by must be 'family', 'measure', or 'feature'.")
        if qc_config and group_by is not None:
            # Deferred masking: keep every row here and record per-family bad ids
            # in meta; ``apply_family_qc_mask`` drops only the relevant families'
            # subjects per analysis unit. ``_run_family_scoped_qc`` already sets
            # the per-family ``qc_result.family_qc``.
            family_masks, qc_result = _run_family_scoped_qc(
                scoring_container,
                subject_col=args.subject_col,
                descriptor_names=feature_names,
                qc_config=qc_config,
            )
            ids = np.asarray(scoring_container.ids, dtype=object).astype(str)
            bad_ids = {family: ids[~mask].tolist() for family, mask in family_masks.items()}
            container.meta = {
                **dict(container.meta),
                "family_qc_keep_masks": family_masks,
                "family_qc_bad_ids": bad_ids,
                "family_qc_group_by": group_by,
                "family_qc_descriptor_names": list(feature_names),
            }
            for family, values in bad_ids.items():
                if values:
                    LOGGER.warning(
                        "Family-scoped QC marked %d observation(s) bad for %s.",
                        len(values),
                        family,
                    )
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
            "subject_scalar_mean",
            "subject_native",
            "recording_flat",
            "recording_time_as_sample",
            "recording_scalar_mean",
            "recording_native",
        }:
            aggregation_unit = getattr(args, "aggregation_unit", "recording")
            if aggregation_unit == "recording":
                container = ensure_recording_id(container, args.subject_col)
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
            elif args.representation in {
                "epoch_scalar_mean",
                "subject_scalar_mean",
                "recording_scalar_mean",
            }:
                represented = _to_epoch_scalar_mean(container)
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

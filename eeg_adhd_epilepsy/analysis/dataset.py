"""High-level dataset builders for EEG analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.descriptors import KNOWN_FAMILY_TOKENS, load_descriptor_table
from coco_pipe.descriptors.qc import aggregate_family_qc
from coco_pipe.io import (
    AGGREGATION_LEVELS,
    BIDSConfig,
    DataContainer,
    aggregate_recordings,
    embedding_label,
    embedding_path,
    load_data,
    load_embeddings,
)
from coco_pipe.io.quality import GROUP_BY_COLUMN, run_qc

from eeg_adhd_epilepsy.analysis.utils.common import qc_keys
from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    get_derivative_root,
)
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
    subject_col: str = "subject",
    target_col: str | None = None,
    desc: str = "base",
    condition: str | list[str] | None = None,
    units: str = "V",
    n_jobs: int = 1,
) -> DataContainer:
    """Load raw BIDS data or saved epoch derivatives into a DataContainer.

    ``condition`` names one event or several; ``n_jobs`` threads read saved epochs.
    """
    subjects = subjects or []
    if use_derivatives:
        epochs_root = get_derivative_root(bids_root, DerivativeStage.PREPROC)
        logger.info(f"Loading saved epochs from {epochs_root}")
        config = BIDSConfig(
            path=epochs_root,
            task=task,
            session=session,
            description=desc,
            datatype="eeg",
            suffix="epo",
            loading_mode="load_existing",
            subjects=subjects if subjects else None,
            event_id=condition if condition else None,
            target_col=target_col,
            units=units,
            n_jobs=n_jobs,
        )
        return load_data(config=config)

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
        units=units,
    )
    container = load_data(config=config)
    logger.info(f"Initial Container Shape: {container.X.shape}, Dims: {container.dims}")
    return container


def foundation_embedding_file(
    root: Path, space: str, desc: str, *, view: str = "window", task: str = "clinical"
) -> Path:
    """The all-recordings file of one embedding space, in its model's dataset under ``root``.

    ``space`` names the space as the analyses do: the model key (``cbramod``), followed by
    ``_pool-attention`` for REVE's attention read-out and ``_align-leace`` for the window
    vectors with subject identity erased. ``desc`` is the epoch variant (``base`` or
    ``baserejected``).
    """
    model, *parts = space.split("_")
    unknown = set(parts) - {"pool-attention", "align-leace"}
    if unknown:
        raise ValueError(f"Unknown embedding space suffix {sorted(unknown)} in {space!r}.")
    label = embedding_label(
        desc,
        pooling="attention" if "pool-attention" in parts else "mean",
        transform="Leace" if "align-leace" in parts else None,
    )
    return embedding_path(Path(root) / f"coco-pipe-{model}", label, view, task=task).fpath


def attach_subject_metadata(
    container: DataContainer,
    meta_df: pd.DataFrame,
    subject_col: str,
) -> DataContainer:
    """Join cohort metadata columns onto an embedding container, keyed by subject.

    Foundation embeddings arrive with only identity coords; this adds every
    ``meta_df`` column not already present (e.g. ``combined_diagnosis``,
    ``patient_group_id``) so downstream evals can read their target/group columns,
    as coco-pipe's BIDS loader does from ``participants.tsv`` for epochs.
    Subjects with no metadata row get NaN, which the evals already treat as
    missing. A no-op if the container carries no recognizable subject coord.
    """
    from coco_pipe.io import normalize_subject_value

    subject_key = subject_col if subject_col in container.coords else "subject"
    if subject_key not in container.coords:
        return container

    lookup = meta_df.copy()
    lookup[subject_col] = lookup[subject_col].map(normalize_subject_value)
    lookup = lookup.drop_duplicates(subject_col).set_index(subject_col)

    obs_subjects = pd.Index(
        [normalize_subject_value(value) for value in np.asarray(container.coords[subject_key])]
    )
    new_coords = dict(container.coords)
    for column in lookup.columns:
        if column in new_coords:
            continue
        new_coords[column] = np.asarray(
            lookup[column].reindex(obs_subjects).to_numpy(), dtype=object
        )
    return DataContainer(
        X=container.X,
        dims=container.dims,
        coords=new_coords,
        ids=container.ids,
        meta=container.meta,
    )


def group_index(
    container: DataContainer, group_filters: list[dict[str, list[Any]]] | None
) -> np.ndarray:
    """Each row's group: the index of the first group filter it meets, -1 for none (all 0
    without group filters)."""
    group = np.full(container.X.shape[0], -1 if group_filters else 0)
    for index, filters in enumerate(group_filters or []):
        member = np.ones(container.X.shape[0], dtype=bool)
        for column, values in filters.items():
            member &= np.isin(
                np.asarray(container.coords[column]).astype(str), [str(v) for v in values]
            )
        group[(group == -1) & member] = index
    return group


def filter_cohort_container(
    container: DataContainer,
    *,
    group_filters: list[dict[str, list[Any]]] | None,
    filter_col: list[str],
    filter_val: list[list[Any]],
    matched: bool = False,
) -> DataContainer:
    """Select the configured cohort from a metadata-enriched container: the group and
    stratum filters, then each child's first recording, then (``matched``) the groups
    matched on sex, age and epilepsy."""
    keep = group_index(container, group_filters) >= 0
    for column, values in zip(filter_col, filter_val):
        if values:
            keep &= np.isin(
                np.asarray(container.coords[column]).astype(str),
                [str(value) for value in values],
            )
    container = keep_first_recording(container.isel(obs=np.flatnonzero(keep)))
    return match_groups(container, group_filters) if matched else container


def metadata_container(container: DataContainer) -> DataContainer:
    """The container's recordings with age, sex (male) and epilepsy as their only features.

    The input of the metadata-only model: what these three explain without any EEG, on the
    rows (recordings and conditions) the EEG analyses use.
    """
    coords = container.coords
    features = np.column_stack(
        [
            pd.to_numeric(np.asarray(coords["age"]), errors="coerce"),
            (np.asarray(coords["sex"]).astype(str) == "M").astype(float),
            pd.to_numeric(np.asarray(coords["epilepsy"]), errors="coerce"),
        ]
    )
    return DataContainer(
        X=features,
        dims=("obs", "feature"),
        coords={
            **{key: value for key, value in coords.items() if not key.startswith("feature")},
            "feature": np.asarray(["age", "sex_male", "epilepsy"], dtype=object),
        },
        ids=container.ids,
        meta={},
    )


MATCH_COVARIATES = ("sex", "age", "epilepsy")


def match_groups(
    container: DataContainer, group_filters: list[dict[str, list[Any]]], random_state: int = 0
) -> DataContainer:
    """The cohort's groups matched on sex, age in whole years and epilepsy, per condition.

    coco-pipe's ``balance`` keeps, in every cell of condition x covariates, a random draw of
    the larger group the size of the smaller one; a child's rows of a condition go together.
    Covariates the groups are defined by are left out.
    """
    if not group_filters or len(group_filters) < 2:
        raise ValueError("Matching needs a cohort with two groups (group_filters).")
    defining = {column for filters in group_filters for column in filters}
    covariates = [f"_{name}" for name in MATCH_COVARIATES if name not in defining]
    age = np.floor(pd.to_numeric(np.asarray(container.coords["age"]), errors="coerce"))
    matched = container.replace(
        coords={
            **container.coords,
            "_group": group_index(container, group_filters),
            # As text: balance bins numeric covariates into quantiles.
            "_sex": np.asarray(container.coords["sex"]).astype(str),
            "_age": pd.Series(age).map(lambda years: f"{years:.0f}").to_numpy(),
            "_epilepsy": np.asarray(container.coords["epilepsy"]).astype(str),
        }
    ).balance(
        target="_group",
        strategy="undersample",
        covariates=["condition", *covariates],
        groups=["subject", "condition"],
        random_state=random_state,
    )
    return matched.replace(
        coords={key: value for key, value in matched.coords.items() if not key.startswith("_")}
    )


def keep_first_recording(container: DataContainer) -> DataContainer:
    """Each child's recording at the youngest age, per condition, among those the cohort kept.

    A child (``patient_group_id``) recorded twice counts once; one whose first recording has
    no epochs of a condition keeps the other for it.
    """
    if not {"patient_group_id", "age"} <= set(container.coords):
        return container
    rows = pd.DataFrame(
        {
            key: np.asarray(container.coords[key])
            for key in ("patient_group_id", "condition", "subject")
        }
    ).assign(age=pd.to_numeric(np.asarray(container.coords["age"]), errors="coerce"))
    first = rows.sort_values(["age", "subject"]).drop_duplicates(["patient_group_id", "condition"])
    keep = pd.MultiIndex.from_frame(rows[["condition", "subject"]]).isin(
        pd.MultiIndex.from_frame(first[["condition", "subject"]])
    )
    return container.isel(obs=np.flatnonzero(keep))


def build_dataset(
    args,
    meta_df: pd.DataFrame | None,
    condition: str,
    target_col: str | None = None,
    *,
    analysis_mode: str | None = None,
    representation: str | None = None,
) -> DataContainer:
    """Top-level dataset orchestrator.

    ``analysis_mode`` and ``representation`` may be passed explicitly to load a
    layout other than the run's current mode; both default to the matching
    ``args`` attribute.
    """
    if meta_df is None:
        subjects = None
    else:
        mask = pd.Series(True, index=meta_df.index)
        for col, vals in zip(args.filter_col, args.filter_val):
            if col in meta_df.columns:
                mask &= meta_df[col].astype(str).isin(vals)

        from coco_pipe.io import normalize_subject_value

        subjects = sorted(
            {
                normalize_subject_value(subject)
                for subject in meta_df.loc[mask, args.subject_col].dropna().unique()
            }
        )

    input_mode = args.input_mode
    effective_input_mode = (
        args.reduced_source_input_mode if input_mode == "reduced_dimensions" else input_mode
    )
    analysis_mode = analysis_mode if analysis_mode is not None else args.analysis_mode
    representation = (
        representation if representation is not None else getattr(args, "representation", "")
    )
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
            units=args.units,
        )
    elif effective_input_mode == "descriptors":
        qc_config = args.qc or {}
        column_prune = qc_config.get("column_prune", {})
        container = load_descriptor_table(
            table_path=Path(args.descriptor_table_path),
            known_families=KNOWN_FAMILY_TOKENS,
            condition=condition,
            target_col=target_col,
            subjects=subjects,
            subject_col=args.subject_col,
            descriptor_families=args.descriptor_families,
            descriptor_max_abs_value=args.descriptor_max_abs_value,
            drop_degenerate_columns=bool(column_prune.get("enabled", False)),
            max_missing_rate=float(column_prune.get("max_missing_rate", 0.20)),
            drop_constant_columns=bool(column_prune.get("drop_constant", True)),
            max_row_drop_rate=column_prune.get("max_row_drop_rate"),
            location_statistic=args.location_statistic,
        )
        scoring_container = container
        feature_names = (
            np.asarray(scoring_container.coords.get("feature", []), dtype=object)
            .astype(str)
            .tolist()
        )
        outlier_config = qc_config.get("outlier", {})
        group_by = outlier_config.get("group_by")
        if group_by is not None and group_by not in GROUP_BY_COLUMN:
            choices = ", ".join(repr(choice) for choice in GROUP_BY_COLUMN)
            raise ValueError(f"qc.outlier.group_by must be one of {choices}; got {group_by!r}.")
        if qc_config and group_by is not None:
            family_masks, qc_result = compute_grouped_qc_masks(
                scoring_container, subject_col=args.subject_col, qc_config=qc_config
            )
            ids = qc_keys(scoring_container)
            bad_ids = {family: ids[~mask].tolist() for family, mask in family_masks.items()}
            container.meta = {
                **dict(container.meta),
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
    elif effective_input_mode == "metadata":
        container = metadata_container(
            load_descriptor_table(
                table_path=Path(args.descriptor_table_path),
                known_families=KNOWN_FAMILY_TOKENS,
                condition=condition,
                target_col=target_col,
                subjects=subjects,
                subject_col=args.subject_col,
                location_statistic=args.location_statistic,
            )
        )
    elif effective_input_mode == "foundation_embeddings":
        container = load_embeddings(
            foundation_embedding_file(
                Path(args.embedding_derivative_root),
                args.embedding_model_key,
                args.desc,
                task=args.task,
            ),
            representation="recording" if representation == "recording" else "epoch",
            conditions=condition,
            subjects=subjects,
            subject_metadata_df=meta_df,
            subject_key=args.subject_col if meta_df is not None else None,
        )
    else:
        raise ValueError(f"Unsupported input mode '{input_mode}'.")
    container = filter_cohort_container(
        container,
        group_filters=args.group_filters,
        filter_col=args.filter_col,
        filter_val=args.filter_val,
        matched=bool(getattr(args, "matched", False)),
    )
    if args.balance_target:
        container = container.balance(
            target=args.balance_target,
            strategy=args.balance_strategy,
        )

    represented = container
    if effective_input_mode == "foundation_embeddings" and representation == "subject":
        container = represented = container.aggregate(by=args.subject_col, stats="mean")
    if effective_input_mode == "raw":
        if representation == "subject":
            container = container.aggregate(by=args.subject_col, stats="mean")
        elif representation == "recording":
            container = aggregate_recordings(container)
        elif representation != "epoch":
            raise ValueError(
                f"Raw representation must be one of {list(AGGREGATION_LEVELS)}, "
                f"got '{representation}'."
            )

        if analysis_mode == "flat":
            represented = container.flatten(preserve="obs")
        elif analysis_mode == "sensor":
            represented = container
        else:
            raise ValueError(
                f"Raw inputs support analysis_mode 'flat' or 'sensor', got '{analysis_mode}'."
            )
    represented.meta = dict(represented.meta)
    represented.meta.update(
        {
            "input_mode": input_mode,
            "source_input_mode": effective_input_mode,
            "condition": condition,
            "analysis_mode": analysis_mode,
            "representation": representation,
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

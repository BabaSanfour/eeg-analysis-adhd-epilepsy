#!/usr/bin/env python3
"""Materialize globally aligned embedding variants for descriptive analyses."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from coco_pipe.decoding import redact_sensitive
from coco_pipe.decoding.foundation_models import FoundationEmbeddingResult
from coco_pipe.dim_reduction import occurrence_aligned_positions
from coco_pipe.io import (
    discover_embedding_derivatives,
    embedding_sidecar_path,
    load_embedding_derivatives,
    normalize_subject_value,
    read_json,
    read_table,
    save_embedding_outputs,
)
from coco_pipe.transforms.subject_alignment import make_subject_transform
from coco_pipe.utils import slug, stable_hash

from eeg_adhd_epilepsy.analysis.dataset import attach_subject_metadata
from eeg_adhd_epilepsy.analysis.variance_diagnostics import (
    DiagnosticTask,
    build_diagnostic_tasks,
    score_streamed_variance_diagnostics,
    score_variance_diagnostics,
    skipped_variance_diagnostics,
    write_variance_diagnostics,
)
from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    _sanitize_bids_token,
    get_bids_derivative_variant_path,
    get_derivative_root,
)
from eeg_adhd_epilepsy.utils.artifacts import freeze_config_used, write_text_atomic
from eeg_adhd_epilepsy.utils.config import resolve_cli_config

LOGGER = logging.getLogger(__name__)

_VOLATILE_KEYS = {
    "bids_root",
    "metadata",
    "reports_root",
    "source_embedding_root",
    "overwrite",
}


def _complete_subject_output_paths(
    source_paths: list[Path],
    *,
    source_root: Path,
    transform_name: str,
    overwrite: bool,
) -> list[Path] | None:
    """Return output paths only when every NPZ/JSON pair for a subject exists."""
    if overwrite:
        return None
    output_paths: list[Path] = []
    for source_path in source_paths:
        output_path = get_bids_derivative_variant_path(
            source_path,
            source_root,
            processing=f"align{transform_name}",
            suffix="embedding",
        )
        output_paths.append(output_path)
        if not output_path.exists() or not embedding_sidecar_path(output_path).exists():
            return None
    return output_paths


def _load_completed_subject_rows(
    output_paths: list[Path],
    *,
    pooled_ids: np.ndarray,
    model_key: str,
    transform_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    container = load_embedding_derivatives(
        output_paths,
        representation="epoch",
        model_key=f"{model_key}_align-{transform_name}",
    )
    pooled_rows = occurrence_aligned_positions(pooled_ids, container.ids)
    if pooled_rows is None:
        raise ValueError(
            "Existing aligned artifacts do not match the current source observations; "
            "rerun with overwrite enabled."
        )
    return np.asarray(container.X, dtype=np.float32), np.asarray(pooled_rows, dtype=int)


def _save_aligned_artifact(
    *,
    source_path: Path,
    source_metadata: Mapping[str, Any],
    aligned_windows: np.ndarray,
    window_start: np.ndarray,
    window_stop: np.ndarray,
    window_index: np.ndarray,
    source_root: Path,
    model_key: str,
    transform_name: str,
    transform_fingerprint: str,
    params: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    output_path = get_bids_derivative_variant_path(
        source_path,
        source_root,
        processing=f"align{transform_name}",
        suffix="embedding",
    )

    relative_source = str(source_path.relative_to(source_root))
    aligned_model_key = f"{model_key}_align-{transform_name}"
    source_token_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key.startswith("token_") or key == "token_layout"
    }
    result = FoundationEmbeddingResult(
        window_embeddings=np.asarray(aligned_windows, dtype=np.float32),
        recording_embedding=np.asarray(aligned_windows.mean(axis=0), dtype=np.float32),
        window_start=np.asarray(window_start, dtype=np.int64),
        window_stop=np.asarray(window_stop, dtype=np.int64),
        window_index=np.asarray(window_index, dtype=np.int64),
        metadata={
            **{
                key: value
                for key, value in source_metadata.items()
                if key not in source_token_metadata
                and key not in {"arrays", "artifact_kind", "representation"}
            },
            "model_key": aligned_model_key,
            "source_model_key": model_key,
            "source_artifact": relative_source,
            "subject_transform": transform_name,
            "subject_transform_params": dict(params),
            "subject_transform_fingerprint": transform_fingerprint,
            "alignment_scope": "global_descriptive",
            **({"source_token_metadata": source_token_metadata} if source_token_metadata else {}),
        },
    )
    save_embedding_outputs(result, output_path, overwrite=overwrite)
    return output_path


def _align_and_save_ra_by_subject(
    token_paths_by_subject: Mapping[str, list[Path]],
    *,
    pooled_ids: np.ndarray,
    source_root: Path,
    model_key: str,
    params: Mapping[str, Any],
    overwrite: bool,
) -> tuple[list[Path], dict[str, Any]]:
    """Write pending RA embeddings without loading completed subject outputs.

    For each subject, native token derivatives are loaded together, reshaped to
    ``(window, token, feature)``, aligned, split back into source artifacts, and
    saved immediately.
    """
    output_paths: list[Path] = []
    first_token_path = next(
        (path for paths in token_paths_by_subject.values() for path in paths), None
    )
    if first_token_path is None:
        raise ValueError("RA received no source artifacts.")
    first_token_metadata = read_json(embedding_sidecar_path(first_token_path))
    token_axes = list(first_token_metadata["token_axes"])
    token_shape = list(first_token_metadata["token_shape"])
    token_feature_axis = str(first_token_metadata["token_feature_axis"])
    token_feature_count = int(token_shape[token_axes.index(token_feature_axis)])
    n_features = token_feature_count * (token_feature_count + 1) // 2
    n_created_observations = 0
    transform_fingerprint = make_subject_transform("ra", **params).fingerprint()
    completed_outputs = {
        subject_id: _complete_subject_output_paths(
            subject_paths,
            source_root=source_root,
            transform_name="ra",
            overwrite=overwrite,
        )
        for subject_id, subject_paths in sorted(token_paths_by_subject.items())
    }
    n_completed = sum(paths is not None for paths in completed_outputs.values())
    LOGGER.info(
        "Skipped %d completed subject(s) for transform 'ra'; %d pending.",
        n_completed,
        len(completed_outputs) - n_completed,
    )
    for subject_id, subject_paths in sorted(token_paths_by_subject.items()):
        existing_paths = completed_outputs[subject_id]
        if existing_paths is not None:
            output_paths.extend(existing_paths)
            continue

        token_container = load_embedding_derivatives(
            subject_paths,
            representation="token",
            model_key=model_key,
        )
        artifact_metadata = token_container.meta["artifact_metadata"]
        token_feature_axis = str(next(iter(artifact_metadata.values()))["token_feature_axis"])
        native_tokens = np.moveaxis(
            np.asarray(token_container.X),
            token_container.dims.index(token_feature_axis),
            -1,
        )
        subject_tokens = native_tokens.reshape(len(native_tokens), -1, native_tokens.shape[-1])
        groups = np.full(len(subject_tokens), subject_id, dtype=object)
        aligner = make_subject_transform("ra", **params)
        aligned_subject_windows = np.asarray(
            aligner.fit_transform(subject_tokens, groups=groups), dtype=np.float32
        )
        pooled_rows = occurrence_aligned_positions(pooled_ids, token_container.ids)
        if pooled_rows is None:
            raise ValueError("RA token observations do not match the pooled observations.")
        if n_features != aligned_subject_windows.shape[1]:
            raise ValueError("RA artifacts have inconsistent feature dimensions.")
        n_created_observations += len(aligned_subject_windows)
        window_start = np.asarray(token_container.coords["window_start"])
        window_stop = np.asarray(token_container.coords["window_stop"])
        window_index = np.asarray(token_container.coords["window_index"])
        for artifact_path, rows in token_container.observation_frame().groupby(
            "artifact_path", sort=False
        ):
            artifact_path = str(artifact_path)
            positions = rows.index.to_numpy(dtype=int)
            output_paths.append(
                _save_aligned_artifact(
                    source_path=Path(artifact_path),
                    source_metadata=artifact_metadata[artifact_path],
                    aligned_windows=aligned_subject_windows[positions],
                    window_start=window_start[positions],
                    window_stop=window_stop[positions],
                    window_index=window_index[positions],
                    source_root=source_root,
                    model_key=model_key,
                    transform_name="ra",
                    transform_fingerprint=transform_fingerprint,
                    params=params,
                    overwrite=overwrite,
                )
            )
    if not output_paths:
        raise ValueError("RA received no source artifacts.")
    return output_paths, {
        "n_observations": len(pooled_ids),
        "n_features": n_features,
        "dtype": "float32",
        "n_created_observations": n_created_observations,
    }


class _AlignedArtifactBatchSource:
    """Re-iterable aligned-artifact reader with exact pooled-row validation."""

    def __init__(
        self,
        artifact_paths: list[Path],
        *,
        pooled_ids: np.ndarray,
        model_key: str,
        transform_name: str,
        max_batch_bytes: int,
        expected_n_features: int | None = None,
    ) -> None:
        if max_batch_bytes < 1:
            raise ValueError("diagnostic_batch_max_bytes must be at least 1.")
        self.artifact_paths = tuple(Path(path) for path in artifact_paths)
        self.pooled_ids = np.asarray(pooled_ids, dtype=object)
        self.model_key = model_key
        self.transform_name = transform_name
        self.max_batch_bytes = int(max_batch_bytes)
        self.expected_n_features = expected_n_features

    @property
    def n_observations(self) -> int:
        return len(self.pooled_ids)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        pooled_positions: dict[str, list[int]] = {}
        for position, observation_id in enumerate(self.pooled_ids.astype(str)):
            pooled_positions.setdefault(observation_id, []).append(position)
        occurrence_counts: dict[str, int] = {}
        assigned = np.zeros(self.n_observations, dtype=bool)
        n_features = self.expected_n_features
        for artifact_path in self.artifact_paths:
            container = load_embedding_derivatives(
                [artifact_path],
                representation="epoch",
                model_key=f"{self.model_key}_align-{self.transform_name}",
            )
            values = np.asarray(container.X)
            if values.ndim != 2:
                raise ValueError(f"Aligned artifact must be two-dimensional: {artifact_path}.")
            if n_features is None:
                n_features = int(values.shape[1])
            elif values.shape[1] != n_features:
                raise ValueError(
                    "Aligned artifacts have inconsistent feature dimensions: "
                    f"expected {n_features}, got {values.shape[1]} in {artifact_path}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Aligned artifact contains nonfinite values: {artifact_path}.")
            artifact_rows = np.empty(len(container.ids), dtype=int)
            for index, observation_id in enumerate(np.asarray(container.ids).astype(str)):
                occurrence = occurrence_counts.get(observation_id, 0)
                available = pooled_positions.get(observation_id, [])
                if not available:
                    raise ValueError(
                        "Aligned artifacts contain an observation absent from the pooled "
                        f"source: {observation_id!r} in {artifact_path}."
                    )
                if occurrence >= len(available):
                    raise ValueError(
                        "Aligned artifacts contain duplicate pooled assignments for "
                        f"observation {observation_id!r}."
                    )
                artifact_rows[index] = available[occurrence]
                occurrence_counts[observation_id] = occurrence + 1
            if assigned[artifact_rows].any() or len(np.unique(artifact_rows)) != len(artifact_rows):
                raise ValueError("Aligned artifacts contain duplicate pooled assignments.")
            assigned[artifact_rows] = True
            # Statistics use float64 sufficient statistics even when artifacts are
            # float32, so size chunks for their largest in-memory representation.
            row_bytes = int(values.shape[1]) * max(int(values.dtype.itemsize), 8)
            if row_bytes > self.max_batch_bytes:
                raise ValueError(
                    "One aligned feature row exceeds diagnostic_batch_max_bytes: "
                    f"{row_bytes} > {self.max_batch_bytes} bytes."
                )
            rows_per_batch = max(1, self.max_batch_bytes // max(row_bytes, 1))
            for start in range(0, len(values), rows_per_batch):
                stop = min(start + rows_per_batch, len(values))
                yield artifact_rows[start:stop], values[start:stop]
        if not assigned.all():
            missing = int((~assigned).sum())
            raise ValueError(
                "Aligned artifacts do not exactly cover the pooled observations: "
                f"{missing} row(s) are missing."
            )


def _align_and_save_vector_by_subject(
    transform: Any,
    pooled_embeddings: np.ndarray,
    subjects: np.ndarray,
    observations: Any,
    artifact_metadata: Mapping[str, Mapping[str, Any]],
    *,
    pooled_ids: np.ndarray,
    window_start: np.ndarray,
    window_stop: np.ndarray,
    window_index: np.ndarray,
    source_root: Path,
    model_key: str,
    transform_name: str,
    params: Mapping[str, Any],
    overwrite: bool,
) -> tuple[np.ndarray | None, list[Path], Any]:
    """Reuse complete subjects and materialize only pending vector outputs."""
    transform_fingerprint = transform.fingerprint()
    source_paths_by_subject: dict[str, list[Path]] = {}
    positions_by_subject: dict[str, np.ndarray] = {}
    for subject_id in sorted(np.unique(subjects)):
        positions = np.flatnonzero(subjects == subject_id)
        positions_by_subject[str(subject_id)] = positions
        subject_observations = observations.iloc[positions]
        source_paths_by_subject[str(subject_id)] = [
            Path(value)
            for value in subject_observations["artifact_path"].drop_duplicates().tolist()
        ]

    completed_outputs = {
        subject_id: _complete_subject_output_paths(
            source_paths,
            source_root=source_root,
            transform_name=transform_name,
            overwrite=overwrite,
        )
        for subject_id, source_paths in source_paths_by_subject.items()
    }
    n_completed = sum(paths is not None for paths in completed_outputs.values())
    LOGGER.info(
        "Skipped %d completed subject(s) for transform '%s'; %d pending.",
        n_completed,
        transform_name,
        len(completed_outputs) - n_completed,
    )

    pending_positions = [
        positions_by_subject[subject_id]
        for subject_id, outputs in completed_outputs.items()
        if outputs is None
    ]
    if pending_positions:
        fit_positions = (
            np.concatenate(pending_positions)
            if transform_name in {"ea_coral", "ea_mean"}
            else np.arange(len(pooled_embeddings))
        )
        transform.fit(
            pooled_embeddings[fit_positions],
            groups=subjects[fit_positions],
        )
        if bool(getattr(transform, "degenerate_", False)):
            return None, [], transform

    aligned_embeddings = np.empty_like(pooled_embeddings, dtype=np.float32)
    assigned_rows = np.zeros(len(pooled_embeddings), dtype=bool)
    output_paths: list[Path] = []
    for subject_id, positions in positions_by_subject.items():
        existing_paths = completed_outputs[subject_id]
        if existing_paths is not None:
            aligned_subject_windows, pooled_rows = _load_completed_subject_rows(
                existing_paths,
                pooled_ids=pooled_ids,
                model_key=model_key,
                transform_name=transform_name,
            )
            if aligned_subject_windows.shape[1] != aligned_embeddings.shape[1]:
                raise ValueError(
                    f"Existing {transform_name} artifacts have an inconsistent feature dimension."
                )
            aligned_embeddings[pooled_rows] = aligned_subject_windows
            assigned_rows[pooled_rows] = True
            output_paths.extend(existing_paths)
            continue

        aligned_subject_windows = np.asarray(
            transform.transform(
                pooled_embeddings[positions],
                groups=subjects[positions],
            ),
            dtype=np.float32,
        )
        aligned_embeddings[positions] = aligned_subject_windows
        assigned_rows[positions] = True
        subject_observations = observations.iloc[positions]
        for artifact_path, rows in subject_observations.groupby("artifact_path", sort=False):
            artifact_path = str(artifact_path)
            artifact_positions = rows.index.to_numpy(dtype=int)
            output_paths.append(
                _save_aligned_artifact(
                    source_path=Path(artifact_path),
                    source_metadata=dict(artifact_metadata[artifact_path]),
                    aligned_windows=aligned_embeddings[artifact_positions],
                    window_start=window_start[artifact_positions],
                    window_stop=window_stop[artifact_positions],
                    window_index=window_index[artifact_positions],
                    source_root=source_root,
                    model_key=model_key,
                    transform_name=transform_name,
                    transform_fingerprint=transform_fingerprint,
                    params=params,
                    overwrite=overwrite,
                )
            )
    if not assigned_rows.all():
        raise ValueError(f"{transform_name} outputs do not exactly cover the pooled observations.")
    return aligned_embeddings, output_paths, transform


def _load_alignment_progress(
    path: Path,
    *,
    config_fingerprint: str,
    source_inventory_signature: str,
    diagnostics_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Load a compatible transform checkpoint or initialize a fresh one."""
    identity = {
        "config_fingerprint": config_fingerprint,
        "source_inventory_signature": source_inventory_signature,
        "diagnostics_path": str(diagnostics_path),
    }
    if not overwrite and path.exists():
        try:
            existing = read_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            existing = {}
        if all(existing.get(key) == value for key, value in identity.items()):
            return {
                **identity,
                "schema_version": 1,
                "completed_diagnostics": list(existing.get("completed_diagnostics", [])),
                "materialized_transforms": list(existing.get("materialized_transforms", [])),
                "skipped_transforms": dict(existing.get("skipped_transforms", {})),
                "artifacts_by_transform": dict(existing.get("artifacts_by_transform", {})),
            }
    return {
        **identity,
        "schema_version": 1,
        "completed_diagnostics": [],
        "materialized_transforms": [],
        "skipped_transforms": {},
        "artifacts_by_transform": {},
    }


def _write_alignment_progress(path: Path, progress: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(dict(progress), indent=2))


def _diagnostic_checkpoint_exists(
    diagnostics_path: Path,
    *,
    transform: str,
    cohort_name: str,
    tasks: list[DiagnosticTask],
) -> bool:
    if not diagnostics_path.exists():
        return False
    try:
        frame = read_table(diagnostics_path, sep=",")
    except (OSError, ValueError, TypeError):
        return False
    required = {"transform", "cohort_name", "selection_fingerprint"}
    if not required.issubset(frame.columns):
        return False
    selected = frame[
        (frame["transform"].astype(str) == transform)
        & (frame["cohort_name"].astype(str) == cohort_name)
    ]
    expected = {task.selection_fingerprint for task in tasks}
    observed = set(selected["selection_fingerprint"].dropna().astype(str))
    return bool(expected) and expected <= observed


def _transform_checkpoint_complete(
    progress: Mapping[str, Any],
    *,
    transform: str,
    source_root: Path,
    diagnostics_path: Path,
    cohort_name: str,
    tasks: list[DiagnosticTask],
) -> bool:
    if transform not in progress.get("completed_diagnostics", []):
        return False
    if not _diagnostic_checkpoint_exists(
        diagnostics_path,
        transform=transform,
        cohort_name=cohort_name,
        tasks=tasks,
    ):
        return False
    if transform == "none" or transform in progress.get("skipped_transforms", {}):
        return True
    relative_paths = progress.get("artifacts_by_transform", {}).get(transform, [])
    return bool(relative_paths) and all(
        (source_root / relative_path).exists()
        and (source_root / relative_path).with_suffix(".json").exists()
        for relative_path in relative_paths
    )


def _checkpoint_transform(
    progress: dict[str, Any],
    path: Path,
    *,
    transform: str,
    artifact_paths: list[Path],
    source_root: Path,
    skipped_reason: str | None = None,
) -> None:
    completed = progress["completed_diagnostics"]
    if transform not in completed:
        completed.append(transform)
    if skipped_reason is None and transform != "none":
        materialized = progress["materialized_transforms"]
        if transform not in materialized:
            materialized.append(transform)
        progress["skipped_transforms"].pop(transform, None)
    elif skipped_reason is not None:
        progress["skipped_transforms"][transform] = skipped_reason
        progress["materialized_transforms"] = [
            value for value in progress["materialized_transforms"] if value != transform
        ]
    progress["artifacts_by_transform"][transform] = [
        str(artifact_path.relative_to(source_root)) for artifact_path in artifact_paths
    ]
    _write_alignment_progress(path, progress)


def run(config: dict[str, Any]) -> Path:
    """Materialize configured global variants and their variance diagnostics."""
    source_root = Path(config["source_embedding_root"]).expanduser()
    model_key = str(config["embedding_model_key"])
    transforms = tuple(str(value).lower() for value in config["transforms"])
    source_pooling = str(config["source_pooling"])

    pooled_paths = [
        path
        for path in discover_embedding_derivatives(
            source_root,
            model_key=model_key,
            kind="embedding",
        )
        if read_json(embedding_sidecar_path(path)).get("within_window_pooling") == source_pooling
    ]
    if not pooled_paths:
        raise FileNotFoundError(
            f"No {model_key!r} pooled derivatives use source_pooling={source_pooling!r}."
        )

    token_paths: list[Path] = []
    if "ra" in transforms:
        token_paths = discover_embedding_derivatives(
            source_root,
            model_key=model_key,
            kind="token",
        )
        if not token_paths:
            raise FileNotFoundError(
                f"RA was requested for {model_key!r}, but no native token derivatives "
                "were found. Re-extract this model with store_tokens: true."
            )

    container = load_embedding_derivatives(
        pooled_paths,
        representation="epoch",
        model_key=model_key,
    )
    pooled_embeddings = np.asarray(container.X, dtype=np.float32)
    observations = container.observation_frame()
    subjects = np.asarray(
        [normalize_subject_value(value) for value in container.coords["subject"]],
        dtype=object,
    )
    artifact_metadata = container.meta["artifact_metadata"]
    window_start = np.asarray(container.coords["window_start"])
    window_stop = np.asarray(container.coords["window_stop"])
    window_index = np.asarray(container.coords["window_index"])

    token_paths_by_subject: dict[str, list[Path]] = {}
    pooled_ids = np.asarray(container.ids, dtype=object)
    if token_paths:
        n_token_windows = 0
        for path in token_paths:
            token_metadata = read_json(embedding_sidecar_path(path))
            subject_id = normalize_subject_value(token_metadata["subject"])
            token_paths_by_subject.setdefault(subject_id, []).append(path)
            n_token_windows += int(token_metadata["token_shape"][0])
        if n_token_windows != len(pooled_embeddings):
            raise ValueError(
                "Native token observations do not exactly cover the selected pooled variant: "
                f"{n_token_windows} token windows != {len(pooled_embeddings)} pooled windows."
            )

    cohort_metadata = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )
    diagnostic_container = (
        attach_subject_metadata(container, cohort_metadata, str(config["subject_col"]))
        if cohort_metadata is not None
        else container
    )
    diagnostic_tasks = build_diagnostic_tasks(diagnostic_container, config)
    diagnostics_root = get_derivative_root(
        Path(config["bids_root"]).expanduser(),
        DerivativeStage.VARIANCE_DIAGNOSTICS,
    ) / slug(model_key)
    diagnostics_path = diagnostics_root / "variance_diagnostics.csv"
    overwrite = bool(config["overwrite"])
    alignment_config = redact_sensitive(
        {key: value for key, value in config.items() if key not in _VOLATILE_KEYS}
    )
    freeze_config_used(
        alignment_config,
        diagnostics_root,
        overwrite=overwrite,
        mismatch_message=(
            "Existing subject-alignment outputs use a different alignment "
            "configuration. Use overwrite to replace them."
        ),
    )
    source_inventory_paths = {*pooled_paths, *token_paths}
    source_inventory_signature = stable_hash(
        sorted(str(path.relative_to(source_root)) for path in source_inventory_paths),
        length=16,
    )
    config_fingerprint = stable_hash(alignment_config, length=16)
    progress_path = source_root / (
        f"_alignment_{_sanitize_bids_token(model_key, 'model_key')}_progress.json"
    )
    progress = _load_alignment_progress(
        progress_path,
        config_fingerprint=config_fingerprint,
        source_inventory_signature=source_inventory_signature,
        diagnostics_path=diagnostics_path,
        overwrite=bool(config["overwrite"]),
    )
    _write_alignment_progress(progress_path, progress)
    cohort_name = str(config["dataset_name"])
    if _transform_checkpoint_complete(
        progress,
        transform="none",
        source_root=source_root,
        diagnostics_path=diagnostics_path,
        cohort_name=cohort_name,
        tasks=diagnostic_tasks,
    ):
        LOGGER.info("Resuming raw variance diagnostics from checkpoint.")
    else:
        raw_diagnostics = score_variance_diagnostics(
            pooled_embeddings,
            diagnostic_tasks,
            config,
            transform="none",
        )
        write_variance_diagnostics(raw_diagnostics, diagnostics_root)
        _checkpoint_transform(
            progress,
            progress_path,
            transform="none",
            artifact_paths=[],
            source_root=source_root,
        )

    transform_params = config.get("transform_params", {}) or {}
    for transform_name in transforms:
        if transform_name == "none":
            continue
        if _transform_checkpoint_complete(
            progress,
            transform=transform_name,
            source_root=source_root,
            diagnostics_path=diagnostics_path,
            cohort_name=cohort_name,
            tasks=diagnostic_tasks,
        ):
            LOGGER.info("Resuming completed transform %s from checkpoint.", transform_name)
            continue
        params = dict(transform_params.get(transform_name, {}) or {})
        LOGGER.info("Materializing global subject transform %s.", transform_name)
        output_paths: list[Path] = []
        streamed_feature_batches: _AlignedArtifactBatchSource | None = None

        if transform_name == "ra":
            output_paths, feature_metadata = _align_and_save_ra_by_subject(
                token_paths_by_subject,
                pooled_ids=pooled_ids,
                source_root=source_root,
                model_key=model_key,
                params=params,
                overwrite=overwrite,
            )
            streamed_feature_batches = _AlignedArtifactBatchSource(
                output_paths,
                pooled_ids=pooled_ids,
                model_key=model_key,
                transform_name="ra",
                max_batch_bytes=int(config.get("diagnostic_batch_max_bytes", 512 * 1024**2)),
                expected_n_features=feature_metadata["n_features"],
            )
        else:
            transform = make_subject_transform(transform_name, **params)
            aligned_embeddings, output_paths, transform = _align_and_save_vector_by_subject(
                transform,
                pooled_embeddings,
                subjects,
                observations,
                artifact_metadata,
                pooled_ids=pooled_ids,
                window_start=window_start,
                window_stop=window_stop,
                window_index=window_index,
                source_root=source_root,
                model_key=model_key,
                transform_name=transform_name,
                params=params,
                overwrite=overwrite,
            )
            if aligned_embeddings is None:
                rank = int(getattr(transform, "rank_", pooled_embeddings.shape[1]))
                n_subjects = int(getattr(transform, "n_subjects_", len(np.unique(subjects))))
                reason = (
                    "Transform was skipped because its fitted subject projector was "
                    f"marked degenerate (rank {rank}/{pooled_embeddings.shape[1]}, "
                    f"{n_subjects} subjects); the aligned representation would be "
                    "collapsed or scientifically unreliable."
                )
                LOGGER.warning(
                    "%s Skipped %s; existing artifacts, if any, were left unchanged.",
                    reason,
                    transform_name,
                )
                skipped_rows = skipped_variance_diagnostics(
                    diagnostic_tasks,
                    config,
                    transform=transform_name,
                    reason=reason,
                    n_features=pooled_embeddings.shape[1],
                )
                write_variance_diagnostics(skipped_rows, diagnostics_root)
                _checkpoint_transform(
                    progress,
                    progress_path,
                    transform=transform_name,
                    artifact_paths=[],
                    source_root=source_root,
                    skipped_reason=reason,
                )
                continue

        if streamed_feature_batches is None:
            transform_diagnostics = score_variance_diagnostics(
                aligned_embeddings,
                diagnostic_tasks,
                config,
                transform=transform_name,
            )
        else:
            transform_diagnostics = score_streamed_variance_diagnostics(
                streamed_feature_batches,
                diagnostic_tasks,
                config,
                transform=transform_name,
                n_observations=len(pooled_embeddings),
            )
        write_variance_diagnostics(transform_diagnostics, diagnostics_root)
        _checkpoint_transform(
            progress,
            progress_path,
            transform=transform_name,
            artifact_paths=output_paths,
            source_root=source_root,
        )

    materialized = set(progress["materialized_transforms"])
    materialized_transforms = [name for name in transforms if name in materialized]
    skipped_transforms = {
        name: progress["skipped_transforms"][name]
        for name in transforms
        if name in progress["skipped_transforms"]
    }
    write_text_atomic(
        source_root / f"_alignment_{_sanitize_bids_token(model_key, 'model_key')}_complete.json",
        json.dumps(
            {
                "config_fingerprint": config_fingerprint,
                "source_inventory_signature": source_inventory_signature,
                "transforms": list(transforms),
                "materialized_transforms": materialized_transforms,
                "skipped_transforms": skipped_transforms,
            },
            indent=2,
        ),
    )
    return source_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_config", type=Path, required=True)
    parser.add_argument("--analysis_config", type=Path, required=True)
    parser.add_argument("--bids_root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source_embedding_root", type=Path)
    parser.add_argument("--embedding_model_key", required=True)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    bids_root = args.bids_root.expanduser()
    cohort_config = args.cohort_config.expanduser()
    analysis_config = args.analysis_config.expanduser()
    metadata = args.metadata.expanduser() if args.metadata else None
    source_embedding_root = (
        args.source_embedding_root.expanduser()
        if args.source_embedding_root
        else get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)
    )

    config = resolve_cli_config(
        cohort_config=cohort_config,
        analysis_config=analysis_config,
        bids_root=str(bids_root),
        metadata=str(metadata) if metadata else None,
        source_embedding_root=str(source_embedding_root),
        embedding_model_key=args.embedding_model_key,
        overwrite=args.overwrite,
    )
    run(config)


if __name__ == "__main__":
    main()

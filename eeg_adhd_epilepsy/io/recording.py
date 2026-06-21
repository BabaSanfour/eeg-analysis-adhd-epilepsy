"""Recording-level grouping helpers shared by analysis loaders."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from coco_pipe.io import DataContainer

__all__ = [
    "add_recording_group_columns",
    "clean_group_value",
    "ensure_recording_id",
    "infer_bids_entity_from_text",
    "recording_id_from_parts",
]


def infer_bids_entity_from_text(value: object, entity: str) -> str | None:
    """Infer a BIDS entity value from a path, filename, or observation id."""
    match = re.search(rf"(?:^|[_/]){entity}-([^_/]+)", str(value))
    if match:
        return match.group(1)
    return None


def clean_group_value(value: object, *, missing: str) -> str:
    """Return a stable string value for grouping coordinates."""
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, (np.ndarray, list, tuple)):
        is_missing = False
    if is_missing:
        return missing
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return missing
    return text


def _strip_entity_prefix(value: str, entity: str) -> str:
    prefix = f"{entity}-"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _clean_entity_value(value: object, entity: str, *, missing: str) -> str:
    return _strip_entity_prefix(clean_group_value(value, missing=missing), entity)


def recording_id_from_parts(
    subject: object,
    session: object = "01",
    run: object = "none",
) -> str:
    """Build the run-aware recording id used for aggregation."""
    subject_value = clean_group_value(subject, missing="unknown")
    session_value = _clean_entity_value(session, "ses", missing="01")
    run_value = _clean_entity_value(run, "run", missing="none")
    return f"{subject_value}_ses-{session_value}_run-{run_value}"


def _aligned_values(
    values: object | None,
    *,
    n_rows: int,
    fallback: Sequence[object],
) -> np.ndarray:
    if values is None:
        return np.asarray(fallback, dtype=object)
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != n_rows:
        return np.asarray(fallback, dtype=object)
    return array


def add_recording_group_columns(
    metadata_df: pd.DataFrame,
    *,
    subject_col: str = "subject",
    id_col: str = "obs_id",
    default_session: str = "01",
    default_run: str = "none",
) -> pd.DataFrame:
    """Add session, run, and recording_id columns to an observation table."""
    out = metadata_df.copy()
    n_rows = len(out)
    obs_values = (
        out[id_col].to_numpy(dtype=object)
        if id_col in out.columns
        else np.asarray([""] * n_rows, dtype=object)
    )

    session_values = _aligned_values(
        out["session"].to_numpy(dtype=object) if "session" in out.columns else None,
        n_rows=n_rows,
        fallback=[None] * n_rows,
    )
    out["session"] = np.asarray(
        [
            _clean_entity_value(
                value,
                "ses",
                missing=infer_bids_entity_from_text(obs_id, "ses") or default_session,
            )
            for value, obs_id in zip(session_values, obs_values)
        ],
        dtype=object,
    )

    run_values = _aligned_values(
        out["run"].to_numpy(dtype=object) if "run" in out.columns else None,
        n_rows=n_rows,
        fallback=[None] * n_rows,
    )
    out["run"] = np.asarray(
        [
            _clean_entity_value(
                value,
                "run",
                missing=infer_bids_entity_from_text(obs_id, "run") or default_run,
            )
            for value, obs_id in zip(run_values, obs_values)
        ],
        dtype=object,
    )

    if subject_col not in out.columns:
        raise KeyError(f"Subject column '{subject_col}' not found in metadata_df.")
    out["recording_id"] = [
        recording_id_from_parts(subject, session, run)
        for subject, session, run in zip(
            out[subject_col].to_numpy(dtype=object),
            out["session"].to_numpy(dtype=object),
            out["run"].to_numpy(dtype=object),
        )
    ]
    return out


def ensure_recording_id(
    container: DataContainer,
    subject_col: str,
    *,
    default_session: str = "01",
    default_run: str = "none",
) -> DataContainer:
    """Return a container with run-aware ``recording_id`` obs coordinates."""
    obs_len = container.X.shape[0]
    coords = dict(container.coords)

    if "recording_id" in coords:
        return container

    id_values = (
        np.asarray(container.ids, dtype=object)
        if container.ids is not None
        else np.asarray([""] * obs_len, dtype=object)
    )
    subject_fallback = id_values if container.ids is not None else np.arange(obs_len).astype(str)
    subject_values = _aligned_values(
        coords.get(subject_col, coords.get("subject")),
        n_rows=obs_len,
        fallback=subject_fallback,
    )
    subject_values = np.asarray(
        [
            clean_group_value(value, missing=str(index))
            for index, value in enumerate(subject_values)
        ],
        dtype=object,
    )

    session_values = _aligned_values(
        coords.get("session"),
        n_rows=obs_len,
        fallback=[None] * obs_len,
    )
    session_values = np.asarray(
        [
            _clean_entity_value(
                value,
                "ses",
                missing=infer_bids_entity_from_text(obs_id, "ses") or default_session,
            )
            for value, obs_id in zip(session_values, id_values)
        ],
        dtype=object,
    )

    run_values = _aligned_values(
        coords.get("run"),
        n_rows=obs_len,
        fallback=[None] * obs_len,
    )
    run_values = np.asarray(
        [
            _clean_entity_value(
                value,
                "run",
                missing=infer_bids_entity_from_text(obs_id, "run") or default_run,
            )
            for value, obs_id in zip(run_values, id_values)
        ],
        dtype=object,
    )

    coords["session"] = session_values
    coords["run"] = run_values
    coords["recording_id"] = np.asarray(
        [
            recording_id_from_parts(subject, session, run)
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
        meta=dict(container.meta or {}),
    )

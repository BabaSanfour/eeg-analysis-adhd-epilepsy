"""Temporary loaders for current REVE and CBraMod embedding exports."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from coco_pipe.io.structures import DataContainer

logger = logging.getLogger(__name__)

_EMBED_PATTERN = re.compile(
    r"^(sub-[^_]+)_desc-([^_]+)_embed_(reve|cbramod)(?:_[^.]*)?\.npy$"
)


def _normalize_subject_id(value: object) -> str:
    token = str(value).strip()
    if token.startswith("sub-"):
        token = token[4:]
    return f"{int(token):04d}"


def _discover_embedding_files(
    embeddings_root: Path,
    model: str,
    desc: str,
    subjects: Optional[Sequence[str]],
) -> list[Path]:
    subject_filter = (
        {_normalize_subject_id(subject) for subject in subjects}
        if subjects is not None
        else None
    )
    matches: list[Path] = []
    for fpath in sorted(embeddings_root.rglob("*.npy")):
        match = _EMBED_PATTERN.match(fpath.name)
        if match is None:
            continue
        subject_tag, desc_tag, model_tag = match.groups()
        if model_tag != model or desc_tag != desc:
            continue
        subject_id = _normalize_subject_id(subject_tag)
        if subject_filter is not None and subject_id not in subject_filter:
            continue
        matches.append(fpath)
    if not matches:
        raise FileNotFoundError(
            f"No {model} embeddings found in {embeddings_root} for desc='{desc}'."
        )
    return matches


def _load_embedding_pair(fpath: Path) -> tuple[np.ndarray, dict]:
    meta_path = fpath.with_name(fpath.name.replace("_embed_", "_metadata_")).with_suffix(".json")
    arr = np.load(fpath)
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return arr, meta


def _resolve_segments_csv(segments_root: Path, subject_id: str, desc: str) -> Path:
    subject_dir = segments_root / f"sub-{subject_id}" / "eeg"
    matches = sorted(subject_dir.glob(f"sub-{subject_id}*desc-{desc}_segments.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No segments.csv found for sub-{subject_id}, desc='{desc}' in {subject_dir}."
        )
    return matches[0]


def _window_bounds(
    n_rows: int,
    model: str,
    meta: dict,
    reve_segment_duration: float,
    cbramod_sampling_rate: float,
) -> pd.DataFrame:
    segment_index = np.arange(n_rows, dtype=int)
    if model == "reve":
        duration = float(reve_segment_duration)
    elif model == "cbramod":
        points_per_patch = float(meta.get("points_per_patch", cbramod_sampling_rate))
        duration = points_per_patch / float(cbramod_sampling_rate)
    else:
        raise ValueError(f"Unsupported model '{model}'.")

    starts = segment_index * duration
    stops = starts + duration
    return pd.DataFrame(
        {
            "segment_index": segment_index,
            "t_start": starts,
            "t_stop": stops,
        }
    )


def _assign_conditions(
    windows: pd.DataFrame,
    segments_df: pd.DataFrame,
    min_overlap_fraction: float,
) -> pd.Series:
    labels: list[Optional[str]] = []
    seg_starts = pd.to_numeric(segments_df["t_start"], errors="coerce").to_numpy(dtype=float)
    seg_stops = pd.to_numeric(segments_df["t_stop"], errors="coerce").to_numpy(dtype=float)
    seg_labels = segments_df["segment_type"].astype(str).to_numpy()

    for _, row in windows.iterrows():
        win_start = float(row["t_start"])
        win_stop = float(row["t_stop"])
        overlaps = np.minimum(win_stop, seg_stops) - np.maximum(win_start, seg_starts)
        overlaps = np.maximum(overlaps, 0.0)
        best_idx = int(np.argmax(overlaps)) if overlaps.size else -1
        best_overlap = float(overlaps[best_idx]) if best_idx >= 0 else 0.0
        required = min_overlap_fraction * (win_stop - win_start)
        labels.append(seg_labels[best_idx] if best_overlap >= required else None)
    return pd.Series(labels, name="condition")


def _metadata_lookup(
    metadata_df: Optional[pd.DataFrame],
    subject_col: str,
) -> Optional[pd.DataFrame]:
    if metadata_df is None:
        return None
    out = metadata_df.copy()
    out[subject_col] = out[subject_col].map(_normalize_subject_id)
    return out.drop_duplicates(subset=[subject_col])


def load_temp_dl_data(
    embeddings_root: Path,
    segments_root: Path,
    model: str,
    desc: str = "base",
    subjects: Optional[Sequence[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    subject_col: str = "Study ID",
    target_col: Optional[str] = None,
    conditions: Optional[Sequence[str]] = None,
    min_overlap_fraction: float = 0.8,
    reve_segment_duration: float = 10.0,
    cbramod_sampling_rate: float = 200.0,
    drop_unassigned: bool = True,
) -> DataContainer:
    """
    Load current REVE/CBraMod outputs and align each row to a condition using segments.csv.

    Notes
    -----
    - REVE rows are treated as fixed contiguous windows of `reve_segment_duration`.
    - CBraMod rows are treated as contiguous patches with duration
      `points_per_patch / cbramod_sampling_rate`.
    - Current CBraMod extraction only covers the first extraction block written by the
      extractor, so condition coverage may be incomplete.
    """
    if model not in {"reve", "cbramod"}:
        raise ValueError("model must be one of {'reve', 'cbramod'}")

    files = _discover_embedding_files(Path(embeddings_root), model, desc, subjects)
    metadata_lookup = _metadata_lookup(metadata_df, subject_col)
    allowed_conditions = set(conditions) if conditions is not None else None

    arrays: list[np.ndarray] = []
    row_frames: list[pd.DataFrame] = []
    component_sizes: Optional[tuple[int, ...]] = None

    for fpath in files:
        match = _EMBED_PATTERN.match(fpath.name)
        if match is None:
            continue
        subject_tag, _, _ = match.groups()
        subject_id = _normalize_subject_id(subject_tag)
        arr, meta = _load_embedding_pair(fpath)

        if arr.ndim not in {2, 3}:
            raise ValueError(f"Unsupported embedding shape {arr.shape} in {fpath}")

        current_component_sizes = arr.shape[1:]
        if component_sizes is None:
            component_sizes = current_component_sizes
        elif component_sizes != current_component_sizes:
            raise ValueError(
                f"Inconsistent embedding shapes across subjects: "
                f"{component_sizes} vs {current_component_sizes}"
            )

        windows = _window_bounds(
            n_rows=arr.shape[0],
            model=model,
            meta=meta,
            reve_segment_duration=reve_segment_duration,
            cbramod_sampling_rate=cbramod_sampling_rate,
        )
        segments_df = pd.read_csv(_resolve_segments_csv(Path(segments_root), subject_id, desc))
        windows["condition"] = _assign_conditions(
            windows=windows,
            segments_df=segments_df,
            min_overlap_fraction=min_overlap_fraction,
        )
        windows[subject_col] = subject_id
        windows["dl_model"] = model
        windows["desc"] = desc
        windows["sample_id"] = [
            f"{subject_id}_{model}_{idx}" for idx in windows["segment_index"].astype(int)
        ]

        if metadata_lookup is not None:
            windows = windows.merge(metadata_lookup, on=subject_col, how="left")

        if drop_unassigned:
            keep_mask = windows["condition"].notna().to_numpy()
            windows = windows.loc[keep_mask].reset_index(drop=True)
            arr = arr[keep_mask]

        if allowed_conditions is not None:
            keep_mask = windows["condition"].isin(allowed_conditions).to_numpy()
            windows = windows.loc[keep_mask].reset_index(drop=True)
            arr = arr[keep_mask]

        if len(windows) == 0:
            logger.info(f"Skipping {fpath.name}: no rows survived condition alignment.")
            continue

        arrays.append(arr)
        row_frames.append(windows)

    if not arrays:
        raise RuntimeError("No aligned DL embeddings were loaded.")

    X = np.concatenate(arrays, axis=0)
    rows = pd.concat(row_frames, ignore_index=True)
    ids = rows["sample_id"].astype(str).to_numpy()

    if X.ndim == 2:
        dims = ("obs", "feature")
        coords = {
            "obs": ids,
            "feature": np.arange(X.shape[1]),
        }
    else:
        dims = ("obs", "component", "feature")
        coords = {
            "obs": ids,
            "component": np.arange(X.shape[1]),
            "feature": np.arange(X.shape[2]),
        }

    for col in rows.columns:
        if col != "sample_id":
            coords[col] = rows[col].to_numpy()

    y = None
    if target_col is not None:
        if target_col not in rows.columns:
            raise KeyError(f"target_col '{target_col}' not found in merged DL rows.")
        y = rows[target_col].astype(str).to_numpy()

    return DataContainer(
        X=X,
        dims=dims,
        coords=coords,
        y=y,
        ids=ids,
        meta={
            "source": "temp_dl_loader",
            "model": model,
            "desc": desc,
        },
    )

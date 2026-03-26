#!/usr/bin/env python3
"""Shared helpers for analysis scripts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from coco_pipe.io.structures import DataContainer
from coco_pipe.report.core import ImageElement, PlotlyElement, Section
from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz.plotly_utils import plot_embedding_interactive
from eeg_adhd_epilepsy.utils.metadata_schema import EPILEPSY_MED_COLS

logger = logging.getLogger(__name__)

PLOT_META_EXCLUDED_COLUMNS = {
    "obs",
    "channel",
    "time",
    "study_id",
    "patient_id",
    "eeg_date",
    *EPILEPSY_MED_COLS,
}

REPRESENTATION_CONFIG = {
    "epoch_flat": (False, "flat"),
    "epoch_time_as_sample": (False, "time_as_sample"),
    "epoch_scalar_mean": (False, "epoch_scalar_mean"),
    "subject_flat": (True, "flat"),
    "subject_time_as_sample": (True, "time_as_sample"),
    "subject_scalar_mean": (True, "epoch_scalar_mean"),
}

_SENSOR_DESCRIPTOR_FILES = (
    "_SUCCESS",
    "sensor_descriptor_bundle.npz",
    "sensor_epoch_features.csv",
    "sensor_epoch_features.parquet",
    "sensor_epoch_features_feature_columns.json",
    "sensor_subject_features.csv",
    "sensor_subject_features.parquet",
    "sensor_subject_features_feature_columns.json",
    "failures.csv",
)

_POOLED_DESCRIPTOR_FILES = (
    "pooled_epoch_features.csv",
    "pooled_epoch_features.parquet",
    "pooled_epoch_features_feature_columns.json",
    "pooled_subject_features.csv",
    "pooled_subject_features.parquet",
    "pooled_subject_features_feature_columns.json",
)


def required_descriptor_files(include_pooled: bool) -> tuple[str, ...]:
    """Return the expected shard file set for descriptor outputs."""
    if include_pooled:
        return _SENSOR_DESCRIPTOR_FILES + _POOLED_DESCRIPTOR_FILES
    return _SENSOR_DESCRIPTOR_FILES


def save_table(
    df: pd.DataFrame,
    base_path: Path,
    feature_columns: list[str] | None = None,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    if feature_columns is not None:
        (base_path.parent / f"{base_path.name}_feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2),
            encoding="utf-8",
        )


def coerce_sample_vector(values, n_samples: int) -> Optional[np.ndarray]:
    """Return 1D array only if values are sample-aligned."""
    if values is None:
        return None
    arr = np.asarray(values).ravel()
    return arr if arr.shape[0] == n_samples else None


def add_embedding_plot(
    section: Section,
    embedding: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    meta: Optional[Dict[str, np.ndarray]],
    title: str,
    dimensions: int,
    interactive: bool,
) -> None:
    """Add one embedding plot element (2D/3D) to a report section."""
    if embedding is None:
        return

    if interactive:
        fig = plot_embedding_interactive(
            embedding=embedding,
            labels=labels,
            meta=meta,
            title=title,
            dimensions=dimensions,
        )
        section.add_element(PlotlyElement(fig))
        return

    dims = (0, 1) if dimensions == 2 else (0, 1, 2)
    fig = viz.plot_embedding(
        X_emb=embedding,
        labels=labels,
        dims=dims,
        title=title,
        interactive=False,
    )
    section.add_element(ImageElement(fig))


def to_epoch_scalar_mean(container: DataContainer) -> DataContainer:
    """Collapse each epoch to one scalar using stack + aggregate."""
    required_dims = {"obs", "channel", "time"}
    if not required_dims.issubset(set(container.dims)):
        raise ValueError(
            f"epoch_scalar_mean requires dims {required_dims}, got {container.dims}"
        )

    scalar_series = container.stack(dims=("obs", "channel", "time"), new_dim="obs")
    epoch_ids = np.array([str(i).rsplit("_", 2)[0] for i in scalar_series.ids])
    epoch_means = scalar_series.aggregate(by=epoch_ids, method="mean")

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


def apply_representation(
    container: DataContainer,
    representation: Literal[
        "epoch_flat",
        "epoch_time_as_sample",
        "epoch_scalar_mean",
        "subject_flat",
        "subject_time_as_sample",
        "subject_scalar_mean",
    ],
    study_id_col: str = "study_id",
) -> DataContainer:
    """Apply subject averaging and feature layout for the selected representation."""
    subject_average, stacking_mode = REPRESENTATION_CONFIG[representation]
    if subject_average:
        container = container.aggregate(by=study_id_col, method="mean")
    if stacking_mode == "flat":
        return container.flatten(preserve="obs")
    if stacking_mode == "time_as_sample":
        return container.stack(dims=("obs", "time"), new_dim="obs")
    if stacking_mode == "epoch_scalar_mean":
        return to_epoch_scalar_mean(container)
    raise ValueError(f"Unknown stacking mode: {stacking_mode}")


def build_meta_dict(container: DataContainer) -> Dict[str, np.ndarray]:
    """Build metadata dictionary for interactive coloring."""
    n_samples = container.X.shape[0]
    return {
        col_name: np.asarray(col_values)
        for col_name, col_values in container.coords.items()
        if col_name not in PLOT_META_EXCLUDED_COLUMNS
        and not col_name.endswith("_bool")
        and not col_name.endswith("_clean")
        and len(col_values) == n_samples
        and 1 < pd.Index(np.asarray(col_values).astype(str)).nunique() <= 200
    }

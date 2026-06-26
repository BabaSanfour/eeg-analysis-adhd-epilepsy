"""Shared study-level decoding helpers.

Small utilities plus a thin re-export layer over coco-pipe decoding primitives,
together with the sweep-preparation and result-summarization logic shared by the
classical and foundation decoding entry points.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.decoding import (
    ChanceAssessmentConfig,
    StatisticalAssessmentConfig,
    prepare_target,
    safe_group_n_splits,
)
from coco_pipe.io import DataContainer

DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
]


def slug(value: Any) -> str:
    """Return a filesystem-safe analysis label."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_.").lower()
    return text or "unnamed"


def grouped_accuracy_assessment(
    method: str = "permutation",
    n_permutations: int = 100,
    store_null: bool = False,
) -> StatisticalAssessmentConfig:
    """Chance assessment with group-level inference, shared by all sweeps.

    Centralizing this keeps the classical and foundation decoding paths on an
    identical inferential contract; if they drift, results stop being comparable.

    ``method="permutation"`` builds an empirical, finite-sample null by shuffling
    labels *within subjects* (``custom_unit_column="group_id"``), so the
    significance threshold reflects the number of subjects rather than the
    theoretical 0.5. ``method="binomial"`` falls back to the analytical
    theoretical chance level (``p0``). Permutation refits the full pipeline per
    shuffle, so cost scales with ``n_permutations``.
    """
    return StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        chance=ChanceAssessmentConfig(
            method=method,
            n_permutations=n_permutations,
            temporal_correction="none",
            store_null_distribution=store_null,
        ),
        unit_of_inference="custom",
        custom_unit_column="group_id",
    )


def require_conditions(config: dict[str, Any]) -> list[str]:
    """Require explicitly configured conditions."""
    conditions = config.get("conditions")
    if not conditions or not isinstance(conditions, list):
        raise ValueError("conditions must be explicitly configured as a non-empty list.")
    return [str(c) for c in conditions]


def require_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Require explicitly configured models."""
    models = config.get("models")
    if not models or not isinstance(models, list):
        raise ValueError("models must be explicitly configured as a non-empty config list.")
    return models


def foundation_provenance(
    model_cfg: Mapping[str, Any],
    spec: Any,
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Provenance row shared by the foundation extraction and decoding sweeps.

    Centralizing the model/window/spec fields keeps the embedding-extraction and
    decoding manifests byte-for-byte comparable; if they drift, joins across the
    two derivative trees stop lining up.
    """
    return {
        "config_hash": config_hash,
        "model_key": str(model_cfg["model_key"]),
        "segment_duration": float(model_cfg["segment_duration"]),
        "overlap": float(model_cfg["overlap"]),
        "use_derivatives": bool(model_cfg["use_derivatives"]),
        "window_source": str(model_cfg["window_source"]),
        "expected_n_times": spec.pretrained_n_times,
        "expected_sfreq": float(spec.pretrained_sfreq),
        "expected_duration": spec.pretrained_window_seconds,
    }


def cohort_signature(groups: Any) -> str:
    """Hash the sorted unique inference groups without exposing identifiers."""
    values = sorted({str(value) for value in groups})
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]


def prepare_decoding_scope(
    container: DataContainer,
    eval_spec: Mapping[str, Any],
    scope: str,
    group_col: str,
    session_col: str,
    subject_col: str,
    requested_splits: int,
) -> tuple[DataContainer, Any, Any, pd.DataFrame, int]:
    """Prepare labels, metadata, groups, IDs, and viable grouped folds."""
    selected, y, groups, metadata = prepare_target(
        container,
        eval_spec,
        group_col=group_col,
    )
    missing = [
        column for column in (subject_col, session_col, "group_id") if column not in metadata
    ]
    if missing:
        raise ValueError(
            "Decoding metadata must explicitly provide subject, session, and "
            f"group columns. Missing: {missing}"
        )
    metadata = metadata.copy()
    normalized_columns = {str(column).casefold() for column in metadata}
    if "subject" not in normalized_columns:
        metadata["Subject"] = metadata[subject_col].astype(str)
    if "session" not in normalized_columns:
        metadata["Session"] = metadata[session_col].astype(str)
    metadata["sample_id"] = [
        f"{scope}_{value}_{index}" for index, value in enumerate(metadata["sample_id"].astype(str))
    ]
    n_splits = safe_group_n_splits(y, groups, requested=requested_splits)
    return selected, y, groups, metadata, n_splits


def result_records(
    result: Any,
    context: Mapping[str, Any],
    output_dir: Path,
    include_p_values: bool = False,
) -> list[dict[str, Any]]:
    """Convert aggregate result rows into sweep records."""
    summary = result.summary().reset_index()
    stats = result.get_statistical_assessment() if include_p_values else pd.DataFrame()
    records = []
    for _, row in summary.iterrows():
        model = row.get("Model")
        record = {
            **dict(context),
            "model": model,
            "status": "success",
            "output_dir": str(output_dir),
            **{str(key): value for key, value in row.items() if key != "Model"},
        }
        if not stats.empty and {"Model", "PValue"}.issubset(stats.columns):
            model_stats = stats[stats["Model"] == model]
            if "Metric" in model_stats:
                model_stats = model_stats[model_stats["Metric"] == "accuracy"]
            if not model_stats.empty:
                record["p_value"] = float(model_stats.iloc[0]["PValue"])
        records.append(record)
    return records


__all__ = [
    "DEFAULT_METRICS",
    "cohort_signature",
    "foundation_provenance",
    "grouped_accuracy_assessment",
    "prepare_decoding_scope",
    "require_conditions",
    "require_models",
    "result_records",
    "slug",
]

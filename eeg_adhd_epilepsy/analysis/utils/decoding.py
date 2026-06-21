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
    completed_for_config,
    config_hash,
    load_completed_result_records,
    prepare_target,
    redact_sensitive,
    safe_group_n_splits,
    write_run_status,
)
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

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


def require_conditions(config: Mapping[str, Any]) -> list[str]:
    """Require explicitly configured experimental conditions."""
    conditions = config.get("conditions")
    if not conditions or not all(str(value).strip() for value in conditions):
        raise ValueError("conditions must be an explicit non-empty config list.")
    return [str(value) for value in conditions]


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
    "completed_for_config",
    "config_hash",
    "grouped_accuracy_assessment",
    "load_completed_result_records",
    "load_yaml_config",
    "prepare_decoding_scope",
    "prepare_target",
    "redact_sensitive",
    "require_conditions",
    "result_records",
    "safe_group_n_splits",
    "slug",
    "write_run_status",
]

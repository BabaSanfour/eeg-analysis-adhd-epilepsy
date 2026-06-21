"""Study-level dimensionality-reduction orchestration helpers."""

from __future__ import annotations

from typing import Any

from coco_pipe.io import DataContainer
from coco_pipe.utils import slug


def build_run_config_payload(
    args: Any,
    reducers: list[str],
    eval_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the scientific configuration that isolates one run namespace."""
    return {
        "dataset_name": args.dataset_name,
        "run_label": args.run_label,
        "input_mode": args.input_mode,
        "analysis_mode": args.analysis_mode,
        "representation": args.representation,
        "conditions": list(args.conditions),
        "run_pooled": bool(args.run_pooled),
        "reducers": list(reducers),
        "n_components_sweep": list(args.n_components_sweep),
        "subject_col": args.subject_col,
        "subjects": list(args.subjects or []),
        "filter_col": list(args.filter_col),
        "filter_val": list(args.filter_val),
        "group_filters": getattr(args, "group_filters", None),
        "balance_target": args.balance_target,
        "balance_strategy": args.balance_strategy if args.balance_target else None,
        "descriptor_families": list(args.descriptor_families or []),
        "descriptor_table_path": getattr(args, "descriptor_table_path", None),
        "descriptor_feature_columns_path": getattr(args, "descriptor_feature_columns_path", None),
        "descriptor_max_abs_value": getattr(args, "descriptor_max_abs_value", None),
        "location_statistic": getattr(args, "location_statistic", None),
        "qc": getattr(args, "qc", None),
        "bids_root": args.bids_root,
        "use_derivatives": bool(args.use_derivatives),
        "task": args.task,
        "segment_duration": args.segment_duration,
        "overlap": args.overlap,
        "desc": args.desc,
        "window_source": getattr(args, "window_source", "auto"),
        "aggregation_unit": getattr(args, "aggregation_unit", None),
        "embedding_derivative_root": getattr(args, "embedding_derivative_root", None),
        "embedding_representation": getattr(args, "embedding_representation", None),
        "embedding_aggregate_by": getattr(args, "embedding_aggregate_by", None),
        "embedding_model_key": getattr(args, "embedding_model_key", None),
        "evals": eval_specs,
    }


def pool_containers(containers: list[DataContainer]) -> DataContainer:
    """Concatenate conditions while preserving deferred family-QC semantics."""
    pooled = DataContainer.concat(containers)
    group_by_values = {
        str(container.meta["family_qc_group_by"])
        for container in containers
        if container.meta.get("family_qc_group_by") is not None
    }
    if len(group_by_values) > 1:
        raise ValueError(
            "Cannot pool containers with different family-QC grouping levels: "
            f"{sorted(group_by_values)}"
        )
    descriptor_name_values = [
        list(container.meta.get("family_qc_descriptor_names", []))
        for container in containers
        if container.meta.get("family_qc_descriptor_names")
    ]
    if descriptor_name_values and any(
        names != descriptor_name_values[0] for names in descriptor_name_values[1:]
    ):
        raise ValueError("Cannot pool containers with different descriptor schemas.")
    pooled.meta = {
        **dict(pooled.meta),
        "family_qc_bad_ids": {
            group: sorted(
                {
                    str(obs_id)
                    for item in containers
                    for obs_id in item.meta.get("family_qc_bad_ids", {}).get(group, [])
                }
            )
            for group in {
                group for item in containers for group in item.meta.get("family_qc_bad_ids", {})
            }
        },
    }
    if group_by_values:
        pooled.meta["family_qc_group_by"] = next(iter(group_by_values))
    if descriptor_name_values:
        pooled.meta["family_qc_descriptor_names"] = descriptor_name_values[0]
    return pooled


def condition_load_failure_record(*, condition: str, args: Any, error: Exception) -> dict[str, Any]:
    """Represent a load failure in the standard fit inventory."""
    return {
        "fit_id": f"load-{slug(condition)}",
        "scope": "condition",
        "condition": condition,
        "analysis_mode": args.analysis_mode,
        "unit_type": "load",
        "unit_name": "input",
        "unit_key": "input",
        "family": None,
        "subfamily": None,
        "input_mode": args.input_mode,
        "representation": args.representation,
        "reducer": None,
        "n_components": None,
        "status": "failed",
        "error": str(error),
        "artifact_path": None,
    }


__all__ = [
    "build_run_config_payload",
    "condition_load_failure_record",
    "pool_containers",
]

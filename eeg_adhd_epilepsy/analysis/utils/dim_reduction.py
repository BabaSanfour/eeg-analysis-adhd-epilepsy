"""Study-level dimensionality-reduction orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY
from coco_pipe.io import (
    AGGREGATION_LEVELS,
    DESCRIPTOR_ONLY_ANALYSIS_MODES,
    read_json,
)
from coco_pipe.utils import slug

from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def validate_inputs(args: Any) -> None:
    """Validate dimensionality reduction specific configuration dependencies."""
    if len(args.filter_col) != len(args.filter_val):
        raise ValueError("--filter_col and --filter_val must be provided in matching pairs.")
    if args.input_mode == "descriptors":
        if not args.descriptor_table_path or not args.descriptor_feature_columns_path:
            raise ValueError(
                "--descriptor_table_path and --descriptor_feature_columns_path are required "
                "when --input_mode descriptors."
            )
        if args.descriptor_max_abs_value is not None and args.descriptor_max_abs_value <= 0:
            raise ValueError("--descriptor_max_abs_value must be positive when provided.")
    if args.input_mode == "foundation_embeddings":
        if not args.embedding_derivative_root:
            raise ValueError("--embedding_derivative_root is required for foundation embeddings.")
        if not args.embedding_model_key:
            raise ValueError("--embedding_model_key is required to keep model spaces separate.")

    # Resolve the cohort run label
    if args.run_label is None:
        args.run_label = args.dataset_name
    if args.run_label is None:
        raise ValueError(
            "Provide run_label (or dataset_name) in the config to name the output cohort folder."
        )

    if args.descriptor_families and args.input_mode != "descriptors":
        raise ValueError("--descriptor_families is only supported for descriptor inputs.")
    if args.descriptor_families:
        invalid_families = [
            family
            for family in args.descriptor_families
            if family not in ("band", "complexity", "param")
        ]
        if invalid_families:
            raise ValueError(
                f"Unknown descriptor families: {invalid_families}. "
                f"Valid families: {['band', 'complexity', 'param']}"
            )


def build_and_validate_mode_specs(
    args: Any,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Per-analysis-mode specs (reducers + n_components), in plan order."""
    analysis_modes = getattr(args, "analysis_modes", None)
    if not isinstance(analysis_modes, Mapping):
        raise ValueError(
            "The analysis config must declare `analysis_modes` (a mode -> spec mapping)."
        )
    specs = {str(mode): dict(spec or {}) for mode, spec in analysis_modes.items()}
    for key in ("reducers", "n_components"):
        missing = [mode for mode, spec in specs.items() if not spec.get(key)]
        if missing:
            raise ValueError(
                f"Every analysis mode must list `{key}`; none configured for: {sorted(missing)}."
            )

    tasks = [
        (mode, str(spec.get("representation") or args.representation or ""))
        for mode, spec in specs.items()
    ]

    input_mode = getattr(args, "input_mode", None)
    for mode, representation in tasks:
        if mode in DESCRIPTOR_ONLY_ANALYSIS_MODES and input_mode != "descriptors":
            raise ValueError(f"analysis_mode='{mode}' is only supported for descriptor inputs.")
        if input_mode == "foundation_embeddings" and mode != "flat":
            raise ValueError("Foundation embeddings currently support analysis_mode='flat' only.")
        if input_mode == "raw":
            if mode not in {"flat", "sensor"}:
                raise ValueError("Raw inputs support only analysis_mode 'flat' or 'sensor'.")
        if input_mode in ("raw", "foundation_embeddings"):
            if representation not in AGGREGATION_LEVELS:
                raise ValueError(
                    f"representation must be one of {list(AGGREGATION_LEVELS)}, "
                    f"got '{representation}'."
                )

    return specs, tasks


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Load one run's persisted leaderboard + provenance into a roll-up summary."""
    leaderboard_path = run_dir / "runs" / "leaderboard.json"
    if not leaderboard_path.exists():
        return None
    try:
        leaderboard = pd.read_json(leaderboard_path, orient="records")
    except ValueError:
        return None
    if leaderboard.empty:
        return None
    summary_path = run_dir / "runs" / "run_summary.json"
    meta = read_json(summary_path) if summary_path.exists() else {}
    representation = meta.get("representation", "")
    if "representation" in leaderboard.columns and not leaderboard["representation"].empty:
        representation = str(leaderboard["representation"].iloc[0])
    return {
        "analysis_mode": meta.get("analysis_mode", ""),
        "representation": representation,
        "run_variant": meta.get("run_variant", run_dir.name),
        "report_path": meta.get("report_path"),
        "leaderboard": leaderboard,
    }


def _selection_config(run_dir: Path) -> tuple[str, str | None]:
    """Recover the run's selection metric/eval from its persisted config snapshot."""
    config_path = run_dir / "config_used.yaml"
    if config_path.exists():
        cfg = load_yaml_config(config_path)
        return (
            str(cfg.get("selection_metric", SEPARATION_METRIC_KEY)),
            cfg.get("selection_eval_name"),
        )
    return SEPARATION_METRIC_KEY, None


def build_output_root(bids_root: Path, args: Any, mode: str, representation: str) -> Path:
    """Build the final output root directory for a given dim reduction run."""
    from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root

    output_base = get_derivative_root(bids_root, DerivativeStage.DIM_REDUCTION)

    input_token = (
        f"foundation_{args.embedding_model_key}"
        if args.input_mode == "foundation_embeddings"
        else args.input_mode
    )

    modifier = representation

    parts = [input_token, mode, modifier, f"cfg-{args.run_config_hash}"]
    run_variant = "_".join(slug(str(part)) for part in parts if part)
    args.run_variant = run_variant

    output_dataset_name = slug(args.run_label)
    return output_base / output_dataset_name / run_variant


def build_input_signature(
    args: Any,
    unit_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build the provenance signature for a specific analysis unit."""
    from pathlib import Path

    filter_specs = [
        {"column": str(col), "values": [str(value) for value in vals]}
        for col, vals in zip(args.filter_col, args.filter_val)
        if vals
    ]

    input_signature: dict[str, Any] = {
        "input_mode": args.input_mode,
        "analysis_mode": args.analysis_mode,
        "run_config_hash": args.run_config_hash,
        "filters": filter_specs,
        "group_filters": args.group_filters,
        "balance_target": args.balance_target,
        "balance_strategy": args.balance_strategy if args.balance_target else None,
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "run_label": args.run_label,
        "qc": args.qc,
    }

    if args.input_mode == "raw":
        input_signature.update(
            {
                "representation": args.representation,
                "bids_root": str(Path(args.bids_root).expanduser()),
                "use_derivatives": bool(args.use_derivatives),
                "task": args.task,
                "segment_duration": float(args.segment_duration),
                "overlap": float(args.overlap),
                "desc": args.desc,
                "window_source": args.window_source,
            }
        )
    elif args.input_mode == "descriptors":
        input_signature.update(
            {
                "descriptor_table_path": str(Path(args.descriptor_table_path).expanduser()),
                "descriptor_families": list(args.descriptor_families or []),
                "descriptor_feature_columns_path": str(
                    Path(args.descriptor_feature_columns_path).expanduser()
                ),
                "descriptor_max_abs_value": args.descriptor_max_abs_value,
                "location_statistic": args.location_statistic,
                "family": unit_spec.get("family"),
            }
        )
    elif args.input_mode == "foundation_embeddings":
        input_signature.update(
            {
                "embedding_derivative_root": str(Path(args.embedding_derivative_root).expanduser()),
                "representation": args.representation or "",
                "embedding_aggregate_by": args.embedding_aggregate_by,
                "embedding_model_key": args.embedding_model_key,
            }
        )
    else:
        raise ValueError(f"Unsupported input_mode '{args.input_mode}'.")

    return input_signature


def build_run_config_payload(
    args: Any,
    reducers: list[str],
    eval_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the scientific configuration that isolates one run namespace."""
    payload = {
        "dataset_name": args.dataset_name,
        "run_label": args.run_label,
        "input_mode": args.input_mode,
        "analysis_mode": args.analysis_mode,
        "conditions": list(args.conditions),
        "run_pooled": bool(args.run_pooled),
        "reducers": list(reducers),
        "n_components_sweep": list(args.n_components_sweep),
        "subject_col": args.subject_col,
        "subjects": list(args.subjects or []),
        "filter_col": list(args.filter_col),
        "filter_val": list(args.filter_val),
        "group_filters": args.group_filters,
        "balance_target": args.balance_target,
        "balance_strategy": args.balance_strategy if args.balance_target else None,
        "qc": args.qc,
        "bids_root": args.bids_root,
        "use_derivatives": bool(args.use_derivatives),
        "task": args.task,
        "evals": eval_specs,
    }

    if args.input_mode == "raw":
        payload.update(
            {
                "representation": args.representation,
                "segment_duration": args.segment_duration,
                "overlap": args.overlap,
                "desc": args.desc,
                "window_source": args.window_source,
            }
        )

    if args.input_mode == "foundation_embeddings":
        payload.update(
            {
                "embedding_derivative_root": args.embedding_derivative_root,
                "representation": args.representation or "",
                "embedding_aggregate_by": args.embedding_aggregate_by,
                "embedding_model_key": args.embedding_model_key,
            }
        )

    if args.input_mode == "descriptors" or args.descriptor_families:
        payload.update(
            {
                "descriptor_families": list(args.descriptor_families or []),
                "descriptor_table_path": args.descriptor_table_path,
                "descriptor_feature_columns_path": args.descriptor_feature_columns_path,
                "descriptor_max_abs_value": args.descriptor_max_abs_value,
                "location_statistic": args.location_statistic,
            }
        )

    return payload


def group_fit_requests(
    requests: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group fit requests into the units of parallel work.

    A *nested* reducer (PCA/SVD) is fitted once at the largest ``n_components``
    and the smaller sweep values are sliced from it, so its whole sweep for one
    analysis unit stays in a single group — see
    :func:`coco_pipe.dim_reduction.run_fit_group`. A *non-nested* reducer
    (UMAP/PHATE/Isomap/…) must refit per dimension, and grouping its sweep would
    run those refits serially inside one worker; instead each ``n_components``
    becomes its own singleton group so the outer pool runs them in parallel.
    This is the dominant win for flat-only foundation runs, which otherwise have
    too few groups to saturate the node. First-seen order is preserved so the
    queued-fit ordering stays stable.
    """
    from collections import OrderedDict

    from coco_pipe.dim_reduction import supports_nested_components

    groups: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
    for request in requests:
        fit_payload = request["fit_payload"]
        reducer = fit_payload["reducer"]
        key: tuple[Any, ...] = (
            fit_payload["scope"],
            fit_payload["condition"],
            fit_payload["unit_key"],
            reducer,
        )
        if not supports_nested_components(reducer):
            key = (*key, fit_payload["n_components"])
        groups.setdefault(key, []).append(request)
    return list(groups.values())


__all__ = [
    "build_and_validate_mode_specs",
    "build_run_config_payload",
    "group_fit_requests",
]

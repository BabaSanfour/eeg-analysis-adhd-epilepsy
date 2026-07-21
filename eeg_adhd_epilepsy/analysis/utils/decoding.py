"""Shared study-level decoding helpers.

Small utilities plus a thin re-export layer over coco-pipe decoding primitives,
together with the sweep-preparation and result-summarization logic shared by the
classical and foundation decoding entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from coco_pipe.decoding import (
    ClassicalModelConfig,
    prepare_target,
    redact_sensitive,
    safe_group_n_splits,
)
from coco_pipe.io import (
    DESCRIPTOR_ONLY_ANALYSIS_MODES,
    DataContainer,
)
from coco_pipe.utils import slug, stable_hash

from eeg_adhd_epilepsy.analysis.utils.common import (
    base_layout_mode,
    require_config,
)
from eeg_adhd_epilepsy.analysis.utils.hashing import normalize_scientific_paths

_NON_SCIENTIFIC_HASH_KEYS = frozenset(
    {
        "derivative_root",
        "reports_only",
        "compare_only",
        "reports_root",
        "n_jobs",
        "overwrite",
        "verbose",
        "report_asset_urls",
        "detailed_unit_reports",
        "detailed_unit_report_modes",
        "foundation_report_sections",
        "write_shared_comparison_report",
    }
)


def scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop orchestration-only keys so run identity ignores them.

    Both the run-variant directory hash and the per-unit resume hash must be
    computed over the same key set: keys in ``_NON_SCIENTIFIC_HASH_KEYS`` (worker
    counts, verbosity, overwrite/reports toggles) change how a run executes but
    not what it computes, so including them would make a resume after e.g. an
    ``n_jobs`` change spuriously mismatch the stored manifest.
    """
    scientific = {
        key: value for key, value in dict(config).items() if key not in _NON_SCIENTIFIC_HASH_KEYS
    }
    return normalize_scientific_paths(scientific)


def resolve_decoding_paths(
    config: Mapping[str, Any], input_mode: str
) -> tuple[Path, Path, Path, pd.DataFrame | None, str, Path, str]:
    """Resolve standard paths and hashes for a decoding run.

    Calculates the stable config hash and constructs identical `run_variant` layouts
    for classical and foundation scripts.
    """
    from coco_pipe.io import read_table

    from eeg_adhd_epilepsy.io.bids import (
        DerivativeStage,
        get_derivative_root,
    )
    from eeg_adhd_epilepsy.io.report_paths import (
        ReportStage,
        default_reports_root,
        summary_report_dir,
    )

    bids_root = Path(config["bids_root"]).expanduser()
    metadata = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )

    cfg_hash = stable_hash(redact_sensitive(scientific_config(config)), length=12)
    run_variant = f"{slug(input_mode)}_cfg-{cfg_hash}"
    dataset_name_slug = slug(config.get("run_label", config["dataset_name"]))

    decoding_root = (
        Path(config["derivative_root"]).expanduser()
        if config.get("derivative_root")
        else get_derivative_root(bids_root, DerivativeStage.DECODING)
    )
    derivative_root = decoding_root / dataset_name_slug / run_variant
    reports_root = Path(config.get("reports_root", default_reports_root(bids_root))).expanduser()
    report_root = (
        summary_report_dir(reports_root, ReportStage.DECODING) / dataset_name_slug / run_variant
    )
    return (
        bids_root,
        derivative_root,
        report_root,
        metadata,
        cfg_hash,
        reports_root,
        dataset_name_slug,
    )


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
        "pooling": str(model_cfg.get("pooling", "mean")),
        "bandpass": (
            [float(v) for v in model_cfg["bandpass"]] if model_cfg.get("bandpass") else None
        ),
        "segment_duration": float(model_cfg["segment_duration"]),
        "overlap": float(model_cfg["overlap"]),
        "use_derivatives": bool(model_cfg["use_derivatives"]),
        "window_source": str(model_cfg["window_source"]),
        "expected_n_times": spec.pretrained_n_times,
        "expected_sfreq": float(spec.pretrained_sfreq),
        "expected_duration": spec.pretrained_window_seconds,
    }


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


@dataclass
class ClassicalPlan:
    """Validated method plan for a classical decoding sweep.

    Holds the derived, checked configuration (input/source/layout modes, the
    analysis-mode order, per-model configs/grids/mode filters, tuning, selection
    passes, and target specs) so the entry point stays a thin loader + runner,
    mirroring the dim-reduction ``build_and_validate_mode_specs`` split.
    """

    input_mode: str
    layout_mode: str
    analysis_modes: list[str]
    model_configs: dict[str, ClassicalModelConfig]
    model_analysis_modes: dict[str, set[str] | None]
    model_grids: dict[str, dict[str, Any]]
    tuning_cfg: dict[str, Any]
    selection_specs: list[dict[str, Any]]
    evals: list[dict[str, Any]]
    transforms: list[str]
    transform_params: dict[str, dict[str, Any]]
    reducer_enabled: bool
    reducer_cfg: dict[str, Any]
    metrics: list[str]


def build_classical_plan(config: Mapping[str, Any]) -> ClassicalPlan:
    """Validate config into a :class:`ClassicalPlan` (raises on bad input)."""
    input_mode = config["input_mode"]
    if input_mode not in {"descriptors", "foundation_embeddings"}:
        raise ValueError(
            f"Invalid input_mode '{input_mode}'. Classical decoding requires "
            "'descriptors' or 'foundation_embeddings'."
        )

    layout_mode = base_layout_mode(input_mode)

    analysis_modes = require_config(
        dict(config),
        "analysis_modes",
        expected_type=list,
        cast_str=True,
    )

    for mode in analysis_modes:
        if mode in DESCRIPTOR_ONLY_ANALYSIS_MODES and input_mode != "descriptors":
            raise ValueError(f"analysis_mode='{mode}' is only supported for descriptor inputs.")
        if input_mode == "foundation_embeddings" and mode != "flat":
            raise ValueError(
                f"analysis_mode='{mode}' is not supported for foundation embeddings; "
                "use analysis_mode='flat'."
            )

    evals_spec = require_config(dict(config), "evals", expected_type=list)
    model_specs = require_config(dict(config), "models", expected_type=dict)

    input_kind = "embeddings" if input_mode == "foundation_embeddings" else "tabular"
    model_configs = {
        name: ClassicalModelConfig(
            estimator=spec["estimator"],
            params=dict(spec.get("params", {})),
            input_kind=input_kind,
        )
        for name, spec in model_specs.items()
    }
    model_analysis_modes = {
        name: (
            {str(mode) for mode in spec["analysis_modes"]} if spec.get("analysis_modes") else None
        )
        for name, spec in model_specs.items()
    }

    model_grids = {
        name: dict(spec["grid"]) for name, spec in model_specs.items() if spec.get("grid")
    }

    selection_specs = require_config(dict(config), "feature_selection", expected_type=list)
    for spec in selection_specs:
        method = str(spec.get("method", "none"))
        if method not in {"none", "sfs"}:
            raise ValueError(
                "feature_selection supports only method='none' and method='sfs'; "
                f"got method='{method}'."
            )
        if method == "sfs" and spec.get("n_features") is None and spec.get("tol") is None:
            name = spec.get("name", "sfs")
            raise ValueError(f"feature_selection entry '{name}' must set either n_features or tol.")

    metrics = require_config(dict(config), "metrics", expected_type=list)
    transforms = [str(value) for value in config.get("transforms", ["none"])]
    if input_mode != "foundation_embeddings" and transforms != ["none"]:
        raise ValueError("Subject transforms are only supported for foundation embeddings.")
    reducer_cfg = dict(config.get("reducer") or {})

    return ClassicalPlan(
        input_mode=input_mode,
        layout_mode=layout_mode,
        analysis_modes=analysis_modes,
        model_configs=model_configs,
        model_analysis_modes=model_analysis_modes,
        model_grids=model_grids,
        tuning_cfg=dict(config.get("tuning") or {}),
        selection_specs=selection_specs,
        evals=evals_spec,
        transforms=transforms,
        transform_params={
            str(name): dict(params or {})
            for name, params in (config.get("transform_params") or {}).items()
        },
        reducer_enabled=bool(reducer_cfg.get("enabled", False)),
        reducer_cfg=reducer_cfg,
        metrics=metrics,
    )


def build_loader_args(
    config: Mapping[str, Any],
    *,
    input_mode: str,
    layout_mode: str,
    segment_duration: float | None = None,
    overlap: float | None = None,
    use_derivatives: bool | None = None,
    window_source: str | None = None,
) -> SimpleNamespace:
    """Explicit dataset-loader args for classical decoding.

    Every field is sourced via ``config.get`` with a default, so the loader no
    longer depends on inline construction inside ``run`` (mirrors the
    dim-reduction ``_run_args_from_config`` split).
    """
    if "filter_col" in config or "filter_val" in config:
        filter_col = list(config.get("filter_col", []) or [])
        filter_val = list(config.get("filter_val", []) or [])
    else:
        filters = config.get("filters", {}) or {}
        filter_col = list(filters)
        filter_val = [filters[column] for column in filters]

    return SimpleNamespace(
        input_mode=input_mode,
        reduced_source_input_mode=config.get("reduced_source_input_mode", "descriptors"),
        analysis_mode=layout_mode,
        bids_root=config.get("bids_root"),
        use_derivatives=(
            use_derivatives
            if use_derivatives is not None
            else bool(config.get("use_derivatives", True))
        ),
        task=config.get("task", "clinical"),
        segment_duration=(
            segment_duration
            if segment_duration is not None
            else float(config.get("segment_duration", 10.0))
        ),
        overlap=overlap if overlap is not None else float(config.get("overlap", 0.0)),
        units=config.get("units", "V"),
        subject_col=config.get("subject_col", "study_id"),
        desc=config.get("desc", "base"),
        descriptor_table_path=config.get("descriptor_table_path"),
        descriptor_feature_columns_path=config.get("descriptor_feature_columns_path"),
        descriptor_families=config.get("descriptor_families"),
        descriptor_max_abs_value=config.get("descriptor_max_abs_value", 1e12),
        location_statistic=config.get("location_statistic"),
        qc=config.get("qc"),
        embedding_derivative_root=config.get("embedding_derivative_root"),
        embedding_aggregate_by=config.get("embedding_aggregate_by"),
        embedding_model_key=config.get("embedding_model_key"),
        filter_col=filter_col,
        filter_val=filter_val,
        group_filters=config.get("group_filters"),
        balance_target=None,
        balance_strategy="undersample",
        representation=config.get("representation", "subject"),
        window_source=(
            window_source if window_source is not None else config.get("window_source", "auto")
        ),
    )


__all__ = [
    "ClassicalPlan",
    "build_classical_plan",
    "build_loader_args",
    "foundation_provenance",
    "prepare_decoding_scope",
    "resolve_decoding_paths",
    "scientific_config",
]

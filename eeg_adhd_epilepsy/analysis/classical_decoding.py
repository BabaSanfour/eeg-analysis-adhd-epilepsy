#!/usr/bin/env python3
"""Leakage-safe classical decoding over descriptors and foundation embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from coco_pipe.decoding import (
    ClassicalModelConfig,
    CVConfig,
    Experiment,
    ExperimentConfig,
    FeatureSelectionConfig,
    ReducerConfig,
    TuningConfig,
    correct_sweep_pvalues,
)
from coco_pipe.io import DataContainer, iter_analysis_units, read_table
from coco_pipe.report import make_decoding_report

from eeg_adhd_epilepsy.analysis.utils.decoding import (
    DEFAULT_METRICS,
    cohort_signature,
    completed_for_config,
    grouped_accuracy_assessment,
    load_completed_result_records,
    prepare_decoding_scope,
    redact_sensitive,
    require_conditions,
    result_records,
    safe_group_n_splits,
    slug,
    write_run_status,
)
from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.units import (
    apply_family_qc_mask,
    families_for_analysis_unit,
)
from eeg_adhd_epilepsy.io.report_paths import default_reports_root
from eeg_adhd_epilepsy.reports.decoding import (
    descriptor_feature_metadata,
    generate_decoding_summary_report,
    generate_head_to_head_report,
)
from eeg_adhd_epilepsy.utils.config import resolve_cli_config

LOGGER = logging.getLogger(__name__)

# Canonical descriptor plan. Keep this order aligned with the summary report.
_DESCRIPTOR_ANALYSIS_MODES = [
    "flat",
    "sensor",
    "subfamily",
    "sensor_within_subfamily",
    "descriptor",
    "descriptor_sensor",
]
_DESCRIPTOR_ANALYSIS_MODE_SET = frozenset(_DESCRIPTOR_ANALYSIS_MODES)


def _is_transductive(container: DataContainer) -> bool:
    """Return whether the input representation used full-cohort information."""
    return bool(container.meta.get("transductive", False))


def _selection_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return mandatory baseline plus explicitly requested SFS passes."""
    normalized = [{"name": "baseline", "method": "none"}]
    names = {"baseline"}
    for raw_spec in config.get("feature_selection") or []:
        spec = dict(raw_spec)
        method = str(spec.get("method", "none")).lower()
        if method == "none":
            continue
        if method != "sfs":
            raise ValueError(
                "feature_selection only supports method='sfs'; "
                "the baseline pass is added automatically."
            )
        name = str(spec.get("name") or "sfs")
        if name in names:
            raise ValueError(f"Duplicate feature-selection name: {name!r}.")
        spec.update(name=name, method="sfs")
        normalized.append(spec)
        names.add(name)
    return normalized


def _selection_specs_for_unit(
    specs: list[dict[str, Any]],
    *,
    analysis_mode: str,
    n_available: int,
) -> list[dict[str, Any]]:
    """Filter optional SFS passes for one analysis unit."""
    selected: list[dict[str, Any]] = []
    for spec in specs:
        if spec["method"] == "none":
            selected.append(spec)
            continue
        requested_modes = spec.get("analysis_modes")
        if requested_modes and analysis_mode not in requested_modes:
            continue
        if analysis_mode == "descriptor_sensor" or n_available <= 1:
            continue
        selected.append(spec)
    return selected


def _feature_selection_config(
    spec: dict[str, Any],
    n_available: int,
    *,
    cv: CVConfig | None = None,
) -> FeatureSelectionConfig:
    method = str(spec.get("method", "none"))
    if method == "none":
        return FeatureSelectionConfig(enabled=False)
    n_features = spec.get("n_features")
    if n_features is not None:
        n_features = min(int(n_features), int(n_available) - 1)
    return FeatureSelectionConfig(
        enabled=True,
        method=method,
        n_features=n_features,
        direction=str(spec.get("direction", "forward")),
        tol=spec.get("tol", 0.001) if n_features is None else spec.get("tol"),
        cv=cv,
        scoring=spec.get("scoring"),
    )


def run(config: dict[str, Any]) -> Path:
    input_mode = config.get("input_mode", "descriptors")
    source_mode = (
        config.get("reduced_source_input_mode", "descriptors")
        if input_mode == "reduced_dimensions"
        else input_mode
    )
    if source_mode not in {"descriptors", "foundation_embeddings"}:
        raise ValueError("Classical decoding supports descriptors or embeddings.")
    bids_root = Path(config["bids_root"]).expanduser()
    metadata = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )
    derivative_root = (
        bids_root
        / "derivatives"
        / "decoding"
        / str(config.get("output_group", "default"))
        / str(config.get("dataset_name", "dataset"))
        / input_mode
    )
    reports_root = Path(config.get("reports_root", default_reports_root(bids_root))).expanduser()
    report_root = (
        reports_root
        / "summary"
        / "decoding"
        / str(config.get("output_group", "default"))
        / str(config.get("dataset_name", "dataset"))
        / input_mode
    )
    conditions = require_conditions(config)
    scopes: list[tuple[str, DataContainer]] = []
    qc_report_results = []
    load_mode = "reduced_dimensions" if input_mode == "reduced_dimensions" else source_mode
    layout_mode = "sensor" if source_mode == "descriptors" else "flat"
    default_analysis_modes = (
        _DESCRIPTOR_ANALYSIS_MODES if source_mode == "descriptors" else ["flat"]
    )
    analysis_modes = list(config.get("analysis_modes", default_analysis_modes))
    if source_mode == "descriptors":
        unsupported_modes = [
            mode for mode in analysis_modes if mode not in _DESCRIPTOR_ANALYSIS_MODE_SET
        ]
        if unsupported_modes:
            raise ValueError(
                "Unsupported descriptor analysis modes. Keep only "
                f"{_DESCRIPTOR_ANALYSIS_MODES}; received {unsupported_modes}."
            )
    filters = config.get("filters", {})
    loader_args = SimpleNamespace(
        input_mode=load_mode,
        reduced_source_input_mode=config.get("reduced_source_input_mode", "descriptors"),
        analysis_mode=layout_mode,
        bids_root=config.get("bids_root"),
        use_derivatives=bool(config.get("use_derivatives", True)),
        task=config.get("task", "clinical"),
        segment_duration=float(config.get("segment_duration", 10.0)),
        overlap=float(config.get("overlap", 0.0)),
        subject_col=config.get("subject_col", "study_id"),
        desc=config.get("desc", "base"),
        descriptor_table_path=config.get("descriptor_table_path"),
        descriptor_feature_columns_path=config.get("descriptor_feature_columns_path"),
        descriptor_families=config.get("descriptor_families"),
        descriptor_max_abs_value=config.get("descriptor_max_abs_value", 1e12),
        location_statistic=config.get("location_statistic"),
        qc=config.get("qc"),
        embedding_derivative_root=config.get("embedding_derivative_root"),
        embedding_representation=config.get("embedding_representation", "recording"),
        embedding_aggregate_by=config.get("embedding_aggregate_by"),
        embedding_model_key=config.get("embedding_model_key"),
        filter_col=list(filters),
        filter_val=[filters[column] for column in filters],
        group_filters=config.get("group_filters"),
        balance_target=None,
        balance_strategy="undersample",
        representation=config.get("representation", "recording_flat"),
        aggregation_unit=config.get("aggregation_unit", "recording"),
    )
    for condition in conditions:
        container = build_dataset(
            loader_args,
            config.get("subjects"),
            metadata,
            condition,
            target_col=None,
        )
        if _is_transductive(container) and not config.get("allow_transductive_input", False):
            raise ValueError(
                "The loaded input is marked transductive because its transformation "
                "was fitted outside the decoding folds. Set allow_transductive_input: "
                "true only for explicitly exploratory analyses."
            )
        scopes.append((condition, container))
        if container.meta.get("qc_result") is not None:
            qc_report_results.append((condition, container.meta["qc_result"]))
    if config.get("run_pooled", True) and len(scopes) > 1:
        pooled = DataContainer.concat([item[1] for item in scopes])
        pooled.meta = {
            **dict(pooled.meta),
            "transductive": any(_is_transductive(item[1]) for item in scopes),
            "family_qc_bad_ids": {
                family: sorted(
                    {
                        str(obs_id)
                        for _, item in scopes
                        for obs_id in item.meta.get("family_qc_bad_ids", {}).get(family, [])
                    }
                )
                for family in {
                    family
                    for _, item in scopes
                    for family in item.meta.get("family_qc_bad_ids", {})
                }
            },
        }
        scopes.append(("pooled", pooled))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evals = config.get("evals", [])
    if not evals:
        raise ValueError("At least one target specification is required in evals.")
    model_specs = config.get("models")
    if not model_specs:
        raise ValueError("`models` must be specified in the config.")

    all_model_configs = {
        name: ClassicalModelConfig(
            estimator=spec["estimator"],
            params=dict(spec.get("params", {})),
            input_kind="embeddings"
            if config.get("input_mode") == "foundation_embeddings"
            else "tabular",
        )
        for name, spec in model_specs.items()
    }
    model_analysis_modes = {name: spec.get("analysis_modes") for name, spec in model_specs.items()}
    # Per-model hyperparameter grids (models without a grid run with fixed params).
    model_grids = {
        name: dict(spec["grid"]) for name, spec in model_specs.items() if spec.get("grid")
    }
    tuning_cfg = config.get("tuning") or {}
    selection_specs = _selection_specs(config)

    for scope, full_container in scopes:
        transductive_input = _is_transductive(full_container)
        for eval_spec in evals:
            target_name = eval_spec.get("name", eval_spec["target_col"])
            try:
                (
                    target_container,
                    y,
                    groups,
                    sample_metadata,
                    _n_splits,
                ) = prepare_decoding_scope(
                    full_container,
                    eval_spec,
                    scope=scope,
                    group_col=config.get("group_col", "patient_group_id"),
                    session_col=config["session_col"],
                    subject_col=config.get("subject_col", "study_id"),
                    requested_splits=int(config.get("cv", {}).get("n_splits", 5)),
                )
            except Exception as exc:
                failures.append(
                    {
                        "scope": scope,
                        "target": target_name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for analysis_mode in analysis_modes:
                model_configs = {
                    name: model_config
                    for name, model_config in all_model_configs.items()
                    if not model_analysis_modes[name] or analysis_mode in model_analysis_modes[name]
                }
                # Grids only for the models active in this mode (tuned models).
                experiment_grids = {
                    name: model_grids[name] for name in model_configs if name in model_grids
                }
                if not model_configs:
                    failures.append(
                        {
                            "scope": scope,
                            "target": target_name,
                            "analysis_mode": analysis_mode,
                            "status": "skipped",
                            "reason": "No models are configured for this analysis mode.",
                        }
                    )
                    continue
                try:
                    units = iter_analysis_units(
                        target_container,
                        analysis_mode,
                        "descriptors" if source_mode == "descriptors" else "foundation_embeddings",
                        config.get("descriptor_families"),
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "scope": scope,
                            "target": target_name,
                            "analysis_mode": analysis_mode,
                            "status": "skipped",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                for unit in units:
                    families = families_for_analysis_unit(
                        target_container,
                        unit,
                        config.get("descriptor_families"),
                    )
                    unit_container, keep_indices = apply_family_qc_mask(
                        unit["container"],
                        families,
                    )
                    unit_y = np.asarray(y)[keep_indices]
                    unit_groups = np.asarray(groups)[keep_indices]
                    unit_metadata = sample_metadata.iloc[keep_indices].reset_index(drop=True)
                    try:
                        unit_n_splits = safe_group_n_splits(
                            unit_y,
                            unit_groups,
                            requested=int(config.get("cv", {}).get("n_splits", 5)),
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "scope": scope,
                                "target": target_name,
                                "analysis_mode": analysis_mode,
                                "unit_name": unit["unit_name"],
                                "unit_key": unit["unit_key"],
                                "family": unit.get("family"),
                                "subfamily": unit.get("subfamily"),
                                "status": "skipped",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    X = np.asarray(unit_container.X)
                    if X.ndim != 2:
                        X = unit_container.flatten(preserve="obs").X
                    flattened = (
                        unit_container
                        if unit_container.dims == ("obs", "feature")
                        else unit_container.flatten(preserve="obs")
                    )
                    names = [
                        str(value)
                        for value in flattened.coords.get(
                            "feature", np.arange(flattened.X.shape[1])
                        )
                    ]
                    unit_selection_specs = _selection_specs_for_unit(
                        selection_specs,
                        analysis_mode=analysis_mode,
                        n_available=X.shape[1],
                    )
                    for selection_spec in unit_selection_specs:
                        selection_mode = str(
                            selection_spec.get("name") or selection_spec.get("method") or "fs"
                        )
                        output_dir = (
                            derivative_root
                            / slug(scope)
                            / slug(target_name)
                            / analysis_mode
                            / slug(unit["unit_key"])
                            / selection_mode
                        )
                        context = {
                            "scope": scope,
                            "target": target_name,
                            "input_mode": input_mode,
                            "analysis_mode": analysis_mode,
                            "unit_name": unit["unit_name"],
                            "unit_key": unit["unit_key"],
                            "family": unit.get("family"),
                            "subfamily": unit.get("subfamily"),
                            "selection_mode": selection_mode,
                            "primary": (analysis_mode == "flat" and not transductive_input),
                            "transductive_input": transductive_input,
                            "cv_strategy": "stratified_group_kfold",
                            "cv_random_state": int(config.get("random_state", 42)),
                            "effective_n_splits": unit_n_splits,
                            "n_samples": int(len(unit_y)),
                            "n_groups": int(np.unique(unit_groups).size),
                            "cohort_signature": cohort_signature(
                                unit_metadata[config.get("subject_col", "study_id")]
                            ),
                        }
                        unit_config = {
                            **config,
                            **context,
                        }
                        if not config.get("overwrite", False) and completed_for_config(
                            output_dir, unit_config
                        ):
                            records.extend(
                                load_completed_result_records(
                                    output_dir,
                                    context=context,
                                )
                            )
                            continue
                        try:
                            reducer_cfg = config.get("reducer", {})
                            reducer_enabled = input_mode == "reduced_dimensions"
                            n_components = reducer_cfg.get("n_components")
                            cv_config = CVConfig(
                                strategy="stratified_group_kfold",
                                n_splits=unit_n_splits,
                                shuffle=True,
                                random_state=int(config.get("random_state", 42)),
                                group_key="group_id",
                            )
                            experiment_config = ExperimentConfig(
                                task="classification",
                                tag=(
                                    f"{target_name}_{analysis_mode}_"
                                    f"{unit['unit_key']}_{selection_mode}"
                                ),
                                random_state=int(config.get("random_state", 42)),
                                models=model_configs,
                                cv=cv_config,
                                tuning=TuningConfig(
                                    enabled=bool(experiment_grids),
                                    scoring=tuning_cfg.get("scoring"),
                                    search_type=tuning_cfg.get("search_type", "grid"),
                                    cv=CVConfig(
                                        strategy="stratified_group_kfold",
                                        n_splits=min(
                                            int(tuning_cfg.get("n_splits", 3)),
                                            unit_n_splits,
                                        ),
                                        shuffle=True,
                                        random_state=int(config.get("random_state", 42)),
                                        group_key="group_id",
                                    ),
                                    n_jobs=int(config.get("n_jobs", 1)),
                                    allow_nongroup_inner_cv=bool(
                                        tuning_cfg.get("allow_nongroup_inner_cv", False)
                                    ),
                                ),
                                grids=experiment_grids or None,
                                reducer=ReducerConfig(
                                    enabled=reducer_enabled,
                                    method="pca",
                                    n_components=n_components,
                                    whiten=bool(reducer_cfg.get("whiten", False)),
                                ),
                                feature_selection=_feature_selection_config(
                                    selection_spec,
                                    X.shape[1],
                                    cv=cv_config,
                                ),
                                statistical_assessment=grouped_accuracy_assessment(
                                    method=config.get("chance_method", "permutation"),
                                    n_permutations=int(config.get("n_permutations", 100)),
                                    store_null=bool(config.get("store_null_distribution", False)),
                                ),
                                metrics=config.get("metrics", DEFAULT_METRICS),
                                use_scaler=True,
                                n_jobs=int(config.get("n_jobs", 1)),
                                verbose=bool(config.get("verbose", False)),
                            )
                            result = Experiment(experiment_config).run(
                                X,
                                unit_y,
                                groups=unit_groups,
                                feature_names=names,
                                sample_ids=unit_metadata["sample_id"].astype(str),
                                sample_metadata=unit_metadata,
                                inferential_unit="group_id",
                            )
                            result.export(output_dir, config=unit_config)
                            report_modes = set(config.get("detailed_unit_report_modes", ["flat"]))
                            if (
                                config.get("detailed_unit_reports", False)
                                and analysis_mode in report_modes
                            ):
                                feature_metadata = (
                                    descriptor_feature_metadata(config, names)
                                    if source_mode == "descriptors"
                                    else pd.DataFrame({"FeatureName": names})
                                )
                                make_decoding_report(
                                    result,
                                    title=(
                                        f"{target_name}: {analysis_mode}/"
                                        f"{unit['unit_name']} ({selection_mode})"
                                    ),
                                    feature_metadata=feature_metadata,
                                    sections="compact",
                                    on_error="placeholder",
                                    asset_urls=config.get(
                                        "report_asset_urls",
                                        "inline",
                                    ),
                                    output_path=str(output_dir / "report.html"),
                                )
                            records.extend(
                                result_records(
                                    result,
                                    context=context,
                                    output_dir=output_dir,
                                    include_p_values=True,
                                )
                            )
                        except Exception as exc:
                            LOGGER.exception("Decoding unit failed: %s", context)
                            output_dir.mkdir(parents=True, exist_ok=True)
                            (output_dir / "_FAILED").write_text("", encoding="utf-8")
                            failure = {
                                **context,
                                "status": "failed",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "output_dir": str(output_dir),
                            }
                            records.append(failure)
                            failures.append(failure)

    results_frame = pd.DataFrame(records)
    if not results_frame.empty and "p_value" in results_frame:
        results_frame = correct_sweep_pvalues(results_frame)
    derivative_root.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(derivative_root / "sweep_results.csv", index=False)
    pd.DataFrame(failures).to_csv(derivative_root / "failures.csv", index=False)
    (derivative_root / "config_used.yaml").write_text(
        yaml.safe_dump(redact_sensitive(config), sort_keys=False),
        encoding="utf-8",
    )
    primary_success = bool(
        not results_frame.empty
        and {"status", "primary"}.issubset(results_frame.columns)
        and (
            (results_frame["status"] == "success")
            & results_frame["primary"].fillna(False).astype(bool)
        ).any()
    )
    any_success = bool(
        not results_frame.empty
        and "status" in results_frame
        and (results_frame["status"] == "success").any()
    )
    failure_count = len(failures)
    status = (
        "SUCCESS"
        if primary_success and not failure_count
        else "PARTIAL"
        if any_success
        else "FAILED"
    )
    write_run_status(derivative_root, status)
    generate_decoding_summary_report(
        report_root / "dataset_summary.html",
        results_frame.to_dict("records"),
        title=f"Classical Decoding: {config.get('dataset_name', 'dataset')}",
        config=redact_sensitive(config),
        qc_results=qc_report_results,
    )
    generate_head_to_head_report(
        bids_root=bids_root,
        reports_root=reports_root,
        output_group=str(config.get("output_group", "default")),
        dataset_name=str(config.get("dataset_name", "dataset")),
        asset_urls=config.get("report_asset_urls", "inline"),
    )
    return derivative_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort_config",
        required=True,
        help="Cohort/dataset config: subjects + clinical question (configs/cohorts/).",
    )
    parser.add_argument(
        "--analysis_config",
        required=True,
        help="Analysis/method config: models, cv, feature_selection (configs/analyses/decoding/).",
    )
    parser.add_argument("--bids_root", default=None, help="Override BIDS root (else from config).")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override worker count.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's overwrite flag.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = resolve_cli_config(
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        bids_root=args.bids_root,
        metadata=args.metadata,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    run(config)


if __name__ == "__main__":
    main()

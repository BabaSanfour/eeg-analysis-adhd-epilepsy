#!/usr/bin/env python3
"""Leakage-safe classical decoding over descriptors and foundation embeddings.

The entry point stays thin, mirroring the dimensionality-reduction workflow:
``run`` normalizes the config into a :class:`ClassicalPlan`, loads each condition
scope once, enumerates the independent decoding units, and runs them through the
coco-pipe per-unit runner (optionally in parallel). All the per-unit lifecycle —
experiment assembly, resume, export, reports, record extraction — lives in
``coco_pipe.decoding`` and ``analysis.utils.decoding``.
"""

from __future__ import annotations

import argparse
import gc
import logging
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import coco_pipe.report
import numpy as np
import pandas as pd
from coco_pipe.decoding import (
    CVConfig,
    DecodingUnit,
    ExperimentConfig,
    FeatureSelectionConfig,
    ReducerConfig,
    TuningConfig,
    correct_sweep_pvalues,
    execute_decoding_sweep_streaming,
    grouped_chance_assessment,
    load_sweep_records,
    redact_sensitive,
    safe_group_n_splits,
)
from coco_pipe.descriptors import build_descriptor_feature_metadata
from coco_pipe.io import DataContainer, read_json
from coco_pipe.utils import slug

from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.common import (
    apply_family_qc_mask,
    container_pool_spec,
    families_for_analysis_unit,
    pool_containers_streaming,
    require_config,
)
from eeg_adhd_epilepsy.analysis.utils.decoding import (
    ClassicalPlan,
    build_classical_plan,
    build_loader_args,
    prepare_decoding_scope,
    resolve_decoding_paths,
)
from eeg_adhd_epilepsy.reports.decoding import (
    generate_decoding_summary_report,
    generate_head_to_head_report,
)
from eeg_adhd_epilepsy.utils.config import resolve_cli_config

LOGGER = logging.getLogger(__name__)

_FS_METADATA_KEYS = {"name", "analysis_modes"}
_MAX_FAILURE_REASONS_TO_LOG = 8


def _log_enumeration_failures(
    failures: list[dict[str, Any]],
    *,
    unit_count: int,
    derivative_root: Path,
) -> None:
    """Log a compact reason summary for skipped enumeration candidates."""
    if not failures:
        return

    reason_counts = Counter(str(item.get("reason", "unknown")) for item in failures)
    if unit_count == 0:
        LOGGER.warning(
            "No classical decoding units were enumerated; %d skip/failure record(s) "
            "will be written to %s.",
            len(failures),
            derivative_root / "failures.csv",
        )
    else:
        LOGGER.info(
            "Classical decoding enumeration skipped %d candidate(s); details will be "
            "written to %s.",
            len(failures),
            derivative_root / "failures.csv",
        )

    log_reason = LOGGER.warning if unit_count == 0 else LOGGER.info
    for reason, count in reason_counts.most_common(_MAX_FAILURE_REASONS_TO_LOG):
        log_reason("Enumeration skip x%d: %s", count, reason)
    remaining = len(reason_counts) - _MAX_FAILURE_REASONS_TO_LOG
    if remaining > 0:
        log_reason("Enumeration skip summary truncated; %d more reason(s).", remaining)


# ---------------------------------------------------------------------------
# Classical unit construction
# ---------------------------------------------------------------------------


def _validate_unit_data(
    X: np.ndarray,
    y: pd.Series,
    groups: pd.Series | None,
    target_name: str,
    requested_splits: int,
    is_classifier: bool = True,
) -> int:
    """Validate data dimensions and class counts; return safe CV fold count."""
    if X.shape[0] == 0:
        raise ValueError(f"No valid observations for {target_name}.")
    if is_classifier:
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError(
                f"Target '{target_name}' has fewer than 2 classes: {counts.to_dict()}."
            )
        if counts.min() < 2:
            raise ValueError(
                f"Target '{target_name}' has classes with < 2 members: {counts.to_dict()}."
            )
    return safe_group_n_splits(y, groups, requested_splits)


def _mode_models(
    plan: ClassicalPlan,
    analysis_mode: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return model configs and grids explicitly enabled for one analysis mode."""
    model_configs = {
        name: model
        for name, model in plan.model_configs.items()
        if plan.model_analysis_modes.get(name) is None
        or analysis_mode in plan.model_analysis_modes[name]
    }
    model_grids = {name: grid for name, grid in plan.model_grids.items() if name in model_configs}
    return model_configs, model_grids


def _build_selection_units(
    plan: ClassicalPlan,
    unit: dict[str, Any],
    X: np.ndarray,
    unit_y: pd.Series,
    unit_groups: pd.Series | None,
    unit_n_splits: int,
    unit_metadata: pd.DataFrame,
    scope: str,
    target_name: str,
    analysis_mode: str,
    derivative_root: Path,
    config: Mapping[str, Any],
    model_configs: dict[str, Any],
    model_grids: dict[str, dict[str, Any]],
    random_state: int,
    overwrite: bool,
):
    """Yield DecodingUnit configs for all valid feature selection passes."""
    flattened = (
        unit["container"]
        if unit["container"].dims == ("obs", "feature")
        else unit["container"].flatten(preserve="obs")
    )
    names = [str(value) for value in flattened.coords.get("feature", np.arange(X.shape[1]))]
    valid_selection_specs = [
        spec
        for spec in plan.selection_specs
        if (
            not spec.get("analysis_modes")
            or analysis_mode in {str(mode) for mode in spec["analysis_modes"]}
        )
        if str(spec.get("method", "none")) == "none" or X.shape[1] > 1
    ]

    for selection_spec in valid_selection_specs:
        selection_mode = str(selection_spec.get("name") or selection_spec.get("method") or "fs")
        stem = (
            f"fit_{slug(scope)}_{slug(target_name)}"
            f"_{analysis_mode}_{slug(unit['unit_key'])}_{selection_mode}"
        )
        output_dir = derivative_root / "artifacts" / "fits" / stem
        context = {
            "scope": scope,
            "target": target_name,
            "input_mode": plan.input_mode,
            "analysis_mode": analysis_mode,
            "unit_name": unit["unit_name"],
            "unit_key": unit["unit_key"],
            "family": unit.get("family"),
            "subfamily": unit.get("subfamily"),
            "selection_mode": selection_mode,
            "primary": analysis_mode == "flat",
        }
        cv_config = CVConfig(
            strategy="stratified_group_kfold",
            n_splits=unit_n_splits,
            shuffle=True,
            random_state=random_state,
            group_key="group_id",
        )
        tuning = (
            TuningConfig(
                enabled=True,
                scoring=plan.tuning_cfg.get("scoring"),
                search_type=plan.tuning_cfg.get("search_type", "grid"),
                cv=CVConfig(
                    strategy="stratified_group_kfold",
                    n_splits=min(int(plan.tuning_cfg.get("n_splits", 3)), unit_n_splits),
                    shuffle=True,
                    random_state=random_state,
                    group_key="group_id",
                ),
                n_jobs=1,
                allow_nongroup_inner_cv=bool(plan.tuning_cfg.get("allow_nongroup_inner_cv", False)),
            )
            if model_grids and bool(plan.tuning_cfg.get("enabled", True))
            else None
        )
        reducer = (
            ReducerConfig(
                enabled=True,
                method="pca",
                n_components=plan.reducer_cfg.get("n_components"),
                whiten=bool(plan.reducer_cfg.get("whiten", False)),
            )
            if plan.reducer_enabled
            else None
        )
        fs_kwargs = {k: v for k, v in selection_spec.items() if k not in _FS_METADATA_KEYS}
        fs_config = (
            FeatureSelectionConfig(enabled=True, cv=cv_config, **fs_kwargs)
            if str(fs_kwargs.get("method", "none")) != "none"
            else None
        )

        experiment_config = ExperimentConfig(
            task="classification",
            tag=f"{target_name}_{analysis_mode}_{unit['unit_key']}_{selection_mode}",
            random_state=random_state,
            models=model_configs,
            grids=model_grids or None,
            cv=cv_config,
            tuning=tuning or TuningConfig(enabled=False),
            reducer=reducer or ReducerConfig(enabled=False),
            feature_selection=fs_config or FeatureSelectionConfig(enabled=False),
            statistical_assessment=grouped_chance_assessment(
                config["chance_method"],
                n_permutations=int(config["n_permutations"]),
                store_null=bool(config["store_null_distribution"]),
            ),
            metrics=list(plan.metrics),
            use_scaler=True,
            n_jobs=1,
            verbose=bool(config["verbose"]),
        )
        yield DecodingUnit(
            experiment_config=experiment_config,
            X=X,
            y=unit_y,
            output_dir=output_dir,
            context=context,
            run_config={**dict(config), **context},
            groups=unit_groups,
            feature_names=names,
            sample_ids=unit_metadata["sample_id"].astype(str),
            sample_metadata=unit_metadata,
            inferential_unit="group_id",
            overwrite=overwrite,
            include_p_values=True,
        )


def _enumerate_scope(
    plan: ClassicalPlan,
    scope: str,
    full_container: DataContainer,
    config: Mapping[str, Any],
    derivative_root: Path,
):
    """Yield DecodingUnit objects or failure dicts for a single scope."""
    from coco_pipe.io import iter_analysis_units

    subject_col = str(config.get("subject_col", "study_id"))
    session_col = str(config.get("session_col", "session_id"))
    requested_splits = int(config["cv"]["n_splits"])
    random_state = int(config["random_state"])
    overwrite = bool(config["overwrite"])

    for eval_spec in plan.evals:
        target_name = eval_spec.get("name", eval_spec["target_col"])
        if target_name == "condition_separation" and scope != "pooled":
            continue
        try:
            target_container, y, groups, sample_metadata, _ = prepare_decoding_scope(
                full_container,
                eval_spec,
                scope=scope,
                group_col=eval_spec.get("group_col", config.get("group_col", "patient_group_id")),
                session_col=session_col,
                subject_col=subject_col,
                requested_splits=requested_splits,
            )
        except Exception as exc:
            yield {
                "scope": scope,
                "target": target_name,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            continue

        for analysis_mode in plan.analysis_modes:
            model_configs, model_grids = _mode_models(plan, analysis_mode)
            if not model_configs:
                yield {
                    "scope": scope,
                    "target": target_name,
                    "analysis_mode": analysis_mode,
                    "status": "skipped",
                    "reason": "No models are configured for this analysis mode.",
                }
                continue
            try:
                analysis_units = iter_analysis_units(
                    target_container,
                    analysis_mode,
                    "descriptors" if plan.input_mode == "descriptors" else "foundation_embeddings",
                    config.get("descriptor_families"),
                )
            except Exception as exc:
                yield {
                    "scope": scope,
                    "target": target_name,
                    "analysis_mode": analysis_mode,
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                continue

            for unit in analysis_units:
                families = families_for_analysis_unit(
                    target_container,
                    unit,
                    config.get("descriptor_families"),
                )
                unit_container, keep_indices = apply_family_qc_mask(unit["container"], families)
                unit["container"] = unit_container
                unit_y = pd.Series(y).iloc[keep_indices].reset_index(drop=True)
                unit_groups = pd.Series(groups).iloc[keep_indices].reset_index(drop=True)
                unit_metadata = sample_metadata.iloc[keep_indices].reset_index(drop=True)
                try:
                    unit_n_splits = _validate_unit_data(
                        unit_container.X,
                        unit_y,
                        unit_groups,
                        target_name,
                        requested_splits,
                    )
                except ValueError as exc:
                    yield {
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
                    continue

                X = np.asarray(unit_container.X)
                if X.ndim != 2:
                    X = unit_container.flatten(preserve="obs").X

                try:
                    yield from _build_selection_units(
                        plan=plan,
                        unit=unit,
                        X=X,
                        unit_y=unit_y,
                        unit_groups=unit_groups,
                        unit_n_splits=unit_n_splits,
                        unit_metadata=unit_metadata,
                        scope=scope,
                        target_name=target_name,
                        analysis_mode=analysis_mode,
                        derivative_root=derivative_root,
                        config=config,
                        model_configs=model_configs,
                        model_grids=model_grids,
                        random_state=random_state,
                        overwrite=overwrite,
                    )
                except ValueError as exc:
                    yield {
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


def _classical_scope_units(
    plan: ClassicalPlan,
    scope: str,
    full_container: DataContainer,
    *,
    config: Mapping[str, Any],
    derivative_root: Path,
    failures: list[dict[str, Any]],
) -> list[DecodingUnit]:
    """Build one scope's decoding units; append enumeration skips to *failures*."""
    units: list[DecodingUnit] = []
    for item in _enumerate_scope(
        plan=plan,
        scope=scope,
        full_container=full_container,
        config=config,
        derivative_root=derivative_root,
    ):
        if isinstance(item, DecodingUnit):
            units.append(item)
        else:
            failures.append(item)
    return units


def _iter_classical_unit_batches(
    plan: ClassicalPlan,
    conditions: list[str],
    loader_args: Any,
    metadata: pd.DataFrame | None,
    *,
    config: Mapping[str, Any],
    derivative_root: Path,
    failures: list[dict[str, Any]],
    enum_failures: list[dict[str, Any]],
    qc_report_results: list[tuple[str, Any]],
    stats: dict[str, int],
) -> Iterator[list[DecodingUnit]]:
    """Yield one scope's decoding units at a time, streaming data per condition.

    Loads each condition's container lazily, builds that scope's units, yields
    them, then frees the container's data — pooling reloads each condition one at
    a time (via container_pool_spec + loaders) rather than holding every condition
    plus pooled, and all their enumerated unit ``X`` copies, at once. Enumeration
    skips accumulate into both *failures* (persisted with runtime failures) and
    *enum_failures* (enumeration-only, for the skip-summary log), which are
    separated here because runtime failures are appended to *failures* only after
    this generator is exhausted.
    """
    run_pooled = bool(config["run_pooled"]) and len(conditions) > 1
    pool_specs: list[dict[str, Any]] = []
    pool_loaders: list[Any] = []
    for condition in conditions:
        container = build_dataset(loader_args, metadata, condition, target_col=None)
        if container.meta.get("qc_result") is not None:
            qc_report_results.append((condition, container.meta["qc_result"]))
        n_before = len(failures)
        units = _classical_scope_units(
            plan,
            condition,
            container,
            config=config,
            derivative_root=derivative_root,
            failures=failures,
        )
        enum_failures.extend(failures[n_before:])
        stats["units"] += len(units)
        if run_pooled:
            pool_specs.append(container_pool_spec(container))
            pool_loaders.append(
                lambda args=loader_args, cond=condition: build_dataset(
                    args, metadata, cond, target_col=None
                )
            )
        yield units
        del container, units
        gc.collect()

    if len(pool_specs) > 1:
        pooled = pool_containers_streaming(pool_specs, pool_loaders)
        n_before = len(failures)
        units = _classical_scope_units(
            plan,
            "pooled",
            pooled,
            config=config,
            derivative_root=derivative_root,
            failures=failures,
        )
        enum_failures.extend(failures[n_before:])
        stats["units"] += len(units)
        yield units
        del pooled, units

    del pool_specs, pool_loaders
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _classical_primary_mask(frame: pd.DataFrame) -> pd.Series:
    """Successful flat-baseline rows — the primary classical result."""
    if not {"status", "primary"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    return (frame["status"] == "success") & frame["primary"].fillna(False).astype(bool)


def run(config: dict[str, Any]) -> Path:
    plan = build_classical_plan(config)
    (
        bids_root,
        derivative_root,
        report_root,
        metadata,
        config_hash,
        reports_root,
        dataset_name_slug,
    ) = resolve_decoding_paths(config, input_mode=plan.input_mode)
    compare_only = bool(config.get("compare_only", False))
    reports_only = bool(config.get("reports_only", False))

    if not compare_only:
        conditions = require_config(config, "conditions", expected_type=list, cast_str=True)
        loader_args = build_loader_args(
            config,
            input_mode=plan.input_mode,
            layout_mode=plan.layout_mode,
        )
        qc_report_results: list[tuple[str, Any]] = []

        if reports_only:
            records = load_sweep_records(derivative_root)
            for condition in conditions:
                container = build_dataset(loader_args, metadata, condition, target_col=None)
                if container.meta.get("qc_result") is not None:
                    qc_report_results.append((condition, container.meta["qc_result"]))
                del container
                gc.collect()
        else:
            failures: list[dict[str, Any]] = []
            enum_failures: list[dict[str, Any]] = []
            stats = {"units": 0}
            records, _ = execute_decoding_sweep_streaming(
                _iter_classical_unit_batches(
                    plan,
                    conditions,
                    loader_args,
                    metadata,
                    config=config,
                    derivative_root=derivative_root,
                    failures=failures,
                    enum_failures=enum_failures,
                    qc_report_results=qc_report_results,
                    stats=stats,
                ),
                failures,
                config=config,
                output_root=derivative_root,
                results_filename="sweep_results.csv",
                primary_mask=_classical_primary_mask,
                leaderboard_group_fields=("scope", "target", "analysis_mode", "selection_mode"),
                reallocate_inner_jobs=True,
                frame_post=lambda frame: (
                    correct_sweep_pvalues(frame) if "p_value" in frame else frame
                ),
                run_metadata={
                    "dataset_name": config["dataset_name"],
                    "input_mode": plan.input_mode,
                    "config_hash": config_hash,
                    "run_variant": derivative_root.name,
                },
            )
            _log_enumeration_failures(
                enum_failures,
                unit_count=stats["units"],
                derivative_root=derivative_root,
            )

        if config.get("detailed_unit_reports", False):
            feature_metadata = None
            if plan.input_mode == "descriptors":
                descriptor_columns_path = config.get("descriptor_feature_columns_path")
                if descriptor_columns_path:
                    path = Path(str(descriptor_columns_path)).expanduser()
                    if path.exists():
                        feature_metadata = build_descriptor_feature_metadata(read_json(path))
            coco_pipe.report.render_unit_reports(
                records,
                modes=config.get("detailed_unit_report_modes", ["flat"]),
                feature_metadata=feature_metadata,
                asset_urls=config.get("report_asset_urls", "inline"),
                title_fn=lambda record: (
                    f"{record.get('target')}: {record.get('analysis_mode')}/"
                    f"{record.get('unit_name')} ({record.get('selection_mode')})"
                ),
            )
        generate_decoding_summary_report(
            report_root / "dataset_summary.html",
            records,
            title=f"Classical Decoding: {config['dataset_name']}",
            config=redact_sensitive(config),
            qc_results=qc_report_results,
        )

    generate_head_to_head_report(
        bids_root=bids_root,
        reports_root=reports_root,
        dataset_name=dataset_name_slug,
        asset_urls=config["report_asset_urls"],
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
        help=(
            "Analysis/method config: models, cv, feature_selection "
            "(configs/analyses/decoding/classical.yaml)."
        ),
    )
    parser.add_argument("--bids_root", default=None, help="Override BIDS root (else from config).")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override worker count.")
    parser.add_argument("--reports_root", default=None, help="Override reports root (else config).")
    parser.add_argument(
        "--descriptor_table_path",
        default=None,
        help="Descriptor table path (dataset path; supply here rather than in the config).",
    )
    parser.add_argument(
        "--descriptor_feature_columns_path",
        default=None,
        help="Descriptor feature-columns JSON path (dataset path; supply here, not in config).",
    )
    parser.add_argument(
        "--representation",
        choices=["epoch", "recording", "subject"],
        default=None,
        help="Representation granularity to reduce (e.g., epoch, subject, recording).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's overwrite flag.",
    )
    parser.add_argument(
        "--reports-only",
        dest="reports_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Regenerate reports from the saved runs/ inventory without refitting.",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Skip decoding and regenerate only the head-to-head comparison report.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = resolve_cli_config(
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        bids_root=args.bids_root,
        metadata=args.metadata,
        n_jobs=args.n_jobs,
        reports_root=args.reports_root,
        descriptor_table_path=args.descriptor_table_path,
        descriptor_feature_columns_path=args.descriptor_feature_columns_path,
        overwrite=args.overwrite,
        representation=args.representation,
        reports_only=args.reports_only,
        compare_only=args.compare_only,
    )
    run(config)


if __name__ == "__main__":
    main()

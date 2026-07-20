#!/usr/bin/env python3
"""Checkpointed dimensionality-reduction analysis for EEG data."""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.dim_reduction import (
    EVAL_RUN_KEY_FIELDS,
    FIT_METRIC_COLUMNS,
    FIT_RUN_KEY_FIELDS,
    POOLED_CONDITION,
    build_availability_record,
    build_eval_request,
    build_fit_request,
    parse_eval_specs,
    run_eval,
    run_fit_group,
    update_runs,
    valid_component_sweep,
    write_run_status,
)
from coco_pipe.io import (
    DataContainer,
    fingerprint_container,
    iter_analysis_units,
    read_json,
    read_table,
    write_json,
)
from coco_pipe.utils import resolve_n_jobs, run_task_batch, slug, stable_hash
from joblib import parallel_backend

from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.common import (
    apply_family_qc_mask,
    base_layout_mode,
    families_for_analysis_unit,
    pool_containers,
    require_config,
)
from eeg_adhd_epilepsy.analysis.utils.dim_reduction import (
    DEFAULT_DIM_REDUCTION_SELECTION_METRIC,
    DIM_REDUCTION_EVAL_METRIC_COLUMNS,
    _load_run,
    _selection_config,
    build_and_validate_mode_specs,
    build_input_signature,
    build_output_root,
    build_run_config_payload,
    group_fit_requests,
    validate_inputs,
)
from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    get_derivative_root,
)
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    summary_report_dir,
)
from eeg_adhd_epilepsy.reports._common import AlignmentDiagnosticsSpec
from eeg_adhd_epilepsy.reports.dim_reduction import (
    collect_mode_leaderboard,
    generate_dataset_report,
    generate_rollup_report,
)
from eeg_adhd_epilepsy.utils.artifacts import freeze_config_used
from eeg_adhd_epilepsy.utils.config import resolve_cli_config
from eeg_adhd_epilepsy.utils.constants import DEFAULT_ANALYSIS_CONDITIONS

logger = logging.getLogger(__name__)
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)


def _collect_scope_fit_requests(
    scope: str,
    condition: str,
    container,
    args,
    reducers: list[str],
    output_root: Path,
    unit_containers_by_key: dict,
    data_availability: list,
) -> list[dict[str, Any]]:
    """Enumerate analysis units and build fit requests for one scope/condition pair."""
    requests: list[dict[str, Any]] = []
    num_skipped = 0
    for unit_spec in iter_analysis_units(
        container,
        args.analysis_mode,
        args.input_mode,
        args.descriptor_families,
    ):
        families = families_for_analysis_unit(
            container,
            unit_spec,
            args.descriptor_families,
        )
        unit_container, _ = apply_family_qc_mask(
            unit_spec["container"],
            families,
        )
        if np.asarray(unit_container.X).ndim != 2:
            unit_container = unit_container.flatten(preserve="obs")
        unit_spec = {**unit_spec, "container": unit_container}
        unit_container = unit_spec["container"]
        unit_containers_by_key[(scope, condition, unit_spec["unit_key"])] = unit_container
        X = np.asarray(unit_container.X)
        valid_components = valid_component_sweep(unit_container, args.n_components_sweep)
        data_availability.append(
            build_availability_record(
                scope=scope,
                condition=condition,
                unit_spec=unit_spec,
                container=unit_container,
                requested_components=args.n_components_sweep,
                valid_components=valid_components,
            )
        )
        if not valid_components:
            logger.warning(
                "Skipping %s/%s: no valid n_components for matrix shape %s.",
                condition,
                unit_spec["unit_name"],
                tuple(X.shape),
            )
            continue
        input_signature = build_input_signature(args, unit_spec)
        container_signature = fingerprint_container(unit_container)

        for reducer_name, n_components in product(reducers, valid_components):
            request = build_fit_request(
                container=unit_container,
                scope=scope,
                condition=condition,
                unit_spec=unit_spec,
                reducer=reducer_name,
                n_components=int(n_components),
                input_signature=input_signature,
                output_root=output_root,
                overwrite=bool(args.overwrite),
                subject_col=args.subject_col,
                container_signature=container_signature,
            )
            if (request["out_path"] / "_SUCCESS").exists() and not request["overwrite"]:
                num_skipped += 1
            else:
                logger.info(
                    "Fitting %s/%s/%s/%s/n%d",
                    condition,
                    args.analysis_mode,
                    unit_spec["unit_name"],
                    reducer_name,
                    int(n_components),
                )
            requests.append(request)

    if num_skipped > 0:
        logger.info(
            "Skipped %d existing fit(s) for %s condition '%s'.",
            num_skipped,
            scope,
            condition,
        )

    return requests


def _get_base_container(
    base_cache: dict[tuple[str, str, str], Any],
    args: Any,
    *,
    condition: str,
    meta_df: Any,
    load_mode: str,
    representation: str,
) -> DataContainer:
    """Load (and cache) one condition's base container in a shared layout.

    Descriptor and foundation inputs load a single layout that every analysis
    mode re-slices, so the cache turns an N-mode sweep into one load per
    condition instead of N. Cached load failures re-raise without retrying.
    """
    key = (condition, representation, load_mode)
    if key in base_cache:
        cached = base_cache[key]
        if isinstance(cached, Exception):
            raise cached
        return cached
    try:
        container = build_dataset(
            args,
            meta_df,
            condition,
            target_col=None,
            analysis_mode=load_mode,
            representation=representation,
        )
    except Exception as exc:
        base_cache[key] = exc
        raise
    base_cache[key] = container
    return container


def _run_shared_memory_batch(
    tasks: list[Any],
    worker_fn: Any,
    max_workers: int,
) -> list[Any]:
    """Run fit/eval batches with thread workers to avoid duplicating large arrays."""
    if max_workers == 1 or len(tasks) <= 1:
        return run_task_batch(tasks, worker_fn, max_workers)
    with parallel_backend("threading"):
        return run_task_batch(tasks, worker_fn, max_workers)


def execute_analysis_mode(
    *,
    args: Any,
    mode: str,
    representation: str,
    reducers: list[str],
    n_components_sweep: list[int],
    meta_df: Any,
    eval_specs: list[dict[str, Any]],
    resolved_n_jobs: int,
    bids_root: Path,
    base_cache: dict[tuple[str, str, str], Any],
    base_load_mode: str,
    reports_root: Path,
) -> dict[str, Any]:
    """Run one analysis mode end-to-end (fits, evals, per-mode report) and report back."""
    args.analysis_mode = mode
    args.representation = representation
    args.n_components_sweep = list(n_components_sweep)

    run_config = build_run_config_payload(args, reducers, eval_specs)
    args.run_config_hash = stable_hash(run_config, length=12)

    output_root = build_output_root(bids_root, args, mode, representation)
    run_variant = args.run_variant
    output_dataset_name = slug(args.run_label)
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {key: value for key, value in vars(args).items() if key != "config"}
    if eval_specs:
        config_snapshot["evals"] = eval_specs
    freeze_config_used(config_snapshot, output_root, overwrite=True)
    fit_runs_path = runs_dir / "fit_runs.json"
    eval_runs_path = runs_dir / "eval_runs.json"
    run_summary_path = runs_dir / "run_summary.json"
    if not args.reports_only:
        write_json(fit_runs_path, [], indent=2)
        write_json(eval_runs_path, [], indent=2)

    base_containers_by_scope: dict[tuple[str, str], DataContainer] = {}
    unit_containers_by_key: dict[tuple[str, str, str], DataContainer] = {}
    data_availability: list[dict[str, Any]] = []
    dataset_stats: list[dict[str, Any]] = []
    report_path: Path | None = None
    fatal_error: str | None = None
    leaderboard = None
    try:
        if args.reports_only:
            if not fit_runs_path.exists():
                raise FileNotFoundError(
                    f"--reports-only requested but {fit_runs_path} does not exist."
                )
            fit_runs = read_json(fit_runs_path)
            if run_summary_path.exists():
                summary_data = read_json(run_summary_path)
                dataset_stats = summary_data.get("run_metadata", {}).get("dataset_stats", [])
        else:
            fit_requests: list[dict[str, Any]] = []
            for condition in args.conditions:
                logger.info(
                    "Loading input for condition '%s' (%s / %s).",
                    condition,
                    args.input_mode,
                    mode,
                )
                base_container = _get_base_container(
                    base_cache,
                    args,
                    condition=condition,
                    meta_df=meta_df,
                    load_mode=base_load_mode,
                    representation=representation,
                )
                base_containers_by_scope[("condition", condition)] = base_container
                fit_requests.extend(
                    _collect_scope_fit_requests(
                        "condition",
                        condition,
                        base_container,
                        args,
                        reducers,
                        output_root,
                        unit_containers_by_key,
                        data_availability,
                    )
                )

            if args.run_pooled:
                source_containers = [
                    base_containers_by_scope[("condition", cond)] for cond in args.conditions
                ]
                pooled_container = pool_containers(source_containers)
                base_containers_by_scope[("pooled", POOLED_CONDITION)] = pooled_container
                fit_requests.extend(
                    _collect_scope_fit_requests(
                        "pooled",
                        POOLED_CONDITION,
                        pooled_container,
                        args,
                        reducers,
                        output_root,
                        unit_containers_by_key,
                        data_availability,
                    )
                )

            logger.info(
                "Queued %d fit request(s) across %d loaded scope(s) for mode '%s'.",
                len(fit_requests),
                len(base_containers_by_scope),
                mode,
            )
            fit_groups = group_fit_requests(fit_requests)
            fit_runs = []
            for group_records in _run_shared_memory_batch(
                fit_groups,
                lambda group: run_fit_group(group, errors="raise"),
                resolved_n_jobs,
            ):
                for record in group_records:
                    update_runs(fit_runs_path, record, key_fields=FIT_RUN_KEY_FIELDS)
                    fit_runs.append(record)

            if eval_specs:
                eval_requests: list[dict[str, Any]] = []
                for fit_record in fit_runs:
                    unit_container = unit_containers_by_key[
                        (
                            fit_record["scope"],
                            fit_record["condition"],
                            fit_record["unit_key"],
                        )
                    ]
                    for eval_spec in eval_specs:
                        # The condition_separation eval is only meaningful for pooled scopes.
                        if (
                            eval_spec["name"] == "condition_separation"
                            and fit_record["scope"] != "pooled"
                        ):
                            continue
                        eval_requests.append(
                            build_eval_request(
                                fit_record=fit_record,
                                eval_spec=eval_spec,
                                container=unit_container,
                                output_root=output_root,
                                overwrite=bool(args.overwrite),
                            )
                        )
                logger.info("Queued %d eval request(s) for mode '%s'.", len(eval_requests), mode)
                for record in _run_shared_memory_batch(
                    eval_requests,
                    lambda request: run_eval(**request, errors="raise"),
                    resolved_n_jobs,
                ):
                    update_runs(eval_runs_path, record, key_fields=EVAL_RUN_KEY_FIELDS)

        if not fit_runs:
            raise RuntimeError(
                f"No fit runs found in {fit_runs_path}. "
                "Dim reduction produced no successful fit inventory."
            )

        if not args.reports_only and base_containers_by_scope:
            import pandas as pd

            for (scope, condition), container in base_containers_by_scope.items():
                stat = {
                    "scope": scope,
                    "condition": condition,
                    "loaded_observations": int(
                        container.meta.get("loaded_obs", container.X.shape[0])
                    ),
                    "samples_used": int(container.X.shape[0]),
                }
                if hasattr(args, "subject_col") and args.subject_col in container.coords:
                    stat["unique_subjects"] = int(
                        pd.Index(
                            np.asarray(container.coords[args.subject_col]).astype(str)
                        ).nunique()
                    )
                if "recording_id" in container.coords:
                    stat["unique_recordings"] = int(
                        pd.Index(np.asarray(container.coords["recording_id"]).astype(str)).nunique()
                    )
                dataset_stats.append(stat)

        report = generate_dataset_report(
            args=args,
            output_root=output_root,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            reducers=reducers,
            meta_df=meta_df,
            containers_by_scope=base_containers_by_scope,
            dataset_stats=dataset_stats,
            eval_specs=eval_specs,
            pooled_condition=POOLED_CONDITION,
        )
        summary_dir = summary_report_dir(reports_root, ReportStage.DIM_REDUCTION, create=True)
        summary_dir = summary_dir / output_dataset_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        report_path = summary_dir / f"{run_variant}_dataset_summary.html"
        report.save(report_path)
        logger.info("Report saved to: %s", report_path)
        leaderboard = collect_mode_leaderboard(
            args=args,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            reducers=reducers,
            pooled_condition=POOLED_CONDITION,
        )
        if leaderboard is not None and not leaderboard.empty:
            leaderboard.to_json(runs_dir / "leaderboard.json", orient="records")
    except Exception as exc:
        fatal_error = str(exc)
        raise
    finally:
        run_summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_run_status(
            output_root,
            fit_runs_path,
            eval_runs_path,
            run_summary_path=run_summary_path,
            fatal_error=fatal_error,
            report_path=report_path,
            run_metadata={
                "dataset_name": args.dataset_name,
                "run_label": args.run_label,
                "input_mode": args.input_mode,
                "analysis_mode": mode,
                "representation": representation,
                "run_variant": run_variant,
                "run_config_hash": args.run_config_hash,
                "output_dataset_name": output_dataset_name,
                "conditions_requested": list(args.conditions),
                "subjects_resolved": (
                    len(base_containers_by_scope[("condition", args.conditions[0])].ids)
                    if args.conditions
                    and ("condition", args.conditions[0]) in base_containers_by_scope
                    else 0
                ),
                "data_availability": data_availability,
                "dataset_stats": dataset_stats,
            },
        )
    return {
        "analysis_mode": mode,
        "representation": representation,
        "input_mode": args.input_mode,
        "run_variant": run_variant,
        "report_path": str(report_path) if report_path else None,
        "leaderboard": leaderboard,
    }


def _run_args_from_config(config: dict[str, Any]) -> SimpleNamespace:
    """Build the per-run args namespace from the merged cohort+analysis config.

    Every field is sourced via ``config.get`` with a default — the run no longer
    depends on argparse defaults (mirrors ``classical_decoding.run``). Per-mode
    fields (``analysis_mode``, ``representation``, ``n_components_sweep``) and
    derived provenance (``run_config_hash``, ``run_variant``, ``run_label``) are
    set later at runtime.
    """
    return SimpleNamespace(
        # Identity / selection
        dataset_name=config.get("dataset_name"),
        run_label=config.get("run_label"),
        conditions=list(config.get("conditions", DEFAULT_CONDITIONS)),
        subjects=config.get("subjects"),
        subject_col=config.get("subject_col", "study_id"),
        run_pooled=bool(config.get("run_pooled", False)),
        selection_metric=config.get("selection_metric", DEFAULT_DIM_REDUCTION_SELECTION_METRIC),
        selection_eval_name=config.get("selection_eval_name"),
        # Input mode + dataset loading
        input_mode=config.get("input_mode", "raw"),
        reduced_source_input_mode=config.get("reduced_source_input_mode", "descriptors"),
        analysis_mode=config.get("analysis_mode", "flat"),
        representation=config.get("representation", ""),
        bids_root=config.get("bids_root"),
        metadata=config.get("metadata"),
        use_derivatives=bool(config.get("use_derivatives", True)),
        task=config.get("task", "clinical"),
        segment_duration=float(config.get("segment_duration", 60.0)),
        overlap=float(config.get("overlap", 0.0)),
        desc=config.get("desc", "base"),
        window_source=config.get("window_source", "auto"),
        units=config.get("units"),
        # Descriptors
        descriptor_table_path=config.get("descriptor_table_path"),
        descriptor_feature_columns_path=config.get("descriptor_feature_columns_path"),
        descriptor_families=config.get("descriptor_families"),
        descriptor_max_abs_value=config.get("descriptor_max_abs_value"),
        location_statistic=config.get("location_statistic"),
        # Foundation embeddings
        embedding_derivative_root=config.get("embedding_derivative_root"),
        derivative_root=config.get("derivative_root"),
        embedding_aggregate_by=config.get("embedding_aggregate_by"),
        embedding_model_key=config.get("embedding_model_key"),
        # Filtering / balancing
        filter_col=list(config.get("filter_col", []) or []),
        filter_val=list(config.get("filter_val", []) or []),
        group_filters=config.get("group_filters"),
        balance_target=config.get("balance_target"),
        balance_strategy=config.get("balance_strategy", "undersample"),
        # Reduction plan (per-mode sweep overrides happen in execute_analysis_mode)
        analysis_modes=config.get("analysis_modes"),
        n_components_sweep=list(config.get("n_components_sweep", []) or []),
        # Execution / reporting
        n_jobs=config.get("n_jobs", 1),
        overwrite=bool(config.get("overwrite", False)),
        reports_only=bool(config.get("reports_only", False)),
        reports_root=config.get("reports_root"),
        qc=config.get("qc", {}),
        interactive=bool(config.get("interactive", False)),
        color_by="subject",
    )


def compare_cohort(
    dim_root: Path,
    dataset_name: str,
    reports_root: Path,
    embedding_model_key: str | None = None,
    bids_root: str | Path | None = None,
    alignment_diagnostics_cohort_name: str | None = None,
    alignment_diagnostics_population: str | None = None,
) -> Path | None:
    """Render one cohort comparison and any explicitly requested diagnostics.

    ``dataset_name`` identifies the reduction-run directory and report label.
    The diagnostics assessment uses the separate exact cohort identity in
    ``alignment_diagnostics_cohort_name``; no BIDS root or cohort name is
    reconstructed from output paths.
    """
    cohort_dir = dim_root / slug(dataset_name)
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"No dim-reduction runs under {cohort_dir}.")

    run_dirs = sorted(
        path.parent.parent for path in cohort_dir.glob("foundation_*/runs/leaderboard.json")
    )
    summaries: list[dict[str, Any]] = []
    selection_metric, selection_eval_name = DEFAULT_DIM_REDUCTION_SELECTION_METRIC, None
    for run_dir in run_dirs:
        summary = _load_run(run_dir)
        if summary is None:
            continue
        leaderboard = summary["leaderboard"].copy()
        model_keys = leaderboard["model"].astype(str)
        aligned = model_keys.str.contains("_align-", regex=False)
        leaderboard.loc[~aligned, "transform"] = "none"
        leaderboard.loc[aligned, "transform"] = model_keys[aligned].str.split("_align-", n=1).str[1]
        leaderboard.loc[aligned, "model"] = model_keys[aligned].str.split("_align-", n=1).str[0]
        summary["leaderboard"] = leaderboard
        summaries.append(summary)
        selection_metric, selection_eval_name = _selection_config(run_dir)

    if not summaries:
        logger.warning("No foundation leaderboards under %s; nothing to compare.", cohort_dir)
        return None

    args = SimpleNamespace(
        run_label=dataset_name,
        dataset_name=dataset_name,
        input_mode="foundation_embeddings",
        selection_metric=selection_metric,
        selection_eval_name=selection_eval_name,
    )
    alignment_diagnostics = (
        AlignmentDiagnosticsSpec(
            base_model_key=embedding_model_key,
            cohort_name=alignment_diagnostics_cohort_name,
            population=alignment_diagnostics_population,
        )
        if alignment_diagnostics_population
        else None
    )
    report = generate_rollup_report(
        args=args,
        summaries=summaries,
        bids_root=bids_root,
        alignment_diagnostics=alignment_diagnostics,
    )

    out_dir = summary_report_dir(reports_root, ReportStage.DIM_REDUCTION, create=True) / slug(
        dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "foundation_model_comparison.html"
    report.save(out_path)

    merged = pd.concat([summary["leaderboard"] for summary in summaries], ignore_index=True)
    merged.to_csv(out_dir / "foundation_model_comparison.csv", index=False)
    logger.info(
        "Cross-model comparison for %s: %d run(s) -> %s",
        dataset_name,
        len(summaries),
        out_path,
    )
    return out_path


def run(config: dict[str, Any]) -> None:
    """Run the full dimensionality-reduction sweep from a merged config dict."""
    bids_root = Path(config["bids_root"]).expanduser()

    if config.get("compare_only"):
        dim_root = (
            Path(config["derivative_root"]).expanduser()
            if config.get("derivative_root")
            else get_derivative_root(bids_root, DerivativeStage.DIM_REDUCTION)
        )
        reports_root = (
            Path(config["reports_root"]).expanduser()
            if config.get("reports_root")
            else Path(default_reports_root(bids_root)).expanduser()
        )
        if config.get("dataset_name"):
            cohorts = [str(config["dataset_name"])]
        else:
            cohorts = sorted(
                path.name
                for path in dim_root.iterdir()
                if path.is_dir() and any(path.glob("foundation_*/runs/leaderboard.json"))
            )
        if not cohorts:
            logger.warning("No cohorts with foundation runs found under %s.", dim_root)
            return

        for cohort in cohorts:
            compare_cohort(
                dim_root,
                cohort,
                reports_root,
                bids_root=bids_root,
                embedding_model_key=config.get("embedding_model_key"),
                alignment_diagnostics_cohort_name=config.get(
                    "alignment_diagnostics_cohort_name",
                    config.get("dataset_name"),
                ),
                alignment_diagnostics_population=config.get("alignment_diagnostics_population"),
            )
        return

    args = _run_args_from_config(config)
    validate_inputs(args)
    resolved_n_jobs = resolve_n_jobs(args.n_jobs)

    meta_df = (
        read_table(Path(config["metadata"]).expanduser(), sep=",")
        if config.get("metadata")
        else None
    )

    eval_specs = parse_eval_specs(
        require_config(config, "evals", expected_type=list),
        args.subject_col,
    )

    valid_selection_metrics = {*FIT_METRIC_COLUMNS, *DIM_REDUCTION_EVAL_METRIC_COLUMNS}
    eval_names = {spec["name"] for spec in eval_specs}

    if args.selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Unknown selection_metric '{args.selection_metric}'. "
            f"Valid metrics: {sorted(valid_selection_metrics)}"
        )
    if args.selection_eval_name and args.selection_eval_name not in eval_names:
        raise ValueError(
            f"Unknown selection_eval_name '{args.selection_eval_name}'. "
            f"Valid eval names: {sorted(eval_names)}"
        )

    reports_root = Path(args.reports_root) if args.reports_root else default_reports_root(bids_root)

    mode_specs, mode_tasks = build_and_validate_mode_specs(args)
    logger.info(
        "Dimensionality-reduction plan: %d task(s) -> %s",
        len(mode_tasks),
        ", ".join(f"{mode}/{representation}" for mode, representation in mode_tasks),
    )
    logger.info("Using %d outer worker(s) for fits/evals.", resolved_n_jobs)

    base_cache: dict[tuple[str, str, str], Any] = {}
    summaries: list[dict[str, Any]] = []
    task_failures: list[dict[str, str]] = []
    for mode, representation in mode_tasks:
        spec = mode_specs[mode]
        reducers_for_this_mode = spec["reducers"]
        sweep_for_this_mode = [int(value) for value in spec["n_components"]]
        base_load_mode = "sensor" if args.input_mode == "raw" else base_layout_mode(args.input_mode)
        try:
            summary = execute_analysis_mode(
                args=args,
                mode=mode,
                representation=representation,
                reducers=reducers_for_this_mode,
                n_components_sweep=sweep_for_this_mode,
                meta_df=meta_df,
                eval_specs=eval_specs,
                resolved_n_jobs=resolved_n_jobs,
                bids_root=bids_root,
                base_cache=base_cache,
                base_load_mode=base_load_mode,
                reports_root=reports_root,
            )
            summaries.append(summary)
        except Exception as exc:  # one mode failing must not abort the whole sweep
            logger.exception("Analysis mode '%s' (%s) failed.", mode, representation)
            task_failures.append(
                {"analysis_mode": mode, "representation": representation, "error": str(exc)}
            )

    if not summaries:
        raise RuntimeError("All dimensionality-reduction tasks failed; see the errors above.")

    if len(mode_tasks) > 1:
        rollup_report = generate_rollup_report(
            args=args,
            summaries=summaries,
            task_failures=task_failures,
        )
        summary_dir = summary_report_dir(reports_root, ReportStage.DIM_REDUCTION, create=True)
        summary_dir = summary_dir / slug(args.run_label or args.dataset_name)
        summary_dir.mkdir(parents=True, exist_ok=True)
        rollup_filename = (
            f"foundation_{slug(str(args.embedding_model_key))}_rollup_leaderboard.html"
            if args.input_mode == "foundation_embeddings"
            else "rollup_leaderboard.html"
        )
        rollup_path = summary_dir / rollup_filename
        rollup_report.save(rollup_path)
        logger.info("Roll-up leaderboard saved to: %s", rollup_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run checkpointed EEG dimensionality reduction.")
    parser.add_argument(
        "--cohort_config",
        help="Cohort/dataset config: subjects + clinical question (configs/cohorts/).",
    )
    parser.add_argument(
        "--analysis_config",
        help="Analysis/method config: analysis_modes (reducers + sweep per mode) "
        "(configs/analyses/dim_reduction/).",
    )
    parser.add_argument(
        "--alignment_transform",
        default=None,
        help="Reduce one materialized alignment variant (for example none, leace, or ra).",
    )
    parser.add_argument("--bids_root", default=None, help="Override BIDS root (else from config).")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override worker count.")
    parser.add_argument(
        "--representation",
        choices=["epoch", "recording", "subject"],
        default=None,
        help="Representation granularity to reduce (e.g., epoch, subject, recording).",
    )
    parser.add_argument(
        "--reports_root",
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root).",
    )
    parser.add_argument(
        "--derivative_root",
        default=None,
        help=(
            "Custom dimensionality-reduction output root. Normal runs write "
            "checkpointed outputs here; --compare_only reads results from here."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Compare one dataset with --compare_only; otherwise discover every dataset.",
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
        help="Regenerate reports from existing fit/eval inventories without refitting.",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Skip reduction and generate cross-model foundation comparison reports.",
    )
    parser.add_argument(
        "--descriptor_table_path",
        default=None,
        help="Override descriptor table path.",
    )
    parser.add_argument(
        "--descriptor_feature_columns_path",
        default=None,
        help="Override descriptor feature columns path.",
    )
    parser.add_argument(
        "--embedding_derivative_root",
        default=None,
        help="Override foundation embedding root.",
    )
    parser.add_argument(
        "--embedding_model_key",
        default=None,
        help="Override foundation embedding model.",
    )
    args = parser.parse_args()

    if args.compare_only:
        if not args.bids_root:
            parser.error("--compare_only requires --bids_root.")
        run(
            {
                "compare_only": True,
                "bids_root": args.bids_root,
                "derivative_root": args.derivative_root,
                "reports_root": args.reports_root,
                "dataset_name": args.dataset_name,
            }
        )
        return

    if not args.cohort_config or not args.analysis_config:
        parser.error("normal runs require --cohort_config and --analysis_config.")

    config = resolve_cli_config(
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        bids_root=args.bids_root,
        metadata=args.metadata,
        n_jobs=args.n_jobs,
        reports_root=args.reports_root,
        derivative_root=args.derivative_root,
        overwrite=args.overwrite,
        reports_only=args.reports_only,
        representation=args.representation,
        compare_only=args.compare_only,
        descriptor_table_path=args.descriptor_table_path,
        descriptor_feature_columns_path=args.descriptor_feature_columns_path,
        embedding_derivative_root=args.embedding_derivative_root,
        embedding_model_key=args.embedding_model_key,
    )
    if args.alignment_transform:
        transform = str(args.alignment_transform)
        base_model_key = str(config["embedding_model_key"])
        config["embedding_model_key"] = (
            base_model_key if transform == "none" else f"{base_model_key}_align-{transform}"
        )
    run(config)


if __name__ == "__main__":
    main()

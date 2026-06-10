#!/usr/bin/env python3
"""Checkpointed dimensionality-reduction analysis for EEG data."""

from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from coco_pipe.dim_reduction.artifacts import (
    EVAL_METRIC_COLUMNS,
    EVAL_RUN_KEY_FIELDS,
    FIT_METRIC_COLUMNS,
    FIT_RUN_KEY_FIELDS,
    SEPARATION_METRIC_KEY,
    _availability_record,
    _write_run_status,
    load_fit_runs,
    update_runs,
)
from coco_pipe.dim_reduction.config import METHODS, parse_eval_specs
from coco_pipe.dim_reduction.pipeline import (
    POOLED_CONDITION,
    _build_eval_task,
    _build_fit_task,
    _execute_eval_task,
    _execute_fit_task,
    _valid_component_sweep,
    build_auto_pooled_eval_spec,
)
from coco_pipe.io import iter_analysis_units
from coco_pipe.io.structures import DataContainer
from coco_pipe.io.utils import normalize_subject_value
from coco_pipe.utils import _resolve_n_jobs, _run_task_batch, _slug

from eeg_adhd_epilepsy.io.analysis import load_container
from eeg_adhd_epilepsy.io.bids import (
    get_reports_root,
    get_stage_summary_dir,
    validate_bids_coverage,
)
from eeg_adhd_epilepsy.io.table import load
from eeg_adhd_epilepsy.reports.dim_reduction import generate_dataset_report
from eeg_adhd_epilepsy.utils.config import DEFAULT_ANALYSIS_CONDITIONS

logger = logging.getLogger(__name__)

DEFAULT_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap"]
EXTENDED_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap", "Pacmap", "Trimap", "LLE", "TSNE"]
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)
DEFAULT_N_COMPONENTS_SWEEP = [2, 3, 5, 10, 20, 50, 75, 100]


def _run_variant(args) -> str:
    parts = [args.analysis_mode]
    aggregation_unit = getattr(args, "aggregation_unit", None)
    if (
        args.input_mode == "raw"
        and args.representation.startswith(("subject_", "recording_"))
        and aggregation_unit
    ):
        parts.append(str(aggregation_unit))
    parts.append(args.representation)
    return "__".join(_slug(part) for part in parts if part)


def _report_variant(args) -> str:
    parts = [f"mode-{_slug(args.analysis_mode).replace('_', '-')}"]
    aggregation_unit = getattr(args, "aggregation_unit", None)
    if (
        args.input_mode == "raw"
        and args.representation.startswith(("subject_", "recording_"))
        and aggregation_unit
    ):
        parts.append(f"unit-{_slug(aggregation_unit).replace('_', '-')}")
    parts.append(f"repr-{_slug(args.representation).replace('_', '-')}")
    return "_".join(parts)


def _resolve_subjects(args, bids_root: Path, meta_df: pd.DataFrame) -> list[str]:
    if args.subjects:
        return [normalize_subject_value(subject) for subject in args.subjects]
    if args.input_mode == "descriptors":
        table_path = Path(args.descriptor_table_path).expanduser()
        descriptor_df = load(str(table_path), sep=None)
        if args.subject_col in descriptor_df.columns:
            subject_series = descriptor_df[args.subject_col]
        elif "subject" in descriptor_df.columns:
            subject_series = descriptor_df["subject"]
        else:
            raise ValueError(
                f"Descriptor table must contain '{args.subject_col}' or 'subject' "
                "to resolve available subjects."
            )
        subjects = sorted(
            {normalize_subject_value(v) for v in subject_series.dropna().unique()}
        )
        logger.info(
            "Resolved %d available subjects from descriptor table %s.",
            len(subjects),
            table_path,
        )
        return subjects

    coverage_root = bids_root / "derivatives" / "preproc" if args.use_derivatives else bids_root
    coverage_desc = args.desc if args.use_derivatives else ""
    coverage_suffix = "epo" if args.use_derivatives else None
    coverage = validate_bids_coverage(
        meta_df,
        coverage_root,
        desc=coverage_desc,
        suffix=coverage_suffix,
        subject_col=args.subject_col,
    )
    subjects = [str(subject) for subject in coverage["present_subjects"]]
    logger.info(
        "Resolved %d available subjects from %s.",
        len(subjects),
        "derivatives" if args.use_derivatives else "BIDS",
    )
    return subjects


def _collect_scope_fit_tasks(
    scope: str,
    condition: str,
    container,
    args,
    reducers: list[str],
    output_root: Path,
    unit_containers_by_key: dict,
    data_availability: list,
) -> list[dict[str, Any]]:
    """Enumerate analysis units and build fit tasks for one scope/condition pair."""
    tasks: list[dict[str, Any]] = []
    for unit_spec in iter_analysis_units(
        container,
        args.analysis_mode,
        args.input_mode,
        args.descriptor_families,
    ):
        unit_containers_by_key[(scope, condition, unit_spec["unit_key"])] = unit_spec["container"]
        valid_components = _valid_component_sweep(unit_spec["container"], args.n_components_sweep)
        data_availability.append(
            _availability_record(
                scope=scope,
                condition=condition,
                unit_spec=unit_spec,
                container=unit_spec["container"],
                requested_components=args.n_components_sweep,
                valid_components=valid_components,
            )
        )
        if not valid_components:
            logger.warning(
                "Skipping %s/%s: no valid n_components for matrix shape %s.",
                condition,
                unit_spec["unit_name"],
                tuple(np.asarray(unit_spec["container"].X).shape),
            )
            continue
        for reducer_name, n_components in product(reducers, valid_components):
            tasks.append(
                _build_fit_task(
                    args=args,
                    scope=scope,
                    condition=condition,
                    unit_spec=unit_spec,
                    reducer_name=reducer_name,
                    n_components=n_components,
                    output_root=output_root,
                )
            )
    return tasks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    bootstrap_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Run checkpointed EEG dimensionality reduction."
    )
    parser.add_argument("--config", default=None, help="Path to dim-reduction YAML config.")

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--bids_root", required=False, default=None)
    dataset_group.add_argument("--metadata", default=None, help="Path to canonical metadata CSV.")
    dataset_group.add_argument(
        "--dataset_name", default=None, help="Name for this dim-reduction run namespace."
    )
    dataset_group.add_argument(
        "--run_label",
        default=None,
        help="Optional human-readable label used in reports and output folders.",
    )
    dataset_group.add_argument(
        "--subject_col", default="study_id", help="Subject identifier column."
    )
    dataset_group.add_argument(
        "--subjects", nargs="+", default=None, help="Specific subjects to process."
    )
    dataset_group.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
    )

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--input_mode", choices=["raw", "descriptors"], default="raw")
    input_group.add_argument("--task", default="clinical")
    input_group.add_argument("--segment_duration", type=float, default=60.0)
    input_group.add_argument("--overlap", type=float, default=0.0)
    input_group.add_argument("--use_derivatives", action="store_true")
    input_group.add_argument("--desc", default="base")
    input_group.add_argument(
        "--representation",
        choices=[
            "epoch_native",
            "epoch_flat",
            "epoch_time_as_sample",
            "epoch_scalar_mean",
            "subject_native",
            "subject_flat",
            "subject_time_as_sample",
            "subject_scalar_mean",
            "recording_native",
            "recording_flat",
            "recording_time_as_sample",
            "recording_scalar_mean",
        ],
        default="epoch_flat",
    )
    input_group.add_argument(
        "--analysis_mode",
        choices=["flat", "sensor", "family", "sensor_within_family"],
        default="flat",
    )
    input_group.add_argument("--descriptor_table_path", default=None)
    input_group.add_argument("--descriptor_feature_columns_path", default=None)
    input_group.add_argument("--descriptor_families", nargs="+", default=None)
    input_group.add_argument(
        "--descriptor_max_abs_value",
        type=float,
        default=1e12,
        help=(
            "For descriptor inputs, drop rows whose selected finite descriptor "
            "features exceed this absolute value threshold."
        ),
    )
    input_group.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    input_group.add_argument(
        "--aggregation_unit",
        choices=["recording", "subject"],
        default="recording",
        help=(
            "Aggregation unit for raw subject_* representations. "
            "'recording' preserves separate session/run rows; "
            "'subject' collapses all runs for a study_id."
        ),
    )

    reduction_group = parser.add_argument_group("Reduction")
    reduction_group.add_argument("--reducers", nargs="+", default=["default"])
    reduction_group.add_argument(
        "--n_components_sweep", nargs="+", type=int, default=DEFAULT_N_COMPONENTS_SWEEP
    )
    reduction_group.add_argument("--run_pooled", action="store_true")
    reduction_group.add_argument("--overwrite", action="store_true")
    reduction_group.add_argument("--reports-only", action="store_true")
    reduction_group.add_argument("--eval_config", default=None)
    reduction_group.add_argument("--n_jobs", type=int, default=1)
    reduction_group.add_argument(
        "--output_group",
        default=None,
        help=(
            "Optional nested output group under derivatives/reports, "
            "e.g. 'medicated_adhd_vs_controls/lis'."
        ),
    )

    filter_group = parser.add_argument_group("Filtering")
    filter_group.add_argument("--filter_col", action="append", default=[])
    filter_group.add_argument("--filter_val", action="append", nargs="+", default=[])
    filter_group.add_argument("--balance_target", default=None)
    filter_group.add_argument(
        "--balance_strategy",
        choices=["undersample", "oversample", "auto"],
        default="undersample",
    )

    report_group = parser.add_argument_group("Report")
    report_group.add_argument(
        "--interactive", action=argparse.BooleanOptionalAction, default=True
    )
    report_group.add_argument("--save_static_figures", action="store_true")
    report_group.add_argument("--compress_viz_with_pca", action="store_true")
    report_group.add_argument(
        "--selection_metric",
        default=SEPARATION_METRIC_KEY,
        help="Metric used to select the best run in report summaries.",
    )
    report_group.add_argument(
        "--selection_eval_name",
        default=None,
        help=(
            "Eval name whose separation metric should drive report selection, "
            "e.g. 'med_adhd_vs_ctrl'."
        ),
    )

    config_eval_specs = None
    if bootstrap_args.config:
        config_path = Path(bootstrap_args.config).expanduser()
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_config, dict):
            raise ValueError(f"Expected mapping payload in {config_path}.")
        config_eval_specs = raw_config.pop("evals", None)
        parser.set_defaults(**raw_config)
    args = parser.parse_args()

    # --- Validation ---
    if args.bids_root is None:
        raise ValueError("--bids_root is required.")
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
        # For descriptor inputs, use the table filename stem as the representation label
        # (the --representation CLI flag is ignored; this sets it once before anything reads it)
        args.representation = Path(args.descriptor_table_path).stem
    if args.run_label is None:
        args.run_label = args.dataset_name
    if args.analysis_mode in {"family", "sensor_within_family"} and args.input_mode != "descriptors":
        raise ValueError(
            f"analysis_mode='{args.analysis_mode}' is only supported for descriptor inputs."
        )
    if args.input_mode == "raw" and args.analysis_mode == "sensor":
        if args.representation not in {"epoch_native", "subject_native", "recording_native"}:
            raise ValueError(
                "Raw sensor mode requires representation 'epoch_native', "
                "'subject_native', or 'recording_native'."
            )
    if args.input_mode == "raw" and args.analysis_mode != "flat":
        if args.analysis_mode != "sensor":
            raise ValueError(
                "Raw inputs currently support only analysis_mode='flat' or 'sensor'."
            )
    if args.input_mode == "raw" and args.analysis_mode == "flat":
        if args.representation in {"epoch_native", "subject_native", "recording_native"}:
            raise ValueError(
                "Native EEG representations are reserved for sensor mode. "
                "Use --analysis_mode sensor with epoch_native, subject_native, "
                "or recording_native."
            )
    if args.input_mode == "raw" and args.representation.startswith("recording_"):
        if args.aggregation_unit != "recording":
            raise ValueError(
                "recording_* representations require --aggregation_unit recording."
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

    resolved_n_jobs = _resolve_n_jobs(args.n_jobs)

    requested_reducers = [value.upper() for value in args.reducers]
    if requested_reducers == ["DEFAULT"]:
        reducers = DEFAULT_REDUCERS
    elif requested_reducers == ["EXTENDED"]:
        reducers = EXTENDED_REDUCERS
    else:
        # Validate against METHODS — the authoritative registry in coco-pipe
        valid_reducers = set(METHODS)
        invalid_reducers = [v for v in requested_reducers if v not in valid_reducers]
        if invalid_reducers:
            raise ValueError(
                f"Unknown reducers: {invalid_reducers}. "
                f"Valid reducers: {sorted(valid_reducers)}"
            )
        reducers = requested_reducers

    bids_root = Path(args.bids_root).expanduser()
    meta_df = load(str(Path(args.metadata)), sep=",")
    subjects = _resolve_subjects(args, bids_root, meta_df)
    raw_eval_specs = config_eval_specs
    if raw_eval_specs is None and args.eval_config:
        raw_eval_specs = (
            yaml.safe_load(Path(args.eval_config).expanduser().read_text(encoding="utf-8")) or []
        )
    eval_specs = parse_eval_specs(raw_eval_specs, args.subject_col)
    auto_pooled_eval_spec = build_auto_pooled_eval_spec(args.conditions, args.run_pooled)
    if auto_pooled_eval_spec is not None and not any(
        spec["name"] == auto_pooled_eval_spec["name"] for spec in eval_specs
    ):
        eval_specs = [*eval_specs, auto_pooled_eval_spec]

    valid_selection_metrics = {*FIT_METRIC_COLUMNS, SEPARATION_METRIC_KEY}
    eval_names = {spec["name"] for spec in eval_specs}
    if args.selection_metric in eval_names and not args.selection_eval_name:
        args.selection_eval_name = args.selection_metric
        args.selection_metric = SEPARATION_METRIC_KEY
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

    output_base = bids_root / "derivatives" / "dim_reduction"
    if args.output_group:
        output_group = Path(str(args.output_group))
        if output_group.is_absolute():
            raise ValueError("--output_group must be relative, not absolute.")
        output_base = output_base / output_group
    run_variant = _run_variant(args)
    output_dataset_name = _slug(args.run_label or args.dataset_name)
    output_root = output_base / output_dataset_name / args.input_mode / run_variant
    output_root.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "eval_config"}
    }
    if eval_specs:
        config_snapshot["evals"] = eval_specs
    (output_root / "config_used.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=True), encoding="utf-8"
    )
    fit_runs_path = output_root / "dim_reduction_fit_runs.json"
    eval_runs_path = output_root / "dim_reduction_eval_runs.json"
    logger.info("Using %d outer worker(s) for fits/evals.", resolved_n_jobs)

    base_containers_by_scope: dict[tuple[str, str], DataContainer] = {}
    unit_containers_by_key: dict[tuple[str, str, str], DataContainer] = {}
    data_availability: list[dict[str, Any]] = []
    condition_load_failures: list[dict[str, str]] = []

    if args.reports_only:
        if not fit_runs_path.exists():
            raise FileNotFoundError(
                f"--reports-only requested but {fit_runs_path} does not exist."
            )
    else:
        fit_tasks: list[dict[str, Any]] = []
        for condition in args.conditions:
            logger.info(
                "Loading input for condition '%s' (%s).", condition, args.input_mode
            )
            try:
                base_container = load_container(
                    args, subjects, meta_df, condition, target_col=None
                )
            except Exception as exc:
                logger.exception("Failed to load condition '%s'.", condition)
                condition_load_failures.append(
                    {"condition": condition, "error": str(exc)}
                )
                continue

            base_containers_by_scope[("condition", condition)] = base_container
            fit_tasks.extend(
                _collect_scope_fit_tasks(
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
            available_conditions = [
                cond
                for cond in args.conditions
                if ("condition", cond) in base_containers_by_scope
            ]
            if available_conditions:
                pooled_container = DataContainer.concat(
                    [
                        base_containers_by_scope[("condition", cond)]
                        for cond in available_conditions
                    ]
                )
                base_containers_by_scope[("pooled", POOLED_CONDITION)] = pooled_container
                fit_tasks.extend(
                    _collect_scope_fit_tasks(
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
            else:
                logger.warning(
                    "Skipping pooled mode: no condition containers were available."
                )

        logger.info(
            "Queued %d fit task(s) across %d loaded scope(s).",
            len(fit_tasks),
            len(base_containers_by_scope),
        )
        for record in _run_task_batch(fit_tasks, _execute_fit_task, resolved_n_jobs):
            update_runs(fit_runs_path, record, key_fields=FIT_RUN_KEY_FIELDS)

        if eval_specs:
            if not fit_runs_path.exists():
                raise RuntimeError(
                    "No fit runs were produced, so post-hoc evaluations cannot run. "
                    "Check the condition load errors above."
                )
            fit_runs = [
                record
                for record in load_fit_runs(fit_runs_path)
                if record.get("status") == "success"
                and record.get("input_mode") == args.input_mode
                and record.get("analysis_mode") == args.analysis_mode
                and record.get("reducer") in reducers
                and int(record.get("n_components", 0)) in args.n_components_sweep
            ]
            if not fit_runs:
                raise RuntimeError(
                    "No successful fit runs were produced, so post-hoc evaluations "
                    "cannot run. Check the fit errors above."
                )
            eval_tasks: list[dict[str, Any]] = []
            for fit_record in fit_runs:
                unit_container = unit_containers_by_key.get(
                    (
                        fit_record["scope"],
                        fit_record["condition"],
                        fit_record["unit_key"],
                    )
                )
                if unit_container is None:
                    logger.warning(
                        "Skipping evals for missing unit scope %s/%s/%s.",
                        fit_record["scope"],
                        fit_record["condition"],
                        fit_record["unit_key"],
                    )
                    continue
                for eval_spec in eval_specs:
                    # The auto-pooled eval spec is only meaningful for pooled scopes
                    if (
                        auto_pooled_eval_spec is not None
                        and eval_spec["name"] == auto_pooled_eval_spec["name"]
                        and fit_record["scope"] != "pooled"
                    ):
                        continue
                    eval_tasks.append(
                        _build_eval_task(
                            fit_record=fit_record,
                            eval_spec=eval_spec,
                            container=unit_container,
                            output_root=output_root,
                            overwrite=args.overwrite,
                        )
                    )
            logger.info("Queued %d eval task(s).", len(eval_tasks))
            for record in _run_task_batch(eval_tasks, _execute_eval_task, resolved_n_jobs):
                update_runs(eval_runs_path, record, key_fields=EVAL_RUN_KEY_FIELDS)

    report_path: Path | None = None
    fatal_error: str | None = None
    try:
        if not fit_runs_path.exists():
            raise RuntimeError(
                f"No fit runs found in {fit_runs_path}. "
                "Dim reduction produced no successful fit inventory."
            )

        report = generate_dataset_report(
            args=args,
            output_root=output_root,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            reducers=reducers,
            subjects=subjects,
            meta_df=meta_df,
            containers_by_scope=base_containers_by_scope or None,
            metric_columns=FIT_METRIC_COLUMNS,
            eval_specs=eval_specs,
            pooled_condition=POOLED_CONDITION,
        )
        reports_root = get_reports_root(bids_root)
        summary_dir = get_stage_summary_dir(reports_root, "dim_reduction", create_dir=True)
        if args.output_group:
            summary_dir = summary_dir / Path(str(args.output_group))
        summary_dir = summary_dir / output_dataset_name / args.input_mode
        summary_dir.mkdir(parents=True, exist_ok=True)
        report_path = summary_dir / f"dataset_summary_{_report_variant(args)}.html"
        report.save(report_path)
        logger.info("Report saved to: %s", report_path)
    except Exception as exc:
        fatal_error = str(exc)
        raise
    finally:
        _write_run_status(
            output_root=output_root,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            fatal_error=fatal_error,
            report_path=report_path,
            run_metadata={
                "dataset_name": args.dataset_name,
                "run_label": args.run_label,
                "input_mode": args.input_mode,
                "analysis_mode": args.analysis_mode,
                "representation": args.representation,
                "aggregation_unit": args.aggregation_unit,
                "run_variant": run_variant,
                "output_dataset_name": output_dataset_name,
                "conditions_requested": list(args.conditions),
                "subjects_resolved": len(subjects),
                "condition_load_failures": condition_load_failures,
                "data_availability": data_availability,
            },
        )


if __name__ == "__main__":
    main()

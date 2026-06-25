#!/usr/bin/env python3
"""Checkpointed dimensionality-reduction analysis for EEG data."""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from coco_pipe.dim_reduction import (
    EVAL_RUN_KEY_FIELDS,
    FIT_METRIC_COLUMNS,
    FIT_RUN_KEY_FIELDS,
    POOLED_CONDITION,
    SEPARATION_METRIC_KEY,
    build_auto_pooled_eval_spec,
    build_availability_record,
    build_eval_request,
    build_fit_request,
    load_fit_runs,
    parse_eval_specs,
    run_eval,
    run_fit,
    update_runs,
    valid_component_sweep,
    write_run_status,
)
from coco_pipe.io import (
    ANALYSIS_MODES,
    DESCRIPTOR_ONLY_ANALYSIS_MODES,
    DataContainer,
    iter_analysis_units,
    normalize_subject_value,
    read_table,
    write_json,
)
from coco_pipe.utils import resolve_n_jobs, run_task_batch, slug, stable_hash

from eeg_adhd_epilepsy.analysis.utils.dim_reduction import (
    build_run_config_payload,
    condition_load_failure_record,
    pool_containers,
)
from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.units import (
    apply_family_qc_mask,
    families_for_analysis_unit,
)
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    summary_report_dir,
)
from eeg_adhd_epilepsy.reports.dim_reduction import generate_dataset_report
from eeg_adhd_epilepsy.utils.config import load_cohort_analysis_config
from eeg_adhd_epilepsy.utils.constants import DEFAULT_ANALYSIS_CONDITIONS

logger = logging.getLogger(__name__)

DEFAULT_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap"]
EXTENDED_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap", "Pacmap", "Trimap", "LLE", "TSNE"]
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)
DEFAULT_N_COMPONENTS_SWEEP = [2, 3, 5, 10, 20, 50, 75, 100]


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
        filter_specs = [
            {"column": str(col), "values": [str(value) for value in vals]}
            for col, vals in zip(args.filter_col, args.filter_val)
            if vals
        ]
        input_signature: dict[str, Any] = {
            "input_mode": args.input_mode,
            "representation": args.representation,
            "analysis_mode": args.analysis_mode,
            "run_config_hash": getattr(args, "run_config_hash", None),
            "descriptor_families": list(getattr(args, "descriptor_families", []) or []),
            "filters": filter_specs,
            "group_filters": getattr(args, "group_filters", None),
            "balance_target": args.balance_target,
            "balance_strategy": args.balance_strategy if args.balance_target else None,
            "unit_type": unit_spec["unit_type"],
            "unit_name": unit_spec["unit_name"],
            "family": unit_spec.get("family"),
            "run_label": getattr(args, "run_label", None),
            "qc": getattr(args, "qc", None),
        }
        if args.input_mode == "raw":
            input_signature.update(
                {
                    "bids_root": str(Path(args.bids_root).expanduser()),
                    "use_derivatives": bool(args.use_derivatives),
                    "task": getattr(args, "task", "clinical"),
                    "segment_duration": float(args.segment_duration),
                    "overlap": float(args.overlap),
                    "desc": args.desc,
                    "window_source": getattr(args, "window_source", "auto"),
                    "aggregation_unit": getattr(args, "aggregation_unit", None),
                }
            )
        elif args.input_mode == "descriptors":
            input_signature.update(
                {
                    "descriptor_table_path": str(Path(args.descriptor_table_path).expanduser()),
                    "descriptor_feature_columns_path": str(
                        Path(args.descriptor_feature_columns_path).expanduser()
                    ),
                    "descriptor_max_abs_value": getattr(args, "descriptor_max_abs_value", None),
                    "location_statistic": getattr(args, "location_statistic", None),
                }
            )
        elif args.input_mode == "foundation_embeddings":
            input_signature.update(
                {
                    "embedding_derivative_root": str(
                        Path(args.embedding_derivative_root).expanduser()
                    ),
                    "embedding_representation": getattr(
                        args, "embedding_representation", "recording"
                    ),
                    "embedding_aggregate_by": getattr(args, "embedding_aggregate_by", None),
                    "embedding_model_key": getattr(args, "embedding_model_key", None),
                }
            )
        else:
            raise ValueError(f"Unsupported input_mode '{args.input_mode}'.")

        for reducer_name, n_components in product(reducers, valid_components):
            logger.info(
                "Fitting %s/%s/%s/%s/n%d",
                condition,
                args.analysis_mode,
                unit_spec["unit_name"],
                reducer_name,
                int(n_components),
            )
            requests.append(
                build_fit_request(
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
                )
            )
    return requests


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cohort_config", default=None)
    pre_parser.add_argument("--analysis_config", default=None)
    bootstrap_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Run checkpointed EEG dimensionality reduction.")
    parser.add_argument(
        "--cohort_config",
        required=True,
        help="Cohort/dataset config: subjects + clinical question (configs/cohorts/).",
    )
    parser.add_argument(
        "--analysis_config",
        required=True,
        help="Analysis/method config: reducers, sweep (configs/analyses/dim_reduction/).",
    )

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
    input_group.add_argument(
        "--input_mode",
        choices=["raw", "descriptors", "foundation_embeddings"],
        default="raw",
    )
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
            "subject_native",
            "subject_flat",
            "subject_time_as_sample",
            "recording_native",
            "recording_flat",
            "recording_time_as_sample",
        ],
        default="epoch_flat",
    )
    input_group.add_argument(
        "--analysis_mode",
        choices=ANALYSIS_MODES,
        default="flat",
    )
    input_group.add_argument("--descriptor_table_path", default=None)
    input_group.add_argument("--descriptor_feature_columns_path", default=None)
    input_group.add_argument("--descriptor_families", nargs="+", default=None)
    input_group.add_argument("--embedding_derivative_root", default=None)
    input_group.add_argument(
        "--embedding_representation",
        choices=["recording", "window"],
        default="recording",
    )
    input_group.add_argument("--embedding_aggregate_by", default=None)
    input_group.add_argument("--embedding_model_key", default=None)
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
    reduction_group.add_argument("--reports_root", type=str, default=None, help="Custom root directory for reports (defaults to sibling of bids_root)")

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
    report_group.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
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
    if bootstrap_args.cohort_config and bootstrap_args.analysis_config:
        raw_config = load_cohort_analysis_config(
            bootstrap_args.cohort_config, bootstrap_args.analysis_config
        )
    else:
        raw_config = None
    if raw_config is not None:
        raw_config = dict(raw_config)
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
    if args.input_mode == "foundation_embeddings":
        if not args.embedding_derivative_root:
            raise ValueError("--embedding_derivative_root is required for foundation embeddings.")
        if not args.embedding_model_key:
            raise ValueError("--embedding_model_key is required to keep model spaces separate.")
        if args.analysis_mode != "flat":
            raise ValueError("Foundation embeddings currently support analysis_mode='flat' only.")
        args.representation = f"foundation_{args.embedding_representation}"
    if args.run_label is None:
        args.run_label = args.dataset_name
    if args.analysis_mode in DESCRIPTOR_ONLY_ANALYSIS_MODES and args.input_mode != "descriptors":
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
            raise ValueError("Raw inputs currently support only analysis_mode='flat' or 'sensor'.")
    if args.input_mode == "raw" and args.analysis_mode == "flat":
        if args.representation in {"epoch_native", "subject_native", "recording_native"}:
            raise ValueError(
                "Native EEG representations are reserved for sensor mode. "
                "Use --analysis_mode sensor with epoch_native, subject_native, "
                "or recording_native."
            )
    if args.input_mode == "raw" and args.representation.startswith("recording_"):
        if args.aggregation_unit != "recording":
            raise ValueError("recording_* representations require --aggregation_unit recording.")
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

    resolved_n_jobs = resolve_n_jobs(args.n_jobs)

    requested_reducers = [str(value) for value in args.reducers]
    if len(requested_reducers) == 1 and requested_reducers[0].lower() == "default":
        reducers = list(DEFAULT_REDUCERS)
    elif len(requested_reducers) == 1 and requested_reducers[0].lower() == "extended":
        reducers = list(EXTENDED_REDUCERS)
    else:
        reducers = requested_reducers

    bids_root = Path(args.bids_root).expanduser()
    meta_df = read_table(Path(args.metadata), sep=",")
    if args.subjects:
        subjects = [normalize_subject_value(subject) for subject in args.subjects]
    else:
        if args.subject_col not in meta_df.columns:
            raise ValueError(f"Metadata table must contain subject column '{args.subject_col}'.")
        subjects = sorted(
            {
                normalize_subject_value(subject)
                for subject in meta_df[args.subject_col].dropna().unique()
            }
        )
        logger.info(
            "Resolved %d subjects from metadata column '%s'.",
            len(subjects),
            args.subject_col,
        )
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

    run_config = build_run_config_payload(args, reducers, eval_specs)
    args.run_config_hash = stable_hash(run_config, length=12)

    output_base = bids_root / "derivatives" / "dim_reduction"
    if args.output_group:
        output_group = Path(str(args.output_group))
        if output_group.is_absolute():
            raise ValueError("--output_group must be relative, not absolute.")
        output_base = output_base / output_group
    aggregation_unit = getattr(args, "aggregation_unit", None)
    representation_label = (
        args.representation.removesuffix(f"_{args.analysis_mode}")
        if args.input_mode == "raw"
        else args.representation
    )
    variant_parts = [args.analysis_mode, args.input_mode, representation_label]
    if args.input_mode == "foundation_embeddings":
        variant_parts.append(args.embedding_model_key)
    if (
        args.input_mode == "raw"
        and representation_label == "subject"
        and aggregation_unit
        and aggregation_unit != "subject"
    ):
        variant_parts.append(f"by_{aggregation_unit}")
    variant_parts.append(f"cfg-{args.run_config_hash}")
    run_variant = "_".join(slug(part) for part in variant_parts if part)
    args.run_variant = run_variant
    output_dataset_name = slug(args.run_label or args.dataset_name)
    output_root = output_base / output_dataset_name / run_variant
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        key: value for key, value in vars(args).items() if key not in {"config", "eval_config"}
    }
    if eval_specs:
        config_snapshot["evals"] = eval_specs
    (output_root / "config_used.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=True), encoding="utf-8"
    )
    fit_runs_path = runs_dir / "fit_runs.json"
    eval_runs_path = runs_dir / "eval_runs.json"
    run_summary_path = runs_dir / "run_summary.json"
    if not args.reports_only:
        write_json(fit_runs_path, [], indent=2)
        write_json(eval_runs_path, [], indent=2)
    logger.info("Using %d outer worker(s) for fits/evals.", resolved_n_jobs)

    base_containers_by_scope: dict[tuple[str, str], DataContainer] = {}
    unit_containers_by_key: dict[tuple[str, str, str], DataContainer] = {}
    data_availability: list[dict[str, Any]] = []
    condition_load_failures: list[dict[str, str]] = []
    report_path: Path | None = None
    fatal_error: str | None = None
    try:
        if args.reports_only:
            if not fit_runs_path.exists():
                raise FileNotFoundError(
                    f"--reports-only requested but {fit_runs_path} does not exist."
                )
        else:
            fit_requests: list[dict[str, Any]] = []
            for condition in args.conditions:
                logger.info("Loading input for condition '%s' (%s).", condition, args.input_mode)
                try:
                    base_container = build_dataset(
                        args, subjects, meta_df, condition, target_col=None
                    )
                except Exception as exc:
                    logger.exception("Failed to load condition '%s'.", condition)
                    condition_load_failures.append({"condition": condition, "error": str(exc)})
                    update_runs(
                        fit_runs_path,
                        condition_load_failure_record(
                            condition=condition,
                            args=args,
                            error=exc,
                        ),
                        key_fields=FIT_RUN_KEY_FIELDS,
                    )
                    continue

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
                available_conditions = [
                    cond
                    for cond in args.conditions
                    if ("condition", cond) in base_containers_by_scope
                ]
                if available_conditions:
                    source_containers = [
                        base_containers_by_scope[("condition", cond)]
                        for cond in available_conditions
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
                else:
                    logger.warning("Skipping pooled mode: no condition containers were available.")

            logger.info(
                "Queued %d fit request(s) across %d loaded scope(s).",
                len(fit_requests),
                len(base_containers_by_scope),
            )
            for record in run_task_batch(
                fit_requests,
                lambda request: run_fit(**request, errors="record"),
                resolved_n_jobs,
            ):
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
                eval_requests: list[dict[str, Any]] = []
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
                        # The auto-pooled eval spec is only meaningful for pooled scopes.
                        if (
                            auto_pooled_eval_spec is not None
                            and eval_spec["name"] == auto_pooled_eval_spec["name"]
                            and fit_record["scope"] != "pooled"
                        ):
                            continue
                        logger.info(
                            "Evaluating %s/%s/%s/%s/n%d [%s]",
                            fit_record["condition"],
                            fit_record["analysis_mode"],
                            fit_record["unit_name"],
                            fit_record["reducer"],
                            fit_record["n_components"],
                            eval_spec["name"],
                        )
                        eval_requests.append(
                            build_eval_request(
                                fit_record=fit_record,
                                eval_spec=eval_spec,
                                container=unit_container,
                                output_root=output_root,
                                overwrite=bool(args.overwrite),
                            )
                        )
                logger.info("Queued %d eval request(s).", len(eval_requests))
                for record in run_task_batch(
                    eval_requests,
                    lambda request: run_eval(**request, errors="record"),
                    resolved_n_jobs,
                ):
                    update_runs(eval_runs_path, record, key_fields=EVAL_RUN_KEY_FIELDS)

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
        reports_root = Path(args.reports_root) if args.reports_root else default_reports_root(bids_root)
        summary_dir = summary_report_dir(reports_root, ReportStage.DIM_REDUCTION, create=True)
        if args.output_group:
            summary_dir = summary_dir / Path(str(args.output_group))
        summary_dir = summary_dir / output_dataset_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        report_path = summary_dir / f"{run_variant}_dataset_summary.html"
        report.save(report_path)
        logger.info("Report saved to: %s", report_path)
    except Exception as exc:
        fatal_error = str(exc)
        raise
    finally:
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
                "analysis_mode": args.analysis_mode,
                "representation": args.representation,
                "aggregation_unit": args.aggregation_unit,
                "run_variant": run_variant,
                "run_config_hash": args.run_config_hash,
                "output_dataset_name": output_dataset_name,
                "conditions_requested": list(args.conditions),
                "subjects_resolved": len(subjects),
                "condition_load_failures": condition_load_failures,
                "data_availability": data_availability,
            },
        )


if __name__ == "__main__":
    main()

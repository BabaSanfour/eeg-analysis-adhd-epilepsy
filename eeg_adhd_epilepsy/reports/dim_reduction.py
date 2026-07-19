"""Dimensionality-reduction report assembly (study wiring).

The generic section builders live in :mod:`coco_pipe.report.dim_reduction_sweep`.
This module supplies the EEG-specific policy — dataset loading, metadata-column
exclusions, the topomap channel vocabulary, the cohort summary and the roll-up
links — and wires them into the shared builders via a
:class:`~coco_pipe.report.DimReductionReportContext`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY, load_fit_runs
from coco_pipe.report import (
    DimReductionReportContext,
    Report,
    build_dataset_report,
    build_reduction_rollup_report,
    build_subject_alignment_diagnostics_section,
    rank_reduction_runs,
)

from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.dim_reduction import (
    DEFAULT_DIM_REDUCTION_SELECTION_METRIC,
    DIM_REDUCTION_EVAL_METRIC_COLUMNS,
)
from eeg_adhd_epilepsy.metadata.schema import EPILEPSY_MED_COLS
from eeg_adhd_epilepsy.reports._common import (
    MODE_TITLES,
    AlignmentDiagnosticsSpec,
    add_overview_cohort_summary,
    family_label,
    load_alignment_diagnostics,
)
from eeg_adhd_epilepsy.utils.constants import BASIC_1020_CHANNELS

logger = logging.getLogger(__name__)

_UNIT_LABELS = {
    "flat": "global",
    "sensor": "sensor",
    "family": "family",
    "subfamily": "subfamily",
    "sensor_within_family": "sensor",
    "sensor_within_subfamily": "sensor",
    "feature": "feature",
    "feature_within_family": "feature",
    "descriptor": "descriptor",
    "descriptor_sensor": "descriptor × sensor",
}

# RF separation is the default primary metric; LR is the first tie-breaker.
_SEPARATION_TIE_BREAKERS = (
    (SEPARATION_METRIC_KEY, False),
    ("trustworthiness", False),
    ("continuity", False),
)

_PLOT_META_EXCLUDED_COLUMNS = {
    "obs",
    "channel",
    "time",
    "subject",
    "study_id",
    "patient_id",
    "patient_group_id",
    "obs_id",
    "run",
    "recording_id",
    "eeg_date",
    *EPILEPSY_MED_COLS,
}
_PLOT_META_EXCLUDED_NORMALIZED = {
    "obs",
    "subject",
    "channel",
    "time",
    "studyid",
    "patientid",
    "patientgroupid",
    "obsid",
    "run",
    "recordingid",
    "eegdate",
    "age",
    "epochcount",
    "firsteeg",
    "psychostimulant",
}
_FIT_FAILURE_COLUMNS = [
    "scope",
    "condition",
    "analysis_mode",
    "family",
    "unit_name",
    "reducer",
    "n_components",
    "status",
    "error",
    "timestamp",
]
_EVAL_FAILURE_COLUMNS = [
    "scope",
    "condition",
    "analysis_mode",
    "family",
    "unit_name",
    "eval_name",
    "reducer",
    "n_components",
    "status",
    "error",
    "timestamp",
]


def _eye_state_extractor(meta: dict[str, np.ndarray]) -> np.ndarray | None:
    """Derive an EO/EC eye-state series from the ``condition`` metadata column."""
    if "condition" not in meta:
        return None

    def _get_eye_state(val: Any) -> str:
        v = str(val).lower()
        if v.startswith("eo"):
            return "EO"
        if v.startswith("ec"):
            return "EC"
        return str(val)

    eye_state = np.array([_get_eye_state(v) for v in meta["condition"]], dtype=object)
    return eye_state if len(np.unique(eye_state)) > 1 else None


def generate_dataset_report(
    args,
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    reducers: Sequence[str],
    meta_df: pd.DataFrame,
    containers_by_scope: dict[tuple[str, str], Any] | None,
    dataset_stats: list[dict[str, Any]] | None,
    eval_specs: Sequence[dict[str, Any]],
    pooled_condition: str,
) -> Report:
    """Build the per-dataset dimensionality-reduction report for one mode."""
    fam_label = family_label(args)
    run_label = args.run_label or args.dataset_name
    report_title = (
        f"Dimensionality Reduction: {run_label} "
        f"[{MODE_TITLES.get(args.analysis_mode, args.analysis_mode)} / "
        f"{args.representation} / {args.input_mode}]"
    )
    if fam_label:
        report_title += f" [{fam_label}]"

    excluded_columns = set(_PLOT_META_EXCLUDED_COLUMNS)
    excluded_normalized = set(_PLOT_META_EXCLUDED_NORMALIZED)
    if getattr(args, "color_by", None) == "subject":
        excluded_columns.difference_update({"subject", args.subject_col})
        excluded_normalized.difference_update(
            {"subject", "".join(ch for ch in args.subject_col.lower() if ch.isalnum())}
        )

    ctx = DimReductionReportContext(
        analysis_mode=args.analysis_mode,
        selection_metric=args.selection_metric,
        reducers=list(reducers),
        conditions=list(args.conditions),
        container_builder=lambda condition: build_dataset(
            args, meta_df, condition, target_col=None
        ),
        output_root=output_root,
        eval_specs=eval_specs,
        interactive=args.interactive,
        input_mode=args.input_mode,
        representation=args.representation,
        family_label=fam_label,
        run_pooled=args.run_pooled,
        pooled_condition=pooled_condition,
        dataset_name=args.dataset_name,
        report_title=report_title,
        excluded_columns=frozenset(excluded_columns),
        excluded_normalized=frozenset(excluded_normalized),
        excluded_normalized_substrings=("psychostimulant",),
        excluded_suffixes=("_bool", "_clean"),
        meta_extractors={"eye_state": _eye_state_extractor},
        topomap_channels=frozenset(BASIC_1020_CHANNELS),
        unit_labels=_UNIT_LABELS,
        fit_failure_columns=_FIT_FAILURE_COLUMNS,
        eval_failure_columns=_EVAL_FAILURE_COLUMNS,
    )

    return build_dataset_report(
        ctx,
        fit_runs_path=fit_runs_path,
        eval_runs_path=eval_runs_path,
        containers_by_scope=containers_by_scope,
        dataset_stats=dataset_stats,
        overview_extras=lambda overview_sec: add_overview_cohort_summary(
            overview_sec,
            args,
            eval_specs,
            containers_by_scope,
            pooled_condition,
        ),
    )


def collect_mode_leaderboard(
    *,
    args: Any,
    fit_runs_path: Path,
    eval_runs_path: Path,
    reducers: list[str] | None = None,
    pooled_condition: str | None = None,
) -> pd.DataFrame:
    """Best run per (scope, condition) for one analysis mode, for the roll-up.

    Returns one row per scope/condition tagged with ``analysis_mode`` (and the
    foundation ``model`` / embedding ``representation`` when relevant), ranked by
    the run's selection metric. Empty when the mode produced no successful fits.

    ``reducers`` restricts the ranking to the reducers the run actually requested
    (a stale inventory may carry others). ``pooled_condition`` standardizes the
    condition label shown for pooled-scope rows so the cross-model roll-up groups
    them consistently.
    """
    fit_runs_df = pd.DataFrame(load_fit_runs(fit_runs_path))
    eval_runs_df = (
        pd.DataFrame(json.loads(eval_runs_path.read_text(encoding="utf-8")))
        if eval_runs_path.exists()
        else pd.DataFrame()
    )
    best = rank_reduction_runs(
        fit_runs_df,
        eval_runs_df,
        selection_metric=args.selection_metric,
        group_by=("scope", "condition"),
        eval_name=getattr(args, "selection_eval_name", None),
        tie_breakers=_SEPARATION_TIE_BREAKERS,
        reducers=reducers,
        metrics=DIM_REDUCTION_EVAL_METRIC_COLUMNS,
    )
    if best.empty:
        return pd.DataFrame()
    best = best.copy()

    # Vectorized column assignment instead of iterrows loop
    best["analysis_mode"] = args.analysis_mode
    best["input_mode"] = args.input_mode
    best["representation"] = args.representation if hasattr(args, "representation") else ""

    keep_cols = [
        "analysis_mode",
        "input_mode",
        "representation",
        "scope",
        "condition",
        "unit_name",
        "reducer",
        "n_components",
    ]

    if pooled_condition is not None and "scope" in best.columns:
        best.loc[best["scope"] == "pooled", "condition"] = pooled_condition

    if args.input_mode == "foundation_embeddings":
        best["model"] = args.embedding_model_key if hasattr(args, "embedding_model_key") else ""
        model_key = str(getattr(args, "embedding_model_key", ""))
        best["transform"] = model_key.split("_align-", 1)[1] if "_align-" in model_key else "none"
        best["representation"] = args.representation or ""
        keep_cols.extend(["model", "transform"])

    for metric in [
        *DIM_REDUCTION_EVAL_METRIC_COLUMNS,
        "trustworthiness",
        "continuity",
        "eval_name",
    ]:
        if metric in best.columns:
            keep_cols.append(metric)

    # Ensure base columns exist gracefully
    for col in keep_cols:
        if col not in best.columns:
            best[col] = ""

    return best[keep_cols]


def generate_rollup_report(
    *,
    args: Any,
    summaries: Sequence[dict[str, Any]],
    task_failures: Sequence[dict[str, str]] = (),
    bids_root: str | Path | None = None,
    alignment_diagnostics: AlignmentDiagnosticsSpec | None = None,
) -> Report:
    """Cross-mode leaderboard answering which representation wins for this cohort.

    Aggregates the per-mode leaderboards into one sortable table plus a
    faithful-vs-discriminative scatter (trustworthiness × separation), so the
    EEG / descriptor-mode / foundation-model comparison lands on a single axis.
    """
    run_label = (
        args.run_label if hasattr(args, "run_label") and args.run_label else args.dataset_name
    )
    asset_urls = getattr(args, "report_asset_urls", None)
    if asset_urls == "cdn":
        asset_urls = None

    frames = [
        summary["leaderboard"]
        for summary in summaries
        if isinstance(summary.get("leaderboard"), pd.DataFrame) and not summary["leaderboard"].empty
    ]
    leaderboard = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    selection_note = (
        f" (eval: {args.selection_eval_name})"
        if hasattr(args, "selection_eval_name") and args.selection_eval_name
        else ""
    )
    strategy_note = (
        f"Best run per analysis mode &times; condition for **{run_label}** ({args.input_mode}), "
        f"ranked by **{args.selection_metric}**{selection_note}.<br/><br/>"
        "💡 A strong representation is both geometrically faithful (high trustworthiness) "
        "and clinically separating (high RF, then LR, balanced accuracy) — "
        "aim for the top-right in the scatter plot below."
    )
    link_rows = [
        {
            "analysis_mode": summary.get("analysis_mode", ""),
            "representation": summary.get("representation", ""),
            "run_variant": summary.get("run_variant", ""),
            "report": f"[View Report]({summary['report_path']})"
            if summary.get("report_path")
            else "",
        }
        for summary in summaries
    ]

    y_metric = (
        args.selection_metric
        if hasattr(args, "selection_metric") and args.selection_metric
        else DEFAULT_DIM_REDUCTION_SELECTION_METRIC
    )

    report = build_reduction_rollup_report(
        leaderboard,
        title=f"Dim Reduction Roll-up: {run_label} ({args.input_mode})",
        y_metric=y_metric,
        sort_col=y_metric if y_metric in leaderboard.columns else None,
        mode_label_map=MODE_TITLES,
        strategy_note=strategy_note,
        link_rows=link_rows,
        task_failures=task_failures,
        asset_urls=asset_urls,
    )
    if alignment_diagnostics is not None:
        if bids_root is None:
            raise ValueError("bids_root is required when alignment_diagnostics is requested.")
        diagnostics = load_alignment_diagnostics(bids_root, alignment_diagnostics)
        section = build_subject_alignment_diagnostics_section(diagnostics)
        if section is None:
            raise ValueError("Requested alignment diagnostics contain no reportable rows.")
        report.add_section(section)
    return report

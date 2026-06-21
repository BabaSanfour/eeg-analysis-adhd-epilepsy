"""Aggregate reports for classical and foundation-model decoding sweeps."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.io import read_json
from coco_pipe.io.quality import QCResult
from coco_pipe.report import (
    ResultCollection,
    build_comparison_section,
    build_result_tabs,
    collect_results,
)
from coco_pipe.report.core import Report, Section
from coco_pipe.report.elements import (
    InteractiveTableElement,
    TableElement,
)
from coco_pipe.report.qc import build_qc_section

LOGGER = logging.getLogger(__name__)

_RESULT_COLUMN_LABELS = {
    "target": "Target",
    "unit_name": "Analysis Unit",
    "family": "Family",
    "subfamily": "Subfamily",
    "model": "Model",
    "selection_mode": "Feature Selection",
    "status": "Status",
    "reason": "Reason",
    "n_samples": "N Observations",
    "n_groups": "N Subjects",
    "accuracy_mean": "Accuracy",
    "accuracy_std": "Accuracy SD",
    "balanced_accuracy_mean": "Balanced Accuracy",
    "balanced_accuracy_std": "Balanced Accuracy SD",
    "f1_mean": "F1",
    "f1_std": "F1 SD",
    "precision_mean": "Precision",
    "precision_std": "Precision SD",
    "recall_mean": "Recall",
    "recall_std": "Recall SD",
    "roc_auc_mean": "ROC AUC",
    "roc_auc_std": "ROC AUC SD",
    "p_value": "P Value",
    "p_value_fdr": "FDR P Value",
    "significant_fdr": "FDR Significant",
}
_RESULT_COLUMN_ORDER = tuple(_RESULT_COLUMN_LABELS)
_CLASSICAL_ANALYSIS_PLAN = (
    (("flat",), "Full Analysis: All Sensors x All Features"),
    (("sensor",), "Sensor-wise Analyses"),
    (("subfamily",), "Subfamily Analyses: All Sensors"),
    (("sensor_within_subfamily",), "Sensor x Subfamily Analyses"),
    (("descriptor", "feature"), "Single Descriptor (all stats): All Sensors"),
    (
        ("descriptor_sensor", "feature_sensor"),
        "Single Descriptor (all stats) x Single Sensor",
    ),
)


def _result_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the concise, performance-first schema used in result tables."""
    columns = [
        column for column in _RESULT_COLUMN_ORDER if column in frame and frame[column].notna().any()
    ]
    return frame.loc[:, columns].rename(columns=_RESULT_COLUMN_LABELS)


def descriptor_feature_metadata(
    config: Mapping[str, Any],
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """Build coco-pipe feature metadata from the configured descriptor schema."""
    path_value = config.get("descriptor_feature_columns_path")
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return None

    from coco_pipe.descriptors import parse_descriptor_feature_column
    from coco_pipe.descriptors.qc import descriptor_subfamily

    known_families = ("band", "param", "complexity")
    try:
        columns = read_json(path)
        parsed = [
            parse_descriptor_feature_column(str(column), known_families) for column in columns
        ]
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.debug("Could not build descriptor feature metadata: %s", exc)
        return None

    rows = [
        {
            "FeatureName": f"{item['sensor']}_{item['feature']}",
            "Sensor": item["sensor"],
            "FeatureFamily": descriptor_subfamily(
                item["family"],
                item["feature"],
            ),
        }
        for item in parsed
    ]
    metadata = pd.DataFrame(rows).drop_duplicates("FeatureName")
    if feature_names is not None:
        requested = {str(value) for value in feature_names}
        metadata = metadata[metadata["FeatureName"].isin(requested)]
    return metadata.reset_index(drop=True)


def _standard_sensor_info(sensor_names: Sequence[str]) -> Any | None:
    """Return standard-1020 MNE info when at least three sensors are known."""
    names = list(dict.fromkeys(str(value) for value in sensor_names))
    if len(names) < 3:
        return None
    try:
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        available = set(montage.ch_names)
        names = [name for name in names if name in available]
        if len(names) < 3:
            return None
        info = mne.create_info(names, sfreq=100.0, ch_types="eeg")
        info.set_montage(montage, on_missing="raise")
        return info
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _collect_mode_results(mode_frame: pd.DataFrame):
    """Collect unique successful result artifacts through coco-pipe."""
    required = {"status", "output_dir"}
    if not required.issubset(mode_frame.columns):
        return None
    successful = mode_frame[
        (mode_frame["status"] == "success") & mode_frame["output_dir"].notna()
    ].copy()
    if successful.empty:
        return None
    context_columns = [
        column
        for column in (
            "scope",
            "target",
            "analysis_mode",
            "unit_key",
            "unit_name",
            "subfamily",
            "selection_mode",
        )
        if column in successful and successful[column].notna().any()
    ]
    unique = successful[context_columns + ["output_dir"]].drop_duplicates()
    items = [
        (
            {column: row[column] for column in context_columns},
            Path(str(row["output_dir"])) / "result.joblib",
        )
        for row in unique.to_dict("records")
    ]
    return collect_results(items, by=context_columns)


def _summary_collection(
    mode_frame: pd.DataFrame,
    *,
    feature_metadata: pd.DataFrame | None,
    include_sensor: bool = False,
) -> ResultCollection | None:
    """Build a lightweight comparison collection from persisted sweep metrics."""
    if "accuracy_mean" not in mode_frame:
        return None
    summary = mode_frame.copy()
    summary["_status"] = summary.get("status", "success")
    summary = summary[summary["_status"] == "success"].copy()
    summary["accuracy_mean"] = pd.to_numeric(
        summary["accuracy_mean"],
        errors="coerce",
    )
    summary = summary[summary["accuracy_mean"].notna()]
    if summary.empty:
        return None

    family_by_feature: dict[str, str] = {}
    if feature_metadata is not None and not feature_metadata.empty:
        for row in feature_metadata.to_dict("records"):
            sensor = str(row.get("Sensor", ""))
            feature_name = str(row.get("FeatureName", ""))
            prefix = f"{sensor}_"
            if sensor and feature_name.startswith(prefix):
                family_by_feature.setdefault(
                    feature_name.removeprefix(prefix),
                    str(row.get("FeatureFamily", "Other")),
                )
    summary["descriptor_family"] = summary["unit_name"].map(family_by_feature).fillna("Other")

    if include_sensor:

        def sensor_name(row: pd.Series) -> str:
            prefix = f"{row['unit_name']}_"
            unit_key = str(row.get("unit_key", ""))
            return unit_key.removeprefix(prefix) if unit_key.startswith(prefix) else ""

        summary["sensor"] = summary.apply(sensor_name, axis=1)
        summary = summary[summary["sensor"] != ""]
        if summary.empty:
            return None

    by = tuple(
        column
        for column in (
            "scope",
            "target",
            "analysis_mode",
            "unit_key",
            "unit_name",
            "selection_mode",
        )
        if column in summary
    )
    return ResultCollection(by=by, results={}, contexts={}, summary=summary)


def generate_decoding_summary_report(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    title: str,
    config: Mapping[str, Any],
    qc_results: Sequence[tuple[str, QCResult]] = (),
    figures_dir: str | Path | None = None,
) -> Path:
    """Generate one ordered analysis section per condition and pooled scope."""
    output = Path(output_path)
    _ = figures_dir  # Retained for API compatibility; coco-pipe embeds figures.
    frame = pd.DataFrame([dict(record) for record in records])
    feature_metadata = descriptor_feature_metadata(config)
    report = Report(
        title=title,
        config=dict(config),
        asset_urls=config.get("report_asset_urls", "inline"),
    )

    overview = Section("Run Overview")
    if frame.empty:
        overview.add_markdown("No decoding analysis units were produced.")
    else:
        overview.add_element(
            TableElement(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Analysis rows",
                            "Successful",
                            "Skipped",
                            "Failed",
                        ],
                        "Value": [
                            len(frame),
                            int((frame["status"] == "success").sum()),
                            int((frame["status"] == "skipped").sum()),
                            int((frame["status"] == "failed").sum()),
                        ],
                    }
                ),
                title="Run Coverage",
            )
        )
    report.add_section(overview)
    for scope, qc_result in qc_results:
        qc_section = build_qc_section(qc_result, compact=True, page_size=10)
        qc_section.title = f"Data Quality (QC): {scope}"
        report.add_section(qc_section)

    configured_scopes = [str(value) for value in config.get("conditions", [])]
    if (
        config.get("run_pooled", True)
        and len(configured_scopes) > 1
        and "pooled" not in configured_scopes
    ):
        configured_scopes.append("pooled")
    if not frame.empty:
        if "scope" not in frame.columns:
            frame["scope"] = "all"
        available_scopes = list(dict.fromkeys(frame["scope"].astype(str)))
    else:
        available_scopes = []
    pooled_requested = "pooled" in configured_scopes or "pooled" in available_scopes
    scope_order = [scope for scope in dict.fromkeys(configured_scopes) if scope != "pooled"]
    scope_order.extend(
        scope for scope in available_scopes if scope not in scope_order and scope != "pooled"
    )
    if pooled_requested:
        scope_order.append("pooled")

    if scope_order and (frame.empty or "analysis_mode" in frame.columns):
        for scope in scope_order:
            scope_frame = (
                frame[frame["scope"].astype(str) == scope].copy()
                if not frame.empty
                else frame.copy()
            )
            scope_label = "POOLED" if scope == "pooled" else scope
            section = Section(scope_label)
            section.add_markdown(
                "Results are ordered from the full descriptor-space analysis "
                "through progressively narrower sensor and feature analyses."
            )
            flat = (
                scope_frame[scope_frame["analysis_mode"] == "flat"]
                if "analysis_mode" in scope_frame
                else pd.DataFrame()
            )
            successful_flat = flat[
                flat.get("status", pd.Series(index=flat.index, dtype=object)) == "success"
            ]
            if successful_flat.empty:
                section.status = "FAIL"
                section.add_markdown("**The full analysis is missing or incomplete.**")

            for analysis_modes, table_title in _CLASSICAL_ANALYSIS_PLAN:
                if "analysis_mode" not in scope_frame:
                    break
                mode_frame = scope_frame[scope_frame["analysis_mode"].isin(analysis_modes)].copy()
                if mode_frame.empty:
                    continue
                if "selection_mode" in mode_frame:
                    mode_frame["_selection_order"] = (
                        mode_frame["selection_mode"].astype(str) != "baseline"
                    ).astype(int)
                sort_columns = [
                    column
                    for column in (
                        "target",
                        "unit_name",
                        "subfamily",
                        "_selection_order",
                        "selection_mode",
                        "model",
                    )
                    if column in mode_frame
                ]
                if sort_columns:
                    mode_frame = mode_frame.sort_values(
                        sort_columns, kind="stable", na_position="last"
                    )
                mode_frame = mode_frame.drop(columns=["_selection_order"], errors="ignore")
                display_frame = _result_display_frame(mode_frame)
                selector_columns = [
                    column
                    for column in (
                        "Target",
                        "Analysis Unit",
                        "Subfamily",
                        "Family",
                        "Model",
                        "Feature Selection",
                        "Status",
                    )
                    if column in display_frame
                ]
                section.add_element(
                    InteractiveTableElement(
                        display_frame,
                        title=table_title,
                        selector_columns=selector_columns,
                    )
                )
                primary_mode = analysis_modes[0]
                if primary_mode in {"descriptor", "feature"}:
                    collection = _summary_collection(
                        mode_frame,
                        feature_metadata=feature_metadata,
                    )
                    if collection is not None:
                        comparison = build_comparison_section(
                            collection,
                            kind="axis_heatmap",
                            axis="unit_name",
                            column="model",
                            group_by=(
                                "target",
                                "selection_mode",
                                "descriptor_family",
                            ),
                            title=f"Single-Descriptor Accuracy: {scope_label}",
                        )
                        if comparison is not None:
                            section.add_element(comparison)
                    continue
                if primary_mode in {"descriptor_sensor", "feature_sensor"}:
                    collection = _summary_collection(
                        mode_frame,
                        feature_metadata=feature_metadata,
                        include_sensor=True,
                    )
                    if collection is not None:
                        comparison = build_comparison_section(
                            collection,
                            kind="grid_heatmap",
                            row="unit_name",
                            column="sensor",
                            group_by=(
                                "target",
                                "selection_mode",
                                "model",
                                "descriptor_family",
                            ),
                            title=(f"Single Descriptor x Sensor Accuracy: {scope_label}"),
                        )
                        if comparison is not None:
                            section.add_element(comparison)
                    continue
                collection = _collect_mode_results(mode_frame)
                if collection is None:
                    continue
                if primary_mode == "flat":
                    nested = build_result_tabs(
                        collection,
                        sections="full",
                        feature_metadata=feature_metadata,
                        on_error="placeholder",
                    )
                    if nested is not None:
                        section.add_element(nested)
                elif primary_mode == "sensor":
                    comparison = build_comparison_section(
                        collection,
                        kind="sensor_topomap",
                        axis="unit_name",
                        group_by=("target", "selection_mode", "Model"),
                        info=_standard_sensor_info(mode_frame["unit_name"]),
                        title=f"Sensor-wise Accuracy: {scope_label}",
                    )
                    if comparison is not None:
                        section.add_element(comparison)
                elif primary_mode == "subfamily":
                    comparison = build_comparison_section(
                        collection,
                        kind="axis_heatmap",
                        axis="subfamily",
                        column="Model",
                        group_by=("target", "selection_mode"),
                        title=f"Subfamily Accuracy: {scope_label}",
                    )
                    if comparison is not None:
                        section.add_element(comparison)
                elif primary_mode == "sensor_within_subfamily":
                    comparison = build_comparison_section(
                        collection,
                        kind="grid_heatmap",
                        row="subfamily",
                        column="unit_name",
                        group_by=("target", "selection_mode", "Model"),
                        title=f"Sensor x Subfamily Accuracy: {scope_label}",
                    )
                    if comparison is not None:
                        section.add_element(comparison)
            report.add_section(section)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save(output)
    return output


def generate_foundation_decoding_report(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    title: str,
    config: Mapping[str, Any],
    capability_records: Sequence[Mapping[str, Any]] = (),
    figures_dir: str | Path | None = None,
) -> Path:
    """Render foundation-decoding comparisons through coco-pipe."""
    output = Path(output_path)
    _ = figures_dir  # Retained for API compatibility; coco-pipe embeds figures.
    frame = pd.DataFrame([dict(record) for record in records])
    asset_urls = config.get("report_asset_urls", "inline")
    report = Report(title=title, config=dict(config), asset_urls=asset_urls)

    overview = Section("Run Overview")
    if frame.empty or "status" not in frame:
        overview.add_markdown("No foundation decoding units were produced.")
        report.add_section(overview)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.save(output)
        return output

    status_counts = frame["status"].value_counts()
    coverage = pd.DataFrame(
        {
            "Metric": ["Analysis rows", "Successful", "Skipped", "Failed"],
            "Value": [
                len(frame),
                int(status_counts.get("success", 0)),
                int(status_counts.get("skipped", 0)),
                int(status_counts.get("failed", 0)),
            ],
        }
    )
    overview.add_element(TableElement(coverage, title="Run Coverage"))
    report.add_section(overview)

    required = {"status", "output_dir", "model_key", "train_mode", "target"}
    if not required.issubset(frame.columns):
        gap = Section("Performance Figures")
        gap.add_markdown(
            "Foundation results are missing the context or output paths required "
            "for coco-pipe comparison figures."
        )
        report.add_section(gap)
    else:
        success = frame[(frame["status"] == "success") & frame["output_dir"].notna()].copy()
        context_columns = [
            column
            for column in ("condition", "target", "model_key", "train_mode")
            if column in success and success[column].notna().any()
        ]
        success["unit"] = (
            success["model_key"].astype(str) + " · " + success["train_mode"].astype(str)
        )
        context_columns.append("unit")
        unique = success[context_columns + ["output_dir"]].drop_duplicates()
        collection = collect_results(
            [
                (
                    {column: row[column] for column in context_columns},
                    Path(str(row["output_dir"])) / "result.joblib",
                )
                for row in unique.to_dict("records")
            ],
            by=context_columns,
        )
        comparison_specs = [
            (
                collection.filter(train_mode="linear_probe"),
                {
                    "kind": "model_bars",
                    "axis": "model_key",
                    "group_by": ("condition", "target"),
                    "title": "Primary Result: Linear Probe Accuracy",
                },
            ),
            (
                collection,
                {
                    "kind": "axis_heatmap",
                    "axis": "model_key",
                    "column": "train_mode",
                    "group_by": ("condition", "target"),
                    "title": "Training-Mode Comparison",
                },
            ),
            (
                collection,
                {
                    "kind": "metric_matrix",
                    "axis": "unit",
                    "group_by": ("condition", "target"),
                    "title": "Metric Matrix",
                },
            ),
            (
                collection,
                {
                    "kind": "spread",
                    "axis": "train_mode",
                    "group_by": ("condition", "target"),
                    "title": "Accuracy Spread by Training Mode",
                },
            ),
        ]
        for comparison_collection, spec in comparison_specs:
            comparison = build_comparison_section(comparison_collection, **spec)
            if comparison is not None:
                report.add_section(comparison)

        per_result = build_result_tabs(
            collection,
            sections="full",
            on_error="placeholder",
        )
        if per_result is not None:
            section = Section("Per-Result Diagnostics")
            section.add_element(per_result)
            report.add_section(section)

    capabilities = pd.DataFrame([dict(record) for record in capability_records])
    if not capabilities.empty:
        capability_section = Section("Foundation Capability Matrix")
        capability_section.add_markdown(
            "Preflight decisions for every model and training mode. Unsupported "
            "combinations are reported explicitly and never silently downgraded."
        )
        capability_columns = [
            column
            for column in (
                "condition",
                "target",
                "model_key",
                "train_mode",
                "status",
                "reason",
                "sfreq",
                "n_channels",
                "n_times",
            )
            if column in capabilities.columns
        ]
        display = capabilities[capability_columns] if capability_columns else capabilities
        capability_section.add_element(
            InteractiveTableElement(
                display,
                title="Preflight Decisions",
                selector_columns=[
                    column
                    for column in ("model_key", "train_mode", "status")
                    if column in display.columns
                ],
            )
        )
        report.add_section(capability_section)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save(output)
    return output


def generate_head_to_head_report(
    *,
    bids_root: str | Path,
    reports_root: str | Path,
    output_group: str,
    dataset_name: str,
    asset_urls: dict[str, str] | str | None = "inline",
) -> tuple[Path, Path] | None:
    """Compare primary classical inputs and linear probes in one table."""
    bids_root = Path(bids_root)
    classical_root = bids_root / "derivatives" / "decoding" / output_group / dataset_name
    frames: list[pd.DataFrame] = []
    if classical_root.exists():
        for path in sorted(classical_root.glob("*/sweep_results.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            required = {"status", "analysis_mode", "selection_mode"}
            if not required.issubset(frame.columns):
                continue
            frame = frame[
                (frame["status"] == "success")
                & (frame["analysis_mode"] == "flat")
                & (frame["selection_mode"] == "baseline")
            ].copy()
            if not frame.empty:
                frame["comparison_space"] = path.parent.name
                frames.append(frame)

    foundation_path = (
        bids_root
        / "derivatives"
        / "foundation_decoding"
        / output_group
        / dataset_name
        / "foundation_results.csv"
    )
    if foundation_path.exists():
        try:
            frame = pd.read_csv(foundation_path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if {"status", "train_mode"}.issubset(frame.columns):
            frame = frame[
                (frame["status"] == "success") & (frame["train_mode"] == "linear_probe")
            ].copy()
            if not frame.empty:
                frame["comparison_space"] = "foundation_linear_probe"
                frames.append(frame)

    if not frames:
        return None

    comparison = pd.concat(frames, ignore_index=True, sort=False)
    signature_columns = [
        column
        for column in (
            "cv_strategy",
            "effective_n_splits",
            "cv_random_state",
        )
        if column in comparison.columns
    ]
    if signature_columns:
        comparison["cv_signature"] = (
            comparison[signature_columns]
            .astype(str)
            .agg(
                " | ".join,
                axis=1,
            )
        )
    derivative_output = classical_root / "head_to_head_comparison.csv"
    derivative_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(derivative_output, index=False)

    report = Report(
        title=f"Head-to-Head Decoding: {dataset_name}",
        config={"output_group": output_group, "dataset_name": dataset_name},
        asset_urls=asset_urls,
    )
    section = Section("Head-to-Head Comparison")
    section.add_markdown(
        "Primary flat baseline results and foundation-model linear probes are "
        "shown together. Full fine-tuning, LoRA, SFS, and feature sweeps remain "
        "secondary or exploratory. Compare rows only when their "
        "grouped-CV and cohort signatures match."
    )
    selectors = [
        column
        for column in (
            "comparison_space",
            "scope",
            "target",
            "model_key",
            "model",
            "cv_signature",
            "cohort_signature",
        )
        if column in comparison.columns
    ]
    section.add_element(
        InteractiveTableElement(
            comparison,
            title="Descriptors vs. Reduced Dimensions vs. Embeddings vs. Linear Probe",
            selector_columns=selectors,
        )
    )
    report.add_section(section)
    value_column = next(
        (column for column in ("accuracy_mean", "accuracy") if column in comparison.columns),
        None,
    )
    if value_column is not None:
        comparison_plot = comparison.copy()
        model_label = pd.Series("model", index=comparison_plot.index, dtype=object)
        if "model" in comparison_plot:
            model_label = comparison_plot["model"].combine_first(model_label)
        if "model_key" in comparison_plot:
            model_label = comparison_plot["model_key"].combine_first(model_label)
        comparison_plot["comparison_label"] = (
            comparison_plot["comparison_space"].astype(str) + " · " + model_label.astype(str)
        )
        comparison_plot["_status"] = "success"
        plot_section = build_comparison_section(
            ResultCollection(
                by=("comparison_label",),
                results={},
                contexts={},
                summary=comparison_plot,
            ),
            kind="head_to_head",
            axis="comparison_label",
            value=value_column,
            group_by=("scope", "target"),
            title="Head-to-Head Accuracy",
            include_table=False,
        )
        if plot_section is not None:
            report.add_section(plot_section)
    report_output = (
        Path(reports_root)
        / "summary"
        / "decoding"
        / output_group
        / dataset_name
        / "head_to_head_comparison.html"
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report.save(report_output)
    return derivative_output, report_output

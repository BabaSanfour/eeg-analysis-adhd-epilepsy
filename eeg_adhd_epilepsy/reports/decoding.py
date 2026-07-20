"""Aggregate reports for classical and foundation-model decoding sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from coco_pipe.descriptors import build_descriptor_feature_metadata
from coco_pipe.io import read_json
from coco_pipe.io.quality import QCResult
from coco_pipe.report import (
    CLASSICAL_MODE_TITLES,
    HEAD_TO_HEAD_DISPLAY_COLUMNS,
    build_alignment_coverage_section,
    build_alignment_tradeoff_section,
    build_classical_mode_elements,
    build_subject_alignment_diagnostics_section,
    collect_comparison_runs,
    enrich_head_to_head_frame,
    feature_selection_section,
    make_decoding_report,
    make_foundation_decoding_report,
    make_head_to_head_report,
    normalize_head_to_head_frame,
    prepare_sweep_frame,
)
from coco_pipe.report.core import Report, Section
from coco_pipe.report.qc import build_qc_section

from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root
from eeg_adhd_epilepsy.io.report_paths import ReportStage, summary_report_dir
from eeg_adhd_epilepsy.reports._common import (
    AlignmentDiagnosticsSpec,
    load_alignment_diagnostics,
)

_CLASSICAL_ANALYSIS_PLAN = tuple(((mode,), title) for mode, title in CLASSICAL_MODE_TITLES.items())


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
    frame = prepare_sweep_frame(records, scope_from=None, default_scope="all")

    feature_metadata = None
    descriptor_columns_path = config.get("descriptor_feature_columns_path")
    if descriptor_columns_path:
        path = Path(str(descriptor_columns_path)).expanduser()
        if path.exists():
            feature_metadata = build_descriptor_feature_metadata(read_json(path))

    metric_label = (
        str(frame["primary_metric_name"].iloc[0])
        if "primary_metric_name" in frame and not frame.empty
        else "unavailable"
    )
    input_mode = config.get("input_mode", "descriptors")

    configured_scopes = [str(value) for value in config.get("conditions", [])]
    if (
        config.get("run_pooled", True)
        and len(configured_scopes) > 1
        and "pooled" not in configured_scopes
    ):
        configured_scopes.append("pooled")
    available_scopes = list(dict.fromkeys(frame["scope"].astype(str))) if not frame.empty else []
    pooled_requested = "pooled" in configured_scopes or "pooled" in available_scopes
    scope_order = [scope for scope in dict.fromkeys(configured_scopes) if scope != "pooled"]
    scope_order.extend(
        scope for scope in available_scopes if scope not in scope_order and scope != "pooled"
    )
    if pooled_requested:
        scope_order.append("pooled")

    qc_sections = []
    for scope, qc_result in qc_results:
        qc_section = build_qc_section(qc_result, compact=True, page_size=10)
        qc_section.title = f"Data Quality (QC): {scope}"
        qc_sections.append(qc_section)

    make_decoding_report(
        records,
        frame=frame,
        title=title,
        dataset_name=str(config.get("dataset_name", "dataset")),
        strategy_note=(
            f"Primary metric: {metric_label}. Flat baseline is the primary "
            f"classical result for {input_mode}; non-flat and SFS runs explain "
            "where the signal localizes."
        ),
        pre_sections=qc_sections,
        feature_metadata=feature_metadata,
        scope_order=scope_order,
        scope_from=None,
        default_scope="all",
        config=config,
        asset_urls=config.get("report_asset_urls", "inline"),
        output_path=output,
    )

    if scope_order and (frame.empty or "analysis_mode" in frame.columns):
        mode_reports: dict[str, Report] = {}
        for scope in scope_order:
            scope_frame = (
                frame[frame["scope"].astype(str) == scope].copy()
                if not frame.empty
                else frame.copy()
            )
            scope_label = "POOLED" if scope == "pooled" else scope
            if "analysis_mode" not in scope_frame:
                continue
            for analysis_modes, table_title in _CLASSICAL_ANALYSIS_PLAN:
                mode_frame = scope_frame[scope_frame["analysis_mode"].isin(analysis_modes)].copy()
                if mode_frame.empty:
                    continue
                mode = analysis_modes[0]
                mode_report = mode_reports.get(mode)
                if mode_report is None:
                    mode_report = Report(
                        title=f"{title} - {table_title}",
                        config=dict(config),
                        asset_urls=config.get("report_asset_urls", "inline"),
                    )
                    mode_reports[mode] = mode_report
                mode_section = Section(scope_label)
                for el in build_classical_mode_elements(
                    mode_frame,
                    table_title,
                    analysis_modes,
                    scope_label,
                    feature_metadata,
                ):
                    mode_section.add_element(el)
                fs_section = feature_selection_section(
                    scope_frame, feature_metadata=feature_metadata
                )
                if fs_section is not None:
                    mode_section.add_element(fs_section)
                mode_report.add_section(mode_section)
        for mode, mode_report in mode_reports.items():
            mode_output = output.with_name(f"{output.stem}_{mode}{output.suffix}")
            mode_report.save(mode_output)

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
    """Render the foundation-decoding sweep report through coco-pipe."""
    output = Path(output_path)
    _ = figures_dir  # Retained for API compatibility; coco-pipe embeds figures.
    frame = prepare_sweep_frame(records, scope_from="condition", default_scope=None)
    metric_label = (
        str(frame["primary_metric_name"].iloc[0])
        if "primary_metric_name" in frame and not frame.empty
        else "unavailable"
    )
    make_foundation_decoding_report(
        records,
        frame=frame,
        capability_records=capability_records,
        title=title,
        dataset_name=str(config.get("dataset_name", "dataset")),
        strategy_note=(
            f"Primary metric: {metric_label}. Linear probes are treated as the "
            "primary foundation comparison; full fine-tuning and LoRA are secondary."
        ),
        per_result_sections=config.get("foundation_report_sections", "compact"),
        config=config,
        asset_urls=config.get("report_asset_urls", "inline"),
        output_path=output,
    )
    return output


def generate_foundation_decoding_comparison(
    *,
    bids_root: str | Path,
    reports_root: str | Path,
    dataset_name: str,
    config: Mapping[str, Any],
    derivative_root: str | Path | None = None,
) -> tuple[Path, Path, Path | None] | None:
    """Aggregate every direct foundation run across models and training modes."""
    decoding_base = (
        Path(derivative_root).expanduser()
        if derivative_root is not None
        else get_derivative_root(Path(bids_root), DerivativeStage.DECODING)
    )
    decoding_root = decoding_base / dataset_name
    result_frames: list[pd.DataFrame] = []
    capability_frames: list[pd.DataFrame] = []
    for path in sorted(decoding_root.glob("foundation*/foundation_results.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["run_variant"] = path.parent.name
            result_frames.append(frame)
    for path in sorted(decoding_root.glob("foundation*/capability_matrix.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["run_variant"] = path.parent.name
            capability_frames.append(frame)

    if not result_frames and not capability_frames:
        return None

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    capabilities = (
        pd.concat(capability_frames, ignore_index=True) if capability_frames else pd.DataFrame()
    )
    comparison_csv = decoding_root / "foundation_decoding_comparison.csv"
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(comparison_csv, index=False)

    capability_csv: Path | None = None
    if not capabilities.empty:
        capability_csv = decoding_root / "foundation_decoding_capabilities.csv"
        capabilities.to_csv(capability_csv, index=False)

    report_path = (
        summary_report_dir(Path(reports_root), ReportStage.DECODING)
        / dataset_name
        / "foundation_decoding_comparison.html"
    )
    generate_foundation_decoding_report(
        report_path,
        results.to_dict("records"),
        title=f"Foundation Decoding Comparison: {dataset_name}",
        config={**dict(config), "dataset_name": dataset_name},
        capability_records=capabilities.to_dict("records"),
    )
    return comparison_csv, report_path, capability_csv


def _apply_legacy_head_to_head_migration_policy(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Fill context omitted by result schemas predating transforms and PCA cells.

    The migration is intentionally isolated from report generation. Missing
    ``transform`` means the historical raw representation; missing
    ``reduction_mode`` means the historical unreduced, all-dimensions cell.
    """
    migrated = comparison.copy()
    defaults = {
        "transform": "none",
        "reduction_mode": "all_dimensions",
    }
    for column, default in defaults.items():
        if column not in migrated:
            migrated[column] = default
        else:
            migrated[column] = migrated[column].fillna(default)
    return migrated


def _build_requested_alignment_diagnostics_sections(
    bids_root: Path,
    specs: Sequence[AlignmentDiagnosticsSpec],
) -> tuple[list[Section], pd.DataFrame]:
    """Load explicitly requested assessments and render one section per model."""
    identities = [(spec.base_model_key, spec.cohort_name, spec.population) for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError("Alignment diagnostics specifications must be unique.")
    sections: list[Section] = []
    loaded_diagnostics: list[pd.DataFrame] = []
    for spec in specs:
        diagnostics = load_alignment_diagnostics(bids_root, spec)
        diagnostics["model"] = spec.base_model_key
        diagnostics["target"] = diagnostics["eval_name"].astype(str)
        loaded_diagnostics.append(diagnostics)
        section = build_subject_alignment_diagnostics_section(
            diagnostics,
            title=(
                "Subject Alignment Diagnostics"
                if len(specs) == 1
                else f"Subject Alignment Diagnostics: {spec.base_model_key}"
            ),
        )
        if section is None:
            raise ValueError(
                "Requested alignment diagnostics contain no reportable rows for "
                f"base_model_key={spec.base_model_key!r}."
            )
        sections.append(section)
    return (
        sections,
        pd.concat(loaded_diagnostics, ignore_index=True) if loaded_diagnostics else pd.DataFrame(),
    )


def _generate_foundation_transform_comparison(
    comparison: pd.DataFrame,
    *,
    decoding_root: Path,
    report_output: Path,
    dataset_name: str,
    alignment_sections: Sequence[Section],
    alignment_diagnostics: pd.DataFrame,
    asset_urls: dict[str, str] | str | None,
) -> tuple[Path, Path] | None:
    """Write the same-decoder transform/PCA comparison against raw embeddings."""
    foundation = comparison[
        comparison["comparison_family"] == "foundation_embedding_flat_baseline"
    ].copy()
    baseline = (foundation["transform"] == "none") & (
        foundation["reduction_mode"] == "all_dimensions"
    )
    if foundation.empty or not baseline.any() or not (~baseline).any():
        return None
    if "embedding_model_key" not in foundation or foundation["embedding_model_key"].isna().any():
        raise ValueError(
            "Foundation transform comparisons require embedding_model_key on every row."
        )

    foundation["embedding_model_key"] = foundation["embedding_model_key"].astype(str)
    foundation["decoder_model"] = (
        foundation["model"].astype(str) if "model" in foundation else "model"
    )
    foundation["comparison_family"] = "foundation_transform"
    foundation.loc[baseline, "comparison_family"] = "foundation_none_all_dimensions"
    foundation["model"] = (
        foundation["embedding_model_key"]
        + " | "
        + foundation["decoder_model"]
        + " | "
        + foundation["transform"].astype(str)
        + "/"
        + foundation["reduction_mode"].astype(str)
    )
    performance = foundation[foundation["reduction_mode"].astype(str) == "all_dimensions"].copy()
    performance["model"] = performance["embedding_model_key"]
    performance["decoder"] = performance["decoder_model"]
    performance["performance"] = performance["primary_metric"]
    diagnostic_sections: list[Section] = []
    coverage_section = build_alignment_coverage_section(alignment_diagnostics)
    if coverage_section is not None:
        diagnostic_sections.append(coverage_section)
    tradeoff_section = build_alignment_tradeoff_section(
        performance.loc[
            :,
            ["model", "decoder", "transform", "scope", "target", "performance"],
        ],
        alignment_diagnostics,
    )
    if tradeoff_section is not None:
        diagnostic_sections.append(tradeoff_section)
    foundation = enrich_head_to_head_frame(foundation)
    foundation_csv = decoding_root / "foundation_transform_comparison.csv"
    normalize_head_to_head_frame(
        foundation,
        display_columns=(
            "embedding_model_key",
            *HEAD_TO_HEAD_DISPLAY_COLUMNS,
            "transform",
            "reduction_mode",
        ),
    ).to_csv(foundation_csv, index=False)
    foundation_report = report_output.with_name("foundation_transform_comparison.html")
    make_head_to_head_report(
        foundation,
        baseline_family="foundation_none_all_dimensions",
        group_columns=(
            "scope",
            "target",
            "embedding_model_key",
            "decoder_model",
        ),
        title=f"Foundation Transform Comparison: {dataset_name}",
        table_title="Erasure and reduction cells",
        intro=(
            "Each foundation-model and decoder pair is matched to its own raw, "
            "all-dimensions cell. Deltas therefore measure label performance "
            "after subject erasure or PCA without crossing embedding spaces or "
            "changing the decoder family."
        ),
        paired_delta_title="Paired Delta vs Raw Foundation Embeddings",
        display_columns=(
            "embedding_model_key",
            *HEAD_TO_HEAD_DISPLAY_COLUMNS,
            "transform",
            "reduction_mode",
        ),
        config={"dataset_name": dataset_name},
        asset_urls=asset_urls,
        extra_sections=[*diagnostic_sections, *alignment_sections],
        output_path=foundation_report,
    )
    return foundation_csv, foundation_report


def generate_head_to_head_report(
    *,
    bids_root: str | Path,
    reports_root: str | Path,
    dataset_name: str,
    asset_urls: dict[str, str] | str | None = "inline",
    alignment_diagnostics_cohort_name: str | None = None,
    alignment_diagnostics_population: str | None = None,
    generate_foundation_transform_report: bool = False,
    derivative_root: str | Path | None = None,
) -> tuple[Path, Path] | None:
    """Compare primary classical inputs and linear probes in one table.

    When an exact diagnostic cohort and population are requested, every explicit
    ``embedding_model_key`` represented by saved-embedding results gets its own
    section. Model identities are never inferred from paths or display labels.
    """
    bids_root = Path(bids_root)
    # Classical and foundation runs share one decoding derivative tree; they are
    # distinguished by their per-run result filenames, not separate roots.
    decoding_base = (
        Path(derivative_root).expanduser()
        if derivative_root is not None
        else get_derivative_root(bids_root, DerivativeStage.DECODING)
    )
    decoding_root = decoding_base / dataset_name
    if not decoding_root.exists():
        return None

    comparison = collect_comparison_runs(
        [
            {
                "paths": sorted(decoding_root.glob("*/sweep_results.csv")),
                "filters": {
                    "status": "success",
                    "analysis_mode": "flat",
                    "selection_mode": "baseline",
                },
                "family": lambda f, p: (
                    "foundation_embedding_flat_baseline"
                    if (
                        str(f["input_mode"].dropna().iloc[0]) == "foundation_embeddings"
                        if "input_mode" in f and f["input_mode"].notna().any()
                        else "foundation" in p.parent.name
                    )
                    else "descriptor_flat_baseline"
                ),
            },
            {
                "paths": sorted(decoding_root.glob("foundation*/foundation_results.csv")),
                "filters": {"status": "success", "train_mode": "linear_probe"},
                "family": "foundation_linear_probe",
            },
        ]
    )
    if comparison.empty:
        return None

    report_output = (
        summary_report_dir(Path(reports_root), ReportStage.DECODING)
        / dataset_name
        / "head_to_head_comparison.html"
    )
    comparison = enrich_head_to_head_frame(comparison)
    comparison = _apply_legacy_head_to_head_migration_policy(comparison)

    alignment_diagnostics: list[AlignmentDiagnosticsSpec] = []
    if (
        alignment_diagnostics_cohort_name is not None
        or alignment_diagnostics_population is not None
    ):
        if not alignment_diagnostics_cohort_name or not alignment_diagnostics_population:
            raise ValueError(
                "Foundation alignment diagnostics require both cohort_name and population."
            )
        foundation = comparison[
            comparison["comparison_family"] == "foundation_embedding_flat_baseline"
        ]
        if not foundation.empty:
            if "embedding_model_key" not in foundation:
                raise ValueError(
                    "Foundation alignment diagnostics require embedding_model_key on every row."
                )
            model_keys = foundation["embedding_model_key"]
            if model_keys.isna().any() or model_keys.astype(str).str.strip().eq("").any():
                raise ValueError(
                    "Foundation alignment diagnostics require embedding_model_key on every row."
                )
            alignment_diagnostics = [
                AlignmentDiagnosticsSpec(
                    base_model_key=model_key,
                    cohort_name=alignment_diagnostics_cohort_name,
                    population=alignment_diagnostics_population,
                )
                for model_key in sorted(model_keys.astype(str).unique())
            ]
    (
        alignment_sections,
        alignment_diagnostics_frame,
    ) = _build_requested_alignment_diagnostics_sections(bids_root, alignment_diagnostics)
    derivative_output = decoding_root / "head_to_head_comparison.csv"
    derivative_output.parent.mkdir(parents=True, exist_ok=True)
    normalize_head_to_head_frame(comparison).to_csv(derivative_output, index=False)

    make_head_to_head_report(
        comparison,
        baseline_family="descriptor_flat_baseline",
        group_columns=("scope", "target", "transform", "reduction_mode"),
        title=f"Head-to-Head Decoding: {dataset_name}",
        intro=(
            "Primary flat baseline results and foundation-model linear probes are "
            "shown together. Full fine-tuning, LoRA, SFS, and feature sweeps remain "
            "secondary or exploratory. Compare rows only when their grouped-CV and "
            "cohort signatures match."
        ),
        paired_delta_title="Paired Delta vs Descriptor Baseline",
        config={"dataset_name": dataset_name},
        asset_urls=asset_urls,
        output_path=report_output,
    )

    if generate_foundation_transform_report:
        _generate_foundation_transform_comparison(
            comparison,
            decoding_root=decoding_root,
            report_output=report_output,
            dataset_name=dataset_name,
            alignment_sections=alignment_sections,
            alignment_diagnostics=alignment_diagnostics_frame,
            asset_urls=asset_urls,
        )
    return derivative_output, report_output

"""Descriptor QC report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from coco_pipe.report.core import Report, Section
from coco_pipe.report.elements import (
    ImageElement,
    InteractiveTableElement,
    TableElement,
)


def _add_optional_table(section: Section, data: pd.DataFrame | None, title: str) -> None:
    if data is not None and not data.empty:
        section.add_element(TableElement(data, title=title))


def _add_optional_interactive_table(
    section: Section,
    data: pd.DataFrame | None,
    title: str,
    *,
    selector_columns: list[str] | None = None,
    default_sort: dict[str, str] | None = None,
    page_size: int = 5,
) -> None:
    if data is not None and not data.empty:
        section.add_element(
            InteractiveTableElement(
                data,
                title=title,
                selector_columns=selector_columns,
                default_sort=default_sort,
                page_size=page_size,
            )
        )


def _add_images(section: Section, figures: Mapping[str, Path], ordered_keys: Sequence[str]) -> None:
    for key in ordered_keys:
        path = figures.get(key)
        if path and path.exists():
            section.add_element(ImageElement(str(path), caption=key.replace("_", " ").title()))


def generate_descriptor_subject_report(
    output_path: Path,
    overview_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    failure_summary_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    figure_paths: Mapping[str, Path],
) -> Path:
    report = Report(
        title=(
            "Descriptor QC Report - "
            f"{overview_df.iloc[0]['Subject']} {overview_df.iloc[0]['Session']} {overview_df.iloc[0]['Condition']}"
        )
    )

    overview = Section("Overview", icon="📋")
    _add_optional_table(overview, overview_df, "Shard Overview")
    report.add_section(overview)

    integrity = Section("Integrity Checks", icon="🧪")
    _add_optional_interactive_table(
        integrity,
        flags_df,
        "QC Flags",
        selector_columns=[column for column in ["level", "scope", "code"] if column in flags_df.columns],
        default_sort={"column": "level", "direction": "desc"} if "level" in flags_df.columns else None,
        page_size=5,
    )
    report.add_section(integrity)

    failures = Section("Failure Summary", icon="⚠️")
    _add_optional_interactive_table(
        failures,
        failure_summary_df,
        "Failure Summary",
        selector_columns=[column for column in ["group", "value"] if column in failure_summary_df.columns],
        default_sort={"column": "count", "direction": "desc"} if "count" in failure_summary_df.columns else None,
        page_size=5,
    )
    report.add_section(failures)

    missingness = Section("Missingness and Numerical Sanity", icon="📉")
    _add_optional_interactive_table(
        missingness,
        feature_missingness_df.round(4) if feature_missingness_df is not None else None,
        "Feature Missingness",
        selector_columns=[column for column in ["family", "scope", "sensor"] if column in feature_missingness_df.columns],
        default_sort={"column": "missing_rate", "direction": "desc"} if "missing_rate" in feature_missingness_df.columns else None,
        page_size=5,
    )
    report.add_section(missingness)

    families = Section("Family-Specific Summary", icon="🧬")
    _add_optional_interactive_table(
        families,
        family_summary_df.round(4) if family_summary_df is not None else None,
        "Family Summary",
        selector_columns=[column for column in ["family"] if column in family_summary_df.columns],
        default_sort={"column": "missing_rate", "direction": "desc"} if "missing_rate" in family_summary_df.columns else None,
        page_size=5,
    )
    report.add_section(families)

    figures = Section("Figures", icon="📈")
    _add_images(
        figures,
        figure_paths,
        (
            "family_missingness",
            "failure_counts_by_family",
            "top_missing_features",
            "param_r_squared_hist",
            "param_fit_error_hist",
        ),
    )
    report.add_section(figures)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path


def generate_descriptor_dataset_report(
    output_path: Path,
    overview_df: pd.DataFrame,
    shard_summary_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    failure_family_df: pd.DataFrame,
    failure_channel_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    low_variance_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    figure_paths: Mapping[str, Path],
) -> Path:
    report = Report(title="Descriptor QC Dataset Report")

    overview = Section("Overview", icon="📋")
    _add_optional_table(overview, overview_df, "Dataset Overview")
    report.add_section(overview)

    shards = Section("Shard-Level QC Summary", icon="🗂️")
    _add_optional_interactive_table(
        shards,
        shard_summary_df,
        "Shard QC Summary",
        selector_columns=[column for column in ["session", "condition", "qc_status"] if column in shard_summary_df.columns],
        default_sort={"column": "qc_status", "direction": "desc"} if "qc_status" in shard_summary_df.columns else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        shards,
        flags_df,
        "Dataset QC Flags",
        selector_columns=[column for column in ["level", "scope", "code"] if column in flags_df.columns],
        default_sort={"column": "level", "direction": "desc"} if "level" in flags_df.columns else None,
        page_size=5,
    )
    report.add_section(shards)

    failures = Section("Failures Summary", icon="⚠️")
    _add_optional_interactive_table(
        failures,
        failure_family_df,
        "Failures by Family",
        selector_columns=[column for column in ["value"] if column in failure_family_df.columns],
        default_sort={"column": "count", "direction": "desc"} if "count" in failure_family_df.columns else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        failures,
        failure_channel_df,
        "Failures by Channel",
        selector_columns=[column for column in ["value"] if column in failure_channel_df.columns],
        default_sort={"column": "count", "direction": "desc"} if "count" in failure_channel_df.columns else None,
        page_size=10,
    )
    report.add_section(failures)

    missingness = Section("Missingness and Degeneracy", icon="📉")
    _add_optional_interactive_table(
        missingness,
        feature_missingness_df.round(4) if feature_missingness_df is not None else None,
        "Feature Missingness",
        selector_columns=[column for column in ["family", "scope", "sensor"] if column in feature_missingness_df.columns],
        default_sort={"column": "missing_rate", "direction": "desc"} if "missing_rate" in feature_missingness_df.columns else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        missingness,
        low_variance_df.round(6) if low_variance_df is not None else None,
        "Low-Variance Features",
        selector_columns=[column for column in ["family"] if column in low_variance_df.columns],
        default_sort={"column": "std", "direction": "asc"} if "std" in low_variance_df.columns else None,
        page_size=10,
    )
    report.add_section(missingness)

    families = Section("Family-Specific Summary", icon="🧬")
    _add_optional_interactive_table(
        families,
        family_summary_df.round(4) if family_summary_df is not None else None,
        "Family Summary",
        selector_columns=[column for column in ["family"] if column in family_summary_df.columns],
        default_sort={"column": "missing_rate", "direction": "desc"} if "missing_rate" in family_summary_df.columns else None,
        page_size=5,
    )
    report.add_section(families)

    figures = Section("Figures", icon="📈")
    _add_images(
        figures,
        figure_paths,
        (
            "shard_status_counts",
            "failure_counts_by_family",
            "failure_counts_by_channel",
            "top_missing_features",
            "low_variance_by_family",
        ),
    )
    report.add_section(figures)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

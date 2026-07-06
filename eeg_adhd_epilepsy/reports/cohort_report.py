"""Cohort report generation over cleaned patient metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from coco_pipe.report.core import Report, Section
from coco_pipe.report.elements import (
    ImageElement,
    InteractiveTableElement,
    PlotlyElement,
    TableElement,
)

from eeg_adhd_epilepsy.reports._common import add_image_list, add_optional_table


def generate_cohort_report(
    output_path: Path,
    report_title: str,
    cohort_name: str,
    cohort_markdown: str,
    cohort_summary_df: pd.DataFrame,
    provenance_reason_df: pd.DataFrame,
    provenance_source_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    combined_diagnosis_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    medication_df: pd.DataFrame,
    valid_opportunities_df: pd.DataFrame,
    figures_by_section: Mapping[str, Sequence[tuple[str, Path]]],
    drug_resistant_overview_df: pd.DataFrame | None = None,
    first_later_drug_resistant_df: pd.DataFrame | None = None,
    source_overlap_df: pd.DataFrame | None = None,
    longitudinal_drug_resistant_patients_df: pd.DataFrame | None = None,
    drug_resistant_first_later_figure: object | None = None,
    recruitment_markdown: str | None = None,
    recruitment_projection_df: pd.DataFrame | None = None,
    recruitment_summary_df: pd.DataFrame | None = None,
    recruitment_pools_df: pd.DataFrame | None = None,
) -> Path:
    report = Report(title=report_title)

    cohort_definition = Section("Cohort Definition", icon="🎯")
    cohort_definition.add_markdown(
        f"Phase 1 cohort report for **{cohort_name}**, built directly from "
        "`patients_metadata_clean.csv`."
    )
    cohort_definition.add_markdown(cohort_markdown)
    add_optional_table(cohort_definition, cohort_summary_df, "Cohort Summary")
    add_image_list(cohort_definition, figures_by_section.get("Cohort Definition", []))
    report.add_section(cohort_definition)

    provenance = Section("Provenance", icon="🧹")
    provenance.add_markdown(
        "These summaries come from `patients_metadata_removed.json` and describe the "
        "metadata-build drops that happened before this report."
    )
    add_optional_table(provenance, provenance_reason_df, "Removed Rows by Reason")
    add_optional_table(provenance, provenance_source_df, "Removed Rows by Source Dataset")
    report.add_section(provenance)

    diagnosis = Section("Diagnosis and Demographics", icon="🧾")
    add_optional_table(diagnosis, diagnosis_df, "Diagnosis Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title == "Diagnosis Prevalence" and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    add_optional_table(diagnosis, combined_diagnosis_df, "Combined Diagnosis Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title == "Combined Diagnosis Counts" and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    add_optional_table(diagnosis, demographics_df, "Demographics Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title in {"Sex by Age Group", "Age by Combined Diagnosis"} and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    report.add_section(diagnosis)

    medication = Section("Medication and Drug Resistance", icon="💊")
    add_optional_table(medication, medication_df, "Medication Summary")
    add_image_list(medication, figures_by_section.get("Medication and Drug Resistance", []))
    report.add_section(medication)

    drug_resistant = Section("Drug Resistant Longitudinal Cohort", icon="🔁")
    drug_resistant.add_markdown(
        "These tables summarize cohort 1 vs cohort 2 overlap and within-patient "
        "first-EEG versus later-EEG availability for drug-resistant analyses."
    )
    if drug_resistant_overview_df is not None:
        add_optional_table(drug_resistant, drug_resistant_overview_df, "Drug-Resistant Overview")
    if first_later_drug_resistant_df is not None:
        add_optional_table(
            drug_resistant,
            first_later_drug_resistant_df,
            "Drug-Resistant First EEG vs Later EEG",
        )
    if drug_resistant_first_later_figure is not None:
        drug_resistant.add_element(PlotlyElement(drug_resistant_first_later_figure, height="420px"))
    add_image_list(drug_resistant, figures_by_section.get("Drug Resistant Longitudinal Cohort", []))
    if source_overlap_df is not None:
        add_optional_table(drug_resistant, source_overlap_df, "Cohort 1 vs Cohort 2 Overlap")
    if longitudinal_drug_resistant_patients_df is not None:
        drug_resistant.add_element(
            TableElement(
                longitudinal_drug_resistant_patients_df,
                title="Drug-Resistant Patients with First and Later Recordings",
            )
        )
    report.add_section(drug_resistant)

    opportunities = Section("Analysis Opportunities", icon="🧠")
    if not valid_opportunities_df.empty:
        opportunities.add_element(
            InteractiveTableElement(
                valid_opportunities_df,
                title="Valid Analysis Opportunities",
                selector_columns=["Sex", "AgeGroup", "Constraint", "Analysis"],
                default_sort={"column": "cohort_n", "direction": "desc"},
            )
        )
    else:
        opportunities.add_markdown("No valid analysis opportunities.")
    report.add_section(opportunities)

    if recruitment_summary_df is not None:
        recruitment = Section("Recruitment Strategy", icon="📈")
        if recruitment_markdown:
            recruitment.add_markdown(recruitment_markdown)
        if recruitment_projection_df is not None and not recruitment_projection_df.empty:
            recruitment_columns = [
                "milestone",
                "family",
                "analysis",
                "constraint",
                "group_1",
                "group_2",
                "projected_n1",
                "projected_n2",
                "required_n1",
                "required_n2",
                "shortfall_n1",
                "shortfall_n2",
                "limiting_group",
            ]
            available_columns = [
                c for c in recruitment_columns if c in recruitment_projection_df.columns
            ]
            recruitment.add_element(
                InteractiveTableElement(
                    recruitment_projection_df[available_columns],
                    title="Recruitment Milestone Explorer",
                    selector_columns=["milestone", "family", "constraint", "analysis"],
                    default_sort={"column": "shortfall_n1", "direction": "desc"},
                )
            )
        if recruitment_pools_df is not None and not recruitment_pools_df.empty:
            pool_columns = [
                "milestone",
                "pool",
                "current_n",
                "target_n",
                "raw_gap",
                "child_planned",
                "net_recruit_needed",
            ]
            available_pool_columns = [c for c in pool_columns if c in recruitment_pools_df.columns]
            recruitment.add_element(
                InteractiveTableElement(
                    recruitment_pools_df[available_pool_columns],
                    title="Recruitment Pools",
                    selector_columns=["milestone"],
                    default_sort={"column": "milestone", "direction": "asc"},
                )
            )
        report.add_section(recruitment)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

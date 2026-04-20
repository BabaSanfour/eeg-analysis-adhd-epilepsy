"""Cohort report generation over cleaned patient metadata."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eeg_adhd_epilepsy.io.table import load
from eeg_adhd_epilepsy.reports.cohort_report import generate_cohort_report
from eeg_adhd_epilepsy.utils.analysis_opportunities_schema import (
    CONSTRAINT_RULES,
    TARGET_ANALYSES,
)
from eeg_adhd_epilepsy.utils.metadata_schema import (
    EPILEPSY_MED_COLS,
    NORMALIZED_PSYCHOSTIMULANT_CATEGORIES,
)
from eeg_adhd_epilepsy.viz.patients import (
    plot_age_by_diagnosis,
    plot_asm_exposure_counts,
    plot_combined_diagnosis_counts,
    plot_diagnosis_prevalence,
    plot_drug_resistance_summary,
    plot_medication_overlap,
    plot_psychostimulant_category_counts,
    plot_sex_age_heatmap,
    plot_source_dataset_counts,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_COHORT_NAME = "full_cohort"
DEFAULT_REPORT_TITLE = "Clean Metadata Cohort Report"
ALL_LABEL = "All"
DEFAULT_RECRUITMENT_MILESTONES = [1500, 2000, 3000, 5000]
DEFAULT_TRACKED_RECRUITMENT_OPPORTUNITIES = [
    {"analysis": "ADHD_Psychostim_Effect_Any", "constraint": "No_Autism"},
    {"analysis": "ADHD_Psychostim_Effect_Any", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "ADHD_Psychostim_Effect_Any", "constraint": "ASM_False"},
    {"analysis": "ADHD_Psychostim_Effect_Methylphenidate", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "ADHD_Psychostim_Effect_Lisdexamfetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "ADHD_Psychostim_Effect_Dextroamphetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "ADHD_Methylphenidate_vs_Lisdexamfetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "ADHD_Methylphenidate_vs_Amphetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "Control_vs_ADHD_Medicated_Any", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "Control_vs_ADHD_Methylphenidate", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "Control_vs_ADHD_Lisdexamfetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "Control_vs_ADHD_Amphetamine", "constraint": "No_Autism+No_Epilepsy"},
    {"analysis": "Epilepsy_ASM_Effect_Any", "constraint": "No_Autism"},
    {"analysis": "Epilepsy_ASM_Effect_Any", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "Epilepsy_ASM_Effect_Any", "constraint": "Psychostim_False"},
    {"analysis": "Epilepsy_ASM_Effect_LEV_Only", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "Epilepsy_ASM_Effect_VPA_Only", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "Epilepsy_LEV_vs_VPA_Only", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "NonEpilepsy_vs_Epilepsy_Unmedicated", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "NonEpilepsy_vs_Epilepsy_ASM_Any", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "NonEpilepsy_vs_Epilepsy_LEV_Only", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "NonEpilepsy_vs_Epilepsy_VPA_Only", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "DrugResistance_Status", "constraint": "No_Constraint"},
    {"analysis": "DrugResistance_Status", "constraint": "No_Autism"},
    {"analysis": "DrugResistance_Status", "constraint": "No_ADHD+No_Autism"},
    {"analysis": "DrugResistance_Status", "constraint": "Psychostim_False"},
]
RECRUITMENT_REQUIRED_GROUPS = {
    "ADHD_Psychostim_Effect_Any": {1500: (100, 150), 2000: (150, 250), 3000: (200, 400), 5000: (200, 800)},
    "ADHD_Psychostim_Effect_Methylphenidate": {1500: (100, 75), 2000: (150, 125), 3000: (200, 200), 5000: (200, 400)},
    "ADHD_Psychostim_Effect_Lisdexamfetamine": {1500: (100, 75), 2000: (150, 125), 3000: (200, 200), 5000: (200, 400)},
    "ADHD_Psychostim_Effect_Dextroamphetamine": {1500: (100, 75), 2000: (150, 125), 3000: (200, 200), 5000: (200, 400)},
    "ADHD_Methylphenidate_vs_Lisdexamfetamine": {1500: (75, 75), 2000: (125, 125), 3000: (200, 200), 5000: (400, 400)},
    "ADHD_Methylphenidate_vs_Amphetamine": {1500: (75, 75), 2000: (125, 125), 3000: (200, 200), 5000: (400, 400)},
    "Control_vs_ADHD_Medicated_Any": {1500: (150, 150), 2000: (250, 250), 3000: (400, 400), 5000: (800, 800)},
    "Control_vs_ADHD_Methylphenidate": {1500: (75, 75), 2000: (125, 125), 3000: (200, 200), 5000: (400, 400)},
    "Control_vs_ADHD_Lisdexamfetamine": {1500: (75, 75), 2000: (125, 125), 3000: (200, 200), 5000: (400, 400)},
    "Control_vs_ADHD_Amphetamine": {1500: (75, 75), 2000: (125, 125), 3000: (200, 200), 5000: (400, 400)},
    "Epilepsy_ASM_Effect_Any": {1500: (100, 250), 2000: (150, 350), 3000: (200, 600), 5000: (200, 1150)},
    "Epilepsy_ASM_Effect_LEV_Only": {1500: (100, 125), 2000: (150, 175), 3000: (200, 300), 5000: (200, 575)},
    "Epilepsy_ASM_Effect_VPA_Only": {1500: (100, 125), 2000: (150, 175), 3000: (200, 300), 5000: (200, 575)},
    "Epilepsy_LEV_vs_VPA_Only": {1500: (125, 125), 2000: (175, 175), 3000: (300, 300), 5000: (575, 575)},
    "NonEpilepsy_vs_Epilepsy_Unmedicated": {1500: (100, 100), 2000: (150, 150), 3000: (200, 200), 5000: (200, 200)},
    "NonEpilepsy_vs_Epilepsy_ASM_Any": {1500: (250, 250), 2000: (350, 350), 3000: (600, 600), 5000: (1150, 1150)},
    "NonEpilepsy_vs_Epilepsy_LEV_Only": {1500: (125, 125), 2000: (175, 175), 3000: (300, 300), 5000: (575, 575)},
    "NonEpilepsy_vs_Epilepsy_VPA_Only": {1500: (125, 125), 2000: (175, 175), 3000: (300, 300), 5000: (575, 575)},
    "DrugResistance_Status": {1500: (150, 100), 2000: (200, 150), 3000: (350, 250), 5000: (700, 450)},
}
RECRUITMENT_POOL_SPECS = [
    {
        "key": "controls",
        "label": "Controls",
        "depth": 0,
        "parent": None,
        "targets": {1500: 350, 2000: 500, 3000: 1000, 5000: 1650},
    },
    {
        "key": "adhd_total",
        "label": "Pure ADHD (Total)",
        "depth": 0,
        "parent": None,
        "targets": {1500: 250, 2000: 400, 3000: 600, 5000: 1000},
    },
    {
        "key": "adhd_unmed",
        "label": "ADHD Unmedicated",
        "depth": 1,
        "parent": "adhd_total",
        "targets": {1500: 100, 2000: 150, 3000: 200, 5000: 200},
    },
    {
        "key": "adhd_med_any",
        "label": "ADHD Medicated Any",
        "depth": 1,
        "parent": "adhd_total",
        "targets": {1500: 150, 2000: 250, 3000: 400, 5000: 800},
    },
    {
        "key": "adhd_methyl",
        "label": "ADHD Methylphenidate",
        "depth": 2,
        "parent": "adhd_med_any",
        "targets": {1500: 75, 2000: 125, 3000: 200, 5000: 400},
    },
    {
        "key": "adhd_amphetamine",
        "label": "ADHD Amphetamine",
        "depth": 2,
        "parent": "adhd_med_any",
        "targets": {1500: 75, 2000: 125, 3000: 200, 5000: 400},
    },
    {
        "key": "epilepsy_total",
        "label": "Pure Epilepsy (Total)",
        "depth": 0,
        "parent": None,
        "targets": {1500: 350, 2000: 500, 3000: 800, 5000: 1350},
    },
    {
        "key": "epilepsy_unmed",
        "label": "Epilepsy Unmedicated",
        "depth": 1,
        "parent": "epilepsy_total",
        "targets": {1500: 100, 2000: 150, 3000: 200, 5000: 200},
    },
    {
        "key": "epilepsy_asm_any",
        "label": "Epilepsy ASM Any",
        "depth": 1,
        "parent": "epilepsy_total",
        "targets": {1500: 250, 2000: 350, 3000: 600, 5000: 1150},
    },
    {
        "key": "epilepsy_lev_only",
        "label": "Epilepsy LEV Only",
        "depth": 2,
        "parent": "epilepsy_asm_any",
        "targets": {1500: 125, 2000: 175, 3000: 300, 5000: 575},
    },
    {
        "key": "epilepsy_vpa_only",
        "label": "Epilepsy VPA Only",
        "depth": 2,
        "parent": "epilepsy_asm_any",
        "targets": {1500: 125, 2000: 175, 3000: 300, 5000: 575},
    },
    {
        "key": "drug_resistant_epilepsy",
        "label": "Drug Resistant Epilepsy",
        "depth": 0,
        "parent": None,
        "targets": {1500: 100, 2000: 150, 3000: 250, 5000: 450},
    },
]
ANALYSIS_OUTPUT_COLUMNS = [
    "Sex",
    "AgeGroup",
    "Constraint",
    "Analysis",
    "Group 1",
    "Group 2",
    "cohort_n",
    "N1",
    "N2",
    "is_valid",
    "skip_reason",
    "dedupe_key",
]
EXCLUSIVE_DIAGNOSIS_CONSTRAINTS = {
    "Control_Only",
    "ADHD_Only",
    "Epilepsy_Only",
    "Autism_Only",
    "ADHD_Epilepsy",
    "ADHD_Autism",
    "Epilepsy_Autism",
    "ADHD_Epilepsy_Autism",
}
SINGLE_STIMULANT_CATEGORY_CONSTRAINTS = {
    "Methylphenidate",
    "Dextroamphetamine",
    "Lisdexamfetamine",
}
DROP_REASON_DISPLAY = {
    "potential_diagnosis": "Non-confirmed diagnosis",
    "non_confirmed_diagnosis": "Non-confirmed diagnosis",
    "invalid_eeg_date": "No EEG files",
    "no_eeg_files": "No EEG files",
    "missing_adhd_or_autism": "Missing diagnosis",
    "missing_diagnosis": "Missing diagnosis",
    "duplicate_same_source_patient_id": "Duplicate same-source patient",
    "medication_mismatch": "Medication mismatch",
}


def _load_cohort_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {
            "name": DEFAULT_COHORT_NAME,
            "title": DEFAULT_REPORT_TITLE,
            "row_filter": [],
            "min_group_n": 1,
            "recruitment": {
                "title": "Recruitment Strategy",
                "milestones": DEFAULT_RECRUITMENT_MILESTONES,
                "tracked_opportunities": None,
            },
        }

    with config_path.open("r", encoding="utf-8") as fobj:
        raw = yaml.safe_load(fobj) or {}
    recruitment_raw = raw.get("recruitment") or {}
    milestones = recruitment_raw.get("milestones") or DEFAULT_RECRUITMENT_MILESTONES
    tracked = recruitment_raw.get("tracked_opportunities")

    return {
        "name": raw.get("name") or config_path.stem,
        "title": raw.get("title") or raw.get("name") or DEFAULT_REPORT_TITLE,
        "row_filter": raw.get("row_filter") or [],
        "min_group_n": int(raw.get("min_group_n", 1)),
        "recruitment": {
            "title": recruitment_raw.get("title") or "Recruitment Strategy",
            "milestones": [int(value) for value in milestones],
            "tracked_opportunities": tracked or None,
        },
    }


def _values_for_filter(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple, set, pd.Index)):
        return list(values)
    return [values]


def _apply_row_filter(df: pd.DataFrame, row_filter: list[dict[str, Any]]) -> pd.DataFrame:
    if not row_filter:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for rule in row_filter:
        column = rule["column"]
        operator = str(rule.get("operator", "==")).strip()
        values = rule.get("values")
        series = df[column]

        if operator == "==":
            current = series.isin(_values_for_filter(values)) if isinstance(values, list) else series == values
        elif operator == "!=":
            current = ~series.isin(_values_for_filter(values)) if isinstance(values, list) else series != values
        elif operator == "in":
            current = series.isin(_values_for_filter(values))
        elif operator == "not in":
            current = ~series.isin(_values_for_filter(values))
        elif operator == ">":
            current = series > values
        elif operator == ">=":
            current = series >= values
        elif operator == "<":
            current = series < values
        elif operator == "<=":
            current = series <= values
        else:
            raise ValueError(f"Unsupported row_filter operator: {operator!r}")

        mask &= current.fillna(False)

    return df.loc[mask].copy()


def _format_row_filter_markdown(config: dict[str, Any], n_rows: int) -> str:
    lines = [
        f"**Cohort Name:** {config['name']}",
        f"**Rows in Cohort:** {n_rows}",
        f"**Minimum Group Size:** {config['min_group_n']}",
    ]
    if not config["row_filter"]:
        lines.append("**Row Filter:** full clean cohort")
        return "\n\n".join(lines)

    lines.append("**Row Filter:**")
    for rule in config["row_filter"]:
        lines.append(
            f"- `{rule['column']}` {rule.get('operator', '==')} `{rule.get('values')}`"
        )
    return "\n\n".join(lines)


def _load_removed_json(removed_json: Path | None) -> dict[str, Any] | None:
    if removed_json is None or not removed_json.exists():
        return None
    with removed_json.open("r", encoding="utf-8") as fobj:
        return json.load(fobj)


def _metric_table(rows: list[tuple[str, Any]], value_name: str = "value") -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", value_name])


def _build_cohort_summary_table(
    df: pd.DataFrame,
    cohort_config: dict[str, Any],
    metadata_path: Path,
) -> pd.DataFrame:
    source_counts = df["source_dataset"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "cohort_name": cohort_config["name"],
                "report_title": cohort_config["title"],
                "metadata_csv": str(metadata_path),
                "rows": len(df),
                "source_datasets": df["source_dataset"].nunique(),
                "source_dataset_counts": ", ".join(
                    f"{name}={source_counts.get(name, 0)}" for name in sorted(source_counts)
                ),
                "mean_age": round(float(df["age"].mean()), 2) if not df.empty else None,
                "median_age": round(float(df["age"].median()), 2) if not df.empty else None,
            }
        ]
    )


def _build_provenance_tables(
    removed_data: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not removed_data:
        empty = pd.DataFrame(columns=["drop_reason", "count"])
        return empty, pd.DataFrame(columns=["source_dataset", "count"])

    summary = removed_data.get("summary", {})
    reason_df = (
        pd.Series(summary.get("drop_reason_counts", {}), name="count")
        .rename_axis("drop_reason")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    if not reason_df.empty:
        reason_df["drop_reason"] = reason_df["drop_reason"].map(
            lambda value: DROP_REASON_DISPLAY.get(str(value), str(value))
        )
    source_df = (
        pd.Series(summary.get("source_dataset_counts", {}), name="count")
        .rename_axis("source_dataset")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return reason_df, source_df


def _build_diagnosis_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in [
        ("adhd", "ADHD"),
        ("autism", "Autism"),
        ("epilepsy", "Epilepsy"),
    ]:
        positive = int(df[column].fillna(0).sum())
        rows.append(
            {
                "diagnosis": label,
                "count": positive,
                "prevalence_pct": round(positive / len(df) * 100, 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_combined_diagnosis_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["combined_diagnosis"]
        .value_counts()
        .rename_axis("combined_diagnosis")
        .reset_index(name="count")
    )


def _build_demographics_summary(df: pd.DataFrame) -> pd.DataFrame:
    sex_counts = df["sex"].value_counts().to_dict()
    age_group_counts = (
        df["age_group"].astype(str).value_counts().to_dict()
        if "age_group" in df.columns
        else {}
    )
    return _metric_table(
        [
            ("rows", len(df)),
            ("female_rows", int(sex_counts.get("F", 0))),
            ("male_rows", int(sex_counts.get("M", 0))),
            ("mean_age", round(float(df["age"].mean()), 2) if len(df) else None),
            ("median_age", round(float(df["age"].median()), 2) if len(df) else None),
            (
                "age_group_counts",
                ", ".join(
                    f"{group}={count}"
                    for group, count in sorted(
                        age_group_counts.items(),
                        key=lambda item: int(item[0].split("-")[0]) if item[0] and item[0] != "nan" else 999,
                    )
                    if group != "nan"
                ),
            ),
        ]
    )


def _build_medication_summary(df: pd.DataFrame) -> pd.DataFrame:
    psych_counts = df["psychostimulant_category"].value_counts().to_dict()
    asm_counts = {column: int(df[column].fillna(0).sum()) for column in EPILEPSY_MED_COLS}
    asm_counts = {key: value for key, value in asm_counts.items() if value > 0}

    rows = [
        ("psychostimulant_rows", int(df["psychostimulant"].fillna(0).sum())),
        ("asm_rows", int(df["asm"].fillna(0).sum())),
        ("asm_resistant_rows", int(df["asm_resistant"].fillna(0).sum())),
        ("other_asm_rows", int(df["other_asm"].fillna(0).sum())),
        (
            "psychostimulant_categories",
            ", ".join(
                f"{name}={psych_counts.get(name, 0)}"
                for name in NORMALIZED_PSYCHOSTIMULANT_CATEGORIES
                if psych_counts.get(name, 0) > 0
            ),
        ),
        (
            "top_asm_counts",
            ", ".join(f"{name}={count}" for name, count in sorted(asm_counts.items(), key=lambda item: item[1], reverse=True)),
        ),
    ]
    return _metric_table(rows)


def _resolve_recruitment_selectors(
    valid_opportunities: pd.DataFrame,
    recruitment_config: dict[str, Any],
) -> pd.DataFrame:
    selectors = (
        recruitment_config["tracked_opportunities"]
        or DEFAULT_TRACKED_RECRUITMENT_OPPORTUNITIES
    )
    rows: list[dict[str, Any]] = []

    for selector in selectors:
        analysis = selector["analysis"]
        constraint = selector["constraint"]
        sex = selector.get("sex", ALL_LABEL)
        age_group = selector.get("age_group", ALL_LABEL)

        matches = valid_opportunities.loc[
            (valid_opportunities["Analysis"] == analysis)
            & (valid_opportunities["Constraint"] == constraint)
            & (valid_opportunities["Sex"] == sex)
            & (valid_opportunities["AgeGroup"] == age_group)
        ].copy()

        if matches.empty:
            raise ValueError(
                "Recruitment selector did not match any valid opportunity: "
                f"{analysis} | {constraint} | {sex} | {age_group}"
            )
        if len(matches) > 1:
            raise ValueError(
                "Recruitment selector matched more than one valid opportunity: "
                f"{analysis} | {constraint} | {sex} | {age_group}"
            )

        match = matches.iloc[0]
        rows.append(
            {
                "analysis": analysis,
                "constraint": constraint,
                "sex": sex,
                "age_group": age_group,
                "group_1": match["Group 1"],
                "group_2": match["Group 2"],
                "current_n1": int(match["N1"]),
                "current_n2": int(match["N2"]),
                "current_cohort_n": int(match["cohort_n"]),
            }
        )

    return pd.DataFrame(rows)


def _recruitment_opportunity_label(row: pd.Series) -> str:
    label = f"{row['analysis']} | {row['constraint']}"
    if row["sex"] != ALL_LABEL:
        label += f" | {row['sex']}"
    if row["age_group"] != ALL_LABEL:
        label += f" | {row['age_group']}"
    return label


def _recruitment_limiting_group(row: pd.Series) -> str:
    if int(row["shortfall_n1"]) == 0 and int(row["shortfall_n2"]) == 0:
        return "none"
    if int(row["shortfall_n1"]) > 0 and int(row["shortfall_n2"]) > 0:
        return "both"
    if int(row["shortfall_n1"]) > 0:
        return str(row["group_1"])
    return str(row["group_2"])


def _analysis_family(analysis_name: str) -> str:
    if analysis_name == "DrugResistance_Status":
        return "Drug Resistance"
    if analysis_name.startswith("ADHD_") or analysis_name.startswith("Control_vs_ADHD"):
        return "ADHD"
    if analysis_name.startswith("Epilepsy_") or analysis_name.startswith("NonEpilepsy_vs_Epilepsy"):
        return "Epilepsy"
    return "Other"


def _recruitment_pool_mask(df: pd.DataFrame, pool_key: str) -> pd.Series:
    if pool_key == "controls":
        return df["combined_diagnosis"] == "Control"
    if pool_key == "adhd_total":
        return (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 0)
    if pool_key == "adhd_unmed":
        return (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 0) & (df["psychostimulant"] == 0)
    if pool_key == "adhd_med_any":
        return (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 0) & (df["psychostimulant"] == 1)
    if pool_key == "adhd_methyl":
        return (
            (df["adhd"] == 1)
            & (df["autism"] == 0)
            & (df["epilepsy"] == 0)
            & (df["psychostimulant_category"] == "Methylphenidate")
        )
    if pool_key == "adhd_amphetamine":
        return (
            (df["adhd"] == 1)
            & (df["autism"] == 0)
            & (df["epilepsy"] == 0)
            & (df["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"]))
        )
    if pool_key == "epilepsy_total":
        return (df["epilepsy"] == 1) & (df["adhd"] == 0) & (df["autism"] == 0)
    if pool_key == "epilepsy_unmed":
        return (df["epilepsy"] == 1) & (df["adhd"] == 0) & (df["autism"] == 0) & (df["asm"] == 0)
    if pool_key == "epilepsy_asm_any":
        return (df["epilepsy"] == 1) & (df["adhd"] == 0) & (df["autism"] == 0) & (df["asm"] == 1)
    if pool_key == "epilepsy_lev_only":
        return (df["epilepsy"] == 1) & (df["adhd"] == 0) & (df["autism"] == 0) & (df["asm_types"] == "LEV")
    if pool_key == "epilepsy_vpa_only":
        return (df["epilepsy"] == 1) & (df["adhd"] == 0) & (df["autism"] == 0) & (df["asm_types"] == "VPA")
    if pool_key == "drug_resistant_epilepsy":
        return (df["epilepsy"] == 1) & (df["asm_resistant"] == 1)
    raise KeyError(f"Unknown recruitment pool key: {pool_key}")


def _required_group_sizes(analysis_name: str, milestone: int) -> tuple[int, int]:
    analysis_targets = RECRUITMENT_REQUIRED_GROUPS.get(analysis_name)
    if analysis_targets is None:
        raise ValueError(f"No recruitment targets defined for analysis {analysis_name!r}.")
    if int(milestone) not in analysis_targets:
        raise ValueError(
            f"No recruitment targets defined for analysis {analysis_name!r} at milestone {milestone}."
        )
    required_n1, required_n2 = analysis_targets[int(milestone)]
    return int(required_n1), int(required_n2)


def _current_feasibility_milestone(filtered_cohort_n: int, milestones: list[int]) -> int:
    eligible = [int(value) for value in milestones if int(value) <= int(filtered_cohort_n)]
    if eligible:
        return max(eligible)
    return min(int(value) for value in milestones)


def _build_recruitment_projection(
    tracked_df: pd.DataFrame,
    filtered_cohort_n: int,
    milestones: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if filtered_cohort_n <= 0:
        raise ValueError("Recruitment projection requires a non-empty filtered cohort.")

    current_milestone = _current_feasibility_milestone(filtered_cohort_n, milestones)
    current_required = tracked_df["analysis"].map(
        lambda analysis: _required_group_sizes(str(analysis), current_milestone)
    )
    current_feasible = pd.Series(
        [
            int(row["current_n1"]) >= required[0] and int(row["current_n2"]) >= required[1]
            for (_, row), required in zip(tracked_df.iterrows(), current_required.tolist())
        ],
        index=tracked_df.index,
    )
    current_feasible_count = int(current_feasible.sum())

    projection_rows: list[dict[str, Any]] = []
    for milestone in milestones:
        rows_to_add = max(0, int(milestone) - filtered_cohort_n)
        scale = max(1.0, float(milestone) / float(filtered_cohort_n))

        for _, tracked_row in tracked_df.iterrows():
            projected_n1 = math.floor(int(tracked_row["current_n1"]) * scale)
            projected_n2 = math.floor(int(tracked_row["current_n2"]) * scale)
            required_n1, required_n2 = _required_group_sizes(
                str(tracked_row["analysis"]),
                int(milestone),
            )
            shortfall_n1 = max(0, required_n1 - projected_n1)
            shortfall_n2 = max(0, required_n2 - projected_n2)
            row = {
                "milestone": int(milestone),
                "rows_to_add": rows_to_add,
                "analysis": tracked_row["analysis"],
                "constraint": tracked_row["constraint"],
                "sex": tracked_row["sex"],
                "age_group": tracked_row["age_group"],
                "group_1": tracked_row["group_1"],
                "group_2": tracked_row["group_2"],
                "current_n1": int(tracked_row["current_n1"]),
                "current_n2": int(tracked_row["current_n2"]),
                "current_cohort_n": int(tracked_row["current_cohort_n"]),
                "filtered_cohort_n": int(filtered_cohort_n),
                "projected_n1": projected_n1,
                "projected_n2": projected_n2,
                "required_n1": required_n1,
                "required_n2": required_n2,
                "shortfall_n1": shortfall_n1,
                "shortfall_n2": shortfall_n2,
                "need_total": shortfall_n1 + shortfall_n2,
                "feasible": shortfall_n1 == 0 and shortfall_n2 == 0,
            }
            row["limiting_group"] = _recruitment_limiting_group(pd.Series(row))
            row["opportunity_label"] = _recruitment_opportunity_label(pd.Series(row))
            projection_rows.append(row)

    projection_df = pd.DataFrame(projection_rows).sort_values(
        ["milestone", "analysis", "constraint", "sex", "age_group"]
    ).reset_index(drop=True)

    previous_feasible = {
        row["opportunity_label"]: (
            int(row["current_n1"]) >= _required_group_sizes(str(row["analysis"]), current_milestone)[0]
            and int(row["current_n2"]) >= _required_group_sizes(str(row["analysis"]), current_milestone)[1]
        )
        for _, row in tracked_df.assign(
            opportunity_label=tracked_df.apply(_recruitment_opportunity_label, axis=1)
        ).iterrows()
    }
    summary_rows: list[dict[str, Any]] = []
    newly_feasible_labels_by_milestone: dict[int, set[str]] = {}
    for milestone in milestones:
        milestone_df = projection_df.loc[projection_df["milestone"] == int(milestone)].copy()
        feasible_map = {
            row["opportunity_label"]: bool(row["feasible"])
            for _, row in milestone_df.iterrows()
        }
        newly_feasible_labels = {
            label
            for label, feasible in feasible_map.items()
            if feasible and not previous_feasible.get(label, False)
        }
        newly_feasible = len(newly_feasible_labels)
        newly_feasible_labels_by_milestone[int(milestone)] = newly_feasible_labels
        summary_rows.append(
            {
                "milestone": int(milestone),
                "rows_to_add": max(0, int(milestone) - filtered_cohort_n),
                "tracked_analyses": len(milestone_df),
                "feasible_count": int(milestone_df["feasible"].sum()),
                "newly_feasible_count": int(newly_feasible),
            }
        )
        previous_feasible = feasible_map

    summary_df = pd.DataFrame(summary_rows)
    projection_df["family"] = projection_df["analysis"].map(_analysis_family)
    projection_df["newly_feasible"] = projection_df.apply(
        lambda row: row["opportunity_label"] in newly_feasible_labels_by_milestone.get(int(row["milestone"]), set()),
        axis=1,
    )
    projection_df["status"] = projection_df.apply(
        lambda row: "Newly Feasible"
        if bool(row["newly_feasible"])
        else ("Feasible" if bool(row["feasible"]) else "Not Feasible"),
        axis=1,
    )
    projection_df = projection_df.sort_values(
        ["milestone", "family", "analysis", "constraint", "sex", "age_group"]
    ).reset_index(drop=True)

    return projection_df, summary_df, current_feasible_count, current_milestone


def _build_recruitment_pools(
    df: pd.DataFrame,
    milestones: list[int],
) -> pd.DataFrame:
    child_map: dict[str, list[str]] = {}
    spec_by_key = {spec["key"]: spec for spec in RECRUITMENT_POOL_SPECS}
    for spec in RECRUITMENT_POOL_SPECS:
        if spec["parent"] is not None:
            child_map.setdefault(str(spec["parent"]), []).append(str(spec["key"]))

    rows: list[dict[str, Any]] = []
    for milestone in milestones:
        milestone_rows: list[dict[str, Any]] = []
        for spec in RECRUITMENT_POOL_SPECS:
            current_n = int(_recruitment_pool_mask(df, str(spec["key"])).sum())
            target_n = int(spec["targets"][int(milestone)])
            milestone_rows.append(
                {
                    "milestone": int(milestone),
                    "pool_key": str(spec["key"]),
                    "pool": ("  " * int(spec["depth"])) + ("↳ " if int(spec["depth"]) > 0 else "") + str(spec["label"]),
                    "depth": int(spec["depth"]),
                    "parent": spec["parent"],
                    "current_n": current_n,
                    "target_n": target_n,
                    "raw_gap": max(0, target_n - current_n),
                    "child_planned": 0,
                    "net_recruit_needed": 0,
                }
            )

        row_map = {row["pool_key"]: row for row in milestone_rows}
        for spec in reversed(RECRUITMENT_POOL_SPECS):
            row = row_map[str(spec["key"])]
            child_keys = child_map.get(str(spec["key"]), [])
            child_planned = sum(int(row_map[child_key]["net_recruit_needed"]) for child_key in child_keys)
            row["child_planned"] = child_planned
            row["net_recruit_needed"] = max(0, int(row["raw_gap"]) - child_planned)

        rows.extend(milestone_rows)

    return pd.DataFrame(rows)


def _build_recruitment_markdown(
    filtered_cohort_n: int,
    recruitment_config: dict[str, Any],
    tracked_count: int,
    current_feasible_count: int,
    current_milestone: int,
) -> str:
    milestones = "/".join(str(int(value)) for value in recruitment_config["milestones"])
    return "\n\n".join(
        [
            f"**Filtered Cohort Size:** {filtered_cohort_n}",
            f"**Milestones:** {milestones}",
            "**Projection Model:** proportional projection from the current filtered cohort",
            f"**Tracked Main Analyses:** {tracked_count}",
            f"**Current Feasibility Benchmark:** N={int(current_milestone)} analysis targets",
            f"**Currently Feasible Analyses:** {current_feasible_count}",
        ]
    )


def _build_recruitment_outputs(
    valid_opportunities: pd.DataFrame,
    cohort_df: pd.DataFrame,
    filtered_cohort_n: int,
    recruitment_config: dict[str, Any],
) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    tracked_df = _resolve_recruitment_selectors(valid_opportunities, recruitment_config)
    projection_df, summary_df, current_feasible_count, current_milestone = _build_recruitment_projection(
        tracked_df=tracked_df,
        filtered_cohort_n=filtered_cohort_n,
        milestones=recruitment_config["milestones"],
    )
    pools_df = _build_recruitment_pools(
        df=cohort_df,
        milestones=recruitment_config["milestones"],
    )

    recruitment_markdown = _build_recruitment_markdown(
        filtered_cohort_n=filtered_cohort_n,
        recruitment_config=recruitment_config,
        tracked_count=len(tracked_df),
        current_feasible_count=current_feasible_count,
        current_milestone=current_milestone,
    )
    return recruitment_markdown, projection_df, summary_df, pools_df, current_milestone


def _constraint_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "No_Constraint":
        return pd.Series(True, index=df.index)
    if name == "No_ADHD":
        return df["adhd"] == 0
    if name == "No_Autism":
        return df["autism"] == 0
    if name == "No_Epilepsy":
        return df["epilepsy"] == 0
    if name == "Psychostim_True":
        return df["psychostimulant"] == 1
    if name == "Psychostim_False":
        return df["psychostimulant"] == 0
    if name == "ASM_True":
        return df["asm"] == 1
    if name == "ASM_False":
        return df["asm"] == 0
    if name == "ASM_Resistant_True":
        return df["asm_resistant"] == 1
    if name == "ASM_Resistant_False":
        return df["asm_resistant"] == 0
    if name == "Control_Only":
        return (df["adhd"] == 0) & (df["autism"] == 0) & (df["epilepsy"] == 0)
    if name == "ADHD_Only":
        return (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 0)
    if name == "Epilepsy_Only":
        return (df["adhd"] == 0) & (df["autism"] == 0) & (df["epilepsy"] == 1)
    if name == "Autism_Only":
        return (df["adhd"] == 0) & (df["autism"] == 1) & (df["epilepsy"] == 0)
    if name == "ADHD_Epilepsy":
        return (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 1)
    if name == "ADHD_Autism":
        return (df["adhd"] == 1) & (df["autism"] == 1) & (df["epilepsy"] == 0)
    if name == "Epilepsy_Autism":
        return (df["adhd"] == 0) & (df["autism"] == 1) & (df["epilepsy"] == 1)
    if name == "ADHD_Epilepsy_Autism":
        return (df["adhd"] == 1) & (df["autism"] == 1) & (df["epilepsy"] == 1)
    if name == "Methylphenidate":
        return df["psychostimulant_category"] == "Methylphenidate"
    if name == "Dextroamphetamine":
        return df["psychostimulant_category"] == "Dextroamphetamine"
    if name == "Lisdexamfetamine":
        return df["psychostimulant_category"] == "Lisdexamfetamine"
    if name == "Combined_Amphetamine":
        return df["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"])
    raise KeyError(f"Unknown constraint: {name}")


def _constraint_status(constraint_names: tuple[str, ...]) -> str | None:
    names = set(constraint_names)
    if len(names & EXCLUSIVE_DIAGNOSIS_CONSTRAINTS) > 1:
        return "contradictory_constraints"

    explicit_categories = names & SINGLE_STIMULANT_CATEGORY_CONSTRAINTS
    if len(explicit_categories) > 1:
        return "contradictory_constraints"
    if "Combined_Amphetamine" in names and "Methylphenidate" in names:
        return "contradictory_constraints"
    if "Combined_Amphetamine" in names and len(names & {"Dextroamphetamine", "Lisdexamfetamine"}) == 2:
        return "contradictory_constraints"
    if "Combined_Amphetamine" in names and len(names & {"Dextroamphetamine", "Lisdexamfetamine"}) == 1:
        return "redundant_constraint_set"

    for rule in CONSTRAINT_RULES:
        if not set(rule.if_all).issubset(names):
            continue
        if rule.contradicts:
            return "contradictory_constraints"
        implied_present = [item for item in rule.implies if item in names and item not in rule.if_all]
        if implied_present:
            return "redundant_constraint_set"
    return None


def _constraint_label(constraint_names: tuple[str, ...]) -> str:
    if constraint_names == ("No_Constraint",):
        return "No_Constraint"
    return "+".join(constraint_names)


def _ordered_age_groups(df: pd.DataFrame) -> list[str]:
    values = [value for value in df["age_group"].dropna().astype(str).unique().tolist() if value]
    return sorted(values, key=lambda value: int(value.split("-")[0]))


def _analysis_group_masks(
    subset: pd.DataFrame,
    analysis_name: str,
) -> tuple[pd.Series, pd.Series, str, str] | None:
    if analysis_name == "DrugResistance_Status":
        return (
            (subset["epilepsy"] == 1) & (subset["asm"] == 1) & (subset["asm_resistant"] == 0),
            (subset["epilepsy"] == 1) & (subset["asm_resistant"] == 1),
            "Not Resistant",
            "Resistant",
        )
    if analysis_name == "Epilepsy_ASM_Effect_Any":
        return (
            (subset["epilepsy"] == 1) & (subset["asm"] == 0),
            (subset["epilepsy"] == 1) & (subset["asm"] == 1),
            "Epilepsy Unmedicated",
            "Epilepsy on ASM",
        )
    if analysis_name == "Epilepsy_ASM_Effect_LEV_Only":
        return (
            (subset["epilepsy"] == 1) & (subset["asm"] == 0),
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "LEV"),
            "Epilepsy Unmedicated",
            "LEV Only",
        )
    if analysis_name == "Epilepsy_ASM_Effect_VPA_Only":
        return (
            (subset["epilepsy"] == 1) & (subset["asm"] == 0),
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "VPA"),
            "Epilepsy Unmedicated",
            "VPA Only",
        )
    if analysis_name == "Epilepsy_LEV_vs_VPA_Only":
        return (
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "LEV"),
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "VPA"),
            "LEV Only",
            "VPA Only",
        )
    if analysis_name == "NonEpilepsy_vs_Epilepsy_Unmedicated":
        return (
            subset["epilepsy"] == 0,
            (subset["epilepsy"] == 1) & (subset["asm"] == 0),
            "Non Epilepsy",
            "Epilepsy Unmedicated",
        )
    if analysis_name == "NonEpilepsy_vs_Epilepsy_ASM_Any":
        return (
            subset["epilepsy"] == 0,
            (subset["epilepsy"] == 1) & (subset["asm"] == 1),
            "Non Epilepsy",
            "Epilepsy on ASM",
        )
    if analysis_name == "NonEpilepsy_vs_Epilepsy_LEV_Only":
        return (
            subset["epilepsy"] == 0,
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "LEV"),
            "Non Epilepsy",
            "LEV Only",
        )
    if analysis_name == "NonEpilepsy_vs_Epilepsy_VPA_Only":
        return (
            subset["epilepsy"] == 0,
            (subset["epilepsy"] == 1) & (subset["asm_types"] == "VPA"),
            "Non Epilepsy",
            "VPA Only",
        )
    if analysis_name == "ADHD_Psychostim_Effect_Any":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 0),
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 1),
            "ADHD Unmedicated",
            "ADHD Medicated",
        )
    if analysis_name == "ADHD_Psychostim_Effect_Methylphenidate":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 0),
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Methylphenidate"),
            "ADHD Unmedicated",
            "Methylphenidate",
        )
    if analysis_name == "ADHD_Psychostim_Effect_Lisdexamfetamine":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 0),
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Lisdexamfetamine"),
            "ADHD Unmedicated",
            "Lisdexamfetamine",
        )
    if analysis_name == "ADHD_Psychostim_Effect_Dextroamphetamine":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 0),
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Dextroamphetamine"),
            "ADHD Unmedicated",
            "Dextroamphetamine",
        )
    if analysis_name == "ADHD_Methylphenidate_vs_Lisdexamfetamine":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Methylphenidate"),
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Lisdexamfetamine"),
            "Methylphenidate",
            "Lisdexamfetamine",
        )
    if analysis_name == "ADHD_Methylphenidate_vs_Amphetamine":
        return (
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Methylphenidate"),
            (subset["adhd"] == 1)
            & (subset["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"])),
            "Methylphenidate",
            "Amphetamine",
        )
    if analysis_name == "Control_vs_ADHD_Medicated_Any":
        return (
            subset["combined_diagnosis"] == "Control",
            (subset["adhd"] == 1) & (subset["psychostimulant"] == 1),
            "Controls",
            "Any Medicated ADHD",
        )
    if analysis_name == "Control_vs_ADHD_Methylphenidate":
        return (
            subset["combined_diagnosis"] == "Control",
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Methylphenidate"),
            "Controls",
            "Methylphenidate",
        )
    if analysis_name == "Control_vs_ADHD_Lisdexamfetamine":
        return (
            subset["combined_diagnosis"] == "Control",
            (subset["adhd"] == 1) & (subset["psychostimulant_category"] == "Lisdexamfetamine"),
            "Controls",
            "Lisdexamfetamine",
        )
    if analysis_name == "Control_vs_ADHD_Amphetamine":
        return (
            subset["combined_diagnosis"] == "Control",
            (subset["adhd"] == 1)
            & (subset["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"])),
            "Controls",
            "Amphetamine",
        )
    return None


def _analysis_applicability_reason(subset: pd.DataFrame, analysis_name: str) -> str | None:
    return None


def _group_empty_reason(analysis_name: str, n1: int, n2: int) -> str:
    if analysis_name in {
        "Epilepsy_ASM_Effect_LEV_Only",
        "Epilepsy_ASM_Effect_VPA_Only",
        "Epilepsy_LEV_vs_VPA_Only",
        "NonEpilepsy_vs_Epilepsy_LEV_Only",
        "NonEpilepsy_vs_Epilepsy_VPA_Only",
        "ADHD_Psychostim_Effect_Methylphenidate",
        "ADHD_Psychostim_Effect_Lisdexamfetamine",
        "ADHD_Psychostim_Effect_Dextroamphetamine",
        "ADHD_Methylphenidate_vs_Lisdexamfetamine",
        "ADHD_Methylphenidate_vs_Amphetamine",
        "Control_vs_ADHD_Methylphenidate",
        "Control_vs_ADHD_Lisdexamfetamine",
        "Control_vs_ADHD_Amphetamine",
    }:
        return "insufficient_category_support"
    if analysis_name in {
        "NonEpilepsy_vs_Epilepsy_Unmedicated",
        "NonEpilepsy_vs_Epilepsy_ASM_Any",
        "Control_vs_ADHD_Medicated_Any",
    } and n2 == 0:
        return "insufficient_category_support"
    return "one_group_empty"


def _dedupe_key(analysis_name: str, group_1: pd.DataFrame, group_2: pd.DataFrame) -> str:
    ids_1 = ",".join(str(int(value)) for value in sorted(group_1["study_id"].astype(int).tolist()))
    ids_2 = ",".join(str(int(value)) for value in sorted(group_2["study_id"].astype(int).tolist()))
    return f"{analysis_name}|{ids_1}|{ids_2}"


def build_analysis_opportunities(df: pd.DataFrame, min_group_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_valid_keys: set[str] = set()

    sex_groups = [
        (ALL_LABEL, pd.Series(True, index=df.index)),
        ("Male", df["sex"] == "M"),
        ("Female", df["sex"] == "F"),
    ]
    age_groups = [(ALL_LABEL, pd.Series(True, index=df.index))]
    age_groups.extend(
        (age_group, df["age_group"] == age_group) for age_group in _ordered_age_groups(df)
    )

    for sex_label, sex_mask in sex_groups:
        sex_df = df.loc[sex_mask].copy()
        for age_label, age_mask in age_groups:
            stratum_df = sex_df if age_label == ALL_LABEL else sex_df.loc[age_mask.loc[sex_df.index]].copy()
            for analysis_spec in TARGET_ANALYSES:
                for constraint_names in analysis_spec.constraint_sets:
                    constraint_label = _constraint_label(constraint_names)
                    base_row = {
                        "Sex": sex_label,
                        "AgeGroup": age_label,
                        "Constraint": constraint_label,
                        "cohort_n": 0,
                        "N1": 0,
                        "N2": 0,
                        "is_valid": False,
                        "skip_reason": None,
                        "dedupe_key": "",
                    }

                    constraint_status = _constraint_status(constraint_names)
                    if constraint_status is None:
                        constraint_mask = pd.Series(True, index=stratum_df.index)
                        if constraint_names != ("No_Constraint",):
                            for name in constraint_names:
                                constraint_mask &= _constraint_mask(stratum_df, name).fillna(False)
                        cohort_df = stratum_df.loc[constraint_mask].copy()
                    else:
                        cohort_df = stratum_df.iloc[0:0].copy()

                    row = {
                        **base_row,
                        "Analysis": analysis_spec.name,
                        "Group 1": analysis_spec.group_1,
                        "Group 2": analysis_spec.group_2,
                    }

                    if constraint_status is not None:
                        row["skip_reason"] = constraint_status
                        rows.append(row)
                        continue

                    row["cohort_n"] = len(cohort_df)
                    if cohort_df.empty:
                        row["skip_reason"] = "empty_filtered_cohort"
                        rows.append(row)
                        continue

                    applicability_reason = _analysis_applicability_reason(cohort_df, analysis_spec.name)
                    if applicability_reason is not None:
                        row["skip_reason"] = applicability_reason
                        rows.append(row)
                        continue

                    group_masks = _analysis_group_masks(cohort_df, analysis_spec.name)
                    if group_masks is None:
                        row["skip_reason"] = "analysis_not_applicable"
                        rows.append(row)
                        continue

                    mask_1, mask_2, label_1, label_2 = group_masks
                    group_1 = cohort_df.loc[mask_1].copy()
                    group_2 = cohort_df.loc[mask_2].copy()
                    row["Group 1"] = label_1
                    row["Group 2"] = label_2
                    row["N1"] = len(group_1)
                    row["N2"] = len(group_2)

                    if row["N1"] == 0 or row["N2"] == 0:
                        row["skip_reason"] = _group_empty_reason(analysis_spec.name, row["N1"], row["N2"])
                        rows.append(row)
                        continue

                    if row["N1"] < min_group_n or row["N2"] < min_group_n:
                        row["skip_reason"] = "too_small_group"
                        rows.append(row)
                        continue

                    ids_1 = tuple(sorted(group_1["study_id"].astype(int).tolist()))
                    ids_2 = tuple(sorted(group_2["study_id"].astype(int).tolist()))
                    if ids_1 == ids_2:
                        row["skip_reason"] = "same_group_membership"
                        rows.append(row)
                        continue

                    dedupe_key = _dedupe_key(analysis_spec.name, group_1, group_2)
                    row["dedupe_key"] = dedupe_key
                    if dedupe_key in seen_valid_keys:
                        row["skip_reason"] = "redundant_constraint_set"
                        rows.append(row)
                        continue

                    row["is_valid"] = True
                    seen_valid_keys.add(dedupe_key)
                    rows.append(row)

    opportunities = pd.DataFrame(rows, columns=ANALYSIS_OUTPUT_COLUMNS)
    return opportunities.sort_values(
        ["is_valid", "Analysis", "Sex", "AgeGroup", "Constraint"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def _create_figures(df: pd.DataFrame, figure_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    sections = {
        "Cohort Definition": [
            ("Source Dataset Counts", figure_dir / "source_dataset_counts.png", plot_source_dataset_counts),
        ],
        "Diagnosis and Demographics": [
            ("Diagnosis Prevalence", figure_dir / "diagnosis_prevalence.png", plot_diagnosis_prevalence),
            ("Combined Diagnosis Counts", figure_dir / "combined_diagnosis_counts.png", plot_combined_diagnosis_counts),
            ("Sex by Age Group", figure_dir / "sex_age_heatmap.png", plot_sex_age_heatmap),
            ("Age by Combined Diagnosis", figure_dir / "age_by_diagnosis.png", plot_age_by_diagnosis),
        ],
        "Medication and Drug Resistance": [
            ("Psychostimulant Categories", figure_dir / "psychostimulant_category_counts.png", plot_psychostimulant_category_counts),
            ("ASM Exposure Counts", figure_dir / "asm_exposure_counts.png", plot_asm_exposure_counts),
            ("Psychostimulant vs ASM Exposure", figure_dir / "medication_overlap.png", plot_medication_overlap),
            ("Drug Resistance Summary", figure_dir / "drug_resistance_summary.png", plot_drug_resistance_summary),
        ],
    }

    rendered: dict[str, list[tuple[str, Path]]] = {}
    for section_name, figures in sections.items():
        rendered[section_name] = []
        for title, out_path, plot_fn in figures:
            plot_fn(df, out_path)
            rendered[section_name].append((title, out_path))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a cohort report over cleaned patient metadata.")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Path to patients_metadata_clean.csv")
    parser.add_argument(
        "--removed_json",
        type=Path,
        default=None,
        help="Optional path to patients_metadata_removed.json",
    )
    parser.add_argument(
        "--cohort_config",
        type=Path,
        default=None,
        help="Optional YAML cohort config with name/title/row_filter/min_group_n and recruitment settings.",
    )
    parser.add_argument(
        "--with_recruitment",
        action="store_true",
        help="Enable the optional recruitment strategy section and recruitment CSV outputs.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for the cohort report.")
    args = parser.parse_args()

    cohort_config = _load_cohort_config(args.cohort_config)
    removed_json = args.removed_json or args.metadata_csv.with_name("patients_metadata_removed.json")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading clean metadata from %s", args.metadata_csv)
    metadata_df = load(str(args.metadata_csv), sep=None)
    cohort_df = _apply_row_filter(metadata_df, cohort_config["row_filter"])
    logger.info("Cohort '%s' contains %d rows", cohort_config["name"], len(cohort_df))

    removed_data = _load_removed_json(removed_json)
    provenance_reason_df, provenance_source_df = _build_provenance_tables(removed_data)

    figure_sections = _create_figures(cohort_df, output_dir / "figures")

    all_opportunities = build_analysis_opportunities(cohort_df, cohort_config["min_group_n"])
    valid_opportunities = (
        all_opportunities.loc[all_opportunities["is_valid"]]
        .copy()
        .sort_values(["Analysis", "Sex", "AgeGroup", "Constraint", "N1", "N2"], ascending=[True, True, True, True, False, False])
        .reset_index(drop=True)
    )
    valid_opportunities = valid_opportunities.drop(
        columns=["is_valid", "skip_reason", "dedupe_key"],
        errors="ignore",
    )
    all_opportunities.to_csv(output_dir / "analysis_opportunities_all.csv", index=False)
    valid_opportunities.to_csv(output_dir / "analysis_opportunities_valid.csv", index=False)

    recruitment_markdown = None
    recruitment_projection_df = None
    recruitment_summary_df = None
    recruitment_pools_df = None
    recruitment_current_milestone = None
    if args.with_recruitment:
        (
            recruitment_markdown,
            recruitment_projection_df,
            recruitment_summary_df,
            recruitment_pools_df,
            recruitment_current_milestone,
        ) = _build_recruitment_outputs(
            valid_opportunities=valid_opportunities,
            cohort_df=cohort_df,
            filtered_cohort_n=len(cohort_df),
            recruitment_config=cohort_config["recruitment"],
        )
        recruitment_projection_df.to_csv(output_dir / "recruitment_projection.csv", index=False)
        recruitment_summary_df.to_csv(output_dir / "recruitment_milestone_summary.csv", index=False)
        recruitment_pools_df.to_csv(output_dir / "recruitment_pools.csv", index=False)

    cohort_summary_df = _build_cohort_summary_table(cohort_df, cohort_config, args.metadata_csv)
    diagnosis_df = _build_diagnosis_summary(cohort_df)
    combined_diagnosis_df = _build_combined_diagnosis_summary(cohort_df)
    demographics_df = _build_demographics_summary(cohort_df)
    medication_df = _build_medication_summary(cohort_df)
    cohort_markdown = _format_row_filter_markdown(cohort_config, len(cohort_df))

    generate_cohort_report(
        output_path=output_dir / "cohort_report.html",
        report_title=cohort_config["title"],
        cohort_name=cohort_config["name"],
        cohort_markdown=cohort_markdown,
        cohort_summary_df=cohort_summary_df,
        provenance_reason_df=provenance_reason_df,
        provenance_source_df=provenance_source_df,
        diagnosis_df=diagnosis_df,
        combined_diagnosis_df=combined_diagnosis_df,
        demographics_df=demographics_df,
        medication_df=medication_df,
        valid_opportunities_df=valid_opportunities,
        figures_by_section=figure_sections,
        recruitment_markdown=recruitment_markdown,
        recruitment_projection_df=recruitment_projection_df,
        recruitment_summary_df=recruitment_summary_df,
        recruitment_pools_df=recruitment_pools_df,
    )

    logger.info("Saved cohort report to %s", output_dir / "cohort_report.html")


if __name__ == "__main__":
    main()

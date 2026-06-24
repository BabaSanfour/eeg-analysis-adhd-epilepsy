"""Schema definitions for cohort analysis opportunities over clean metadata.

This module is the single source of truth for *which* studies are possible and
*how* each cohort filter and comparison group is defined. Every constraint and
analysis carries an executable ``predicate`` (a callable over the metadata
DataFrame); the cohort engine in :mod:`eeg_adhd_epilepsy.metadata.cohort`
dispatches to these predicates rather than re-implementing the membership logic,
so there is exactly one definition per concept.

Data invariants (guaranteed by the metadata builder, relied on here)
--------------------------------------------------------------------
- ``psychostimulant == 1`` implies ``adhd == 1``
- ``asm == 1`` implies ``epilepsy == 1``
- ``asm_resistant == 1`` implies ``asm == 1`` and ``epilepsy == 1``
- ``asm_types`` is a ``+``-joined set of ASM columns (e.g. ``"LEV"``,
  ``"LEV+VPA"``) or ``"No_ASM"``; an exact match such as ``asm_types == "LEV"``
  therefore means *LEV monotherapy*, not "LEV among others".

Deduplication policy (implemented in the cohort engine)
-------------------------------------------------------
Opportunities are deduplicated by *comparison meaning*, not display labels:
two rows are duplicates when they share the same analysis name and the same
ordered pair of group ``study_id`` membership sets. One representative is kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

Predicate = Callable[[pd.DataFrame], "pd.Series"]


@dataclass(frozen=True)
class ConstraintSpec:
    """A cohort filter: a human description plus the executable membership rule."""

    name: str
    description: str
    predicate: Predicate


@dataclass(frozen=True)
class AnalysisSpec:
    """A two-group comparison and the constraint backgrounds it is run under.

    ``group_1_predicate``/``group_2_predicate`` are the executable definitions of
    each comparison arm; ``group_1``/``group_2`` are their display labels.
    ``applicability`` is a human note on when the comparison is meaningful — it is
    operationally enforced by the engine's empty/too-small-group validity gates,
    not by a separate check.
    """

    name: str
    group_1: str
    group_2: str
    description: str
    applicability: str
    constraint_sets: tuple[tuple[str, ...], ...]
    group_1_predicate: Predicate
    group_2_predicate: Predicate


@dataclass(frozen=True)
class RuleSpec:
    if_all: tuple[str, ...]
    implies: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    notes: str = ""


TARGET_CONSTRAINTS = (
    ConstraintSpec(
        "No_Constraint",
        "No additional cohort filter.",
        lambda df: pd.Series(True, index=df.index),
    ),
    ConstraintSpec("No_ADHD", "Keep rows without ADHD.", lambda df: df["adhd"] == 0),
    ConstraintSpec("No_Autism", "Keep rows without autism.", lambda df: df["autism"] == 0),
    ConstraintSpec("No_Epilepsy", "Keep rows without epilepsy.", lambda df: df["epilepsy"] == 0),
    ConstraintSpec(
        "Psychostim_True",
        "Keep psychostimulant-exposed rows.",
        lambda df: df["psychostimulant"] == 1,
    ),
    ConstraintSpec(
        "Psychostim_False",
        "Keep rows without psychostimulant exposure.",
        lambda df: df["psychostimulant"] == 0,
    ),
    ConstraintSpec("ASM_True", "Keep ASM-exposed rows.", lambda df: df["asm"] == 1),
    ConstraintSpec("ASM_False", "Keep rows without ASM exposure.", lambda df: df["asm"] == 0),
    ConstraintSpec(
        "ASM_Resistant_True", "Keep drug-resistant rows.", lambda df: df["asm_resistant"] == 1
    ),
    ConstraintSpec(
        "ASM_Resistant_False", "Keep non-resistant rows.", lambda df: df["asm_resistant"] == 0
    ),
    ConstraintSpec("First_EEG", "Keep first-EEG rows only.", lambda df: df["first_eeg"] == 1),
    ConstraintSpec(
        "DrugResistant_First_EEG",
        "Keep resistant rows only when they are first EEG recordings.",
        lambda df: df["asm_resistant"].ne(1) | df["first_eeg"].eq(1),
    ),
    ConstraintSpec(
        "Control_Only",
        "No ADHD, autism, or epilepsy.",
        lambda df: (df["adhd"] == 0) & (df["autism"] == 0) & (df["epilepsy"] == 0),
    ),
    ConstraintSpec(
        "ADHD_Only",
        "ADHD without autism or epilepsy.",
        lambda df: (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 0),
    ),
    ConstraintSpec(
        "Epilepsy_Only",
        "Epilepsy without ADHD or autism.",
        lambda df: (df["adhd"] == 0) & (df["autism"] == 0) & (df["epilepsy"] == 1),
    ),
    ConstraintSpec(
        "Autism_Only",
        "Autism without ADHD or epilepsy.",
        lambda df: (df["adhd"] == 0) & (df["autism"] == 1) & (df["epilepsy"] == 0),
    ),
    ConstraintSpec(
        "ADHD_Epilepsy",
        "ADHD and epilepsy without autism.",
        lambda df: (df["adhd"] == 1) & (df["autism"] == 0) & (df["epilepsy"] == 1),
    ),
    ConstraintSpec(
        "ADHD_Autism",
        "ADHD and autism without epilepsy.",
        lambda df: (df["adhd"] == 1) & (df["autism"] == 1) & (df["epilepsy"] == 0),
    ),
    ConstraintSpec(
        "Epilepsy_Autism",
        "Epilepsy and autism without ADHD.",
        lambda df: (df["adhd"] == 0) & (df["autism"] == 1) & (df["epilepsy"] == 1),
    ),
    ConstraintSpec(
        "ADHD_Epilepsy_Autism",
        "ADHD, epilepsy, and autism together.",
        lambda df: (df["adhd"] == 1) & (df["autism"] == 1) & (df["epilepsy"] == 1),
    ),
    ConstraintSpec(
        "Methylphenidate",
        "Keep methylphenidate rows only.",
        lambda df: df["psychostimulant_category"] == "Methylphenidate",
    ),
    ConstraintSpec(
        "Dextroamphetamine",
        "Keep dextroamphetamine rows only.",
        lambda df: df["psychostimulant_category"] == "Dextroamphetamine",
    ),
    ConstraintSpec(
        "Lisdexamfetamine",
        "Keep lisdexamfetamine rows only.",
        lambda df: df["psychostimulant_category"] == "Lisdexamfetamine",
    ),
    ConstraintSpec(
        "Combined_Amphetamine",
        "Keep amphetamine-family stimulant rows.",
        lambda df: df["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"]),
    ),
)

CONSTRAINT_BY_NAME = {spec.name: spec for spec in TARGET_CONSTRAINTS}

# Diagnosis filters that partition the cohort into mutually exclusive groups; at
# most one may appear in a single constraint set.
EXCLUSIVE_DIAGNOSIS_CONSTRAINTS = frozenset(
    {
        "Control_Only",
        "ADHD_Only",
        "Epilepsy_Only",
        "Autism_Only",
        "ADHD_Epilepsy",
        "ADHD_Autism",
        "Epilepsy_Autism",
        "ADHD_Epilepsy_Autism",
    }
)
# Single psychostimulant-category filters; at most one may appear in a set.
SINGLE_STIMULANT_CATEGORY_CONSTRAINTS = frozenset(
    {"Methylphenidate", "Dextroamphetamine", "Lisdexamfetamine"}
)


ADHD_MEDICATION_BACKGROUND_CONSTRAINTS = (
    ("No_Epilepsy",),
    ("No_Autism",),
    ("No_Autism", "No_Epilepsy"),
    ("ASM_False",),
    ("No_Autism", "ASM_False"),
    ("ASM_Resistant_False",),
    ("No_Autism", "ASM_Resistant_False"),
)

CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS = (
    ("No_Epilepsy",),
    ("No_Autism",),
    ("No_Autism", "No_Epilepsy"),
    ("ASM_False",),
    ("No_Autism", "ASM_False"),
    ("ASM_Resistant_False",),
    ("No_Autism", "ASM_Resistant_False"),
)

DRUG_RESISTANCE_CONSTRAINTS = (
    ("No_Constraint",),
    ("DrugResistant_First_EEG",),
    ("No_ADHD",),
    ("No_ADHD", "DrugResistant_First_EEG"),
    ("No_Autism",),
    ("No_Autism", "DrugResistant_First_EEG"),
    ("No_ADHD", "No_Autism"),
    ("No_ADHD", "No_Autism", "DrugResistant_First_EEG"),
    ("Psychostim_False",),
    ("Psychostim_False", "DrugResistant_First_EEG"),
)

DRUG_RESISTANCE_LONGITUDINAL_CONSTRAINTS = (
    ("No_Constraint",),
    ("No_ADHD",),
    ("No_Autism",),
    ("No_ADHD", "No_Autism"),
    ("Psychostim_False",),
)

EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS = (
    ("No_ADHD",),
    ("No_Autism",),
    ("No_ADHD", "No_Autism"),
    ("Psychostim_False",),
    ("No_ADHD", "Psychostim_False"),
    ("No_Autism", "Psychostim_False"),
    ("ASM_Resistant_False",),
)

NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS = (
    ("No_ADHD",),
    ("No_Autism",),
    ("No_ADHD", "No_Autism"),
    ("Psychostim_False",),
    ("No_ADHD", "Psychostim_False"),
    ("No_Autism", "Psychostim_False"),
)


TARGET_ANALYSES = (
    AnalysisSpec(
        "DrugResistance_Status",
        "Not Resistant",
        "Resistant",
        "Compare ASM-treated but non-resistant epilepsy rows against drug-resistant rows.",
        "Only valid inside epilepsy/ASM cohorts with both resistance groups present.",
        DRUG_RESISTANCE_CONSTRAINTS,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 1) & (s["asm_resistant"] == 0),
        lambda s: (s["epilepsy"] == 1) & (s["asm_resistant"] == 1),
    ),
    AnalysisSpec(
        "DrugResistance_First_vs_Later",
        "First EEG",
        "Later EEG",
        "Within-subject comparison of drug-resistant patients "
        "with both first and later recordings.",
        "Only valid for drug-resistant patients with at least one first EEG and one later EEG.",
        DRUG_RESISTANCE_LONGITUDINAL_CONSTRAINTS,
        lambda s: (s["asm_resistant"] == 1) & (s["first_eeg"] == 1),
        lambda s: (s["asm_resistant"] == 1) & (s["first_eeg"] == 0),
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_Any",
        "Epilepsy Unmedicated",
        "Epilepsy on ASM",
        "Compare epilepsy rows without ASM against epilepsy rows on any ASM.",
        "Only valid when both epilepsy groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 0),
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 1),
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_LEV_Only",
        "Epilepsy Unmedicated",
        "LEV Only",
        "Compare epilepsy rows without ASM against epilepsy rows on LEV monotherapy.",
        "Only valid when both groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 0),
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "LEV"),
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_VPA_Only",
        "Epilepsy Unmedicated",
        "VPA Only",
        "Compare epilepsy rows without ASM against epilepsy rows on VPA monotherapy.",
        "Only valid when both groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 0),
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "VPA"),
    ),
    AnalysisSpec(
        "Epilepsy_LEV_vs_VPA_Only",
        "LEV Only",
        "VPA Only",
        "Compare the two main monotherapy ASM groups in epilepsy.",
        "Only valid when both monotherapy groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "LEV"),
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "VPA"),
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_Unmedicated",
        "Non Epilepsy",
        "Epilepsy Unmedicated",
        "Compare non-epilepsy rows against epilepsy rows without ASM.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
        lambda s: s["epilepsy"] == 0,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 0),
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_ASM_Any",
        "Non Epilepsy",
        "Epilepsy on ASM",
        "Compare non-epilepsy rows against epilepsy rows on any ASM.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
        lambda s: s["epilepsy"] == 0,
        lambda s: (s["epilepsy"] == 1) & (s["asm"] == 1),
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_LEV_Only",
        "Non Epilepsy",
        "LEV Only",
        "Compare non-epilepsy rows against epilepsy rows on LEV monotherapy.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
        lambda s: s["epilepsy"] == 0,
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "LEV"),
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_VPA_Only",
        "Non Epilepsy",
        "VPA Only",
        "Compare non-epilepsy rows against epilepsy rows on VPA monotherapy.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
        lambda s: s["epilepsy"] == 0,
        lambda s: (s["epilepsy"] == 1) & (s["asm_types"] == "VPA"),
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Any",
        "ADHD Unmedicated",
        "ADHD Medicated",
        "Compare unmedicated ADHD rows against any medicated ADHD rows.",
        "Only valid when both ADHD groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 0),
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 1),
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Methylphenidate",
        "ADHD Unmedicated",
        "Methylphenidate",
        "Compare unmedicated ADHD rows against methylphenidate ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 0),
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Methylphenidate"),
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Lisdexamfetamine",
        "ADHD Unmedicated",
        "Lisdexamfetamine",
        "Compare unmedicated ADHD rows against lisdexamfetamine ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 0),
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Lisdexamfetamine"),
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Dextroamphetamine",
        "ADHD Unmedicated",
        "Dextroamphetamine",
        "Compare unmedicated ADHD rows against dextroamphetamine ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 0),
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Dextroamphetamine"),
    ),
    AnalysisSpec(
        "ADHD_Methylphenidate_vs_Lisdexamfetamine",
        "Methylphenidate",
        "Lisdexamfetamine",
        "Compare methylphenidate ADHD rows against lisdexamfetamine ADHD rows.",
        "Only valid when both categories exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Methylphenidate"),
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Lisdexamfetamine"),
    ),
    AnalysisSpec(
        "ADHD_Methylphenidate_vs_Amphetamine",
        "Methylphenidate",
        "Amphetamine",
        "Compare methylphenidate ADHD rows against the combined amphetamine family.",
        "Only valid when both categories exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Methylphenidate"),
        lambda s: (s["adhd"] == 1)
        & (s["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"])),
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Medicated_Any",
        "Controls",
        "Any Medicated ADHD",
        "Compare controls against any medicated ADHD rows.",
        "Only valid when controls and stimulant-exposed rows are both present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
        lambda s: s["combined_diagnosis"] == "Control",
        lambda s: (s["adhd"] == 1) & (s["psychostimulant"] == 1),
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Methylphenidate",
        "Controls",
        "Methylphenidate",
        "Compare controls against medicated ADHD rows on methylphenidate.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
        lambda s: s["combined_diagnosis"] == "Control",
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Methylphenidate"),
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Lisdexamfetamine",
        "Controls",
        "Lisdexamfetamine",
        "Compare controls against medicated ADHD rows on lisdexamfetamine.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
        lambda s: s["combined_diagnosis"] == "Control",
        lambda s: (s["adhd"] == 1) & (s["psychostimulant_category"] == "Lisdexamfetamine"),
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Amphetamine",
        "Controls",
        "Amphetamine",
        "Compare controls against medicated ADHD rows on the combined amphetamine family.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
        lambda s: s["combined_diagnosis"] == "Control",
        lambda s: (s["adhd"] == 1)
        & (s["psychostimulant_category"].isin(["Lisdexamfetamine", "Dextroamphetamine"])),
    ),
)

ANALYSIS_BY_NAME = {spec.name: spec for spec in TARGET_ANALYSES}


CONSTRAINT_RULES = (
    RuleSpec(
        if_all=("No_ADHD",),
        implies=("Psychostim_False",),
        notes="No ADHD rows cannot be psychostimulant-positive.",
    ),
    RuleSpec(
        if_all=("No_Epilepsy",),
        implies=("ASM_False",),
        notes="No epilepsy rows cannot be ASM-positive.",
    ),
    RuleSpec(
        if_all=("ASM_Resistant_True",),
        implies=("ASM_True",),
        notes="Drug-resistant rows must also be ASM-positive.",
    ),
    RuleSpec(
        if_all=("Methylphenidate",),
        implies=("Psychostim_True",),
        notes="Medication category implies stimulant exposure.",
    ),
    RuleSpec(
        if_all=("Dextroamphetamine",),
        implies=("Psychostim_True",),
        notes="Medication category implies stimulant exposure.",
    ),
    RuleSpec(
        if_all=("Lisdexamfetamine",),
        implies=("Psychostim_True",),
        notes="Medication category implies stimulant exposure.",
    ),
    RuleSpec(
        if_all=("Combined_Amphetamine",),
        implies=("Psychostim_True",),
        notes="Medication family implies stimulant exposure.",
    ),
    RuleSpec(
        if_all=("Psychostim_True", "Psychostim_False"),
        contradicts=("Psychostim_True", "Psychostim_False"),
        notes="A cohort cannot be both stimulant-positive and stimulant-negative.",
    ),
    RuleSpec(
        if_all=("ASM_True", "ASM_False"),
        contradicts=("ASM_True", "ASM_False"),
        notes="A cohort cannot be both ASM-positive and ASM-negative.",
    ),
    RuleSpec(
        if_all=("ASM_Resistant_True", "ASM_False"),
        contradicts=("ASM_Resistant_True", "ASM_False"),
        notes="Drug-resistant rows must be ASM-positive.",
    ),
    RuleSpec(
        if_all=("ASM_Resistant_True", "No_Epilepsy"),
        contradicts=("ASM_Resistant_True", "No_Epilepsy"),
        notes="Drug resistance only exists within epilepsy rows.",
    ),
    RuleSpec(
        if_all=("No_ADHD", "Methylphenidate"),
        contradicts=("No_ADHD", "Methylphenidate"),
        notes="Medication category cannot coexist with a no-ADHD filter.",
    ),
    RuleSpec(
        if_all=("No_ADHD", "Dextroamphetamine"),
        contradicts=("No_ADHD", "Dextroamphetamine"),
        notes="Medication category cannot coexist with a no-ADHD filter.",
    ),
    RuleSpec(
        if_all=("No_ADHD", "Lisdexamfetamine"),
        contradicts=("No_ADHD", "Lisdexamfetamine"),
        notes="Medication category cannot coexist with a no-ADHD filter.",
    ),
    RuleSpec(
        if_all=("No_ADHD", "Combined_Amphetamine"),
        contradicts=("No_ADHD", "Combined_Amphetamine"),
        notes="Medication category cannot coexist with a no-ADHD filter.",
    ),
)


# Vocabulary of reasons an enumerated opportunity is not a valid study. The
# cohort engine asserts every emitted reason is one of these.
SKIP_REASONS = (
    "contradictory_constraints",
    "redundant_constraint_set",
    "analysis_not_applicable",
    "empty_filtered_cohort",
    "one_group_empty",
    "same_group_membership",
    "too_small_group",
    "insufficient_category_support",
    "insufficient_longitudinal_pairs",
)

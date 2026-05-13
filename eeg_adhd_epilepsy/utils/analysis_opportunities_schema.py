"""Schema definitions for cohort analysis opportunities over clean metadata."""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    description: str
    rule: str


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    group_1: str
    group_2: str
    description: str
    applicability: str
    constraint_sets: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RuleSpec:
    if_all: tuple[str, ...]
    implies: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    notes: str = ""


TARGET_CONSTRAINTS = (
    ConstraintSpec("No_Constraint", "No additional cohort filter.", "all rows"),
    ConstraintSpec("No_ADHD", "Keep rows without ADHD.", "adhd == 0"),
    ConstraintSpec("No_Autism", "Keep rows without autism.", "autism == 0"),
    ConstraintSpec("No_Epilepsy", "Keep rows without epilepsy.", "epilepsy == 0"),
    ConstraintSpec("Psychostim_True", "Keep psychostimulant-exposed rows.", "psychostimulant == 1"),
    ConstraintSpec("Psychostim_False", "Keep rows without psychostimulant exposure.", "psychostimulant == 0"),
    ConstraintSpec("ASM_True", "Keep ASM-exposed rows.", "asm == 1"),
    ConstraintSpec("ASM_False", "Keep rows without ASM exposure.", "asm == 0"),
    ConstraintSpec("ASM_Resistant_True", "Keep drug-resistant rows.", "asm_resistant == 1"),
    ConstraintSpec("ASM_Resistant_False", "Keep non-resistant rows.", "asm_resistant == 0"),
    ConstraintSpec("First_EEG", "Keep first-EEG rows only.", "first_eeg == 1"),
    ConstraintSpec(
        "DrugResistant_First_EEG",
        "Keep resistant rows only when they are first EEG recordings.",
        "asm_resistant == 0 or first_eeg == 1",
    ),
    ConstraintSpec("Control_Only", "No ADHD, autism, or epilepsy.", "adhd == 0 and autism == 0 and epilepsy == 0"),
    ConstraintSpec("ADHD_Only", "ADHD without autism or epilepsy.", "adhd == 1 and autism == 0 and epilepsy == 0"),
    ConstraintSpec("Epilepsy_Only", "Epilepsy without ADHD or autism.", "adhd == 0 and autism == 0 and epilepsy == 1"),
    ConstraintSpec("Autism_Only", "Autism without ADHD or epilepsy.", "adhd == 0 and autism == 1 and epilepsy == 0"),
    ConstraintSpec("ADHD_Epilepsy", "ADHD and epilepsy without autism.", "adhd == 1 and autism == 0 and epilepsy == 1"),
    ConstraintSpec("ADHD_Autism", "ADHD and autism without epilepsy.", "adhd == 1 and autism == 1 and epilepsy == 0"),
    ConstraintSpec("Epilepsy_Autism", "Epilepsy and autism without ADHD.", "adhd == 0 and autism == 1 and epilepsy == 1"),
    ConstraintSpec(
        "ADHD_Epilepsy_Autism",
        "ADHD, epilepsy, and autism together.",
        "adhd == 1 and autism == 1 and epilepsy == 1",
    ),
    ConstraintSpec(
        "Methylphenidate",
        "Keep methylphenidate rows only.",
        "psychostimulant_category == 'Methylphenidate'",
    ),
    ConstraintSpec(
        "Dextroamphetamine",
        "Keep dextroamphetamine rows only.",
        "psychostimulant_category == 'Dextroamphetamine'",
    ),
    ConstraintSpec(
        "Lisdexamfetamine",
        "Keep lisdexamfetamine rows only.",
        "psychostimulant_category == 'Lisdexamfetamine'",
    ),
    ConstraintSpec(
        "Combined_Amphetamine",
        "Keep amphetamine-family stimulant rows.",
        "psychostimulant_category in {'Lisdexamfetamine', 'Dextroamphetamine'}",
    ),
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
        "Compare non-resistant vs drug-resistant rows.",
        "Only valid inside epilepsy/ASM cohorts with both resistance groups present.",
        DRUG_RESISTANCE_CONSTRAINTS,
    ),
    AnalysisSpec(
        "DrugResistance_First_vs_Later",
        "First EEG",
        "Later EEG",
        "Within-subject comparison of drug-resistant patients with both first and later recordings.",
        "Only valid for drug-resistant patients with at least one first EEG and one later EEG.",
        DRUG_RESISTANCE_LONGITUDINAL_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_Any",
        "Epilepsy Unmedicated",
        "Epilepsy on ASM",
        "Compare epilepsy rows without ASM against epilepsy rows on any ASM.",
        "Only valid when both epilepsy groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_LEV_Only",
        "Epilepsy Unmedicated",
        "LEV Only",
        "Compare epilepsy rows without ASM against epilepsy rows on LEV monotherapy.",
        "Only valid when both groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Epilepsy_ASM_Effect_VPA_Only",
        "Epilepsy Unmedicated",
        "VPA Only",
        "Compare epilepsy rows without ASM against epilepsy rows on VPA monotherapy.",
        "Only valid when both groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Epilepsy_LEV_vs_VPA_Only",
        "LEV Only",
        "VPA Only",
        "Compare the two main monotherapy ASM groups in epilepsy.",
        "Only valid when both monotherapy groups exist after filtering.",
        EPILEPSY_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_Unmedicated",
        "Non Epilepsy",
        "Epilepsy Unmedicated",
        "Compare non-epilepsy rows against epilepsy rows without ASM.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_ASM_Any",
        "Non Epilepsy",
        "Epilepsy on ASM",
        "Compare non-epilepsy rows against epilepsy rows on any ASM.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_LEV_Only",
        "Non Epilepsy",
        "LEV Only",
        "Compare non-epilepsy rows against epilepsy rows on LEV monotherapy.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "NonEpilepsy_vs_Epilepsy_VPA_Only",
        "Non Epilepsy",
        "VPA Only",
        "Compare non-epilepsy rows against epilepsy rows on VPA monotherapy.",
        "Only valid when both groups are present.",
        NON_EPILEPSY_VS_EPILEPSY_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Any",
        "ADHD Unmedicated",
        "ADHD Medicated",
        "Compare unmedicated ADHD rows against any medicated ADHD rows.",
        "Only valid when both ADHD groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Methylphenidate",
        "ADHD Unmedicated",
        "Methylphenidate",
        "Compare unmedicated ADHD rows against methylphenidate ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Lisdexamfetamine",
        "ADHD Unmedicated",
        "Lisdexamfetamine",
        "Compare unmedicated ADHD rows against lisdexamfetamine ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Psychostim_Effect_Dextroamphetamine",
        "ADHD Unmedicated",
        "Dextroamphetamine",
        "Compare unmedicated ADHD rows against dextroamphetamine ADHD rows.",
        "Only valid when both groups exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Methylphenidate_vs_Lisdexamfetamine",
        "Methylphenidate",
        "Lisdexamfetamine",
        "Compare methylphenidate ADHD rows against lisdexamfetamine ADHD rows.",
        "Only valid when both categories exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "ADHD_Methylphenidate_vs_Amphetamine",
        "Methylphenidate",
        "Amphetamine",
        "Compare methylphenidate ADHD rows against the combined amphetamine family.",
        "Only valid when both categories exist after filtering.",
        ADHD_MEDICATION_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Medicated_Any",
        "Controls",
        "Any Medicated ADHD",
        "Compare controls against any medicated ADHD rows.",
        "Only valid when controls and stimulant-exposed rows are both present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Methylphenidate",
        "Controls",
        "Methylphenidate",
        "Compare controls against medicated ADHD rows on methylphenidate.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Lisdexamfetamine",
        "Controls",
        "Lisdexamfetamine",
        "Compare controls against medicated ADHD rows on lisdexamfetamine.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
    ),
    AnalysisSpec(
        "Control_vs_ADHD_Amphetamine",
        "Controls",
        "Amphetamine",
        "Compare controls against medicated ADHD rows on the combined amphetamine family.",
        "Only valid when both groups are present.",
        CONTROL_VS_ADHD_BACKGROUND_CONSTRAINTS,
    ),
)


DATA_IMPLICATIONS = (
    "psychostimulant == 1 implies adhd == 1",
    "asm == 1 implies epilepsy == 1",
    "asm_resistant == 1 implies epilepsy == 1",
    "asm_resistant == 1 implies asm == 1",
)


CONSTRAINT_RULES = (
    RuleSpec(if_all=("No_ADHD",), implies=("Psychostim_False",), notes="No ADHD rows cannot be psychostimulant-positive."),
    RuleSpec(if_all=("No_Epilepsy",), implies=("ASM_False",), notes="No epilepsy rows cannot be ASM-positive."),
    RuleSpec(if_all=("ASM_Resistant_True",), implies=("ASM_True",), notes="Drug-resistant rows must also be ASM-positive."),
    RuleSpec(if_all=("Methylphenidate",), implies=("Psychostim_True",), notes="Medication category implies stimulant exposure."),
    RuleSpec(if_all=("Dextroamphetamine",), implies=("Psychostim_True",), notes="Medication category implies stimulant exposure."),
    RuleSpec(if_all=("Lisdexamfetamine",), implies=("Psychostim_True",), notes="Medication category implies stimulant exposure."),
    RuleSpec(if_all=("Combined_Amphetamine",), implies=("Psychostim_True",), notes="Medication family implies stimulant exposure."),
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
    "subset_specific_choice_required",
)


DEDUPLICATION_POLICY = (
    "Deduplicate by actual comparison meaning, not display labels alone.",
    "Use `study_id` membership for group 1 and group 2 as the canonical cohort identity.",
    "Two rows are duplicates if they share the same analysis name and the same ordered pair of group membership sets.",
    "Keep one representative row and record alternate labels or constraints later if needed.",
)

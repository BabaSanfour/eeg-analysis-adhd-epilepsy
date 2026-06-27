"""Behavioral tests for the analysis-opportunity (possible-study) engine.

These pin the meaningful behavior of the schema + :func:`build_analysis_opportunities`
— group-membership semantics, a few canonical valid studies, deduplication, and
the skip-reason vocabulary — without a brittle full-table snapshot.
"""

from __future__ import annotations

import pandas as pd

from eeg_adhd_epilepsy.metadata.analysis_opportunities_schema import (
    ANALYSIS_BY_NAME,
    SKIP_REASONS,
    TARGET_ANALYSES,
    TARGET_CONSTRAINTS,
)
from eeg_adhd_epilepsy.metadata.cohort import build_analysis_opportunities

MIN_GROUP_N = 2


def _synthetic_cohort() -> pd.DataFrame:
    """A deterministic clean-metadata cohort exercising every analysis family."""

    def make(
        sid,
        pid,
        sex,
        age,
        *,
        adhd=0,
        autism=0,
        epi=0,
        ps=0,
        pscat="None",
        asm=0,
        res=0,
        first=1,
        atypes="No_ASM",
        cdx="Control",
    ):
        return dict(
            study_id=sid,
            patient_id=pid,
            sex=sex,
            age_group=age,
            adhd=adhd,
            autism=autism,
            epilepsy=epi,
            psychostimulant=ps,
            psychostimulant_category=pscat,
            asm=asm,
            asm_resistant=res,
            first_eeg=first,
            asm_types=atypes,
            combined_diagnosis=cdx,
        )

    rows: list[dict] = []
    counter = {"sid": 1000}

    def add(n, **kw):
        for _ in range(n):
            counter["sid"] += 1
            rows.append(make(counter["sid"], counter["sid"], **kw))

    # Controls
    add(4, sex="M", age="10-12", cdx="Control")
    add(4, sex="F", age="10-12", cdx="Control")
    add(3, sex="M", age="13-15", cdx="Control")
    # ADHD unmedicated
    add(4, sex="M", age="10-12", adhd=1, cdx="ADHD")
    add(3, sex="F", age="10-12", adhd=1, cdx="ADHD")
    # ADHD medicated: methylphenidate / lisdexamfetamine / dextroamphetamine
    add(4, sex="M", age="10-12", adhd=1, ps=1, pscat="Methylphenidate", cdx="ADHD")
    add(3, sex="M", age="10-12", adhd=1, ps=1, pscat="Lisdexamfetamine", cdx="ADHD")
    add(2, sex="F", age="10-12", adhd=1, ps=1, pscat="Dextroamphetamine", cdx="ADHD")
    # Epilepsy: unmedicated / LEV mono / VPA mono / LEV+VPA poly
    add(4, sex="M", age="13-15", epi=1, asm=0, cdx="Epilepsy")
    add(3, sex="M", age="13-15", epi=1, asm=1, atypes="LEV", cdx="Epilepsy")
    add(3, sex="F", age="13-15", epi=1, asm=1, atypes="VPA", cdx="Epilepsy")
    add(2, sex="M", age="13-15", epi=1, asm=1, atypes="LEV+VPA", cdx="Epilepsy")
    # Drug-resistant epilepsy with longitudinal first + later EEG (same patient_id)
    for k in range(3):
        counter["sid"] += 1
        pid = 9000 + k
        rows.append(
            make(
                counter["sid"],
                pid,
                "M",
                "13-15",
                epi=1,
                asm=1,
                res=1,
                first=1,
                atypes="LEV",
                cdx="Epilepsy",
            )
        )
        counter["sid"] += 1
        rows.append(
            make(
                counter["sid"],
                pid,
                "M",
                "13-15",
                epi=1,
                asm=1,
                res=1,
                first=0,
                atypes="LEV",
                cdx="Epilepsy",
            )
        )
    # Autism comorbidity
    add(2, sex="F", age="10-12", adhd=1, autism=1, cdx="ADHD+Autism")

    return pd.DataFrame(rows)


def _valid(opps: pd.DataFrame, analysis: str, constraint: str) -> pd.Series:
    """Return the single valid all-sex/all-age row for an (analysis, constraint)."""
    match = opps[
        opps["is_valid"]
        & (opps["Analysis"] == analysis)
        & (opps["Constraint"] == constraint)
        & (opps["Sex"] == "All")
        & (opps["AgeGroup"] == "All")
    ]
    assert len(match) == 1, f"expected one valid row, got {len(match)}"
    return match.iloc[0]


def test_schema_predicates_are_wellformed() -> None:
    """Every constraint/analysis predicate returns an obs-aligned boolean mask."""
    cohort = _synthetic_cohort()
    n = len(cohort)
    for spec in TARGET_CONSTRAINTS:
        mask = spec.predicate(cohort)
        assert len(mask) == n
        assert mask.dtype == bool
    for spec in TARGET_ANALYSES:
        for predicate in (spec.group_1_predicate, spec.group_2_predicate):
            mask = predicate(cohort)
            assert len(mask) == n
            assert mask.dtype == bool


def test_monotherapy_and_control_group_semantics() -> None:
    """'LEV Only' means LEV monotherapy (exact asm_types); Controls via diagnosis."""
    cohort = _synthetic_cohort()

    lev = ANALYSIS_BY_NAME["Epilepsy_ASM_Effect_LEV_Only"].group_2_predicate(cohort)
    # Only LEV-monotherapy rows — LEV+VPA polytherapy is excluded.
    assert set(cohort.loc[lev, "asm_types"]) == {"LEV"}

    ctrl = ANALYSIS_BY_NAME["Control_vs_ADHD_Medicated_Any"].group_1_predicate(cohort)
    assert set(cohort.loc[ctrl, "combined_diagnosis"]) == {"Control"}


def test_expected_valid_studies_have_correct_group_sizes() -> None:
    opps = build_analysis_opportunities(_synthetic_cohort(), min_group_n=MIN_GROUP_N)

    # ASM-treated non-resistant (3 mono LEV + 3 mono VPA + 2 LEV+VPA) vs 6 resistant.
    status = _valid(opps, "DrugResistance_Status", "No_Constraint")
    assert (status["N1"], status["N2"]) == (8, 6)

    # 'LEV Only' counts resistant LEV rows too (asm_types == "LEV"): 3 mono + 6 resistant.
    lev = _valid(opps, "Epilepsy_ASM_Effect_LEV_Only", "No_ADHD")
    assert (lev["N1"], lev["N2"]) == (4, 9)

    # Within-subject longitudinal: 3 patients each with a first and later EEG.
    longi = _valid(opps, "DrugResistance_First_vs_Later", "No_Constraint")
    assert (longi["N1"], longi["N2"], longi["paired_patients"]) == (3, 3, 3)


def test_valid_opportunities_are_deduplicated() -> None:
    opps = build_analysis_opportunities(_synthetic_cohort(), min_group_n=MIN_GROUP_N)
    valid = opps[opps["is_valid"]]
    assert len(valid) > 0
    # Each valid study is a distinct comparison (unique group-membership signature).
    assert valid["dedupe_key"].is_unique


def test_engine_emits_valid_and_skipped_rows_with_declared_reasons() -> None:
    opps = build_analysis_opportunities(_synthetic_cohort(), min_group_n=MIN_GROUP_N)
    assert opps["is_valid"].any()
    assert (~opps["is_valid"]).any()
    emitted = set(opps["skip_reason"].dropna().unique())
    assert emitted <= set(SKIP_REASONS), emitted - set(SKIP_REASONS)

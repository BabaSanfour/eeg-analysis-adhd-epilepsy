"""Tests for the cohort-summary overview builder in reports._common."""

from types import SimpleNamespace

import pandas as pd
from coco_pipe.report.core import Section

from eeg_adhd_epilepsy.reports._common import add_overview_cohort_summary


class _FakeContainer:
    """Minimal container exposing the observation frame used by the summary."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def observation_frame(self) -> pd.DataFrame:
        return self._frame.copy()


def _args():
    return SimpleNamespace(
        run_pooled=False,
        conditions=["EO"],
        subject_col="study_id",
    )


def _eval_specs():
    return [{"name": "Diagnosis", "target_col": "dx", "label_map": {}}]


def _run(frame: pd.DataFrame) -> Section:
    section = Section("Overview")
    add_overview_cohort_summary(
        section,
        _args(),
        _eval_specs(),
        {("condition", "EO"): _FakeContainer(frame)},
        pooled_condition="pooled",
    )
    return section


def test_cohort_summary_counts_unique_patients_not_recordings():
    # patient 1 has three recordings (one duplicated), patient 2 has one.
    frame = pd.DataFrame(
        [
            {"study_id": "s1a", "patient_id": 1, "dx": "A", "sex": "F"},
            {"study_id": "s1a", "patient_id": 1, "dx": "A", "sex": "F"},
            {"study_id": "s1b", "patient_id": 1, "dx": "A", "sex": "F"},
            {"study_id": "s2", "patient_id": 2, "dx": "B", "sex": "M"},
        ]
    )
    rendered = _run(frame).render()
    # Two unique subjects, not four recordings (markdown bold -> <strong>).
    assert "<strong>2</strong> unique subjects" in rendered


def test_cohort_summary_falls_back_to_subject_id_when_patient_id_missing():
    # Patient 3 has no patient_id; it must still be counted (via study_id),
    # neither dropped from the header nor merged with other id-less rows.
    frame = pd.DataFrame(
        [
            {"study_id": "s1", "patient_id": 1.0, "dx": "A"},
            {"study_id": "s2", "patient_id": 2.0, "dx": "B"},
            {"study_id": "s3", "patient_id": None, "dx": "A"},
            {"study_id": "s4", "patient_id": None, "dx": "B"},
        ]
    )
    rendered = _run(frame).render()
    # Four distinct subjects: two with ids, two recovered via study_id.
    assert "<strong>4</strong> unique subjects" in rendered


def test_cohort_summary_header_matches_class_table_totals():
    # The "N unique subjects" header must equal the sum of per-class counts,
    # even when patient_id is partially missing.
    frame = pd.DataFrame(
        [
            {"study_id": "s1a", "patient_id": 1.0, "dx": "A"},
            {"study_id": "s1b", "patient_id": 1.0, "dx": "A"},
            {"study_id": "s2", "patient_id": None, "dx": "B"},
            {"study_id": "s3", "patient_id": None, "dx": "A"},
        ]
    )
    section = _run(frame)
    rendered = section.render()
    # patient 1 (2 recordings) + s2 + s3 = 3 unique subjects; class A=2, B=1.
    assert "<strong>3</strong> unique subjects" in rendered

    # The class table the builder produced must sum to the same 3 subjects.
    counts_table = next(
        child
        for child in section.children
        if getattr(child, "title", "").startswith("Primary class counts")
    )
    assert int(counts_table.data["n_subjects"].sum()) == 3


def test_cohort_summary_labels_subjects_when_only_subject_id_present():
    # No patient_id column -> counts collapse to the subject/study id and are
    # labelled "subjects", not "patients".
    frame = pd.DataFrame(
        [
            {"study_id": "s1", "dx": "A"},
            {"study_id": "s1", "dx": "A"},  # same subject, second recording
            {"study_id": "s2", "dx": "B"},
        ]
    )
    rendered = _run(frame).render()
    assert "<strong>2</strong> unique subjects" in rendered
    assert "patients" not in rendered


def test_cohort_summary_labels_observations_when_no_identifier():
    # Neither patient_id nor the subject_col is present: the rows are raw
    # observations (recordings/epochs) and must not be mislabelled as subjects.
    frame = pd.DataFrame(
        [
            {"dx": "A"},
            {"dx": "A"},
            {"dx": "B"},
        ]
    )
    rendered = _run(frame).render()
    assert "<strong>3</strong> unique observations" in rendered
    assert "subjects" not in rendered and "patients" not in rendered

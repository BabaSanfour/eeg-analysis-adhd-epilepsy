import numpy as np
import pytest
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis.utils.decoding import (
    cohort_signature,
    prepare_decoding_scope,
    prepare_target,
    safe_group_n_splits,
)


def test_cohort_signature_is_stable_and_does_not_expose_ids():
    first = cohort_signature(["patient-2", "patient-1", "patient-1"])
    second = cohort_signature(["patient-1", "patient-2"])
    assert first == second
    assert len(first) == 16
    assert "patient" not in first


def test_prepare_target_uses_patient_groups_and_label_map():
    container = DataContainer(
        X=np.arange(12, dtype=float).reshape(6, 2),
        dims=("obs", "feature"),
        coords={
            "feature": ["a", "b"],
            "combined_diagnosis": ["Control"] * 3 + ["ADHD"] * 3,
            "patient_group_id": ["c1", "c2", "c3", "a1", "a2", "a3"],
            "subject": ["01", "02", "03", "04", "05", "06"],
            "session": ["01"] * 6,
        },
        ids=np.array([f"r{idx}" for idx in range(6)]),
    )
    selected, y, groups, frame = prepare_target(
        container,
        {
            "target_col": "combined_diagnosis",
            "label_map": {"Control": "0", "ADHD": "1"},
            "positive_class": "1",
        },
    )
    assert selected.X.shape == (6, 2)
    assert y.tolist() == [0, 0, 0, 1, 1, 1]
    assert groups.tolist() == ["c1", "c2", "c3", "a1", "a2", "a3"]
    assert frame["group_id"].tolist() == groups.tolist()


def test_prepare_target_requires_explicit_positive_class():
    container = DataContainer(
        X=np.ones((4, 1)),
        dims=("obs", "feature"),
        coords={
            "feature": ["x"],
            "target": ["Control", "Control", "ADHD", "ADHD"],
            "patient_group_id": ["c1", "c2", "a1", "a2"],
        },
    )
    with pytest.raises(ValueError, match="positive_class"):
        prepare_target(
            container,
            {"target_col": "target"},
        )


def test_safe_group_n_splits_reduces_to_smallest_class_group_count():
    y = np.array([0, 0, 0, 1, 1])
    groups = np.array(["a", "b", "c", "d", "e"])
    assert safe_group_n_splits(y, groups, requested=5) == 2


def test_prepare_scope_adds_canonical_subject_and_session_columns():
    container = DataContainer(
        X=np.ones((4, 1)),
        dims=("obs", "feature"),
        coords={
            "feature": ["x"],
            "target": ["Control", "Control", "ADHD", "ADHD"],
            "study_id": ["01", "02", "03", "04"],
            "visit_id": ["v1", "v1", "v2", "v2"],
            "patient_group_id": ["c1", "c2", "a1", "a2"],
        },
        ids=np.array(["r1", "r2", "r3", "r4"]),
    )
    _, _, _, metadata, _ = prepare_decoding_scope(
        container,
        {
            "target_col": "target",
            "positive_class": "ADHD",
        },
        scope="EO",
        group_col="patient_group_id",
        subject_col="study_id",
        session_col="visit_id",
        requested_splits=2,
    )
    assert metadata["Subject"].tolist() == ["01", "02", "03", "04"]
    assert metadata["Session"].tolist() == ["v1", "v1", "v2", "v2"]

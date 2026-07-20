from types import SimpleNamespace

import numpy as np
import pytest
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis.utils.decoding import (
    build_loader_args,
    foundation_provenance,
    prepare_decoding_scope,
    prepare_target,
    resolve_decoding_paths,
    safe_group_n_splits,
)


def _fm_spec():
    return SimpleNamespace(
        pretrained_n_times=2000,
        pretrained_sfreq=200.0,
        pretrained_window_seconds=10.0,
    )


def _base_model_cfg(**overrides):
    cfg = {
        "model_key": "reve",
        "segment_duration": 10.0,
        "overlap": 0.0,
        "use_derivatives": True,
        "window_source": "derivative",
    }
    cfg.update(overrides)
    return cfg


def test_build_loader_args_sets_raw_units_default_and_override():
    default_args = build_loader_args({}, input_mode="raw", layout_mode="sensor")
    assert default_args.units == "V"

    override_args = build_loader_args(
        {"units": "uV"},
        input_mode="raw",
        layout_mode="sensor",
    )
    assert override_args.units == "uV"


def test_decoding_derivative_root_relocates_output_without_changing_run_identity(tmp_path):
    bids_root = tmp_path / "project" / "BIDS"
    scratch_root = tmp_path / "scratch" / "BIDS" / "derivatives" / "decoding"
    base_config = {
        "bids_root": str(bids_root),
        "dataset_name": "Relocation Cohort",
        "reports_root": str(tmp_path / "reports"),
        "input_mode": "descriptors",
    }

    default_paths = resolve_decoding_paths(base_config, input_mode="descriptors")
    relocated_paths = resolve_decoding_paths(
        {**base_config, "derivative_root": str(scratch_root)},
        input_mode="descriptors",
    )

    assert relocated_paths[1] == scratch_root / default_paths[1].parent.name / default_paths[1].name
    assert relocated_paths[4] == default_paths[4]
    assert relocated_paths[1].name == default_paths[1].name


def test_foundation_provenance_defaults_pooling_to_mean():
    prov = foundation_provenance(_base_model_cfg(), _fm_spec(), config_hash="abc")
    assert prov["pooling"] == "mean"


def test_foundation_provenance_defaults_bandpass_to_none():
    prov = foundation_provenance(_base_model_cfg(), _fm_spec(), config_hash="abc")
    assert prov["bandpass"] is None


def test_foundation_provenance_records_bandpass():
    prov = foundation_provenance(
        _base_model_cfg(window_source="re_epoch", bandpass=[0.5, 40.0]),
        _fm_spec(),
        config_hash="abc",
    )
    assert prov["bandpass"] == [0.5, 40.0]


def test_foundation_provenance_distinguishes_pooling_variants():
    # The pooling field is the join key that keeps same-model/same-window
    # embedding variants (e.g. REVE mean vs attention) distinguishable.
    mean = foundation_provenance(_base_model_cfg(), _fm_spec(), config_hash="abc")
    attn = foundation_provenance(
        _base_model_cfg(pooling="attention"), _fm_spec(), config_hash="abc"
    )
    assert mean["pooling"] == "mean"
    assert attn["pooling"] == "attention"
    assert mean != attn


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

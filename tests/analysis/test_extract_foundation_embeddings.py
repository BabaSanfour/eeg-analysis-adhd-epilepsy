import pytest

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe


def test_freeze_config_used_strips_volatile_keys(tmp_path):
    """config_used.yaml must omit per-task keys so every array task agrees byte-for-byte."""
    derivative_root = tmp_path / "deriv"
    config = {
        "dataset_name": "synthetic",
        "task": "clinical",
        "subjects": ["sub-0001"],
        "bids_root": "/data/BIDS",
        "metadata": "/data/meta.csv",
        "derivative_root": "/scratch/deriv",
    }
    efe._freeze_config_used(config, derivative_root)

    text = (derivative_root / "config_used.yaml").read_text(encoding="utf-8")
    assert "dataset_name" in text
    assert "task" in text
    for volatile in ("subjects", "bids_root", "metadata", "derivative_root"):
        assert volatile not in text


def test_freeze_config_used_tolerates_differing_volatile_keys(tmp_path):
    """A second task processing other subjects must not trip the drift guard."""
    derivative_root = tmp_path / "deriv"
    base = {"dataset_name": "synthetic", "task": "clinical", "subjects": ["sub-0001"]}
    efe._freeze_config_used(base, derivative_root)
    # Same analysis config, different per-task subjects -> identical config_used text.
    efe._freeze_config_used({**base, "subjects": ["sub-0002"]}, derivative_root)


def test_freeze_config_used_rejects_real_config_drift(tmp_path):
    """A genuine analysis-config change against an existing root is rejected."""
    derivative_root = tmp_path / "deriv"
    base = {"dataset_name": "synthetic", "task": "clinical", "subjects": ["sub-0001"]}
    efe._freeze_config_used(base, derivative_root)
    with pytest.raises(ValueError, match="different configuration"):
        efe._freeze_config_used({**base, "task": "other"}, derivative_root)


def test_run_skips_condition_with_no_data(tmp_path, monkeypatch):
    """A shard whose subject lacks a condition (no EC/EO) is skipped, not fatal."""
    import pandas as pd

    def _no_data(*args, **kwargs):
        # Mirrors coco_pipe's loader when no epochs match the condition.
        raise RuntimeError("No valid data found in /fake/preproc")

    monkeypatch.setattr(efe, "build_container", _no_data)

    derivative_root = tmp_path / "deriv"
    config = {
        "dataset_name": "synthetic",
        "task": "clinical",
        "bids_root": str(tmp_path / "bids"),
        "subject_col": "study_id",
        "subjects": ["0001"],
        "conditions": ["EO_baseline", "EC_baseline"],
        "models": [
            {
                "model_key": "cbramod",
                "segment_duration": 10.0,
                "overlap": 0.0,
                "use_derivatives": True,
                "window_source": "derivative",
            }
        ],
    }

    # Must not raise despite every load failing.
    result_root = efe.run(config, derivative_root, shard_token="row-0001")
    assert result_root == derivative_root

    failures = pd.read_csv(derivative_root / "_failures" / "row-0001.csv")
    # One skip row per condition, both flagged as missing-condition (not crashed).
    assert set(failures["condition"]) == {"EO_baseline", "EC_baseline"}
    assert (failures["status"] == "skipped").all()
    assert (failures["reason"] == "no_data_for_condition").all()


def test_run_propagates_unrelated_runtime_error(tmp_path, monkeypatch):
    """A load failure that is NOT a missing-condition must crash, not be skipped."""

    def _boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(efe, "build_container", _boom)

    config = {
        "dataset_name": "synthetic",
        "task": "clinical",
        "bids_root": str(tmp_path / "bids"),
        "subject_col": "study_id",
        "subjects": ["0001"],
        "conditions": ["EO_baseline"],
        "models": [
            {
                "model_key": "cbramod",
                "segment_duration": 10.0,
                "overlap": 0.0,
                "use_derivatives": True,
                "window_source": "derivative",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="disk exploded"):
        efe.run(config, tmp_path / "deriv", shard_token="row-0001")

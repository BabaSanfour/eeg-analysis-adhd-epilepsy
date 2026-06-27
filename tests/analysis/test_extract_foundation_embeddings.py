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

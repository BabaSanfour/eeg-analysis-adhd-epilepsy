import json

import numpy as np
import pytest
from coco_pipe.decoding import redact_sensitive
from coco_pipe.decoding.foundation_models import FoundationEmbeddingResult
from coco_pipe.io import DataContainer, load_embedding_derivatives, save_embedding_outputs

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.utils.artifacts import freeze_config_used


def _freeze_foundation_config(config, derivative_root):
    return freeze_config_used(
        config,
        derivative_root,
        volatile_keys=efe._VOLATILE_CONFIG_KEYS,
        sanitize=redact_sensitive,
        overwrite=bool(config.get("overwrite", False)),
    )


def test_coco_pipe_save_embedding_outputs_keeps_pooled_and_tokens_separate(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        recording_embedding=np.arange(4, dtype=float),
        window_start=np.arange(3),
        window_stop=np.arange(1, 4),
        window_index=np.arange(3),
        metadata={
            "model_key": "demo",
            "recording_id": "sub-0001_ses-01_run-01",
            "subject": "0001",
            "token_layout": "native",
            "token_layout_version": 1,
            "token_source": "test_native_output",
            "token_axes": ["window", "token", "feature"],
            "token_observation_axes": ["token"],
            "token_feature_axis": "feature",
        },
        token_embeddings=np.arange(60, dtype=float).reshape(3, 5, 4),
    )
    artifact = tmp_path / "sub-0001_demo_embedding.npz"
    requested_token_path = tmp_path / "independent_demo_tokens.npz"
    _, token_path = save_embedding_outputs(
        result,
        artifact,
        token_path=requested_token_path,
        overwrite=False,
    )

    pooled = load_embedding_derivatives(tmp_path, model_key="demo", representation="epoch")
    np.testing.assert_allclose(pooled.X, result.window_embeddings)
    assert token_path == requested_token_path
    tokens = load_embedding_derivatives(tmp_path, model_key="demo", representation="token")
    np.testing.assert_allclose(tokens.X, result.token_embeddings)

    # A complete pair resumes without requiring overwrite.
    assert save_embedding_outputs(
        result,
        artifact,
        token_path=token_path,
        overwrite=False,
    ) == (artifact, token_path)


def test_extraction_pooling_variants_share_one_unchanged_native_token_derivative(
    tmp_path, monkeypatch
):
    native_tokens = np.arange(2 * 2 * 3 * 4, dtype=np.float16).reshape(2, 2, 3, 4)
    container = DataContainer(
        X=np.zeros((2, 2, 100), dtype=np.float32),
        dims=("obs", "channel", "time"),
        coords={
            "obs": np.asarray(["window-0", "window-1"], dtype=object),
            "channel": np.asarray(["Fz", "Cz"], dtype=object),
            "time": np.arange(100),
            "subject": np.asarray(["0001", "0001"], dtype=object),
            "study_id": np.asarray(["0001", "0001"], dtype=object),
            "session": np.asarray(["01", "01"], dtype=object),
            "run": np.asarray(["01", "01"], dtype=object),
        },
        ids=np.asarray(["window-0", "window-1"], dtype=object),
        meta={"sfreq": 10.0, "units": "uV"},
    )

    class _AvailableCapability:
        status = "available"
        reason = None

        @staticmethod
        def to_dict():
            return {"status": "available"}

    class _FakeExtractor:
        def __init__(self, model_key, *, pooling, store_tokens, **kwargs):
            assert model_key == "reve"
            assert store_tokens
            self.pooling = pooling

        def extract(self, X, *, window_start, window_stop, metadata, **kwargs):
            value = 1.0 if self.pooling == "mean" else 2.0
            windows = np.full((len(X), 4), value, dtype=np.float32)
            return FoundationEmbeddingResult(
                window_embeddings=windows,
                recording_embedding=windows.mean(axis=0),
                window_start=window_start,
                window_stop=window_stop,
                window_index=np.arange(len(X)),
                metadata={
                    **metadata,
                    "model_key": "reve",
                    "within_window_pooling": self.pooling,
                    "token_layout": "native",
                    "token_layout_version": 1,
                    "token_source": "reve.last_hidden_state",
                    "token_axes": ["window", "channel", "time_patch", "feature"],
                    "token_observation_axes": ["channel", "time_patch"],
                    "token_feature_axis": "feature",
                },
                token_embeddings=native_tokens.copy(),
            )

    spec = type(
        "Spec",
        (),
        {
            "pretrained_sfreq": 10.0,
            "pretrained_n_times": 100,
            "pretrained_window_seconds": 10.0,
        },
    )()
    monkeypatch.setattr(efe, "build_container", lambda **kwargs: container)
    monkeypatch.setattr(efe, "get_foundation_model_spec", lambda model_key: spec)
    monkeypatch.setattr(efe, "check_capability", lambda *args, **kwargs: _AvailableCapability())
    monkeypatch.setattr(
        efe,
        "normalize_inclusive_endpoint",
        lambda raw, **kwargs: (raw, None),
    )
    monkeypatch.setattr(efe, "FoundationEmbeddingExtractor", _FakeExtractor)

    common = {
        "model_key": "reve",
        "segment_duration": 10.0,
        "overlap": 0.0,
        "use_derivatives": True,
        "window_source": "derivative",
        "store_tokens": True,
    }
    derivative_root = tmp_path / "derivatives"
    efe.run(
        {
            "dataset_name": "synthetic",
            "task": "clinical",
            "bids_root": str(tmp_path / "bids"),
            "subject_col": "study_id",
            "subjects": ["0001"],
            "conditions": ["EO_baseline"],
            "models": [common, {**common, "pooling": "attention"}],
        },
        derivative_root,
    )

    pooled_paths = sorted(derivative_root.rglob("*_embedding.npz"))
    token_paths = sorted(derivative_root.rglob("*_tokens.npz"))
    assert len(pooled_paths) == 2
    assert len(token_paths) == 1
    assert {
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["model_key"]
        for path in pooled_paths
    } == {"reve", "reve_pool-attention"}
    mean_embeddings = load_embedding_derivatives(
        derivative_root,
        model_key="reve",
        representation="epoch",
    )
    attention_embeddings = load_embedding_derivatives(
        derivative_root,
        model_key="reve_pool-attention",
        representation="epoch",
    )
    np.testing.assert_array_equal(mean_embeddings.X, np.ones((2, 4)))
    np.testing.assert_array_equal(attention_embeddings.X, np.full((2, 4), 2.0))
    loaded_tokens = load_embedding_derivatives(
        token_paths,
        model_key="reve",
        representation="token",
    )
    assert loaded_tokens.X.dtype == native_tokens.dtype
    np.testing.assert_array_equal(loaded_tokens.X, native_tokens)
    token_metadata = json.loads(token_paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert token_metadata["model_key"] == "reve"
    assert "within_window_pooling" not in token_metadata
    assert token_metadata["token_axes"] == ["window", "channel", "time_patch", "feature"]


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
    _freeze_foundation_config(config, derivative_root)

    text = (derivative_root / "config_used.yaml").read_text(encoding="utf-8")
    assert "dataset_name" in text
    assert "task" in text
    for volatile in ("subjects", "bids_root", "metadata", "derivative_root"):
        assert volatile not in text


def test_freeze_config_used_tolerates_differing_volatile_keys(tmp_path):
    """A second task processing other subjects must not trip the drift guard."""
    derivative_root = tmp_path / "deriv"
    base = {"dataset_name": "synthetic", "task": "clinical", "subjects": ["sub-0001"]}
    _freeze_foundation_config(base, derivative_root)
    # Same analysis config, different per-task subjects -> identical config_used text.
    _freeze_foundation_config({**base, "subjects": ["sub-0002"]}, derivative_root)


def test_freeze_config_used_rejects_real_config_drift(tmp_path):
    """A genuine analysis-config change against an existing root is rejected."""
    derivative_root = tmp_path / "deriv"
    base = {"dataset_name": "synthetic", "task": "clinical", "subjects": ["sub-0001"]}
    _freeze_foundation_config(base, derivative_root)
    with pytest.raises(ValueError, match="different configuration"):
        _freeze_foundation_config({**base, "task": "other"}, derivative_root)


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

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from coco_pipe.decoding.foundation_models import FoundationEmbeddingResult
from coco_pipe.io import load_embedding_derivatives, save_embedding_derivative
from coco_pipe.report import build_subject_alignment_diagnostics_section
from coco_pipe.report.elements import InteractiveTableElement
from coco_pipe.utils import slug

from eeg_adhd_epilepsy.analysis import align_subject_embeddings
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root


def test_alignment_producer_variants_reload_and_report(tmp_path, monkeypatch, caplog):
    rng = np.random.default_rng(4)
    source_root = tmp_path / "source"
    metadata_rows = []
    for subject in range(4):
        metadata_rows.append(
            {
                "study_id": f"{subject + 1:04d}",
                "patient_group_id": f"group-{subject + 1:04d}",
                "diagnosis": subject % 2,
            }
        )
        center = rng.standard_normal(8) * 4.0
        for condition_index, condition in enumerate(("EO", "EC")):
            windows = (
                rng.standard_normal((6, 8)) + center + condition_index * np.r_[2.0, np.zeros(7)]
            )
            recording_id = f"sub-{subject + 1:04d}_ses-01_run-01"
            result = FoundationEmbeddingResult(
                window_embeddings=windows,
                recording_embedding=windows.mean(0),
                window_start=np.arange(6) * 100,
                window_stop=(np.arange(6) + 1) * 100,
                window_index=np.arange(6),
                metadata={
                    "model_key": "demo",
                    "recording_id": recording_id,
                    "subject": f"{subject + 1:04d}",
                    "condition": condition,
                    "within_window_pooling": "mean",
                    "token_layout": "native",
                    "token_layout_version": 1,
                    "token_source": "test_native_output",
                    "token_axes": ["window", "feature", "channel", "time_patch"],
                    "token_observation_axes": ["channel", "time_patch"],
                    "token_feature_axis": "feature",
                },
                token_embeddings=np.moveaxis(
                    windows[:, None, None, :]
                    + rng.standard_normal((len(windows), 2, 2, windows.shape[1])) * 0.2,
                    -1,
                    1,
                ),
            )
            eeg_dir = source_root / f"sub-{subject + 1:04d}" / "ses-01" / "eeg"
            pooled_path = (
                eeg_dir
                / f"sub-{subject + 1:04d}_ses-01_task-{condition}_run-01_desc-demo_embedding.npz"
            )
            save_embedding_derivative(
                FoundationEmbeddingResult(
                    window_embeddings=result.window_embeddings,
                    recording_embedding=result.recording_embedding,
                    window_start=result.window_start,
                    window_stop=result.window_stop,
                    window_index=result.window_index,
                    metadata=result.metadata,
                ),
                pooled_path,
            )
            save_embedding_derivative(
                result,
                eeg_dir
                / (
                    f"sub-{subject + 1:04d}_ses-01_task-{condition}_run-01_"
                    "desc-demotokens_tokens.npz"
                ),
            )

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    config = {
        "dataset_name": "synthetic",
        "bids_root": str(tmp_path / "bids"),
        "source_embedding_root": str(source_root),
        "embedding_model_key": "demo",
        "source_pooling": "mean",
        "metadata": str(metadata_path),
        "subject_col": "study_id",
        "transforms": ["none", "leace", "ea_mean", "ra"],
        "diagnostic_populations": [
            "transform_training_population",
            "clinical_task_subset",
        ],
        "transform_params": {},
        "evals": [
            {
                "name": "condition_separation",
                "target_col": "condition",
                "positive_class": "EC",
            },
            {
                "name": "diagnosis",
                "target_col": "diagnosis",
                "positive_class": "1",
            },
        ],
        "n_null_permutations": 2,
        "random_state": 2,
        "diagnostic_batch_max_bytes": 512,
        "participation_ratio_max_gram_bytes": 1,
        "subject_probe_epochs": 1,
        "overwrite": False,
    }

    real_discover = align_subject_embeddings.discover_embedding_derivatives

    def reverse_token_discovery(*args, **kwargs):
        paths = real_discover(*args, **kwargs)
        return list(reversed(paths)) if kwargs.get("kind") == "token" else paths

    monkeypatch.setattr(
        align_subject_embeddings,
        "discover_embedding_derivatives",
        reverse_token_discovery,
    )
    real_load = align_subject_embeddings.load_embedding_derivatives
    token_load_batches = []

    def track_token_loads(paths, *args, **kwargs):
        if kwargs.get("representation") == "token":
            assert isinstance(paths, list)
            subjects = {
                json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["subject"]
                for path in paths
            }
            assert len(subjects) == 1
            token_load_batches.append(paths)
        return real_load(paths, *args, **kwargs)

    monkeypatch.setattr(
        align_subject_embeddings,
        "load_embedding_derivatives",
        track_token_loads,
    )

    missing_tokens_root = tmp_path / "missing_tokens"
    shutil.copytree(source_root, missing_tokens_root)
    for token_artifact in missing_tokens_root.rglob("*_tokens.*"):
        token_artifact.unlink()
    with pytest.raises(FileNotFoundError, match="Re-extract this model with store_tokens"):
        align_subject_embeddings.run(
            {
                **config,
                "source_embedding_root": str(missing_tokens_root),
                "transforms": ["ra"],
            }
        )

    incomplete_tokens_root = tmp_path / "incomplete_tokens"
    shutil.copytree(source_root, incomplete_tokens_root)
    missing_token = next(incomplete_tokens_root.rglob("*_tokens.npz"))
    missing_token.unlink()
    missing_token.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="do not exactly cover"):
        align_subject_embeddings.run(
            {
                **config,
                "source_embedding_root": str(incomplete_tokens_root),
                "transforms": ["ra"],
            }
        )
    assert not list(incomplete_tokens_root.rglob("*_proc-alignra_*"))

    assert align_subject_embeddings.run(config) == source_root
    raw = load_embedding_derivatives(source_root, representation="epoch", model_key="demo")
    assert raw.X.shape == (48, 8)
    for transform in ("leace", "ea_mean", "ra"):
        loaded = load_embedding_derivatives(
            source_root,
            representation="epoch",
            model_key=f"demo_align-{transform}",
        )
        expected_features = 36 if transform == "ra" else 8
        assert loaded.X.shape == (48, expected_features)
        assert loaded.X.dtype == np.float32
        assert np.isfinite(loaded.X).all()
    aligned_sidecar = next(source_root.rglob("*_proc-alignleace_*_embedding.json"))
    aligned_metadata = json.loads(aligned_sidecar.read_text(encoding="utf-8"))
    assert not Path(aligned_metadata["source_artifact"]).is_absolute()

    diagnostics_root = get_derivative_root(
        tmp_path / "bids", DerivativeStage.VARIANCE_DIAGNOSTICS
    ) / slug("demo")
    assert (diagnostics_root / "config_used.yaml").exists()
    diagnostics = pd.read_csv(diagnostics_root / "variance_diagnostics.csv")
    assert set(diagnostics["transform"]) == set(config["transforms"])
    assert set(diagnostics["population"]) == {
        "transform_training_population",
        "clinical_task_subset",
    }
    assert diagnostics["selection_fingerprint"].notna().all()
    assert set(diagnostics["eval_name"]) == {"condition_separation", "diagnosis"}
    clinical = diagnostics[diagnostics["population"] == "clinical_task_subset"]
    training = diagnostics[diagnostics["population"] == "transform_training_population"]
    assert set(clinical["scope"]) == {"EO", "EC", "pooled"}
    assert set(training["scope"]) == {"transform_fit_all"}
    assert {"between_subject_eta2", "between_subject_excess_over_null"}.issubset(
        set(diagnostics["metric"])
    )
    raw_fingerprints = set(
        diagnostics.loc[diagnostics["transform"] == "none", "selection_fingerprint"]
    )
    for transform in ("leace", "ea_mean", "ra"):
        transform_fingerprints = set(
            diagnostics.loc[diagnostics["transform"] == transform, "selection_fingerprint"]
        )
        assert transform_fingerprints <= raw_fingerprints
    success = json.loads(
        (source_root / "_alignment_demo_complete.json").read_text(encoding="utf-8")
    )
    assert success["transforms"] == config["transforms"]
    assert success["config_fingerprint"]
    assert success["source_inventory_signature"]
    assert token_load_batches
    progress = json.loads(
        (source_root / "_alignment_demo_progress.json").read_text(encoding="utf-8")
    )
    assert set(progress["completed_diagnostics"]) == set(config["transforms"])
    ra_rank = diagnostics[
        (diagnostics["transform"] == "ra")
        & diagnostics["metric"].str.startswith("variance_participation_ratio")
    ]
    assert set(ra_rank["status"]) == {"skipped"}
    assert ra_rank["reason"].str.contains("feature Gram matrix").all()

    real_score_diagnostics = align_subject_embeddings.score_variance_diagnostics
    real_streamed_score_diagnostics = align_subject_embeddings.score_streamed_variance_diagnostics

    def fail_if_recomputed(*args, **kwargs):
        pytest.fail("A completed diagnostic checkpoint was recomputed.")

    monkeypatch.setattr(
        align_subject_embeddings,
        "score_variance_diagnostics",
        fail_if_recomputed,
    )
    monkeypatch.setattr(
        align_subject_embeddings,
        "score_streamed_variance_diagnostics",
        fail_if_recomputed,
    )
    assert align_subject_embeddings.run(config) == source_root
    with pytest.raises(ValueError, match="different alignment configuration"):
        align_subject_embeddings.run({**config, "n_null_permutations": 3})
    monkeypatch.setattr(
        align_subject_embeddings,
        "score_variance_diagnostics",
        real_score_diagnostics,
    )
    monkeypatch.setattr(
        align_subject_embeddings,
        "score_streamed_variance_diagnostics",
        real_streamed_score_diagnostics,
    )

    resume_root = tmp_path / "resume_source"
    shutil.copytree(source_root, resume_root)
    resume_config = {
        **config,
        "bids_root": str(tmp_path / "resume_bids"),
        "source_embedding_root": str(resume_root),
        "transforms": ["ea_mean", "ra"],
    }
    for transform_name in ("ea_mean", "ra"):
        processing = align_subject_embeddings._sanitize_bids_token(
            f"align{transform_name}", "processing"
        )
        for partial_artifact in (resume_root / "sub-0004").rglob(f"*_proc-{processing}_*"):
            partial_artifact.unlink()

    real_make_transform = align_subject_embeddings.make_subject_transform
    resumed_vector_fit_groups = []
    resumed_vector_groups = []

    def track_resumed_transform(name, **params):
        transform = real_make_transform(name, **params)
        if name == "ea_mean":
            real_fit = transform.fit
            real_transform = transform.transform

            def track_fit(features, y=None, groups=None):
                resumed_vector_fit_groups.append(set(np.asarray(groups).astype(str)))
                return real_fit(features, y=y, groups=groups)

            def track_transform(features, groups=None):
                resumed_vector_groups.append(set(np.asarray(groups).astype(str)))
                return real_transform(features, groups=groups)

            transform.fit = track_fit
            transform.transform = track_transform
        return transform

    monkeypatch.setattr(
        align_subject_embeddings,
        "make_subject_transform",
        track_resumed_transform,
    )
    real_ra = align_subject_embeddings._align_and_save_ra_by_subject

    def fail_ra_import(*args, **kwargs):
        raise ImportError("pyriemann is unavailable")

    monkeypatch.setattr(
        align_subject_embeddings,
        "_align_and_save_ra_by_subject",
        fail_ra_import,
    )
    with pytest.raises(ImportError, match="pyriemann"):
        align_subject_embeddings.run(resume_config)
    assert resumed_vector_fit_groups == [{"0004"}]
    assert resumed_vector_groups == [{"0004"}]
    interrupted_progress = json.loads(
        (resume_root / "_alignment_demo_progress.json").read_text(encoding="utf-8")
    )
    assert interrupted_progress["completed_diagnostics"] == ["none", "ea_mean"]

    resumed_scores = []

    def track_resumed_scores(features, tasks, score_config, *, transform, n_observations):
        resumed_scores.append(transform)
        return real_streamed_score_diagnostics(
            features,
            tasks,
            score_config,
            transform=transform,
            n_observations=n_observations,
        )

    monkeypatch.setattr(
        align_subject_embeddings,
        "_align_and_save_ra_by_subject",
        real_ra,
    )
    monkeypatch.setattr(
        align_subject_embeddings,
        "score_streamed_variance_diagnostics",
        track_resumed_scores,
    )
    token_load_batches.clear()
    caplog.set_level("INFO", logger=align_subject_embeddings.__name__)
    assert align_subject_embeddings.run(resume_config) == resume_root
    assert resumed_scores == ["ra"]
    assert len(token_load_batches) == 1
    assert {
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["subject"]
        for path in token_load_batches[0]
    } == {"0004"}
    assert "Skipped 3 completed subject(s) for transform 'ra'; 1 pending." in caplog.text
    monkeypatch.setattr(
        align_subject_embeddings,
        "score_streamed_variance_diagnostics",
        real_streamed_score_diagnostics,
    )

    degenerate_root = tmp_path / "degenerate_source"
    shutil.copytree(source_root, degenerate_root)
    for stale_leace_artifact in degenerate_root.rglob("*_proc-alignleace_*"):
        stale_leace_artifact.unlink()
    monkeypatch.setattr(
        align_subject_embeddings,
        "make_subject_transform",
        real_make_transform,
    )

    class _DegenerateLeace:
        def fingerprint(self):
            return "degenerate-leace"

        def fit(self, X, y=None, groups=None):
            self.degenerate_ = True
            self.rank_ = X.shape[1]
            return self

    def make_degenerate_leace(name, **params):
        return _DegenerateLeace() if name == "leace" else real_make_transform(name, **params)

    monkeypatch.setattr(
        align_subject_embeddings,
        "make_subject_transform",
        make_degenerate_leace,
    )
    degenerate_config = {
        **config,
        "bids_root": str(tmp_path / "degenerate_bids"),
        "source_embedding_root": str(degenerate_root),
        "transforms": ["leace", "ea_mean"],
        "overwrite": True,
    }
    assert align_subject_embeddings.run(degenerate_config) == degenerate_root
    assert not list(degenerate_root.rglob("*_proc-alignleace_*"))
    retained = load_embedding_derivatives(
        degenerate_root,
        representation="epoch",
        model_key="demo_align-ea_mean",
    )
    assert retained.X.shape == (48, 8)

    degenerate_diagnostics = pd.read_csv(
        get_derivative_root(
            tmp_path / "degenerate_bids",
            DerivativeStage.VARIANCE_DIAGNOSTICS,
        )
        / slug("demo")
        / "variance_diagnostics.csv"
    )
    skipped = degenerate_diagnostics[degenerate_diagnostics["transform"] == "leace"]
    assert set(skipped["status"]) == {"skipped"}
    assert skipped["reason"].str.contains("degenerate").all()
    section = build_subject_alignment_diagnostics_section(degenerate_diagnostics)
    audit = next(
        child
        for child in section.children
        if isinstance(child, InteractiveTableElement)
        and child.title == "Quantitative before/after audit"
    )
    assert audit.data["reason"].fillna("").str.contains("degenerate").any()

    completion = json.loads(
        (degenerate_root / "_alignment_demo_complete.json").read_text(encoding="utf-8")
    )
    assert completion["materialized_transforms"] == ["ea_mean"]
    assert set(completion["skipped_transforms"]) == {"leace"}


def test_aligned_artifact_batch_source_bounds_batches_and_validates(monkeypatch):
    pooled_ids = np.asarray(["a", "b", "c"], dtype=object)
    containers = {
        "first.npz": SimpleNamespace(
            X=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ids=np.asarray(["a", "b"], dtype=object),
        ),
        "second.npz": SimpleNamespace(
            X=np.asarray([[5.0, 6.0]], dtype=np.float32),
            ids=np.asarray(["c"], dtype=object),
        ),
    }

    def fake_load(paths, **kwargs):
        assert kwargs == {"representation": "epoch", "model_key": "demo_align-ra"}
        return containers[Path(paths[0]).name]

    monkeypatch.setattr(align_subject_embeddings, "load_embedding_derivatives", fake_load)
    source = align_subject_embeddings._AlignedArtifactBatchSource(
        [Path("first.npz"), Path("second.npz")],
        pooled_ids=pooled_ids,
        model_key="demo",
        transform_name="ra",
        max_batch_bytes=16,
    )
    batches = list(source)
    assert [len(rows) for rows, _ in batches] == [1, 1, 1]
    np.testing.assert_array_equal(np.concatenate([rows for rows, _ in batches]), np.arange(3))

    missing = align_subject_embeddings._AlignedArtifactBatchSource(
        [Path("first.npz")],
        pooled_ids=pooled_ids,
        model_key="demo",
        transform_name="ra",
        max_batch_bytes=16,
    )
    with pytest.raises(ValueError, match="do not exactly cover"):
        list(missing)

    containers["second.npz"] = SimpleNamespace(
        X=np.asarray([[np.nan, 6.0]], dtype=np.float32),
        ids=np.asarray(["c"], dtype=object),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        list(source)


def test_aligned_artifact_batch_source_rejects_duplicate_and_inconsistent_rows(
    monkeypatch,
):
    pooled_ids = np.asarray(["a", "b"], dtype=object)
    containers = {
        "first.npz": SimpleNamespace(
            X=np.ones((1, 2), dtype=np.float32),
            ids=np.asarray(["a"], dtype=object),
        ),
        "duplicate.npz": SimpleNamespace(
            X=np.ones((1, 2), dtype=np.float32),
            ids=np.asarray(["a"], dtype=object),
        ),
        "dimension.npz": SimpleNamespace(
            X=np.ones((1, 3), dtype=np.float32),
            ids=np.asarray(["b"], dtype=object),
        ),
    }
    monkeypatch.setattr(
        align_subject_embeddings,
        "load_embedding_derivatives",
        lambda paths, **kwargs: containers[Path(paths[0]).name],
    )

    duplicate = align_subject_embeddings._AlignedArtifactBatchSource(
        [Path("first.npz"), Path("duplicate.npz")],
        pooled_ids=pooled_ids,
        model_key="demo",
        transform_name="ra",
        max_batch_bytes=64,
    )
    with pytest.raises(ValueError, match="absent|duplicate"):
        list(duplicate)

    inconsistent = align_subject_embeddings._AlignedArtifactBatchSource(
        [Path("first.npz"), Path("dimension.npz")],
        pooled_ids=pooled_ids,
        model_key="demo",
        transform_name="ra",
        max_batch_bytes=64,
    )
    with pytest.raises(ValueError, match="inconsistent feature dimensions"):
        list(inconsistent)

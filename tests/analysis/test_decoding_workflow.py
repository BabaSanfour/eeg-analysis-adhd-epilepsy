import json
import logging

import numpy as np
import pandas as pd
import pytest
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis import classical_decoding as decoding
from eeg_adhd_epilepsy.analysis.utils.decoding import build_classical_plan


def test_classical_empty_sweep_logging_summarizes_failures(tmp_path, caplog):
    failures = [
        {"reason": "ValueError: not enough classes"},
        {"reason": "ValueError: not enough classes"},
        {"reason": "RuntimeError: missing target"},
    ]

    caplog.set_level(logging.WARNING, logger=decoding.LOGGER.name)
    decoding._log_enumeration_failures(
        failures,
        unit_count=0,
        derivative_root=tmp_path,
    )

    assert "No classical decoding units were enumerated" in caplog.text
    assert "Enumeration skip x2: ValueError: not enough classes" in caplog.text
    assert str(tmp_path / "failures.csv") in caplog.text


def test_classical_decoding_end_to_end_with_synthetic_container(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n_groups = 10
    groups = np.repeat([f"p{idx:02d}" for idx in range(n_groups)], 2)
    labels = np.repeat(["Control"] * 5 + ["ADHD"] * 5, 2)
    X = rng.normal(size=(20, 6))
    X[:, 0] += (labels == "ADHD").astype(float) * 1.5
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": [f"band_theta_ch-F{idx}" for idx in range(6)],
            "combined_diagnosis": labels,
            "patient_group_id": groups,
            "subject": groups,
            "session": ["01"] * len(groups),
            "condition": ["EO_baseline"] * len(groups),
        },
        ids=np.array([f"r{idx:03d}" for idx in range(len(groups))]),
    )
    monkeypatch.setattr(decoding, "build_dataset", lambda *args, **kwargs: container)

    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    config = {
        "bids_root": str(bids_root),
        "dataset_name": "synthetic",
        "input_mode": "descriptors",
        "subject_col": "subject",
        "session_col": "session",
        "group_col": "patient_group_id",
        "conditions": ["EO_baseline"],
        "run_pooled": False,
        "analysis_modes": ["flat"],
        "models": {
            "logreg_l1": {
                "estimator": "LogisticRegression",
                "analysis_modes": ["flat"],
                "params": {
                    "penalty": "l1",
                    "solver": "liblinear",
                    "class_weight": "balanced",
                    "max_iter": 1000,
                },
            }
        },
        "feature_selection": [{"name": "baseline", "method": "none"}],
        "metrics": ["accuracy", "balanced_accuracy", "roc_auc"],
        "evals": [
            {
                "name": "adhd",
                "target_col": "combined_diagnosis",
                "label_map": {"Control": "0", "ADHD": "1"},
                "positive_class": "1",
            }
        ],
        "cv": {"n_splits": 5},
        "chance_method": "binomial",
        "n_permutations": 5,
        "store_null_distribution": False,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": False,
        "overwrite": False,
        "report_asset_urls": {
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
    }
    output = decoding.run(config)
    assert (output / "_SUCCESS").exists()
    assert (output / "sweep_results.csv").exists()
    unit_root = output / "artifacts" / "fits" / "fit_eo_baseline_adhd_flat_all_baseline"
    assert (unit_root / "predictions.csv").exists()
    assert list(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic").glob(
            "descriptors_cfg-*/dataset_summary.html"
        )
    )
    assert (output.parent / "head_to_head_comparison.csv").exists()
    assert (
        tmp_path / "reports" / "summary" / "decoding" / "synthetic" / "head_to_head_comparison.html"
    ).exists()

    assert (
        "Model"
        in np.genfromtxt(unit_root / "summary.csv", delimiter=",", dtype=str, max_rows=1).tolist()
    )
    assert (unit_root / "model_artifacts.csv").exists()

    # dim_reduction-parity run tracking: runs/ inventory + summary + leaderboard.
    runs_dir = output / "runs"
    assert (runs_dir / "sweep_runs.json").exists()
    summary = json.loads((runs_dir / "run_summary.json").read_text())
    assert summary["status"] == "SUCCESS"
    assert summary["unit_success"] >= 1
    assert summary["run_variant"] == output.name
    leaderboard = json.loads((runs_dir / "leaderboard.json").read_text())
    assert leaderboard
    assert leaderboard[0]["primary_metric_name"] in {"balanced_accuracy", "accuracy"}

    # --reports-only reproduces reports from disk without refitting.
    predictions_mtime = (unit_root / "predictions.csv").stat().st_mtime_ns
    summary_html = next(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic").glob(
            "descriptors_cfg-*/dataset_summary.html"
        )
    )
    summary_html.unlink()
    decoding.run({**config, "reports_only": True})
    assert summary_html.exists()
    assert (unit_root / "predictions.csv").stat().st_mtime_ns == predictions_mtime

    resumed = decoding.run(config)
    resumed_results = np.genfromtxt(resumed / "sweep_results.csv", delimiter=",", dtype=str)
    assert "resumed" in resumed_results
    assert (resumed / "_SUCCESS").exists()
    sweep = pd.read_csv(output / "sweep_results.csv")
    assert "p_value_fdr" in sweep.columns


def _compact_config(tmp_path, *, input_mode, analysis_modes):
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir(parents=True)
    return {
        "bids_root": str(bids_root),
        "dataset_name": f"synthetic_{input_mode}",
        "input_mode": input_mode,
        "subject_col": "subject",
        "session_col": "session",
        "group_col": "patient_group_id",
        "conditions": ["EO_baseline"],
        "run_pooled": False,
        "analysis_modes": analysis_modes,
        "models": {
            "logreg": {
                "estimator": "LogisticRegression",
                "analysis_modes": analysis_modes,
                "params": {"solver": "liblinear", "max_iter": 200},
            }
        },
        "feature_selection": [{"name": "baseline", "method": "none"}],
        "metrics": ["accuracy"],
        "evals": [
            {
                "name": "adhd",
                "target_col": "combined_diagnosis",
                "label_map": {"Control": "0", "ADHD": "1"},
                "positive_class": "1",
            }
        ],
        "cv": {"n_splits": 2},
        "chance_method": "binomial",
        "n_permutations": 5,
        "store_null_distribution": False,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": False,
        "overwrite": False,
        "report_asset_urls": {
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
    }


def _observation_coords():
    groups = np.repeat([f"p{idx:02d}" for idx in range(8)], 2)
    labels = np.repeat(["Control"] * 4 + ["ADHD"] * 4, 2)
    return (
        groups,
        labels,
        {
            "combined_diagnosis": labels,
            "patient_group_id": groups,
            "subject": groups,
            "session": ["01"] * len(groups),
            "condition": ["EO_baseline"] * len(groups),
        },
    )


def test_condition_separation_is_only_enumerated_for_pooled_scope(tmp_path):
    groups, _labels, coords = _observation_coords()
    container = DataContainer(
        X=np.ones((len(groups), 2)),
        dims=("obs", "feature"),
        coords={**coords, "feature": ["f1", "f2"]},
        ids=np.asarray([f"r{idx:03d}" for idx in range(len(groups))]),
    )
    config = _compact_config(
        tmp_path,
        input_mode="descriptors",
        analysis_modes=["flat"],
    )
    config["evals"] = [
        {
            "name": "condition_separation",
            "target_col": "condition",
            "group_col": "patient_group_id",
            "positive_class": "EC_baseline",
        }
    ]
    plan = build_classical_plan(config)

    failures: list = []
    units = decoding._classical_scope_units(
        plan,
        "EO_baseline",
        container,
        config=config,
        derivative_root=tmp_path,
        failures=failures,
    )

    assert units == []
    assert failures == []


def test_foundation_embedding_decoding_flat_mode(tmp_path, monkeypatch):
    groups, labels, coords = _observation_coords()
    rng = np.random.default_rng(11)
    X = rng.normal(size=(len(groups), 12))
    X[:, 0] += (labels == "ADHD").astype(float)
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={**coords, "feature": [f"embedding_{idx:04d}" for idx in range(12)]},
        ids=np.asarray([f"r{idx:03d}" for idx in range(len(groups))]),
    )
    monkeypatch.setattr(decoding, "build_dataset", lambda *args, **kwargs: container)

    embedding_config = _compact_config(
        tmp_path / "embedding",
        input_mode="foundation_embeddings",
        analysis_modes=["flat"],
    )
    embedding_output = decoding.run(embedding_config)
    embedding_sweep = pd.read_csv(embedding_output / "sweep_results.csv")
    assert set(embedding_sweep["status"]) == {"success"}
    assert (embedding_output / "_SUCCESS").exists()


def test_foundation_embedding_decoding_rejects_nonflat_mode(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="foundation_embeddings",
        analysis_modes=["flat", "sensor"],
    )
    with pytest.raises(ValueError, match="analysis_mode='sensor'"):
        build_classical_plan(config)


def test_nonflat_descriptor_sweeps(tmp_path, monkeypatch):
    groups, labels, coords = _observation_coords()
    rng = np.random.default_rng(15)
    X = rng.normal(size=(len(groups), 2, 2))
    X[:, 0, 0] += (labels == "ADHD").astype(float)
    container = DataContainer(
        X=X,
        dims=("obs", "sensor", "feature"),
        coords={
            **coords,
            "sensor": ["Fz", "Cz"],
            "feature": ["alpha", "entropy"],
            "feature_family": ["band", "complexity"],
            "feature_subfamily": ["alpha", "entropy"],
            "feature_descriptor": ["alpha", "entropy"],
        },
        ids=np.asarray([f"r{idx:03d}" for idx in range(len(groups))]),
    )
    monkeypatch.setattr(decoding, "build_dataset", lambda *args, **kwargs: container)
    config = _compact_config(
        tmp_path,
        input_mode="descriptors",
        analysis_modes=["flat", "sensor", "descriptor_sensor"],
    )
    config["feature_selection"] = [
        {"name": "baseline", "method": "none"},
        {
            "name": "sfs",
            "method": "sfs",
            "direction": "forward",
            "tol": 0.0,
            "analysis_modes": ["sensor"],
        },
    ]
    output = decoding.run(config)
    sweep = pd.read_csv(output / "sweep_results.csv")
    assert {"flat", "sensor", "descriptor_sensor"}.issubset(set(sweep["analysis_mode"]))
    assert (
        output
        / "artifacts"
        / "fits"
        / "fit_eo_baseline_adhd_sensor_fz_sfs"
        / "selected_features.csv"
    ).exists()
    fits_root = output / "artifacts" / "fits"
    assert list(fits_root.glob("*descriptor_sensor*baseline*"))
    assert not list(fits_root.glob("*descriptor_sensor*sfs*"))


def test_classical_plan_rejects_reduced_dimensions(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="reduced_dimensions",
        analysis_modes=["flat"],
    )
    with pytest.raises(ValueError, match="Invalid input_mode"):
        build_classical_plan(config)


def test_classical_plan_requires_explicit_feature_selection(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="descriptors",
        analysis_modes=["flat"],
    )
    config.pop("feature_selection")
    with pytest.raises(ValueError, match="feature_selection"):
        build_classical_plan(config)


def test_foundation_run_wires_sweep_and_capability_without_gpu(tmp_path, monkeypatch):
    """Drive foundation_decoding.run() through the skip path (no neural training).

    Verifies the foundation-specific plumbing around the shared sweep scaffold:
    scope build, capability_matrix.csv, the runs/ inventory, the report, and a
    --reports-only re-run — all without model checkpoints or a GPU.
    """
    from eeg_adhd_epilepsy.analysis import foundation_decoding as fdn

    bids_root = tmp_path / "BIDS"
    bids_root.mkdir(parents=True)
    config = {
        "bids_root": str(bids_root),
        "dataset_name": "synthetic_foundation",
        "conditions": ["EO_baseline"],
        "run_pooled": False,
        "models": [
            {
                "model_key": "labram",
                "segment_duration": 10.0,
                "overlap": 0.0,
                "use_derivatives": True,
                "window_source": "derivative",
                "window_mismatch_policy": "raise",
                "backend": "auto",
                "backend_kwargs": {},
                "lora": {},
                "trainer": {"linear_probe": {}, "full": {}, "lora": {}},
            }
        ],
        "train_modes": ["linear_probe"],
        "training_defaults": {"linear_probe": {}},
        "metrics": ["accuracy"],
        "cv": {"n_splits": 2},
        "chance_method": "binomial",
        "n_permutations": 5,
        "store_null_distribution": False,
        "session_col": "session",
        "subject_col": "subject",
        "device": "cpu",
        "precision": "fp32",
        "on_unsupported": "skip",
        "class_weight": "balanced",
        "n_jobs": 1,
        "random_state": 42,
        "verbose": False,
        "overwrite": False,
        "report_asset_urls": {
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
    }

    # Bypass data loading + coco-pipe model spec; the streamed scope generator
    # yields no unit batches and records a single capability skip (as
    # on_unsupported=skip would produce for a real backend).
    skip = {
        "condition": "EO_baseline",
        "target": "adhd",
        "model_key": "labram",
        "train_mode": "linear_probe",
        "primary": True,
        "status": "skipped",
        "reason": "backend unavailable in test",
        "capability": {
            "condition": "EO_baseline",
            "target": "adhd",
            "model_key": "labram",
            "train_mode": "linear_probe",
            "status": "unavailable",
            "reason": "backend unavailable in test",
        },
    }
    monkeypatch.setattr(
        fdn,
        "_iter_foundation_unit_batches",
        lambda config, metadata, cfg_hash, derivative_root, failures: (
            failures.append(skip) or iter(())
        ),
    )

    output = fdn.run(config)
    assert (output / "runs" / "sweep_runs.json").exists()
    summary = json.loads((output / "runs" / "run_summary.json").read_text())
    assert summary["status"] == "FAILED"  # only a skip, no successful units
    capability = pd.read_csv(output / "capability_matrix.csv")
    assert (capability["status"] == "unavailable").any()
    assert list(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic_foundation").glob(
            "foundation_cfg-*/dataset_summary.html"
        )
    )

    # --reports-only re-run reads the persisted inventory and regenerates reports.
    summary_html = next(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic_foundation").glob(
            "foundation_cfg-*/dataset_summary.html"
        )
    )
    summary_html.unlink()
    fdn.run({**config, "reports_only": True})
    assert summary_html.exists()

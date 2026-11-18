import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from coco_pipe.decoding import ExperimentResult
from coco_pipe.io import (
    DataContainer,
)

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


def test_classical_resume_logging_summarizes_completed_units(tmp_path, caplog):
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "run_manifest.json").write_text('{"status": "success"}')
    units = [
        SimpleNamespace(output_dir=completed, overwrite=False),
        SimpleNamespace(output_dir=tmp_path / "pending", overwrite=False),
        SimpleNamespace(output_dir=completed, overwrite=True),
    ]

    caplog.set_level(logging.INFO, logger=decoding.LOGGER.name)
    decoding._log_scope_resume_summary(units, scope="EO_baseline")

    assert "Skipped 1 existing decoding unit(s) for scope 'EO_baseline'; 2 pending." in caplog.text


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
    unit_root = (
        output / "artifacts" / "fits" / "fit_eo_baseline_adhd_flat_all_none_all_dimensions_baseline"
    )
    assert sorted(path.name for path in unit_root.iterdir()) == [
        "result.joblib",
        "run_manifest.json",
    ]
    assert list(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic").glob(
            "descriptors_cfg-*/dataset_summary.html"
        )
    )
    assert (output.parent / "head_to_head_comparison.csv").exists()
    assert (
        tmp_path / "reports" / "summary" / "decoding" / "synthetic" / "head_to_head_comparison.html"
    ).exists()

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
    result_mtime = (unit_root / "result.joblib").stat().st_mtime_ns
    summary_html = next(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic").glob(
            "descriptors_cfg-*/dataset_summary.html"
        )
    )
    summary_html.unlink()
    decoding.run({**config, "reports_only": True})
    assert summary_html.exists()
    assert (unit_root / "result.joblib").stat().st_mtime_ns == result_mtime

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
    load_calls = {"count": 0}

    def load_container(*args, **kwargs):
        load_calls["count"] += 1
        return container

    monkeypatch.setattr(decoding, "build_dataset", load_container)

    embedding_config = _compact_config(
        tmp_path / "embedding",
        input_mode="foundation_embeddings",
        analysis_modes=["flat"],
    )
    embedding_config["embedding_model_key"] = "demo"
    embedding_config["transforms"] = ["none", "leace"]
    embedding_config["transform_params"] = {"leace": {"condition_column": "condition"}}
    for defaulted_key in (
        "session_col",
        "verbose",
        "overwrite",
    ):
        embedding_config.pop(defaulted_key)
    embedding_output = decoding.run(embedding_config)
    embedding_sweep = pd.read_csv(embedding_output / "sweep_results.csv")
    assert set(embedding_sweep["status"]) == {"success"}
    assert set(embedding_sweep["transform"]) == {"none", "leace"}
    # One load: none and fold-local leace share the raw vectors.
    assert load_calls["count"] == 1
    assert (embedding_output / "_SUCCESS").exists()
    assert (embedding_output.parent / "foundation_transform_comparison.csv").exists()
    assert (
        tmp_path
        / "embedding"
        / "reports"
        / "summary"
        / "decoding"
        / "synthetic_foundation_embeddings"
        / "foundation_transform_comparison.html"
    ).exists()


def test_foundation_embedding_decoding_rejects_nonflat_mode(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="foundation_embeddings",
        analysis_modes=["flat", "sensor"],
    )
    with pytest.raises(ValueError, match="analysis_mode='sensor'"):
        build_classical_plan(config)


def test_foundation_plan_builds_transform_and_reduction_axes(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="foundation_embeddings",
        analysis_modes=["flat"],
    )
    config["transforms"] = ["none", "leace"]
    config["transform_params"] = {"leace": {"eta2": 0.5, "condition_column": "condition"}}
    config["reducer"] = {"enabled": True, "n_components": 0.9}
    plan = build_classical_plan(config)
    assert plan.transforms == ["none", "leace"]
    assert plan.reducer_enabled
    with pytest.raises(ValueError, match="Unknown transforms"):
        build_classical_plan({**config, "transforms": ["none", "ea_mean"]})

    groups, _labels, coords = _observation_coords()
    container = DataContainer(
        X=np.random.default_rng(3).normal(size=(len(groups), 8)),
        dims=("obs", "feature"),
        coords={**coords, "feature": [f"e{i}" for i in range(8)]},
        ids=np.asarray([f"r{idx:03d}" for idx in range(len(groups))]),
    )
    failures: list = []
    units = decoding._classical_scope_units(
        plan,
        "EO_baseline",
        container,
        config=config,
        derivative_root=tmp_path,
        failures=failures,
    )
    assert failures == []
    assert len(units) == 4
    assert {unit.context["transform"] for unit in units} == {"none", "leace"}
    assert {unit.context["reduction_mode"] for unit in units} == {
        "all_dimensions",
        "pca",
    }
    leace = next(unit for unit in units if unit.context["transform"] == "leace")
    assert leace.experiment_config.erasure.enabled
    assert leace.experiment_config.erasure.condition_column == "condition"

    # a list of sizes: one mode each, sizes as wide as the unit skipped, all dimensions dropped
    config["reducer"] = {"enabled": True, "n_components": [2, 8, 0.9], "keep_all_dimensions": False}
    config["transforms"] = ["none"]
    units = decoding._classical_scope_units(
        build_classical_plan(config),
        "EO_baseline",
        container,
        config=config,
        derivative_root=tmp_path,
        failures=failures,
    )
    assert {
        u.context["reduction_mode"]: u.experiment_config.reducer.n_components for u in units
    } == {
        "pca_2": 2,
        "pca_0.9": 0.9,
    }


def test_grouped_descriptor_sweeps(tmp_path, monkeypatch):
    groups, labels, coords = _observation_coords()
    rng = np.random.default_rng(15)
    X = rng.normal(size=(len(groups), 4))
    X[:, 0] += (labels == "ADHD").astype(float)
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            **coords,
            "feature": [
                "band_alpha_ch-Fz",
                "complexity_entropy_ch-Fz",
                "band_alpha_ch-Cz",
                "complexity_entropy_ch-Cz",
            ],
            "feature_family": ["band", "complexity", "band", "complexity"],
            "feature_channel": ["Fz", "Fz", "Cz", "Cz"],
            "feature_measure": ["alpha", "entropy", "alpha", "entropy"],
            "feature_subfamily": ["alpha", "entropy", "alpha", "entropy"],
            "feature_descriptor": ["alpha", "entropy", "alpha", "entropy"],
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
        {
            "name": "sfsepoch",
            "method": "sfs",
            "direction": "forward",
            "tol": 0.0,
            "representations": ["epoch"],
        },
    ]
    config["representation"] = "recording"
    config["chance_method"] = "permutation"
    config["permutation_modes"] = ["flat"]
    output = decoding.run(config)
    sweep = pd.read_csv(output / "sweep_results.csv")
    assert {"flat", "sensor", "descriptor_sensor"}.issubset(set(sweep["analysis_mode"]))
    fits_root = output / "artifacts" / "fits"

    def null_method(unit):
        path = fits_root / f"fit_eo_baseline_adhd_{unit}" / "result.joblib"
        assessment = ExperimentResult.load(path).get_statistical_assessment()
        return set(pd.DataFrame(assessment)["NullMethod"])

    # only flat units without feature selection are permuted
    assert "binomial" not in null_method("flat_all_none_all_dimensions_baseline")
    assert null_method("sensor_fz_none_all_dimensions_baseline") == {"binomial"}
    assert null_method("sensor_fz_none_all_dimensions_sfs") == {"binomial"}
    sfs_result = ExperimentResult.load(
        output
        / "artifacts"
        / "fits"
        / "fit_eo_baseline_adhd_sensor_fz_none_all_dimensions_sfs"
        / "result.joblib"
    )
    assert not sfs_result.get_selected_features().empty
    assert list(fits_root.glob("*descriptor_sensor*baseline*"))
    assert not list(fits_root.glob("*descriptor_sensor*sfs*"))
    # a variant limited to another representation is not run
    assert not list(fits_root.glob("*sfsepoch*"))


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


def test_foundation_targets_keep_only_the_named_evals():
    from eeg_adhd_epilepsy.analysis import foundation_decoding as fdn

    evals = [{"name": "adhd", "target_col": "a"}, {"name": "sex", "target_col": "s"}]
    assert fdn._selected_evals({"evals": evals}) == evals
    assert fdn._selected_evals({"evals": evals, "targets": ["sex"]}) == [evals[1]]
    with pytest.raises(ValueError, match="age"):
        fdn._selected_evals({"evals": evals, "targets": ["age"]})


def test_foundation_model_entry_runs_under_its_name_and_modes(tmp_path, monkeypatch):
    """A model entry runs under its name and modes; seeds and a grid set each fit."""
    import coco_pipe.decoding.foundation_models as fm

    from eeg_adhd_epilepsy.analysis import foundation_decoding as fdn

    container = SimpleNamespace(meta={"sfreq": 200.0}, coords={"channel": ["Fz", "Cz"]})
    monkeypatch.setattr(fm, "normalize_inclusive_endpoint", lambda c, **_: (c, None))
    monkeypatch.setattr(
        fm,
        "check_capability",
        lambda key, **kwargs: SimpleNamespace(
            status="unavailable", reason=f"{key} {kwargs['backend_kwargs']}", to_dict=dict
        ),
    )
    scope = (None, pd.Series([0, 1]), np.array([0, 1]), None, 2)
    monkeypatch.setattr(fdn, "prepare_decoding_scope", lambda *args, **kwargs: scope)
    model_cfg = {
        "model_key": "reve",
        "name": "reve_scratch",
        "train_modes": ["full"],
        "segment_duration": 10.0,
        "window_mismatch_policy": "raise",
        "backend": "auto",
        "backend_kwargs": {"pretrained": False, "seed": 0},
    }
    records = list(
        fdn._enumerate_foundation_scope(
            condition="EO_baseline",
            model_cfg=model_cfg,
            container=container,
            spec=None,
            provenance={},
            evals=[{"name": "adhd", "target_col": "a", "group_col": "g"}],
            train_modes=["linear_probe", "full", "lora"],
            config={
                "session_col": "session",
                "subject_col": "subject",
                "cv": {"n_splits": 2},
                "random_state": 0,
                "on_unsupported": "skip",
            },
            derivative_root=tmp_path,
        )
    )
    assert [(r["model_key"], r["train_mode"]) for r in records] == [("reve_scratch", "full")]
    assert records[0]["reason"] == "reve {'pretrained': False, 'seed': 0}"
    assert "foundation_reve_scratch_full" in records[0]["output_dir"]

    # seeds: one fit each on the same split; a mode's grid is searched on inner folds
    monkeypatch.setattr(
        fm,
        "check_capability",
        lambda key, **kwargs: SimpleNamespace(status="available", to_dict=dict),
    )
    container = SimpleNamespace(X=np.zeros((2, 2, 2000)), **vars(container))
    metadata = pd.DataFrame({"sample_id": ["a", "b"]})
    scope = (container, pd.Series([0, 1]), np.array([0, 1]), metadata, 2)
    model_cfg |= {"grid": {"full": {"trainer__lr": [1e-4, 1e-3]}}, "trainer": {"full": {}}}
    units = list(
        fdn._enumerate_foundation_scope(
            condition="EO_baseline",
            model_cfg=model_cfg,
            container=container,
            spec=None,
            provenance={},
            evals=[{"name": "adhd", "target_col": "a", "group_col": "g"}],
            train_modes=["full"],
            config={
                "session_col": "session",
                "subject_col": "subject",
                "cv": {"n_splits": 2},
                "random_state": 0,
                "seeds": [1, 2],
                "training_defaults": {"full": {}},
                "device": "cpu",
                "class_weight": "balanced",
                "chance_method": "binomial",
                "n_permutations": 1,
                "store_null_distribution": False,
                "metrics": ["balanced_accuracy"],
                "verbose": False,
                "overwrite": False,
                "on_unsupported": "skip",
            },
            derivative_root=tmp_path,
        )
    )
    configs = [unit.experiment_config for unit in units]
    assert [c.random_state for c in configs] == [1, 2]
    assert {c.cv.random_state for c in configs} == {0}
    assert configs[0].grids == {"reve_scratch_full": {"trainer__lr": [1e-4, 1e-3]}}
    assert configs[0].tuning.enabled and configs[0].tuning.cv.group_key == "group_id"
    assert str(units[1].output_dir).endswith("foundation_reve_scratch_full_seed2")


def test_foundation_pooled_only_runs_only_the_pooled_scope(tmp_path, monkeypatch):
    """pooled_only loads every condition but builds units for the pooled scope alone."""
    from eeg_adhd_epilepsy.analysis import foundation_decoding as fdn

    monkeypatch.setattr(fdn, "build_loader_args", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(fdn, "get_foundation_model_spec", lambda key: None)
    monkeypatch.setattr(fdn, "foundation_provenance", lambda *a, **k: {})
    monkeypatch.setattr(fdn, "build_dataset", lambda args, meta, cond, **k: cond)
    monkeypatch.setattr(fdn, "container_pool_spec", lambda container: container)
    monkeypatch.setattr(fdn, "pool_containers_streaming", lambda specs, loaders: "+".join(specs))
    monkeypatch.setattr(
        fdn, "_foundation_scope_units", lambda cond, model, data, *a, **k: [(cond, data)]
    )
    config = {
        "conditions": ["EO", "EC"],
        "models": [
            {"model_key": "reve", "segment_duration": 10, "overlap": 0, "use_derivatives": False}
        ],
        "train_modes": ["full"],
        "evals": [{"name": "adhd"}],
    }

    def scopes(cfg):
        return list(fdn._iter_foundation_unit_batches(cfg, None, "h", tmp_path, []))

    eo, ec, pooled = [("EO", "EO")], [("EC", "EC")], [("pooled", "EO+EC")]
    assert scopes(config) == [eo, ec]
    assert scopes({**config, "run_pooled": True}) == [eo, ec, pooled]
    assert scopes({**config, "pooled_only": True}) == [pooled]


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
    shared_comparison_calls = []
    monkeypatch.setattr(
        fdn,
        "generate_head_to_head_report",
        lambda **kwargs: shared_comparison_calls.append(kwargs),
    )
    config["write_shared_comparison_report"] = False

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
    assert shared_comparison_calls == []

    # --reports-only re-run reads the persisted inventory and regenerates reports.
    summary_html = next(
        (tmp_path / "reports" / "summary" / "decoding" / "synthetic_foundation").glob(
            "foundation_cfg-*/dataset_summary.html"
        )
    )
    summary_html.unlink()
    fdn.run({**config, "reports_only": True})
    assert summary_html.exists()

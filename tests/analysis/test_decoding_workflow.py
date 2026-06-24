import numpy as np
import pandas as pd
import pytest
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis import classical_decoding as decoding


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
        "output_group": "tests",
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
                "params": {
                    "penalty": "l1",
                    "solver": "liblinear",
                    "class_weight": "balanced",
                    "max_iter": 1000,
                },
            }
        },
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
        "n_jobs": 1,
        "random_state": 42,
        "report_asset_urls": {
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
    }
    output = decoding.run(config)
    assert (output / "_SUCCESS").exists()
    assert (output / "sweep_results.csv").exists()
    assert (
        output / "eo_baseline" / "adhd" / "flat" / "all" / "baseline" / "predictions.csv"
    ).exists()
    assert (
        tmp_path
        / "reports"
        / "summary"
        / "decoding"
        / "tests"
        / "synthetic"
        / "descriptors"
        / "dataset_summary.html"
    ).exists()
    assert (output.parent / "head_to_head_comparison.csv").exists()
    assert (
        tmp_path
        / "reports"
        / "summary"
        / "decoding"
        / "tests"
        / "synthetic"
        / "head_to_head_comparison.html"
    ).exists()

    unit_root = output / "eo_baseline" / "adhd" / "flat" / "all" / "baseline"
    assert (
        "Model"
        in np.genfromtxt(unit_root / "summary.csv", delimiter=",", dtype=str, max_rows=1).tolist()
    )
    assert (unit_root / "model_artifacts.csv").exists()

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
        "output_group": "tests",
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
                "params": {"solver": "liblinear", "max_iter": 200},
            }
        },
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
        "n_jobs": 1,
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


def test_embedding_and_reduced_dimension_decoding_modes(tmp_path, monkeypatch):
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
        analysis_modes=["flat", "sensor"],
    )
    embedding_output = decoding.run(embedding_config)
    embedding_sweep = pd.read_csv(embedding_output / "sweep_results.csv")
    assert set(embedding_sweep["status"]) == {"success"}
    failures = pd.read_csv(embedding_output / "failures.csv")
    assert "skipped" in set(failures["status"])
    assert (embedding_output / "_PARTIAL").exists()

    reduced_config = _compact_config(
        tmp_path / "reduced",
        input_mode="reduced_dimensions",
        analysis_modes=["flat"],
    )
    reduced_config["reduced_source_input_mode"] = "foundation_embeddings"
    reduced_config["reducer"] = {"n_components": 0.8}
    reduced_output = decoding.run(reduced_config)
    assert (
        reduced_output / "eo_baseline" / "adhd" / "flat" / "all" / "baseline" / "predictions.csv"
    ).exists()
    assert "p_value_fdr" in pd.read_csv(reduced_output / "sweep_results.csv").columns


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
        {
            "name": "sfs",
            "method": "sfs",
            "direction": "forward",
            "analysis_modes": ["sensor", "descriptor_sensor"],
        }
    ]
    output = decoding.run(config)
    sweep = pd.read_csv(output / "sweep_results.csv")
    assert {"flat", "sensor", "descriptor_sensor"}.issubset(set(sweep["analysis_mode"]))
    assert (
        output / "eo_baseline" / "adhd" / "sensor" / "fz" / "sfs" / "selected_features.csv"
    ).exists()
    descriptor_sensor_root = output / "eo_baseline" / "adhd" / "descriptor_sensor"
    assert list(descriptor_sensor_root.glob("*/baseline"))
    assert not list(descriptor_sensor_root.glob("*/sfs"))


def test_selection_plan_is_baseline_plus_requested_sfs():
    assert decoding._selection_specs({}) == [{"name": "baseline", "method": "none"}]
    specs = decoding._selection_specs(
        {
            "feature_selection": [
                {
                    "name": "forward",
                    "method": "sfs",
                    "analysis_modes": ["flat"],
                }
            ]
        }
    )
    assert [spec["name"] for spec in specs] == ["baseline", "forward"]
    assert [
        spec["name"]
        for spec in decoding._selection_specs_for_unit(
            specs,
            analysis_mode="sensor",
            n_available=4,
        )
    ] == ["baseline"]
    assert [
        spec["name"]
        for spec in decoding._selection_specs_for_unit(
            specs,
            analysis_mode="flat",
            n_available=4,
        )
    ] == ["baseline", "forward"]


def test_selection_plan_rejects_k_best_and_skips_one_column_sfs():
    with pytest.raises(ValueError, match="only supports method='sfs'"):
        decoding._selection_specs(
            {"feature_selection": [{"name": "top10", "method": "k_best", "n_features": 10}]}
        )
    specs = decoding._selection_specs({"feature_selection": [{"name": "sfs", "method": "sfs"}]})
    for analysis_mode in ("flat", "descriptor_sensor"):
        selected = decoding._selection_specs_for_unit(
            specs,
            analysis_mode=analysis_mode,
            n_available=1,
        )
        assert [spec["name"] for spec in selected] == ["baseline"]


def test_sfs_feature_count_stays_below_available_columns():
    config = decoding._feature_selection_config(
        {"name": "sfs", "method": "sfs", "n_features": 10},
        n_available=3,
    )
    assert config.n_features == 2


def test_removed_descriptor_analysis_modes_are_rejected(tmp_path):
    config = _compact_config(
        tmp_path,
        input_mode="descriptors",
        analysis_modes=["flat", "family"],
    )
    with pytest.raises(ValueError, match="Unsupported descriptor analysis modes"):
        decoding.run(config)


def test_transductive_input_requires_explicit_opt_in(tmp_path, monkeypatch):
    groups, labels, coords = _observation_coords()
    container = DataContainer(
        X=np.ones((len(groups), 3)),
        dims=("obs", "feature"),
        coords={**coords, "feature": ["pc1", "pc2", "pc3"]},
        ids=np.asarray([f"r{idx:03d}" for idx in range(len(groups))]),
        meta={"transductive": True},
    )
    monkeypatch.setattr(decoding, "build_dataset", lambda *args, **kwargs: container)
    config = _compact_config(
        tmp_path,
        input_mode="reduced_dimensions",
        analysis_modes=["flat"],
    )
    with pytest.raises(ValueError, match="marked transductive"):
        decoding.run(config)

    config["allow_transductive_input"] = True
    output = decoding.run(config)
    sweep = pd.read_csv(output / "sweep_results.csv")
    assert sweep["transductive_input"].all()
    assert not sweep["primary"].all()


def test_pooled_scope_preserves_transductive_flag(tmp_path, monkeypatch):
    groups, labels, coords = _observation_coords()
    transductive = DataContainer(
        X=np.ones((len(groups), 3)),
        dims=("obs", "feature"),
        coords={**coords, "feature": ["pc1", "pc2", "pc3"]},
        ids=np.asarray([f"a{idx:03d}" for idx in range(len(groups))]),
        meta={"transductive": True},
    )
    inductive = DataContainer(
        X=np.ones((len(groups), 3)),
        dims=("obs", "feature"),
        coords={**coords, "feature": ["pc1", "pc2", "pc3"]},
        ids=np.asarray([f"b{idx:03d}" for idx in range(len(groups))]),
    )
    containers = iter([transductive, inductive])
    monkeypatch.setattr(
        decoding,
        "build_dataset",
        lambda *args, **kwargs: next(containers),
    )
    config = _compact_config(
        tmp_path,
        input_mode="reduced_dimensions",
        analysis_modes=["flat"],
    )
    config.update(
        {
            "conditions": ["EO_baseline", "EC_baseline"],
            "run_pooled": True,
            "allow_transductive_input": True,
        }
    )
    output = decoding.run(config)
    sweep = pd.read_csv(output / "sweep_results.csv")
    pooled = sweep[sweep["scope"] == "pooled"]
    assert not pooled.empty
    assert pooled["transductive_input"].all()
    assert not pooled["primary"].all()

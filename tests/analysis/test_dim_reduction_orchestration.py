"""Multi-mode orchestration, budgeting, and roll-up for dimensionality reduction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY
from coco_pipe.io import DataContainer
from coco_pipe.utils import slug
from joblib.parallel import get_active_backend

import eeg_adhd_epilepsy.analysis.dimensionality_reduction as dim_reduction
from eeg_adhd_epilepsy.analysis.utils.common import base_layout_mode
from eeg_adhd_epilepsy.analysis.utils.dim_reduction import (
    DEFAULT_DIM_REDUCTION_SELECTION_METRIC,
    SEPARATION_RF_METRIC_KEY,
    build_and_validate_mode_specs,
    group_fit_requests,
)
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root
from eeg_adhd_epilepsy.reports._common import AlignmentDiagnosticsSpec
from eeg_adhd_epilepsy.reports.dim_reduction import (
    collect_mode_leaderboard,
    generate_rollup_report,
)

# --- Mode / budgeting resolution -------------------------------------------------


def _mode_args(analysis_modes, *, input_mode, representation=None):
    """Minimal args for build_and_validate_mode_specs (the only fields it reads)."""
    return SimpleNamespace(
        analysis_modes=analysis_modes,
        input_mode=input_mode,
        representation=representation,
    )


def test_base_layout_mode():
    assert base_layout_mode("descriptors") == "sensor"
    assert base_layout_mode("raw") == "flat"
    assert base_layout_mode("foundation_embeddings") == "flat"


def test_build_mode_specs_from_analysis_modes_mapping():
    specs, _ = build_and_validate_mode_specs(
        _mode_args(
            {
                "flat": {"reducers": ["PCA", "UMAP"], "n_components": [2, 5, 10]},
                "feature": {"reducers": ["PCA"], "n_components": [2, 3]},
            },
            input_mode="descriptors",
            representation="features",
        )
    )
    # Declaration order preserved; each mode fully declares its reducers + sweep.
    assert list(specs) == ["flat", "feature"]
    assert specs["flat"] == {"reducers": ["PCA", "UMAP"], "n_components": [2, 5, 10]}
    assert specs["feature"] == {"reducers": ["PCA"], "n_components": [2, 3]}


def test_build_mode_specs_preserves_extra_spec_keys():
    # Any extra key in a mode spec rides through verbatim (raw's averaging
    # granularity is now top-level, not per-spec, but a per-spec override is still
    # preserved on the returned spec).
    specs, _ = build_and_validate_mode_specs(
        _mode_args(
            {
                "flat": {"family": "band", "reducers": ["PCA", "UMAP"], "n_components": [2, 5, 10]},
                "feature": {"family": "complexity", "reducers": ["PCA"], "n_components": [2, 3]},
            },
            input_mode="descriptors",
            representation="features",
        )
    )
    assert list(specs) == ["flat", "feature"]
    assert specs["feature"]["family"] == "complexity"
    assert specs["feature"]["reducers"] == ["PCA"]


def test_build_mode_specs_requires_reducers_per_mode():
    # A mode that names no reducers is a config error, not a silent skip.
    with pytest.raises(ValueError, match="must list `reducers`"):
        build_and_validate_mode_specs(
            _mode_args(
                {
                    "flat": {"reducers": ["PCA"], "n_components": [2]},
                    "family": {"n_components": [2]},
                },
                input_mode="descriptors",
            )
        )


def test_build_mode_specs_requires_n_components_per_mode():
    # n_components is the only source of truth for the sweep — every mode needs it.
    with pytest.raises(ValueError, match="must list `n_components`"):
        build_and_validate_mode_specs(
            _mode_args({"flat": {"reducers": ["PCA"]}}, input_mode="descriptors")
        )


def test_build_mode_specs_requires_a_source():
    with pytest.raises(ValueError, match="analysis_modes"):
        build_and_validate_mode_specs(_mode_args(None, input_mode="raw"))


# --- Task expansion / per-mode validation ---------------------------------------


def test_resolve_mode_tasks_raw_uses_top_level_representation():
    # Raw's averaging granularity (epoch|subject) is one per run, taken from args;
    # specs carry only the sensor axis (analysis_mode) + reducers.
    _, tasks = build_and_validate_mode_specs(
        _mode_args(
            {
                "flat": {"reducers": ["PCA"], "n_components": [2]},
                "sensor": {"reducers": ["PCA"], "n_components": [2]},
            },
            input_mode="raw",
            representation="subject",
        )
    )
    assert tasks == [("flat", "subject"), ("sensor", "subject")]


def test_resolve_mode_tasks_descriptors_share_representation():
    # Specs with no representation fall back to the input's single representation.
    _, tasks = build_and_validate_mode_specs(
        _mode_args(
            {
                "flat": {"reducers": ["PCA"], "n_components": [2]},
                "family": {"reducers": ["PCA"], "n_components": [2]},
                "sensor": {"reducers": ["PCA"], "n_components": [2]},
            },
            input_mode="descriptors",
            representation="features",
        )
    )
    assert tasks == [("flat", "features"), ("family", "features"), ("sensor", "features")]


def test_validate_rejects_descriptor_mode_on_raw():
    with pytest.raises(ValueError, match="descriptor inputs"):
        build_and_validate_mode_specs(
            _mode_args(
                {"family": {"reducers": ["PCA"], "n_components": [2]}},
                input_mode="raw",
                representation="subject",
            )
        )


def test_validate_raw_axes_are_independent():
    # The two raw axes don't constrain each other: flat and sensor each accept
    # either averaging granularity (epoch or subject).
    for representation in ("epoch", "subject"):
        _, tasks = build_and_validate_mode_specs(
            _mode_args(
                {
                    "flat": {"reducers": ["PCA"], "n_components": [2]},
                    "sensor": {"reducers": ["PCA"], "n_components": [2]},
                },
                input_mode="raw",
                representation=representation,
            )
        )
        assert tasks == [("flat", representation), ("sensor", representation)]
    # A granularity outside {epoch, subject} is rejected.
    with pytest.raises(ValueError, match="epoch.*subject|subject.*epoch"):
        build_and_validate_mode_specs(
            _mode_args(
                {"sensor": {"reducers": ["PCA"], "n_components": [2]}},
                input_mode="raw",
                representation="subject_native",
            )
        )


def test_validate_foundation_flat_only():
    with pytest.raises(ValueError, match="flat"):
        build_and_validate_mode_specs(
            _mode_args(
                {"sensor": {"reducers": ["PCA"], "n_components": [2]}},
                input_mode="foundation_embeddings",
                representation="foundation_recording",
            )
        )


# --- Shared sensor-layout base container flattens for the flat unit -------------


def test_collect_flat_unit_flattens_sensor_layout_container(tmp_path):
    # A descriptor base container is loaded once in (obs, sensor, feature) layout
    # and reused for every mode; the flat unit must be flattened to 2D.
    container = DataContainer(
        X=np.arange(4 * 2 * 2, dtype=float).reshape(4, 2, 2),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": np.asarray(["Fz", "Cz"], dtype=object),
            "feature": np.asarray(["alpha_mean", "beta_mean"], dtype=object),
            "feature_family": np.asarray(["band", "band"], dtype=object),
        },
        ids=np.asarray(["r1", "r2", "r3", "r4"], dtype=object),
    )
    args = SimpleNamespace(
        input_mode="descriptors",
        representation="features",
        analysis_mode="flat",
        descriptor_families=None,
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        descriptor_table_path=str(tmp_path / "features.csv"),
        descriptor_feature_columns_path=str(tmp_path / "columns.json"),
        descriptor_max_abs_value=None,
        location_statistic=None,
        run_label="test",
        run_config_hash="cfg",
        overwrite=True,
        subject_col="study_id",
        n_components_sweep=[2, 3],
        qc={},
    )
    requests = dim_reduction._collect_scope_fit_requests(
        scope="condition",
        condition="EO_baseline",
        container=container,
        args=args,
        reducers=["PCA"],
        output_root=tmp_path,
        unit_containers_by_key={},
        data_availability=[],
    )
    # One flat unit; the flattened matrix has 2 sensors x 2 features = 4 columns,
    # so the full [2, 3] sweep is valid.
    assert len(requests) == 2
    assert {request["fit_payload"]["n_components"] for request in requests} == {2, 3}
    assert {request["fit_payload"]["unit_name"] for request in requests} == {"all"}


def test_dim_reduction_batches_use_thread_backend_for_parallel_work():
    def worker(task):
        return get_active_backend()[0].__class__.__name__, task

    records = dim_reduction._run_shared_memory_batch([1, 2], worker, max_workers=2)

    assert records == [("ThreadingBackend", 1), ("ThreadingBackend", 2)]


# --- Leaderboard collection + roll-up rendering ---------------------------------


def _write_inventories(tmp_path):
    fit_runs = [
        {
            "fit_id": "fit_pca",
            "scope": "condition",
            "condition": "EO_baseline",
            "analysis_mode": "flat",
            "input_mode": "descriptors",
            "representation": "features",
            "reducer": "PCA",
            "n_components": 5,
            "unit_name": "all",
            "status": "success",
            "trustworthiness": 0.80,
            "continuity": 0.75,
        },
        {
            "fit_id": "fit_umap",
            "scope": "condition",
            "condition": "EO_baseline",
            "analysis_mode": "flat",
            "input_mode": "descriptors",
            "representation": "features",
            "reducer": "UMAP",
            "n_components": 5,
            "unit_name": "all",
            "status": "success",
            "trustworthiness": 0.95,
            "continuity": 0.90,
        },
    ]
    eval_runs = [
        {
            "fit_id": "fit_pca",
            "scope": "condition",
            "condition": "EO_baseline",
            "analysis_mode": "flat",
            "input_mode": "descriptors",
            "representation": "features",
            "reducer": "PCA",
            "n_components": 5,
            "eval_name": "med_adhd_vs_ctrl",
            "target_col": "dx",
            "status": "success",
            SEPARATION_RF_METRIC_KEY: 0.64,
            SEPARATION_METRIC_KEY: 0.62,
        },
        {
            "fit_id": "fit_umap",
            "scope": "condition",
            "condition": "EO_baseline",
            "analysis_mode": "flat",
            "input_mode": "descriptors",
            "representation": "features",
            "reducer": "UMAP",
            "n_components": 5,
            "eval_name": "med_adhd_vs_ctrl",
            "target_col": "dx",
            "status": "success",
            SEPARATION_RF_METRIC_KEY: 0.73,
            SEPARATION_METRIC_KEY: 0.71,
        },
    ]
    fit_path = tmp_path / "fit_runs.json"
    eval_path = tmp_path / "eval_runs.json"
    fit_path.write_text(json.dumps(fit_runs), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_runs), encoding="utf-8")
    return fit_path, eval_path


def _leaderboard_args():
    return SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="flat",
        representation="features",
        n_components_sweep=[5],
        conditions=["EO_baseline"],
        run_pooled=False,
        selection_metric=DEFAULT_DIM_REDUCTION_SELECTION_METRIC,
        selection_eval_name="med_adhd_vs_ctrl",
        descriptor_families=None,
        descriptor_max_abs_value=None,
        run_config_hash=None,
        embedding_model_key=None,
        run_label="cohortX",
        dataset_name="cohortX",
    )


def test_collect_mode_leaderboard_picks_best_by_separation(tmp_path):
    fit_path, eval_path = _write_inventories(tmp_path)
    board = collect_mode_leaderboard(
        args=_leaderboard_args(),
        fit_runs_path=fit_path,
        eval_runs_path=eval_path,
        reducers=["PCA", "UMAP"],
        pooled_condition="pooled_all",
    )
    assert len(board) == 1  # one row per (scope, condition)
    row = board.iloc[0]
    assert row["analysis_mode"] == "flat"
    assert row["reducer"] == "UMAP"  # higher separation than PCA
    assert list(board.columns).index(SEPARATION_RF_METRIC_KEY) < list(board.columns).index(
        SEPARATION_METRIC_KEY
    )
    assert row[SEPARATION_RF_METRIC_KEY] == 0.73
    assert row[SEPARATION_METRIC_KEY] == 0.71


def test_generate_rollup_report_builds_leaderboard_and_scatter(tmp_path):
    fit_path, eval_path = _write_inventories(tmp_path)
    args = _leaderboard_args()
    board = collect_mode_leaderboard(
        args=args,
        fit_runs_path=fit_path,
        eval_runs_path=eval_path,
        reducers=["PCA", "UMAP"],
        pooled_condition="pooled_all",
    )
    summaries = [
        {
            "analysis_mode": "flat",
            "representation": "features",
            "run_variant": "flat_descriptors_features_cfg-abc",
            "report_path": str(tmp_path / "flat.html"),
            "leaderboard": board,
        },
        {
            "analysis_mode": "family",
            "representation": "features",
            "run_variant": "family_descriptors_features_cfg-def",
            "report_path": str(tmp_path / "family.html"),
            "leaderboard": pd.DataFrame(),  # a mode with no successful runs
        },
    ]
    report = generate_rollup_report(
        args=args,
        summaries=summaries,
        task_failures=[
            {"analysis_mode": "descriptor_sensor", "representation": "features", "error": "boom"}
        ],
    )
    titles = [getattr(section, "title", None) for section in report.children]
    assert "Roll-up Overview" in titles
    assert "Leaderboard" in titles
    assert "Per-mode reports" in titles
    assert "Task Failures" in titles


def test_foundation_leaderboard_tags_alignment_transform(tmp_path):
    fit_path, eval_path = _write_inventories(tmp_path)
    args = _leaderboard_args()
    args.input_mode = "foundation_embeddings"
    args.embedding_model_key = "labram_align-leace"
    board = collect_mode_leaderboard(
        args=args,
        fit_runs_path=fit_path,
        eval_runs_path=eval_path,
        reducers=["PCA", "UMAP"],
    )
    assert set(board["transform"]) == {"leace"}


def test_rollup_adds_subject_alignment_diagnostics(tmp_path):
    args = _leaderboard_args()
    args.embedding_model_key = "demo"
    bids_root = tmp_path / "BIDS"
    diagnostics_spec = AlignmentDiagnosticsSpec(
        base_model_key="demo",
        cohort_name="cohortX",
        population="clinical_task_subset",
    )
    root = get_derivative_root(bids_root, DerivativeStage.VARIANCE_DIAGNOSTICS) / slug(
        args.embedding_model_key
    )
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "transform": transform,
                "cohort_name": "cohortX",
                "population": "clinical_task_subset",
                "selection_fingerprint": "cohort-selection",
                "scope": "pooled",
                "eval_name": "condition_separation",
                "target_col": "condition",
                "metric": metric,
                "value": value,
            }
            for transform, value in (("none", 0.8), ("leace", 0.2))
            for metric in (
                "subject_probe_linear_balanced_accuracy",
                "between_subject_eta2",
            )
        ]
    ).to_csv(root / "variance_diagnostics.csv", index=False)
    report = generate_rollup_report(
        args=args,
        summaries=[],
        bids_root=bids_root,
        alignment_diagnostics=diagnostics_spec,
    )
    titles = [getattr(section, "title", None) for section in report.children]
    assert "Subject Alignment Diagnostics" in titles


def test_alignment_diagnostics_spec_requires_explicit_identity():
    with pytest.raises(
        ValueError,
        match="exact non-empty base model key",
    ):
        AlignmentDiagnosticsSpec(
            base_model_key="",
            cohort_name="cohortX",
            population="clinical_task_subset",
        )

    with pytest.raises(ValueError, match="explicit cohort_name and population"):
        AlignmentDiagnosticsSpec(
            base_model_key="demo",
            cohort_name="",
            population="clinical_task_subset",
        )


def test_rollup_requires_bids_root_for_requested_diagnostics():
    spec = AlignmentDiagnosticsSpec(
        base_model_key="demo",
        cohort_name="cohortX",
        population="clinical_task_subset",
    )
    with pytest.raises(ValueError, match="bids_root is required"):
        generate_rollup_report(
            args=_leaderboard_args(),
            summaries=[],
            alignment_diagnostics=spec,
        )


def test_rollup_does_not_probe_legacy_diagnostic_locations(tmp_path):
    args = _leaderboard_args()
    pd.DataFrame([{"metric": "between_subject_eta2", "value": 0.4}]).to_csv(
        tmp_path / "variance_diagnostics.csv", index=False
    )

    report = generate_rollup_report(args=args, summaries=[])

    titles = [getattr(section, "title", None) for section in report.children]
    assert "Subject Alignment Diagnostics" not in titles


# --- End-to-end orchestration through main() ------------------------------------


def _synthetic_descriptor_container():
    n_obs, n_sensor, n_feature = 12, 2, 2
    rng = np.random.default_rng(0)
    diagnosis = np.array(["ADHD"] * 6 + ["Control"] * 6, dtype=object)
    return DataContainer(
        X=rng.normal(size=(n_obs, n_sensor, n_feature)),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": np.asarray(["Fz", "Cz"], dtype=object),
            "feature": np.asarray(["alpha", "sampen"], dtype=object),
            "feature_family": np.asarray(["band", "complexity"], dtype=object),
            "study_id": np.asarray([f"{i:04d}" for i in range(1, n_obs + 1)], dtype=object),
            "condition": np.asarray(["EO_baseline"] * n_obs, dtype=object),
            "combined_diagnosis": diagnosis,
            "patient_group_id": np.asarray(list(range(10, 10 + n_obs)), dtype=object),
        },
        ids=np.asarray([f"{i:04d}_ses-01_run-01" for i in range(1, n_obs + 1)], dtype=object),
        meta={"loaded_obs": n_obs},
    )


def test_main_sweeps_modes_in_process_and_writes_rollup(tmp_path, monkeypatch):
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    reports_root = tmp_path / "reports"
    metadata = tmp_path / "meta.csv"
    pd.DataFrame({"study_id": [1, 2, 3, 4, 5, 6]}).to_csv(metadata, index=False)

    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(
        "dataset_name: smoke\n"
        "subject_col: study_id\n"
        "conditions: [EO_baseline]\n"
        "run_pooled: false\n"
        "evals:\n"
        "  - name: med_adhd_vs_ctrl\n"
        "    target_col: combined_diagnosis\n"
        "    group_col: patient_group_id\n"
        "    label_map: {Control: '0', ADHD: '1'}\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(
        "input_mode: descriptors\n"
        "selection_metric: separation_rf_balanced_accuracy\n"
        "selection_eval_name: med_adhd_vs_ctrl\n"
        # Mode-centric plan: each mode fully declares its run. flat sweeps [2, 3]
        # over PCA + UMAP; family runs PCA only over [2]. Exercises build_mode_specs,
        # the per-mode reducer set, and the per-mode sweep end-to-end.
        "analysis_modes:\n"
        "  flat: {reducers: [PCA, UMAP], n_components: [2, 3]}\n"
        "  family: {reducers: [PCA], n_components: [2]}\n"
        # Descriptor input paths are config-driven (not CLI flags); build_dataset
        # is stubbed below so these need only satisfy validation.
        f"descriptor_table_path: {tmp_path / 'features.parquet'}\n"
        f"descriptor_feature_columns_path: {tmp_path / 'columns.json'}\n"
        "qc: {}\n",
        encoding="utf-8",
    )

    load_calls = {"n": 0}

    def fake_build_dataset(args, meta_df, condition, target_col=None, **kwargs):
        load_calls["n"] += 1
        return _synthetic_descriptor_container()

    # A descriptor base container is loaded once per condition and reused across
    # modes; stub the heavy per-mode report so we test orchestration, not viz.
    monkeypatch.setattr(dim_reduction, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(
        dim_reduction,
        "generate_dataset_report",
        lambda **kwargs: SimpleNamespace(save=lambda path: path.write_text("ok")),
    )

    argv = [
        "dimensionality_reduction",
        "--bids_root",
        str(bids_root),
        "--metadata",
        str(metadata),
        "--reports_root",
        str(reports_root),
        "--cohort_config",
        str(cohort),
        "--analysis_config",
        str(analysis),
        "--n_jobs",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    dim_reduction.main()

    # Two modes (flat, family) share ONE load of the single condition.
    assert load_calls["n"] == 1

    fit_inventories = list((bids_root / "derivatives" / "dim_reduction").rglob("fit_runs.json"))
    assert len(fit_inventories) == 2  # one run namespace per mode
    statuses = set()
    reducer_sets = []
    ncomp_by_mode: dict[str, set[int]] = {}
    for path in fit_inventories:
        records = json.loads(path.read_text(encoding="utf-8"))
        reducer_sets.append({record.get("reducer") for record in records})
        for record in records:
            statuses.add(record.get("status"))
            if record.get("status") == "success":
                ncomp_by_mode.setdefault(record["analysis_mode"], set()).add(
                    int(record["n_components"])
                )
    assert "success" in statuses
    # flat's spec lists [PCA, UMAP]; family's lists [PCA] -> one namespace has
    # {PCA, UMAP}, the other only {PCA}. Proves the per-mode reducer set is applied.
    assert sorted(reducer_sets, key=len) == [{"PCA"}, {"PCA", "UMAP"}]
    # flat sweeps the global [2, 3]; family is capped to [2] by its mode spec.
    assert ncomp_by_mode["flat"] == {2, 3}
    assert ncomp_by_mode["family"] == {2}

    rollups = list(reports_root.rglob("rollup_leaderboard.html"))
    assert len(rollups) == 1


def _fit_request(reducer, n_components):
    return {
        "fit_payload": {
            "scope": "condition",
            "condition": "EO_baseline",
            "unit_key": "all",
            "reducer": reducer,
            "n_components": n_components,
        }
    }


def test_group_fit_requests_splits_non_nested_reducer_sweep():
    # PCA is nested (slice-once -> one group); UMAP refits per dimension, so its
    # sweep must split into one parallel task per n_components.
    requests = [
        _fit_request("PCA", 2),
        _fit_request("PCA", 5),
        _fit_request("UMAP", 2),
        _fit_request("UMAP", 5),
        _fit_request("UMAP", 10),
    ]
    groups = group_fit_requests(requests)
    sizes = sorted(len(group) for group in groups)
    assert sizes == [1, 1, 1, 2]  # PCA group of 2, three singleton UMAP groups
    pca_groups = [g for g in groups if g[0]["fit_payload"]["reducer"] == "PCA"]
    umap_groups = [g for g in groups if g[0]["fit_payload"]["reducer"] == "UMAP"]
    assert len(pca_groups) == 1 and len(pca_groups[0]) == 2
    assert len(umap_groups) == 3 and all(len(g) == 1 for g in umap_groups)


def test_collect_mode_leaderboard_ignores_offtarget_eval(tmp_path):
    # A fit that separates sex strongly but the target contrast weakly must not be
    # ranked on the off-target eval. Guards the selection_eval_name filter.
    fit_runs = [
        {
            "fit_id": "fit_a",
            "scope": "condition",
            "condition": "EO_baseline",
            "reducer": "PCA",
            "n_components": 5,
            "unit_name": "all",
            "status": "success",
        },
        {
            "fit_id": "fit_b",
            "scope": "condition",
            "condition": "EO_baseline",
            "reducer": "UMAP",
            "n_components": 5,
            "unit_name": "all",
            "status": "success",
        },
    ]
    eval_runs = [
        # fit_a wins on the off-target eval, loses on the selection eval.
        {
            "fit_id": "fit_a",
            "eval_name": "sex_separation",
            "status": "success",
            SEPARATION_RF_METRIC_KEY: 0.99,
            SEPARATION_METRIC_KEY: 0.99,
        },
        {
            "fit_id": "fit_a",
            "eval_name": "med_adhd_vs_ctrl",
            "status": "success",
            SEPARATION_RF_METRIC_KEY: 0.55,
            SEPARATION_METRIC_KEY: 0.55,
        },
        {
            "fit_id": "fit_b",
            "eval_name": "med_adhd_vs_ctrl",
            "status": "success",
            SEPARATION_RF_METRIC_KEY: 0.80,
            SEPARATION_METRIC_KEY: 0.80,
        },
    ]
    fit_path = tmp_path / "fit_runs.json"
    eval_path = tmp_path / "eval_runs.json"
    fit_path.write_text(json.dumps(fit_runs), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_runs), encoding="utf-8")

    board = collect_mode_leaderboard(
        args=_leaderboard_args(),
        fit_runs_path=fit_path,
        eval_runs_path=eval_path,
        reducers=["PCA", "UMAP"],
        pooled_condition="pooled_all",
    )
    assert len(board) == 1
    row = board.iloc[0]
    assert row["reducer"] == "UMAP"  # the selection eval, not sex_separation, decides
    assert row[SEPARATION_RF_METRIC_KEY] == 0.80
    assert row[SEPARATION_METRIC_KEY] == 0.80


def _write_foundation_run(cohort_dir, variant, model, representation, separation):
    run_dir = cohort_dir / variant
    (run_dir / "runs").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "analysis_mode": "flat",
                "input_mode": "foundation_embeddings",
                "representation": representation,
                "scope": "condition",
                "condition": "EO_baseline",
                "unit_name": "all",
                "reducer": "UMAP",
                "n_components": 10,
                "model": model,
                "trustworthiness": 0.9,
                SEPARATION_RF_METRIC_KEY: separation,
                SEPARATION_METRIC_KEY: separation,
            }
        ]
    ).to_json(run_dir / "runs" / "leaderboard.json", orient="records")
    (run_dir / "runs" / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "report_path": str(run_dir / "rep.html"),
                "analysis_mode": "flat",
                "representation": representation,
                "run_variant": variant,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config_used.yaml").write_text(
        "selection_metric: separation_rf_balanced_accuracy\n"
        "selection_eval_name: med_adhd_vs_ctrl\n",
        encoding="utf-8",
    )


def test_compare_cohort_merges_per_model_leaderboards(tmp_path):
    from eeg_adhd_epilepsy.analysis.dimensionality_reduction import compare_cohort

    dim_root = tmp_path / "dim_reduction"
    cohort_dir = dim_root / "cohortx"
    _write_foundation_run(
        cohort_dir, "foundation_cbramod_flat_recording_cfg-a", "cbramod", "recording", 0.62
    )
    _write_foundation_run(
        cohort_dir, "foundation_labram_flat_subject_cfg-b", "labram", "subject", 0.74
    )
    _write_foundation_run(
        cohort_dir,
        "foundation_cbramod_align-ra_flat_epoch_cfg-c",
        "cbramod_align-ra",
        "epoch",
        0.70,
    )

    reports_root = tmp_path / "reports"
    out_path = compare_cohort(dim_root, "cohortx", reports_root)
    assert out_path is not None and out_path.exists()

    merged = pd.read_csv(out_path.parent / "foundation_model_comparison.csv")
    assert sorted(merged["model"]) == ["cbramod", "cbramod", "labram"]
    assert sorted(merged["transform"]) == ["none", "none", "ra"]
    assert sorted(merged["representation"]) == ["epoch", "recording", "subject"]


def test_compare_only_cli_does_not_require_analysis_configs(tmp_path, monkeypatch):
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    derivative_root = tmp_path / "dim_reduction"
    reports_root = tmp_path / "reports"
    captured = {}

    monkeypatch.setattr(dim_reduction, "run", lambda config: captured.update(config))
    monkeypatch.setattr(
        "sys.argv",
        [
            "dimensionality_reduction",
            "--compare_only",
            "--bids_root",
            str(bids_root),
            "--derivative_root",
            str(derivative_root),
            "--reports_root",
            str(reports_root),
            "--dataset_name",
            "cohortx",
        ],
    )

    dim_reduction.main()

    assert captured == {
        "compare_only": True,
        "bids_root": str(bids_root),
        "derivative_root": str(derivative_root),
        "reports_root": str(reports_root),
        "dataset_name": "cohortx",
    }

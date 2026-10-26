from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from coco_pipe.descriptors import load_descriptor_table, save_descriptor_table
from coco_pipe.dim_reduction import run_eval
from coco_pipe.io import ANALYSIS_MODES, DataContainer, iter_analysis_units

import eeg_adhd_epilepsy.analysis.dimensionality_reduction as dim_reduction
import eeg_adhd_epilepsy.reports.dim_reduction as dim_report
from eeg_adhd_epilepsy.analysis import align_subject_embeddings as ase
from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.analysis import merge_foundation_embeddings as mfe
from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.dimensionality_reduction import _collect_scope_fit_requests
from eeg_adhd_epilepsy.analysis.utils.common import (
    apply_family_qc_mask,
    container_pool_spec,
    families_for_analysis_unit,
    pool_containers,
    pool_containers_streaming,
    qc_keys,
)
from eeg_adhd_epilepsy.analysis.utils.dim_reduction import scale_features
from tests.analysis.test_extract_foundation_embeddings import _config as _extraction_config


def test_descriptor_loader_filters_subjects(tmp_path):
    feature_column = "mean_band_log_abs_alpha_chgrp-midline"
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": [1, 2],
            "condition": ["EO_baseline", "EO_baseline"],
            "recording_id": ["0001_ses-01_run-01", "0002_ses-01_run-01"],
            feature_column: [0.1, 0.2],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=[feature_column])

    container = load_descriptor_table(
        table_path=table_path,
        condition="EO_baseline",
        subjects=["sub-0002"],
        subject_col="study_id",
    )

    assert container.X.shape == (1, 1)
    assert container.coords["study_id"].tolist() == [2]
    assert container.ids.tolist() == ["0002_ses-01_run-01"]


def test_descriptor_analysis_attaches_family_qc(tmp_path):
    feature_column = "mean_band_log_abs_alpha_chgrp-midline"
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": ["0001", "0002", "0003", "0004"],
            "condition": ["EO_baseline"] * 4,
            "recording_id": [f"{subject}_ses-01_run-01" for subject in range(1, 5)],
            feature_column: [0.9, 1.0, 1.1, 1.05],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=[feature_column])
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="flat",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        qc={},
        location_statistic=None,
    )

    container = build_dataset(
        args,
        meta_df=None,
        condition="EO_baseline",
    )

    qc_result = container.meta["qc_result"]
    assert qc_result.family_qc is not None
    assert qc_result.family_qc["family"].tolist() == ["band"]
    assert qc_result.n_rows_entering_qc == 4


def test_flat_descriptor_analysis_attaches_family_qc(tmp_path):
    feature_columns = [
        "band_log_abs_alpha_ch-Fz",
        "complexity_sample_entropy_ch-Fz",
        "band_log_abs_alpha_ch-Cz",
        "complexity_sample_entropy_ch-Cz",
    ]
    table_path = tmp_path / "features.parquet"
    table = {
        "study_id": [f"{subject:04d}" for subject in range(1, 5)],
        "condition": ["EO_baseline"] * 4,
        "recording_id": [f"{subject:04d}_ses-01_run-01" for subject in range(1, 5)],
    }
    for offset, column in enumerate(feature_columns):
        table[column] = [offset + value for value in (0.9, 1.0, 1.1, 1.05)]
    pd.DataFrame(table).pipe(
        save_descriptor_table, table_path.with_suffix(""), feature_columns=feature_columns
    )
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="flat",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        qc={},
        location_statistic=None,
    )

    container = build_dataset(
        args,
        meta_df=None,
        condition="EO_baseline",
    )

    assert container.dims == ("obs", "feature")
    qc_result = container.meta["qc_result"]
    assert qc_result.family_qc is not None
    assert qc_result.family_qc["family"].tolist() == ["band", "complexity"]


def test_family_scoped_qc_keeps_band_outlier_for_complexity(tmp_path):
    feature_columns = [
        "band_log_abs_alpha_ch-Fz",
        "complexity_sample_entropy_ch-Fz",
    ]
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": [f"{subject:04d}" for subject in range(1, 7)],
            "condition": ["EO_baseline"] * 6,
            "recording_id": [f"{subject:04d}_ses-01_run-01" for subject in range(1, 7)],
            feature_columns[0]: [1.0, 1.1, 0.9, 1.05, 0.95, 100.0],
            feature_columns[1]: [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=feature_columns)
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="sensor",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        location_statistic=None,
        qc={
            "column_prune": {"enabled": False},
            "outlier": {
                "z_threshold": 3.0,
                "group_by": "family",
                "epoch_outlier_fraction": 0.5,
                "subject_outlier_fraction": 0.5,
            },
            "min_obs": 5,
        },
    )

    container = build_dataset(args, None, "EO_baseline")
    # One row per subject ⇒ the epoch-level (L2) MAD pass is short-circuited.
    qc_result = container.meta["qc_result"]
    assert qc_result.thresholds["subject_aggregated"] is True
    assert qc_result.epoch_drop_threshold is None
    # Deferred masking: the scope drops nothing globally; the drop is recorded
    # per group, not as a misleading combined "subjects_dropped".
    assert qc_result.n_obs_out == qc_result.n_obs_in == 6
    assert qc_result.subjects_dropped == []
    band_dropped = {str(record.subject_id) for record in qc_result.per_family_dropped["band"]}
    assert band_dropped == {"0006"}  # study_id of the alpha-outlier subject
    assert qc_result.per_family_dropped["complexity"] == []
    units = iter_analysis_units(container, "family", "descriptors")
    retained = {}
    for unit in units:
        families = families_for_analysis_unit(container, unit)
        clean, _ = apply_family_qc_mask(unit["container"], families)
        retained[unit["family"]] = clean.ids.astype(str).tolist()

    assert "0006_ses-01_run-01" not in retained["band"]
    assert "0006_ses-01_run-01" in retained["complexity"]


def test_measure_grouping_keeps_alpha_outlier_for_beta(tmp_path):
    # Two measures of the SAME family (band): a subject extreme in alpha should
    # be dropped from alpha analyses but kept for beta when group_by="measure".
    feature_columns = [
        "band_log_abs_alpha_ch-Fz",
        "band_log_abs_beta_ch-Fz",
    ]
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": [f"{subject:04d}" for subject in range(1, 7)],
            "condition": ["EO_baseline"] * 6,
            "recording_id": [f"{subject:04d}_ses-01_run-01" for subject in range(1, 7)],
            feature_columns[0]: [1.0, 1.1, 0.9, 1.05, 0.95, 100.0],
            feature_columns[1]: [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=feature_columns)
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="feature",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        location_statistic=None,
        qc={
            "column_prune": {"enabled": False},
            "outlier": {
                "z_threshold": 3.0,
                "group_by": "measure",
                "subject_outlier_fraction": 0.5,
            },
            "min_obs": 5,
        },
    )

    container = build_dataset(args, None, "EO_baseline")
    assert container.meta["qc_result"].thresholds["group_by"] == "measure"
    retained = {}
    for unit in iter_analysis_units(container, "feature", "descriptors"):
        families = families_for_analysis_unit(container, unit)
        clean, _ = apply_family_qc_mask(unit["container"], families)
        retained[unit["unit_name"]] = clean.ids.astype(str).tolist()

    alpha_unit = next(name for name in retained if "alpha" in name)
    beta_unit = next(name for name in retained if "beta" in name)
    assert "0006_ses-01_run-01" not in retained[alpha_unit]
    assert "0006_ses-01_run-01" in retained[beta_unit]


def test_subfamily_grouping_keeps_logabs_outlier_for_relative(tmp_path):
    # Two sub-families of the SAME family (band): a subject extreme in the
    # log-absolute sub-family should be dropped from log-abs analyses but kept
    # for the relative-power analysis when group_by="subfamily".
    feature_columns = [
        "band_log_abs_alpha_ch-Fz",
        "band_rel_alpha_ch-Fz",
    ]
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": [f"{subject:04d}" for subject in range(1, 7)],
            "condition": ["EO_baseline"] * 6,
            "recording_id": [f"{subject:04d}_ses-01_run-01" for subject in range(1, 7)],
            feature_columns[0]: [1.0, 1.1, 0.9, 1.05, 0.95, 100.0],
            feature_columns[1]: [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=feature_columns)
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="feature",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        location_statistic=None,
        qc={
            "column_prune": {"enabled": False},
            "outlier": {
                "z_threshold": 3.0,
                "group_by": "subfamily",
                "subject_outlier_fraction": 0.5,
            },
            "min_obs": 5,
        },
    )

    container = build_dataset(args, None, "EO_baseline")
    assert container.meta["qc_result"].thresholds["group_by"] == "subfamily"
    # bad ids are keyed by sub-family label
    assert set(container.meta["family_qc_bad_ids"]) == {"log_abs", "rel"}
    retained = {}
    for unit in iter_analysis_units(container, "feature", "descriptors"):
        families = families_for_analysis_unit(container, unit)
        clean, _ = apply_family_qc_mask(unit["container"], families)
        retained[unit["unit_name"]] = clean.ids.astype(str).tolist()

    log_abs_unit = next(name for name in retained if "log_abs" in name)
    rel_unit = next(name for name in retained if "rel" in name and "log_abs" not in name)
    assert "0006_ses-01_run-01" not in retained[log_abs_unit]
    assert "0006_ses-01_run-01" in retained[rel_unit]


def test_wide_unit_is_not_emptied_by_the_union_of_single_group_flags():
    # Ten QC groups, each flagging a different observation: a unit spanning all
    # of them must keep those observations (each is flagged in 1 of 10 groups)
    # and drop only the one flagged in most groups.
    groups = [f"g{index}" for index in range(10)]
    bad_ids = {group: [f"r{index}", "r10"] for index, group in enumerate(groups)}
    container = DataContainer(
        X=np.zeros((12, 1)),
        dims=("obs", "feature"),
        coords={"feature": np.asarray(["band_alpha_ch-Fz"], dtype=object)},
        ids=np.asarray([f"r{index}" for index in range(12)], dtype=object),
        meta={"family_qc_group_by": "measure", "family_qc_bad_ids": bad_ids},
    )

    wide, _ = apply_family_qc_mask(container, groups)
    assert wide.ids.astype(str).tolist() == [f"r{index}" for index in range(10)] + ["r11"]

    # A unit spanning one group still drops everything that group flags.
    narrow, _ = apply_family_qc_mask(container, ["g3"])
    assert set(narrow.ids.astype(str)) == {f"r{i}" for i in range(12)} - {"r3", "r10"}

    # A fraction of 0.0 restores the any-group union.
    union, _ = apply_family_qc_mask(
        container, groups, {"outlier": {"unit_outlier_group_fraction": 0.0}}
    )
    assert union.ids.astype(str).tolist() == ["r11"]


def test_column_prune_removes_all_nan_feature_without_row_drop(tmp_path):
    feature_columns = [
        "band_log_abs_alpha_ch-Fz",
        "complexity_sample_entropy_ch-Fz",
    ]
    table_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "study_id": ["0001", "0002", "0003"],
            "condition": ["EO_baseline"] * 3,
            "recording_id": [
                "0001_ses-01_run-01",
                "0002_ses-01_run-01",
                "0003_ses-01_run-01",
            ],
            feature_columns[0]: [1.0, 2.0, 3.0],
            feature_columns[1]: [np.nan, np.nan, np.nan],
        }
    ).pipe(save_descriptor_table, table_path.with_suffix(""), feature_columns=feature_columns)
    args = SimpleNamespace(
        input_mode="descriptors",
        analysis_mode="flat",
        descriptor_table_path=str(table_path),
        descriptor_families=None,
        descriptor_max_abs_value=None,
        subject_col="study_id",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        location_statistic=None,
        qc={
            "column_prune": {
                "enabled": True,
                "max_missing_rate": 0.20,
                "drop_constant": True,
            },
            "outlier": {},
        },
    )

    container = build_dataset(args, None, "EO_baseline")

    assert container.X.shape == (3, 1)
    assert container.meta["qc_result"].n_dropped_nan_inf == 0
    assert container.meta["qc_result"].feature_columns_dropped["column"].tolist() == [
        feature_columns[1]
    ]


def test_foundation_embedding_loader_reads_a_space_of_the_project_files(bids_root, tmp_path):
    config = _extraction_config(bids_root)
    config["alignment"] = {"leace": {"eta2": 0.5}}
    scratch, project = tmp_path / "scratch", tmp_path / "project"
    efe.run(config, scratch, model_key="cbramod", weights=[None], subjects=["0001", "0003"])
    mfe.run(config, scratch, project, model_key="cbramod", descs=["base"], weights=[None])
    ase.run(config, scratch, project, model_key="cbramod", seed=None)
    args = SimpleNamespace(
        input_mode="foundation_embeddings",
        embedding_derivative_root=str(project),
        embedding_model_key="cbramod",
        desc="base",
        task="clinical",
        representation="recording",
        subject_col="subject",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
        analysis_mode="flat",
        qc={},
        location_statistic=None,
    )

    # The cohort's metadata scopes the subjects and is attached to them.
    meta_df = pd.DataFrame({"subject": ["0003"], "group": ["case"]})
    container = build_dataset(args, meta_df=meta_df, condition="EO_baseline")
    assert set(container.coords["subject"]) == {"0003"}
    assert set(container.coords["condition"]) == {"EO_baseline"}
    assert set(container.coords["group"]) == {"case"}
    assert container.X.shape == (1, 200)

    # A space's suffix picks its label: the window vectors with subject identity erased.
    args.embedding_model_key, args.representation = "cbramod_align-leace", "epoch"
    erased = build_dataset(args, meta_df=None, condition="EO_baseline")
    assert erased.meta["description"] == "baseLeace" and erased.X.shape[1] == 200


def test_fit_request_collection_skips_invalid_n_components(tmp_path):
    container = DataContainer(
        X=np.zeros((3, 5)),
        dims=("obs", "feature"),
        coords={"feature": np.arange(5)},
        ids=np.asarray(["obs1", "obs2", "obs3"], dtype=object),
    )
    args = SimpleNamespace(
        input_mode="raw",
        analysis_mode="flat",
        descriptor_families=None,
        n_components_sweep=[2, 3, 4, 10],
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        representation="epoch",
        bids_root=str(tmp_path),
        use_derivatives=False,
        task="clinical",
        segment_duration=60.0,
        overlap=0.0,
        desc="base",
        window_source="auto",
        qc=None,
        run_label="test",
        run_config_hash="cfg",
        overwrite=True,
        subject_col="study_id",
    )
    availability = []

    requests = _collect_scope_fit_requests(
        scope="condition",
        condition="EO_baseline",
        container=container,
        args=args,
        reducers=["PCA"],
        output_root=tmp_path,
        unit_containers_by_key={},
        data_availability=availability,
    )

    assert [request["fit_payload"]["n_components"] for request in requests] == [2, 3]
    assert availability[0]["skipped_n_components"] == [4, 10]


def test_fit_request_collection_passes_per_reducer_execution_params(tmp_path):
    container = DataContainer(
        X=np.zeros((3, 5)),
        dims=("obs", "feature"),
        coords={"feature": np.arange(5)},
        ids=np.asarray(["obs1", "obs2", "obs3"], dtype=object),
    )
    args = SimpleNamespace(
        input_mode="raw",
        analysis_mode="flat",
        descriptor_families=None,
        n_components_sweep=[2],
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        representation="epoch",
        bids_root=str(tmp_path),
        use_derivatives=False,
        task="clinical",
        segment_duration=60.0,
        overlap=0.0,
        desc="base",
        window_source="auto",
        qc=None,
        run_label="test",
        run_config_hash="cfg",
        overwrite=False,
        subject_col="study_id",
        reducer_params={"UMAP": {"n_jobs": 1}},
    )

    requests = _collect_scope_fit_requests(
        scope="condition",
        condition="EO_baseline",
        container=container,
        args=args,
        reducers=["PCA", "UMAP"],
        output_root=tmp_path,
        unit_containers_by_key={},
        data_availability=[],
    )

    params_by_reducer = {
        request["fit_payload"]["reducer"]: request["fit_payload"]["reducer_params"]
        for request in requests
    }
    assert params_by_reducer == {"PCA": {}, "UMAP": {"n_jobs": 1}}


def test_raw_recording_representation_builds_recording_id(monkeypatch, tmp_path):
    source = DataContainer(
        X=np.asarray([1.0, 3.0, 5.0, 7.0]).reshape(4, 1, 1),
        dims=("obs", "channel", "time"),
        coords={
            "channel": np.asarray(["Fz"], dtype=object),
            "time": np.asarray([0.0]),
            "study_id": np.asarray(["0001", "0001", "0001", "0001"], dtype=object),
            "subject": np.asarray(["0001", "0001", "0001", "0001"], dtype=object),
            "session": np.asarray(["01", "01", "01", "01"], dtype=object),
            "run": np.asarray(["01", "01", "02", "02"], dtype=object),
            "recording_id": np.asarray(
                ["sub-0001_ses-01_task-clinical_run-01"] * 2
                + ["sub-0001_ses-01_task-clinical_run-02"] * 2,
                dtype=object,
            ),
            "condition": np.asarray(["EO_baseline"] * 4, dtype=object),
        },
        ids=np.asarray(["epoch-1", "epoch-2", "epoch-3", "epoch-4"], dtype=object),
        meta={},
    )
    monkeypatch.setattr(
        "eeg_adhd_epilepsy.analysis.dataset.build_container",
        lambda **_kwargs: source,
    )
    args = SimpleNamespace(
        input_mode="raw",
        analysis_mode="flat",
        representation="recording",
        bids_root=str(tmp_path),
        use_derivatives=True,
        task="clinical",
        segment_duration=60.0,
        overlap=0.0,
        subject_col="study_id",
        desc="base",
        window_source="auto",
        units="V",
        group_filters=None,
        filter_col=[],
        filter_val=[],
        balance_target=None,
    )

    container = build_dataset(args, None, "EO_baseline")

    assert container.X[:, 0].tolist() == [2.0, 6.0]
    assert container.coords["recording_id"].tolist() == [
        "sub-0001_ses-01_task-clinical_run-01",
        "sub-0001_ses-01_task-clinical_run-02",
    ]
    # Named like the recording rows of descriptors and embeddings.
    assert container.ids.tolist() == [
        "sub-0001_ses-01_task-clinical_run-01_cond-EO_baseline",
        "sub-0001_ses-01_task-clinical_run-02_cond-EO_baseline",
    ]


def test_scale_features_puts_mixed_scale_features_on_one_scale():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(500.0, 200.0, 200), rng.normal(0.5, 0.1, 200)])
    X[0, 0] = 1e6
    container = DataContainer(X=X, dims=("obs", "feature"), ids=np.arange(200))

    scaled = scale_features(container, {"method": "robust", "clip": 8.0})

    assert np.allclose(np.median(scaled.X, axis=0), 0.0)
    assert np.allclose(np.subtract(*np.percentile(scaled.X, [75, 25], axis=0)), 1.0)
    assert scaled.X[0, 0] == 8.0
    assert scaled.ids.tolist() == container.ids.tolist()
    assert scale_features(container, None) is container


def test_dim_reduction_preserves_exact_reducer_names():
    # Reducer names from each mode's `reducers` list are passed through verbatim to
    # coco_pipe; case-normalizing them would break the registry lookup.
    source = inspect.getsource(dim_reduction)

    assert ".upper()" not in source
    assert ".lower()" not in source


def test_dim_reduction_avoids_private_dim_reduction_imports():
    source = inspect.getsource(dim_reduction)

    assert "coco_pipe.dim_reduction.artifacts" not in source
    assert "coco_pipe.dim_reduction.pipeline import _" not in source
    assert "def _jsonable" not in source
    assert "def _stable_signature" not in source
    assert "def _matrix_signature" not in source


def test_dim_reduction_exposes_every_coco_pipe_analysis_mode():
    assert set(ANALYSIS_MODES) == {
        "flat",
        "sensor",
        "family",
        "subfamily",
        "sensor_within_family",
        "sensor_within_subfamily",
        "feature",
        "feature_within_family",
        "descriptor",
        "descriptor_sensor",
    }
    assert set(dim_report._UNIT_LABELS) == set(ANALYSIS_MODES)


def test_dim_reduction_report_uses_public_coco_pipe_imports():
    source = inspect.getsource(dim_report)

    assert "coco_pipe.dim_reduction.artifacts" not in source
    assert "coco_pipe.dim_reduction.core" not in source
    assert "coco_pipe.io.structures" not in source


def test_pooled_qc_drops_only_the_flagged_condition_of_a_recording():
    # EO and EC rows of a recording share an ID; r1 is flagged in EO, r2 in EC
    containers = []
    for condition, bad_row in (("EO", 0), ("EC", 1)):
        containers.append(
            DataContainer(
                X=np.zeros((2, 1)),
                dims=("obs", "feature"),
                coords={
                    "feature": np.asarray(["band_alpha_ch-Fz"], dtype=object),
                    "feature_family": np.asarray(["band"], dtype=object),
                    "feature_channel": np.asarray(["Fz"], dtype=object),
                    "condition": np.asarray([condition, condition], dtype=object),
                },
                ids=np.asarray(["r1", "r2"], dtype=object),
                meta={"condition": condition, "family_qc_group_by": "subfamily"},
            )
        )
        flagged = qc_keys(containers[-1])[[bad_row]].tolist()
        containers[-1].meta["family_qc_bad_ids"] = {"log_abs": flagged}

    pooled = pool_containers(containers)
    assert pooled.meta["family_qc_group_by"] == "subfamily"

    kept, _ = apply_family_qc_mask(pooled, ["log_abs"])
    assert list(zip(kept.coords["condition"], kept.ids, strict=True)) == [
        ("EO", "r2"),
        ("EC", "r1"),
    ]


def _pool_equivalence_containers(channels_per_condition):
    """Two conditions with real payloads, obs metadata, and auxiliary coords."""
    containers = []
    for index, (condition, channels) in enumerate(channels_per_condition):
        n_obs = 2 + index
        rng = np.random.default_rng(index)
        containers.append(
            DataContainer(
                X=rng.normal(size=(n_obs, len(channels), 4)).astype(np.float32),
                dims=("obs", "channel", "time"),
                coords={
                    "channel": np.asarray(channels, dtype=object),
                    "time": np.arange(4),
                    "subject": np.asarray([f"{condition}{i}" for i in range(n_obs)], dtype=object),
                    "condition": np.asarray([condition] * n_obs, dtype=object),
                },
                y=np.asarray([i % 2 for i in range(n_obs)]),
                ids=np.asarray([f"{condition}_r{i}" for i in range(n_obs)], dtype=object),
                meta={
                    "condition": condition,
                    "sfreq": 200.0,
                    "family_qc_group_by": "family",
                    "family_qc_bad_ids": {"fam": [f"{condition}_r0"]},
                },
            )
        )
    return containers


@pytest.mark.parametrize(
    "channels_per_condition",
    [
        [("EO", ["A", "B", "C"]), ("EC", ["A", "B", "C"])],  # identical channels
        [("EO", ["A", "B", "C"]), ("EC", ["A", "B", "D"])],  # channel intersection
    ],
)
def test_streaming_pool_matches_eager_pool(channels_per_condition):
    containers = _pool_equivalence_containers(channels_per_condition)
    eager = pool_containers(containers)
    # Loaders re-create identical containers, mimicking a per-condition reload.
    streamed = pool_containers_streaming(
        [container_pool_spec(container) for container in containers],
        [(lambda c=container: c) for container in containers],
    )

    assert streamed.X.shape == eager.X.shape
    np.testing.assert_array_equal(streamed.X, eager.X)
    np.testing.assert_array_equal(streamed.ids, eager.ids)
    np.testing.assert_array_equal(streamed.y, eager.y)
    assert set(streamed.coords) == set(eager.coords)
    for key in eager.coords:
        np.testing.assert_array_equal(
            np.asarray(streamed.coords[key]), np.asarray(eager.coords[key])
        )
    assert streamed.meta["family_qc_bad_ids"] == eager.meta["family_qc_bad_ids"]
    assert streamed.meta["family_qc_group_by"] == eager.meta["family_qc_group_by"]
    assert streamed.meta["sfreq"] == eager.meta["sfreq"] == 200.0


def test_fit_identity_changes_with_matrix_content_and_qc(tmp_path):
    base_args = dict(
        input_mode="descriptors",
        representation="features",
        analysis_mode="flat",
        descriptor_families=None,
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        descriptor_table_path=str(tmp_path / "features.parquet"),
        descriptor_max_abs_value=None,
        location_statistic=None,
        run_label="test",
        run_config_hash="cfg",
        overwrite=False,
        subject_col="study_id",
        n_components_sweep=[1],
    )

    def fit_id(value, qc):
        container = DataContainer(
            X=np.asarray([[value], [2.0]]),
            dims=("obs", "feature"),
            coords={"feature": np.asarray(["alpha"], dtype=object)},
            ids=np.asarray(["r1", "r2"], dtype=object),
        )
        args = SimpleNamespace(**base_args, qc=qc)
        return _collect_scope_fit_requests(
            "condition",
            "EO_baseline",
            container,
            args,
            ["PCA"],
            tmp_path,
            {},
            [],
        )[0]["fit_payload"]["fit_id"]

    assert fit_id(1.0, {}) != fit_id(3.0, {})
    assert fit_id(1.0, {}) != fit_id(1.0, {"column_prune": {"enabled": True}})


def test_descriptor_sensor_units_have_distinct_fit_ids(tmp_path):
    container = DataContainer(
        X=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        dims=("obs", "feature"),
        coords={
            "feature": np.asarray(["band_alpha_mean_ch-Fz", "band_alpha_mean_ch-Cz"], dtype=object),
            "feature_family": np.asarray(["band", "band"], dtype=object),
            "feature_channel": np.asarray(["Fz", "Cz"], dtype=object),
            "feature_measure": np.asarray(["alpha_mean", "alpha_mean"], dtype=object),
            "feature_subfamily": np.asarray(["log_abs", "log_abs"], dtype=object),
            "feature_descriptor": np.asarray(["alpha", "alpha"], dtype=object),
        },
        ids=np.asarray(["r1", "r2"], dtype=object),
    )
    args = SimpleNamespace(
        input_mode="descriptors",
        representation="features",
        analysis_mode="descriptor_sensor",
        descriptor_families=None,
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        descriptor_table_path=str(tmp_path / "features.parquet"),
        descriptor_max_abs_value=None,
        location_statistic=None,
        run_label="test",
        run_config_hash="cfg",
        overwrite=False,
        subject_col="study_id",
        n_components_sweep=[1],
        qc={},
    )

    requests = _collect_scope_fit_requests(
        "condition",
        "EO_baseline",
        container,
        args,
        ["PCA"],
        tmp_path,
        {},
        [],
    )

    assert len(requests) == 2
    assert len({request["fit_payload"]["fit_id"] for request in requests}) == 2


def test_foundation_fit_request_records_embedding_provenance(tmp_path):
    container = DataContainer(
        X=np.zeros((2, 3)),
        dims=("obs", "feature"),
        coords={
            "feature": np.arange(3),
            "study_id": np.asarray(["0001", "0002"], dtype=object),
        },
        ids=np.asarray(["rec1", "rec2"], dtype=object),
    )
    args = SimpleNamespace(
        input_mode="foundation_embeddings",
        analysis_mode="flat",
        descriptor_families=None,
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        qc=None,
        embedding_derivative_root=str(tmp_path / "embeddings"),
        representation="recording",
        embedding_aggregate_by="study_id",
        embedding_model_key="cbramod",
        run_label="test",
        run_config_hash="cfg",
        overwrite=True,
        subject_col="study_id",
        n_components_sweep=[2],
    )

    requests = _collect_scope_fit_requests(
        scope="condition",
        condition="EO_baseline",
        container=container,
        args=args,
        reducers=["PCA"],
        output_root=tmp_path,
        unit_containers_by_key={},
        data_availability=[],
    )
    request = requests[0]
    payload = request["fit_payload"]

    assert payload["embedding_model_key"] == "cbramod"
    assert payload["representation"] == "recording"
    assert payload["embedding_aggregate_by"] == "study_id"
    assert payload["input_signature"]["embedding_model_key"] == "cbramod"


def test_separation_eval_passes_patient_groups(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate_embedding(embedding, **kwargs):
        captured["labels"] = kwargs["labels"]
        captured["groups"] = kwargs["groups"]
        return {
            "metrics": {"separation_logreg_balanced_accuracy": 0.75},
            "records": [],
            "metadata": {},
            "artifacts": {},
        }

    monkeypatch.setattr(
        "coco_pipe.dim_reduction.pipeline.evaluate_embedding",
        fake_evaluate_embedding,
    )
    container = DataContainer(
        X=np.zeros((4, 2)),
        dims=("obs", "feature"),
        ids=np.asarray(["r1", "r2", "r3", "r4"], dtype=object),
        coords={
            "adhd": np.asarray([1, 0, 1, 0], dtype=object),
            "patient_group_id": np.asarray([10, 11, 10, 12], dtype=object),
            "study_id": np.asarray(["0001", "0002", "0001", "0003"], dtype=object),
        },
    )
    fit_payload = {
        "fit_id": "fit1",
        "scope": "condition",
        "condition": "EO_baseline",
        "analysis_mode": "flat",
        "unit_type": "global",
        "unit_name": "all_features",
        "unit_key": "all_features",
        "family": None,
        "input_mode": "raw",
        "representation": "subject",
        "run_label": "test",
        "reducer": "PCA",
        "n_components": 2,
        "descriptor_families": [],
        "descriptor_max_abs_value": None,
    }
    fit_artifact = {
        "fit": fit_payload,
        "ids": np.asarray(["r1", "r2", "r3", "r4"], dtype=object),
        "embedding": np.zeros((4, 2)),
    }

    run_eval(
        fit_artifact=fit_artifact,
        container=container,
        eval_spec={
            "name": "adhd",
            "target_col": "adhd",
            "group_col": "patient_group_id",
            "filters": [],
            "label_map": {},
        },
        out_path=tmp_path / "eval",
        output_root=tmp_path,
        overwrite=False,
    )

    assert captured["labels"].tolist() == ["1", "0", "1", "0"]
    assert captured["groups"].tolist() == ["10", "11", "10", "12"]

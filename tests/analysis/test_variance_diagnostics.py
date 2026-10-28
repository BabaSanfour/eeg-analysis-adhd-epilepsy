from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from coco_pipe.diagnostics import crossed_ss_fractions
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.analysis import merge_foundation_embeddings as mfe
from eeg_adhd_epilepsy.analysis import variance_diagnostics as vd
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root
from tests.analysis.test_extract_foundation_embeddings import _config


def test_run_scores_a_space_from_the_project_files_per_epoch_variant(bids_root, tmp_path):
    extraction = _config(bids_root)
    scratch, project = tmp_path / "scratch", tmp_path / "project"
    efe.run(extraction, scratch, model_key="cbramod", weights=[None], subjects=["0001", "0003"])
    mfe.run(extraction, scratch, project, model_key="cbramod", descs=["base"], weights=[None])
    config = {
        "dataset_name": "standalone_test",
        "bids_root": str(bids_root),
        "embedding_derivative_root": str(project),
        "embedding_model_key": "cbramod",
        "desc": "base",
        "diagnostic_transform": "none",
        "metadata": str(bids_root / "participants.tsv"),
        "diagnostic_populations": ["clinical_task_subset"],
        "evals": [
            {
                "name": "condition_separation",
                "target_col": "condition",
                "positive_class": "EC_baseline",
                "group_col": "subject",
            }
        ],
        "n_null_permutations": 2,
        "random_state": 3,
    }

    output_path = vd.run(config)

    assert output_path == (
        get_derivative_root(bids_root, DerivativeStage.VARIANCE_DIAGNOSTICS)
        / "base"
        / "cbramod"
        / "variance_diagnostics.csv"
    )
    diagnostics = pd.read_csv(output_path)
    assert set(diagnostics["transform"]) == {"none"}
    assert set(diagnostics["scope"]) == {"pooled"}
    assert "between_subject_eta2" in set(diagnostics["metric"])


def test_cross_fitted_leace_erases_each_subject_with_directions_learned_without_it():
    rng = np.random.default_rng(0)
    subject = np.repeat([f"{i:02d}" for i in range(10)], 40)
    condition = np.where(rng.random(len(subject)) < 0.4, "EO", "EC")
    x = rng.standard_normal((len(subject), 12))
    x[:, :3] += rng.standard_normal((10, 3))[np.repeat(np.arange(10), 40)] * 4.0
    x[:, 3] += 3.0 * (condition == "EO")
    container = DataContainer(
        X=x,
        dims=("obs", "feature"),
        coords={"subject": subject, "condition": condition},
        ids=np.arange(len(subject)).astype(str),
    )

    erased, ranks = vd.cross_fitted_leace(container, {"eta2": 0.5})

    assert len(ranks) == 5 and set(ranks) == {3}
    before = crossed_ss_fractions(x, subject, condition)["subject_frac"]
    after = crossed_ss_fractions(erased, subject, condition)["subject_frac"]
    assert before > 0.7 and after < 0.2
    shift = erased[condition == "EO", 3].mean() - erased[condition == "EC", 3].mean()
    assert shift == pytest.approx(3.0, abs=0.3)


def test_write_variance_diagnostics_replaces_each_incoming_assessment(tmp_path):
    root = tmp_path / "diag"
    common = {
        "cohort_name": "main_cohort",
        "population": "clinical_task_subset",
        "scope": "pooled",
        "eval_name": "diagnosis",
        "target_col": "diagnosis",
        "metric": "between_subject_eta2",
    }
    vd.write_variance_diagnostics(
        [{"transform": "none", "value": 0.4, **common}],
        root,
    )
    # An alignment run adds "leace" without clobbering the standalone "none" row.
    vd.write_variance_diagnostics(
        [{"transform": "leace", "value": 0.1, **common}],
        root,
    )
    merged = pd.read_csv(root / "variance_diagnostics.csv")
    assert set(merged["transform"]) == {"none", "leace"}

    # Re-running the standalone "none" replaces only its own rows.
    vd.write_variance_diagnostics(
        [{"transform": "none", "value": 0.9, **common}],
        root,
    )
    merged = pd.read_csv(root / "variance_diagnostics.csv")
    assert set(merged["transform"]) == {"none", "leace"}
    none_value = merged.loc[merged["transform"] == "none", "value"].iloc[0]
    assert none_value == 0.9

    # The same transform/eval from another cohort remains an independent cell.
    vd.write_variance_diagnostics(
        [
            {
                "transform": "none",
                "cohort_name": "other_cohort",
                "population": "clinical_task_subset",
                "value": 0.2,
                **{
                    key: value
                    for key, value in common.items()
                    if key not in {"cohort_name", "population"}
                },
            }
        ],
        root,
    )
    merged = pd.read_csv(root / "variance_diagnostics.csv")
    assert len(merged[merged["transform"] == "none"]) == 2


def test_diagnostic_tasks_apply_cohort_filters_label_maps_and_age_groups():
    X = np.arange(30, dtype=float).reshape(6, 5)
    subject = np.asarray(["1", "2", "3", "4", "5", "6"], dtype=object)
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(6)],
            "patient_group_id": subject,
            "condition": ["EO"] * 6,
            "adhd": [0, 1, 1, 0, 1, 0],
            "psychostimulant": [0, 1, 0, 0, 1, 1],
            "combined_diagnosis": [
                "Control",
                "ADHD",
                "ADHD",
                "Control",
                "ADHD+Epilepsy",
                "Control",
            ],
            "age_group": ["5-8", "9-12", "13-18", "9-12", "13-18", "5-8"],
        }
    )
    config = {
        "group_filters": [
            {"adhd": [0], "psychostimulant": [0]},
            {"adhd": [1], "psychostimulant": [1]},
        ],
        "diagnostic_populations": ["clinical_task_subset"],
        "evals": [
            {
                "name": "med_adhd_vs_ctrl",
                "target_col": "combined_diagnosis",
                "positive_class": "1",
                "label_map": {
                    "Control": "0",
                    "ADHD": "1",
                    "ADHD+Epilepsy": "1",
                },
            },
            {
                "name": "age_separation",
                "target_col": "age_group",
                "class_order": ["5-8", "9-12", "13-18"],
            },
        ],
    }

    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            **{column: frame[column].to_numpy() for column in frame if column != "sample_id"},
            "subject": subject,
            "recording_id": subject,
            "start": np.zeros(6),
        },
        ids=frame["sample_id"].to_numpy(),
    )
    tasks = vd.build_diagnostic_tasks(container, config)

    diagnosis = next(task for task in tasks if task.eval_name == "med_adhd_vs_ctrl")
    assert diagnosis.observation_ids.tolist() == [
        "sample-0",
        "sample-1",
        "sample-3",
        "sample-4",
    ]
    assert diagnosis.labels.tolist() == [0, 1, 0, 1]
    age = next(task for task in tasks if task.eval_name == "age_separation")
    assert age.labels.tolist() == [0, 1, 1, 2]
    assert age.selection_fingerprint != diagnosis.selection_fingerprint


def test_diagnostic_class_order_limits_diagnostics_not_training_population():
    ages = np.asarray(["0-4", "5-8", "9-12", "13-18"], dtype=object)
    subjects = np.asarray(["1", "2", "3", "4"], dtype=object)
    container = DataContainer(
        X=np.arange(20, dtype=float).reshape(4, 5),
        dims=("obs", "feature"),
        coords={
            "subject": subjects,
            "patient_group_id": subjects,
            "condition": np.asarray(["EO"] * 4, dtype=object),
            "age_group": ages,
            "recording_id": subjects,
            "start": np.zeros(4),
        },
        ids=np.asarray([f"sample-{index}" for index in range(4)], dtype=object),
    )
    config = {
        "filter_col": ["age_group"],
        "filter_val": [["5-8", "9-12", "13-18"]],
        "diagnostic_populations": [
            "transform_training_population",
            "clinical_task_subset",
        ],
        "evals": [
            {
                "name": "age_separation",
                "target_col": "age_group",
                "class_order": ["5-8", "9-12", "13-18"],
            }
        ],
    }

    tasks = vd.build_diagnostic_tasks(container, config)
    global_task = next(task for task in tasks if task.population == "transform_training_population")
    clinical_task = next(task for task in tasks if task.population == "clinical_task_subset")

    assert global_task.observation_ids.tolist() == ["sample-1", "sample-2", "sample-3"]
    assert global_task.labels.tolist() == [0, 1, 2]
    assert json.loads(global_task.target_encoding) == {
        "5-8": 0,
        "9-12": 1,
        "13-18": 2,
    }
    assert clinical_task.labels.tolist() == [0, 1, 2]
    assert json.loads(clinical_task.target_encoding) == {
        "5-8": 0,
        "9-12": 1,
        "13-18": 2,
    }


def _diagnostics_config(**overrides):
    return {
        "n_null_permutations": 2,
        "random_state": 0,
        "dataset_name": "test_cohort",
        "subject_probe_cap": 100,
        "subject_probe_n_splits": 2,
        "subject_probe_seed": 42,
        "subject_probe_epochs": 1,
        **overrides,
    }


def _make_diagnostics_task(rng, n_subjects=4, per_subject=6, n_features=4):
    n_obs = n_subjects * per_subject
    subjects = np.repeat([f"sub-{i}" for i in range(n_subjects)], per_subject)
    labels = np.repeat(np.arange(n_subjects) % 2, per_subject)
    features = rng.standard_normal((n_obs, n_features)).astype(np.float32)
    indices = np.arange(n_obs)
    task = vd.DiagnosticTask(
        population="clinical_task_subset",
        scope="pooled",
        eval_name="diagnosis",
        target_col="diagnosis",
        indices=indices,
        subjects=subjects,
        labels=labels,
        observation_ids=indices.astype(str),
        blocks=np.tile(np.arange(per_subject) >= per_subject // 2, n_subjects),
        selection_fingerprint="fp",
        target_encoding="{}",
    )
    return task, indices, features


def _task_subset(task, eval_name, rows):
    return replace(
        task,
        eval_name=eval_name,
        indices=task.indices[rows],
        subjects=task.subjects[rows],
        labels=task.labels[rows],
        observation_ids=task.observation_ids[rows],
        blocks=task.blocks[rows],
    )


def _make_parallel_diagnostics_tasks(rng):
    task, indices, features = _make_diagnostics_task(rng, n_subjects=6)
    tasks = [
        task,
        _task_subset(task, "first_half", slice(None, 18)),
        _task_subset(task, "second_half", slice(18, None)),
    ]
    return tasks, indices, features


def test_parallel_diagnostics_match_sequential():
    rng = np.random.default_rng(0)
    tasks, _, features = _make_parallel_diagnostics_tasks(rng)

    def score(n_jobs):
        config = _diagnostics_config(n_jobs=n_jobs)
        return vd.score_variance_diagnostics(features, tasks, config, transform="none")

    sequential_rows = score(1)
    parallel_rows = score(3)

    # Same rows in the same task order: parallelism must not change results or
    # reorder output relative to the sequential loop.
    assert [row["eval_name"] for row in parallel_rows] == [
        row["eval_name"] for row in sequential_rows
    ]
    assert parallel_rows == sequential_rows

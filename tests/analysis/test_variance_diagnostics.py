from __future__ import annotations

import numpy as np
import pandas as pd
from coco_pipe.decoding.foundation_models import FoundationEmbeddingResult
from coco_pipe.io import DataContainer, save_embedding_derivative

from eeg_adhd_epilepsy.analysis import variance_diagnostics as vd
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root


def _make_raw_embeddings(source_root, seed=0):
    rng = np.random.default_rng(seed)
    metadata_rows = []
    for subject in range(4):
        metadata_rows.append(
            {
                "study_id": f"{subject + 1:04d}",
                "patient_group_id": f"group-{subject + 1:04d}",
                "diagnosis": subject % 2,
            }
        )
        center = rng.standard_normal(6) * 4.0
        for condition_index, condition in enumerate(("EO", "EC")):
            windows = rng.standard_normal((5, 6)) + center
            recording_id = f"sub-{subject + 1:04d}_ses-01_run-{condition_index + 1:02d}"
            save_embedding_derivative(
                FoundationEmbeddingResult(
                    window_embeddings=windows,
                    recording_embedding=windows.mean(0),
                    window_start=np.arange(5),
                    window_stop=np.arange(1, 6),
                    window_index=np.arange(5),
                    metadata={
                        "model_key": "demo",
                        "recording_id": recording_id,
                        "subject": f"{subject + 1:04d}",
                        "condition": condition,
                    },
                ),
                source_root / f"{recording_id}_{condition}_embedding.npz",
            )
    return metadata_rows


def test_standalone_run_scores_raw_embeddings_without_alignment(tmp_path):
    source_root = tmp_path / "source"
    metadata_rows = _make_raw_embeddings(source_root)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    config = {
        "dataset_name": "standalone_test",
        "bids_root": str(tmp_path / "bids"),
        "embedding_derivative_root": str(source_root),
        "embedding_model_key": "demo",
        "diagnostic_transform": "none",
        "metadata": str(metadata_path),
        "subject_col": "study_id",
        "diagnostic_populations": ["clinical_task_subset"],
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
        "random_state": 3,
    }

    output_path = vd.run(config)
    assert output_path == (
        get_derivative_root(tmp_path / "bids", DerivativeStage.VARIANCE_DIAGNOSTICS)
        / "demo"
        / "variance_diagnostics.csv"
    )
    diagnostics = pd.read_csv(output_path)

    # Raw-only run: every row is the "none" baseline, no alignment materialized.
    assert set(diagnostics["transform"]) == {"none"}
    assert set(diagnostics["eval_name"]) == {"condition_separation", "diagnosis"}
    assert {"between_subject_eta2", "between_subject_excess_over_null"}.issubset(
        set(diagnostics["metric"])
    )
    assert not list(
        (tmp_path / "bids" / "derivatives" / "eeg_foundation_embeddings").glob("_alignment_*")
    )


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

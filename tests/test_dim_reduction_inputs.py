from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from coco_pipe.io.structures import DataContainer

from eeg_adhd_epilepsy.analysis.dimensionality_reduction import (
    _resolve_subjects,
    _valid_component_sweep,
    run_eval,
)
from eeg_adhd_epilepsy.io.analysis import _ensure_recording_id
from eeg_adhd_epilepsy.io.table import load_tabular_data


def test_descriptor_loader_filters_subjects(tmp_path):
    feature_column = "mean_band_log_abs_alpha_chgrp-midline"
    table_path = tmp_path / "features.csv"
    feature_columns_path = tmp_path / "feature_columns.json"
    pd.DataFrame(
        {
            "study_id": [1, 2],
            "condition": ["EO_baseline", "EO_baseline"],
            "recording_id": ["0001_ses-01_run-01", "0002_ses-01_run-01"],
            feature_column: [0.1, 0.2],
        }
    ).to_csv(table_path, index=False)
    feature_columns_path.write_text(json.dumps([feature_column]), encoding="utf-8")

    container = load_tabular_data(
        table_path=table_path,
        feature_columns_path=feature_columns_path,
        condition="EO_baseline",
        subjects=["sub-0002"],
        subject_col="study_id",
    )

    assert container.X.shape == (1, 1)
    assert container.coords["study_id"].tolist() == [2]
    assert container.ids.tolist() == ["0002_ses-01_run-01"]


def test_run_aware_recording_id_is_added_from_ids():
    container = DataContainer(
        X=np.zeros((2, 2, 3)),
        dims=("obs", "channel", "time"),
        ids=np.asarray(
            [
                "sub-0001_ses-01_task-clinical_run-01_ep-0",
                "sub-0001_ses-01_task-clinical_run-02_ep-0",
            ],
            dtype=object,
        ),
        coords={
            "study_id": np.asarray(["0001", "0001"], dtype=object),
            "channel": np.asarray(["Fz", "Cz"], dtype=object),
            "time": np.arange(3),
        },
    )

    out = _ensure_recording_id(container, "study_id")

    assert out.coords["recording_id"].tolist() == [
        "0001_ses-01_run-01",
        "0001_ses-01_run-02",
    ]


def test_dim_reduction_resolves_descriptor_subjects_from_table(tmp_path):
    table_path = tmp_path / "features.csv"
    pd.DataFrame({"study_id": ["1", "sub-0003"], "condition": ["EO", "EC"]}).to_csv(
        table_path,
        index=False,
    )
    args = SimpleNamespace(
        input_mode="descriptors",
        subjects=None,
        descriptor_table_path=str(table_path),
        subject_col="study_id",
    )

    assert _resolve_subjects(args, tmp_path, pd.DataFrame()) == ["0001", "0003"]


def test_invalid_n_components_are_skipped():
    container = DataContainer(
        X=np.zeros((3, 5)),
        dims=("obs", "feature"),
        coords={"feature": np.arange(5)},
    )

    assert _valid_component_sweep(container, [2, 3, 4, 10]) == [2, 3]


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
        "eeg_adhd_epilepsy.analysis.dimensionality_reduction.evaluate_embedding",
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
        "representation": "subject_flat",
        "aggregation_unit": "recording",
        "run_label": "test",
        "reducer": "PCA",
        "n_components": 2,
        "descriptor_families": [],
        "descriptor_max_abs_value": None,
    }
    fit_artifact = {
        "ids": np.asarray(["r1", "r2", "r3", "r4"], dtype=object),
        "embedding": np.zeros((4, 2)),
    }

    run_eval(
        fit_payload=fit_payload,
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

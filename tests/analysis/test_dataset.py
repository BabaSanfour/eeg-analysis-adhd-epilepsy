from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis.dataset import (
    attach_subject_metadata,
    build_container,
    filter_cohort_container,
)


def _embedding_container(subjects):
    n = len(subjects)
    return DataContainer(
        X=np.random.default_rng(0).normal(size=(n, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": np.asarray(["embedding_0000", "embedding_0001"], dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "condition": np.asarray(["EO_baseline"] * n, dtype=object),
        },
        ids=np.asarray([f"rec-{i}" for i in range(n)], dtype=object),
        meta={},
    )


def test_attach_subject_metadata_joins_eval_columns():
    # Foundation embeddings arrive without the eval target/group columns; the join
    # must add them by subject so the supervised separation evals can resolve.
    container = _embedding_container(["0001", "0001", "0002"])
    meta = pd.DataFrame(
        {
            "study_id": [1, 2],
            "combined_diagnosis": ["ADHD", "Control"],
            "patient_group_id": ["g1", "g2"],
        }
    )
    out = attach_subject_metadata(container, meta, "study_id")
    assert list(out.coords["combined_diagnosis"]) == ["ADHD", "ADHD", "Control"]
    assert list(out.coords["patient_group_id"]) == ["g1", "g1", "g2"]
    # existing identity coords are not clobbered
    assert list(out.coords["subject"]) == ["0001", "0001", "0002"]


def test_attach_subject_metadata_missing_subject_is_nan():
    container = _embedding_container(["9999"])
    meta = pd.DataFrame({"study_id": [1], "combined_diagnosis": ["ADHD"]})
    out = attach_subject_metadata(container, meta, "study_id")
    assert pd.isna(out.coords["combined_diagnosis"][0])  # evals treat NaN as missing


def test_filter_cohort_container_applies_group_union_and_column_filter():
    container = _embedding_container(["0001", "0002", "0003", "0004"])
    container.coords.update(
        {
            "diagnosis": np.asarray(["ADHD", "Control", "ADHD", "Control"]),
            "medicated": np.asarray(["yes", "no", "no", "yes"]),
            "age_group": np.asarray(["5-8", "5-8", "9-12", "13-18"]),
        }
    )

    selected = filter_cohort_container(
        container,
        group_filters=[
            {"diagnosis": ["ADHD"], "medicated": ["yes"]},
            {"diagnosis": ["Control"], "medicated": ["yes"]},
        ],
        filter_col=["age_group"],
        filter_val=[["5-8", "13-18"]],
    )

    assert selected.ids.tolist() == ["rec-0", "rec-3"]


def test_build_container_rejects_bandpass_without_reepoch():
    # Filtering happens on the continuous recording inside the re-epoch path;
    # the saved epoch derivatives are already windowed, so bandpass there is
    # ill-posed and must raise rather than silently no-op.
    with pytest.raises(ValueError, match="bandpass is only supported"):
        build_container(
            bids_root=Path("/nonexistent"),
            use_derivatives=True,
            window_source="derivative",
            bandpass=(0.5, 40.0),
        )

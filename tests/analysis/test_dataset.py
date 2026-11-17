import numpy as np
import pandas as pd
from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis.dataset import (
    attach_subject_metadata,
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
    container = container.replace(
        coords={
            **container.coords,
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


def test_filter_cohort_container_keeps_each_childs_first_recording_per_condition():
    # Child g1 has two recordings, 0001 (age 8) and 0002; 0001 has no EC epochs,
    # so EC falls back to 0002. Child g2 has one recording.
    container = _embedding_container(["0001", "0001", "0002", "0002", "0003", "0003"])
    container = container.replace(
        coords={
            **container.coords,
            "condition": np.asarray(
                ["EO_baseline"] * 3 + ["EC_baseline", "EO_baseline", "EC_baseline"]
            ),
            "patient_group_id": np.asarray(["g1", "g1", "g1", "g1", "g2", "g2"], dtype=object),
            "age": np.asarray([8.0, 8.0, 10.5, 10.5, 12.0, 12.0], dtype=object),
        }
    )

    selected = filter_cohort_container(container, group_filters=None, filter_col=[], filter_val=[])

    assert selected.ids.tolist() == ["rec-0", "rec-1", "rec-3", "rec-4", "rec-5"]


def test_filter_cohort_container_matched_keeps_equal_groups_per_cell():
    # Controls: 3 boys aged 8, 1 girl aged 9; ADHD: 1 boy aged 8, 2 girls aged 9, 1 girl
    # aged 10 (no control to match). Each child has two epochs.
    subjects = [f"{i:04d}" for i in range(8)]
    adhd = [0, 0, 0, 0, 1, 1, 1, 1]
    sex = ["M", "M", "M", "F", "M", "F", "F", "F"]
    age = [8.2, 8.5, 8.9, 9.1, 8.4, 9.3, 9.8, 10.2]
    container = _embedding_container([s for s in subjects for _ in range(2)])
    container = container.replace(
        coords={
            **container.coords,
            "patient_group_id": np.repeat(subjects, 2).astype(object),
            "adhd": np.repeat(adhd, 2),
            "sex": np.repeat(sex, 2),
            "age": np.repeat(age, 2),
            "epilepsy": np.zeros(16, dtype=int),
            "autism": np.zeros(16, dtype=int),
        }
    )

    selected = filter_cohort_container(
        container,
        group_filters=[{"adhd": [0]}, {"adhd": [1]}],
        filter_col=[],
        filter_val=[],
        matched=True,
    )

    kept = pd.DataFrame(
        {
            "subject": selected.coords["subject"],
            "adhd": selected.coords["adhd"],
            "sex": selected.coords["sex"],
            "age": np.floor(selected.coords["age"]),
        }
    )
    assert kept.groupby("subject").size().eq(2).all()  # every epoch of a kept child
    children = kept.drop_duplicates("subject")
    cells = children.groupby(["sex", "age", "adhd"]).size().unstack("adhd")
    assert cells.to_dict("index") == {("F", 9.0): {0: 1, 1: 1}, ("M", 8.0): {0: 1, 1: 1}}

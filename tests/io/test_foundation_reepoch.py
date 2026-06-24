"""Tests for the cleaned-continuous re_epoch path (LaBraM 15 s windows)."""

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_adhd_epilepsy.analysis.utils.foundation import resolve_foundation_input_plan
from eeg_adhd_epilepsy.io import readers
from eeg_adhd_epilepsy.analysis import dataset

_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _synthetic_block_raw(sfreq: float = 200.0, duration_s: float = 60.0) -> mne.io.BaseRaw:
    info = mne.create_info(_CHANNELS, sfreq, ch_types="eeg")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(_CHANNELS), int(sfreq * duration_s))) * 1e-6
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onset=[0.0], duration=[duration_s], description=["BLOCK_EO_baseline"])
    )
    return raw


def test_reepoch_plan_uses_cleaned_continuous_source():
    plan = resolve_foundation_input_plan(
        {"segment_duration": 10.0, "use_derivatives": True},
        {
            "model_key": "labram",
            "segment_duration": 15.0,
            "window_source": "re_epoch",
            "backend_kwargs": {"interpolate_channels": True},
        },
    )
    assert plan.window_source == "re_epoch"
    assert plan.use_derivatives is False
    assert plan.segment_duration == 15.0
    assert plan.skip_reason is None


def test_load_cleaned_continuous_container(tmp_path, monkeypatch):
    raw = _synthetic_block_raw()
    monkeypatch.setattr(dataset, "read_preproc_stage", lambda *a, **k: (raw, {}, []))
    metadata = pd.DataFrame(
        {
            "study_id": [1],
            "combined_diagnosis": ["ADHD"],
            "patient_group_id": ["p1"],
        }
    )

    container = dataset.reepoch_eeg(
        bids_root=tmp_path,
        subjects=["0001"],
        task="clinical",
        segment_duration=15.0,
        overlap=0.0,
        condition="EO_baseline",
        metadata_df=metadata,
        subject_col="study_id",
    )

    assert container.dims == ("obs", "channel", "time")
    assert container.X.shape[0] >= 1  # 60 s / 15 s windows
    assert container.X.shape[1] == len(_CHANNELS)
    # MNE inclusive endpoint: 15 s * 200 Hz (+1); normalize_inclusive_endpoint drops it.
    assert container.X.shape[2] in (3000, 3001)
    assert set(np.asarray(container.coords["condition"]).tolist()) == {"EO_baseline"}
    assert set(np.asarray(container.coords["patient_group_id"]).tolist()) == {"p1"}
    assert set(np.asarray(container.coords["combined_diagnosis"]).tolist()) == {"ADHD"}
    assert container.meta["window_source"] == "re_epoch_cleaned_continuous"
    assert container.meta["autoreject_applied"] is False


def test_load_cleaned_continuous_missing_condition_raises(tmp_path, monkeypatch):
    raw = _synthetic_block_raw()
    monkeypatch.setattr(dataset, "read_preproc_stage", lambda *a, **k: (raw, {}, []))
    # No EC_baseline block exists
    with pytest.raises(
        RuntimeError, match="No cleaned-continuous epochs.*First issues.*EC_baseline"
    ):
        dataset.reepoch_eeg(
            bids_root=tmp_path,
            subjects=["0001"],
            task="clinical",
            segment_duration=15.0,
            overlap=0.0,
            condition="EC_baseline",
            metadata_df=None,
            subject_col="study_id",
        )

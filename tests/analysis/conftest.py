"""Small preproc derivatives for the embedding tests: three subjects, 70 s each."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_adhd_epilepsy.preproc.epochs import make_epochs_from_preproc_raw
from tests.conftest import _SYNTHETIC_CHANNELS

SFREQ = 200.0


def _write_recording(preproc_root: Path, subject: str, blocks: dict[str, tuple[float, float]]):
    """A continuous desc-base recording with BLOCK_ annotations, and its 10 s epochs."""
    eeg_dir = preproc_root / f"sub-{subject}" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    stem = f"sub-{subject}_ses-01_task-clinical_run-01_desc-base"
    info = mne.create_info(_SYNTHETIC_CHANNELS, SFREQ, "eeg")
    data = np.random.default_rng(int(subject)).normal(size=(len(info.ch_names), 70 * 200)) * 1e-5
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[onset for onset, _ in blocks.values()],
            duration=[duration for _, duration in blocks.values()],
            description=[f"BLOCK_{name}" for name in blocks],
        )
    )
    raw.save(eeg_dir / f"{stem}_eeg.fif", verbose="ERROR")
    make_epochs_from_preproc_raw(raw, 10.0, save_path=eeg_dir / f"{stem}_epo.fif")


@pytest.fixture
def bids_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    root = tmp_path / "BIDS"
    preproc_root = root / "derivatives" / "preproc"
    _write_recording(preproc_root, "0001", {"EO_baseline": (0, 30), "EC_baseline": (30, 30)})
    _write_recording(preproc_root, "0002", {"HV_EC": (0, 30)})
    _write_recording(preproc_root, "0003", {"EO_baseline": (0, 30), "EC_baseline": (30, 30)})
    pd.DataFrame({"participant_id": ["sub-0001", "sub-0002", "sub-0003"]}).to_csv(
        root / "participants.tsv", sep="\t", index=False
    )
    return root

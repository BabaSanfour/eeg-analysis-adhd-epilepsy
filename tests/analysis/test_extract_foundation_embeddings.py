"""Embedding extraction on small preproc derivatives, with the cached CBraMod and LaBraM on CPU."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from coco_pipe.io import embedding_path, read_json

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
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


def _config(bids_root: Path, conditions=("EO_baseline", "EC_baseline")) -> dict:
    seeds = [0, 1]
    return {
        "bids_root": str(bids_root),
        "desc": "base",
        "task": "clinical",
        "conditions": list(conditions),
        "device": "cpu",
        "chunk_subjects": 1,
        "token_dtype": "float16",
        "models": {
            "cbramod": {
                "segment_duration": 10.0,
                "window_source": "derivative",
                "random_seeds": seeds,
            },
            "labram": {
                "segment_duration": 15.0,
                "window_source": "re_epoch",
                "random_seeds": seeds,
            },
        },
    }


def _windows(path: Path) -> list[tuple[str, int]]:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        return list(zip(dataset["condition"].values.tolist(), dataset["epoch"].values.tolist()))


def test_extraction_writes_every_view_and_a_resubmission_skips_done_recordings(bids_root, tmp_path):
    derivatives = tmp_path / "scratch"
    root = derivatives / "coco-pipe-cbramod"
    recording = "sub-0001_ses-01_task-clinical_run-01"
    other = "sub-0003_ses-01_task-clinical_run-01"
    options = {
        "model_key": "cbramod",
        "weights": [None, 0],
        "subjects": ["0001", "0002", "0003"],
    }

    records = efe.run(_config(bids_root), derivatives, **options)

    for label in ("base", "baseRandom0"):
        statuses = [(r.get("recording_id") or r["subject"], r["status"]) for r in records[label]]
        assert statuses == [(recording, "written"), ("0002", "skipped"), (other, "written")]
        assert read_json(root / f"desc-{label}_manifest.json")["records"][0]["status"] == "written"
        for view in ("window", "token", "channel", "patch"):
            path = embedding_path(root, label, view, recording_id=recording).fpath
            assert _windows(path) == [("EO_baseline", n) for n in range(3)] + [
                ("EC_baseline", n) for n in range(3, 6)
            ]
        token = embedding_path(root, label, "token", recording_id=recording).fpath
        with xr.open_dataset(token, engine="h5netcdf") as dataset:
            assert dataset["embeddings"].dtype == np.float16
            assert dataset["source"].item() == (
                f"bids:preproc:sub-0001/ses-01/eeg/{recording}_desc-base_epo.fif"
            )
    description = read_json(root / "dataset_description.json")
    assert description["DatasetLinks"] == {
        "preproc": (bids_root / "derivatives" / "preproc").absolute().as_uri()
    }

    # A resubmission embeds only the first recording of each weight set again and
    # rewrites nothing.
    files = sorted(root.rglob("*embeddings.nc"))
    written = {path: path.stat().st_mtime_ns for path in files}
    again = efe.run(_config(bids_root), derivatives, **options)
    assert [r["status"] for r in again["base"]] == ["written", "skipped", "done"]
    assert {path: path.stat().st_mtime_ns for path in files} == written

    # Files that hold other conditions than asked are rewritten.
    ec_only = efe.run(_config(bids_root, conditions=["EC_baseline"]), derivatives, **options)
    assert [r["status"] for r in ec_only["baseRandom0"]] == ["written", "skipped", "written"]
    window = embedding_path(root, "baseRandom0", "window", recording_id=other).fpath
    assert _windows(window) == [("EC_baseline", n) for n in range(3, 6)]


def test_extraction_recuts_windows_for_labram(bids_root, tmp_path):
    root = tmp_path / "scratch" / "coco-pipe-labram"
    efe.run(
        _config(bids_root),
        tmp_path / "scratch",
        model_key="labram",
        weights=[1],
        subjects=["0001"],
    )

    recording = "sub-0001_ses-01_task-clinical_run-01"
    path = embedding_path(root, "baseRandom1", "window", recording_id=recording).fpath
    assert _windows(path) == [("EO_baseline", 0), ("EO_baseline", 1)] + [
        ("EC_baseline", 2),
        ("EC_baseline", 3),
    ]
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        assert dataset["source"].item().endswith(f"{recording}_desc-base_eeg.fif")


def test_extraction_stops_on_a_recording_that_cannot_be_read(bids_root, tmp_path):
    epochs = next((bids_root / "derivatives" / "preproc" / "sub-0001").rglob("*_epo.fif"))
    epochs.write_bytes(b"not a fif file")

    with pytest.raises(RuntimeError, match="could not be loaded"):
        efe.run(
            _config(bids_root),
            tmp_path / "scratch",
            model_key="cbramod",
            weights=[0],
            subjects=["0001"],
        )

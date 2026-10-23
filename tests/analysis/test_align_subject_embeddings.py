"""Subject alignment of the embeddings of small preproc derivatives, with the cached CBraMod."""

import numpy as np
from coco_pipe.io import find_embeddings, load_embeddings, read_json
from coco_pipe.transforms import RiemannAlign

from eeg_adhd_epilepsy.analysis import align_subject_embeddings as ase
from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from tests.analysis.test_extract_foundation_embeddings import _config


def test_alignment_writes_one_file_per_transform_holding_every_recording(bids_root, tmp_path):
    config = _config(bids_root)
    config["alignment"] = {
        "window": ["ea_mean", "ea_coral", "leace"],
        "token": ["ea_mean", "ea_coral", "ra"],
        "per_channel": True,
    }
    scratch, project = tmp_path / "scratch", tmp_path / "project"
    efe.run(config, scratch, model_key="cbramod", weights=[None], subjects=["0001", "0003"])

    records = ase.run(config, scratch, project, model_key="cbramod", seed=None, n_jobs=2)

    root = project / "coco-pipe-cbramod"
    recordings = [f"sub-{subject}_ses-01_task-clinical_run-01" for subject in ("0001", "0003")]
    assert sorted(records) == [
        "baseEaCoral",
        "baseEaCoralCh",
        "baseEaMean",
        "baseEaMeanCh",
        "baseLeace",
        "baseRa",
    ]
    for label, label_records in records.items():
        assert [record["recording_id"] for record in label_records] == recordings
        assert read_json(root / f"desc-{label}_manifest.json")["records"] == label_records
    assert {path.name for path in root.glob("*embeddings.nc")} == {
        f"task-clinical_desc-{name}embeddings.nc"
        for name in (
            "baseEaMean_window",
            "baseEaCoral_window",
            "baseLeace_window",
            "baseEaMean_token",
            "baseEaCoral_token",
            "baseEaMeanCh_token",
            "baseEaCoralCh_token",
            "baseRa_cov",
        )
    }
    assert (root / "task-clinical_desc-baseRa_covembeddings.json").exists()

    # Each subject's tokens are centered on every channel.
    per_channel = load_embeddings(root / "task-clinical_desc-baseEaMeanCh_tokenembeddings.nc")
    subjects = np.asarray(per_channel.coords["subject"]).astype(str)
    for subject in ("0001", "0003"):
        means = per_channel.X[subjects == subject].mean(axis=(0, 2))
        np.testing.assert_allclose(means, 0, atol=1e-2)

    # RA one subject at a time, over two processes, gives what fitting every window does.
    tokens = load_embeddings(find_embeddings(scratch / "coco-pipe-cbramod", "base", "token"))
    flat = tokens.X.reshape(len(tokens.X), -1, tokens.X.shape[-1])
    groups = np.asarray(tokens.coords["subject"]).astype(str)
    expected = RiemannAlign().fit_transform(flat, groups=groups)
    ra = load_embeddings(root / "task-clinical_desc-baseRa_covembeddings.nc")
    np.testing.assert_allclose(ra.X, expected, rtol=1e-4, atol=1e-6)

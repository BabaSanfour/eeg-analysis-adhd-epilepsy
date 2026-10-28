"""LEACE-erased window vectors of small preproc derivatives, with the cached CBraMod."""

from coco_pipe.io import load_embeddings, read_json

from eeg_adhd_epilepsy.analysis import align_subject_embeddings as ase
from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.analysis import merge_foundation_embeddings as mfe
from tests.analysis.test_extract_foundation_embeddings import _config


def test_leace_writes_one_file_holding_every_recording_or_says_why_not(bids_root, tmp_path):
    config = _config(bids_root)
    config["alignment"] = {"leace": {"eta2": 0.5}}
    scratch, project = tmp_path / "scratch", tmp_path / "project"
    efe.run(config, scratch, model_key="cbramod", weights=[None], subjects=["0001", "0003"])
    mfe.run(config, scratch, project, model_key="cbramod", descs=["base"], weights=[None])

    records = ase.run(config, project, model_key="cbramod", seed=None)

    root = project / "coco-pipe-cbramod"
    recordings = [f"sub-{subject}_ses-01_task-clinical_run-01" for subject in ("0001", "0003")]
    assert [record["recording_id"] for record in records] == recordings
    assert read_json(root / "desc-baseLeace_manifest.json")["records"] == records
    path = root / "task-clinical_desc-baseLeace_windowembeddings.nc"
    assert path.exists()
    transform = load_embeddings(path).meta["config"]["Transform"]
    assert transform["name"] == "leace" and transform["fitted_on"].startswith("every window")

    # A LEACE that would erase every direction writes no file and removes the earlier one.
    config["alignment"] = {"leace": {"eta2": 0.0}}
    records = ase.run(config, project, model_key="cbramod", seed=None)
    assert records[0]["status"] == "skipped"
    assert "would erase 200 of 200" in records[0]["reason"]
    assert not path.exists()

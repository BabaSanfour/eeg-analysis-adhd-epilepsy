import pandas as pd
from coco_pipe.io import read_json, write_embedding_manifest

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root


def test_finalize_unions_all_task_shards(tmp_path, monkeypatch):
    """Merge must union every task's rows; last-writer-wins would lose shards."""
    monkeypatch.setattr(efe, "make_foundation_embedding_report", lambda *args, **kwargs: None)
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    config = {"bids_root": str(bids_root), "dataset_name": "synthetic"}
    derivative_root = get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)

    # Two subject-array tasks, disjoint subjects, each writing its own shard.
    write_embedding_manifest(
        derivative_root / "_partials" / "row-0001",
        [
            {"recording_id": "sub-0001_ses-01_run-01", "model_key": "cbramod", "status": "success"},
            {
                "recording_id": "sub-0001_ses-01_run-01",
                "model_key": "labram",
                "status": "skipped",
                "reason": "unsupported",
            },
        ],
    )
    write_embedding_manifest(
        derivative_root / "_partials" / "row-0002",
        [
            {
                "recording_id": "sub-0002_ses-01_run-01",
                "model_key": "cbramod",
                "status": "failed",
                "reason": "boom",
            }
        ],
    )

    efe.finalize(config, derivative_root)

    manifest = read_json(derivative_root / "run_manifest.json")
    assert len(manifest["records"]) == 3  # nothing dropped across shards

    failures = pd.read_csv(derivative_root / "failures.csv")
    assert len(failures) == 2  # only the non-success rows
    assert set(failures["status"]) == {"skipped", "failed"}
    assert (derivative_root / "dataset_description.json").exists()

import json
import sys

import pandas as pd
import pytest
import yaml
from coco_pipe.io import read_json

from eeg_adhd_epilepsy.analysis import merge_foundation_embeddings as mfe
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root


def test_scan_artifacts_keeps_pooling_and_alignment_model_keys_separate(tmp_path):
    raw_path = tmp_path / "sub-0001_desc-demo_embedding.npz"
    attention_path = tmp_path / "sub-0001_desc-demoAttention_embedding.npz"
    aligned_path = tmp_path / "sub-0001_proc-alignra_desc-demo_embedding.npz"
    for path, model_key in (
        (raw_path, "demo"),
        (attention_path, "demo_pool-attention"),
        (aligned_path, "demo_align-ra"),
    ):
        path.touch()
        path.with_suffix(".json").write_text(
            json.dumps({"model_key": model_key}),
            encoding="utf-8",
        )

    by_model, records = mfe._scan_artifacts(tmp_path)

    expected = {"demo", "demo_pool-attention", "demo_align-ra"}
    assert set(by_model) == expected
    assert {record["model_key"] for record in records} == expected


def test_merge_reads_config_used_and_unions_failure_shards(tmp_path, monkeypatch):
    """Merge takes its config from the frozen config_used.yaml and unions every task's
    ``_failures`` shard; last-writer-wins would lose shards."""
    monkeypatch.setattr(mfe, "make_foundation_embedding_report", lambda *args, **kwargs: None)
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    derivative_root = get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)
    derivative_root.mkdir(parents=True, exist_ok=True)

    # extract_foundation_embeddings would have frozen this; merge reads it instead of --config.
    (derivative_root / "config_used.yaml").write_text(
        yaml.safe_dump({"dataset_name": "synthetic"}, sort_keys=True), encoding="utf-8"
    )

    # Two array tasks, disjoint subjects, each leaving only its per-task failures shard.
    failures_dir = derivative_root / "_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "recording_id": "sub-0001_ses-01_run-01",
                "model_key": "labram",
                "status": "skipped",
                "reason": "unsupported",
            }
        ]
    ).to_csv(failures_dir / "row-0001.csv", index=False)
    pd.DataFrame(
        [
            {
                "recording_id": "sub-0002_ses-01_run-01",
                "model_key": "cbramod",
                "status": "failed",
                "reason": "boom",
            }
        ]
    ).to_csv(failures_dir / "row-0002.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_foundation_embeddings",
            "--bids_root",
            str(bids_root),
            "--derivative_root",
            str(derivative_root),
            "--reports_root",
            str(tmp_path / "reports"),
        ],
    )
    mfe.main()

    manifest = read_json(derivative_root / "run_manifest.json")
    assert len(manifest["records"]) == 2  # nothing dropped across shards

    failures = pd.read_csv(derivative_root / "failures.csv")
    assert len(failures) == 2  # both non-success rows survive
    assert set(failures["status"]) == {"skipped", "failed"}
    assert (derivative_root / "dataset_description.json").exists()


def test_merge_requires_config_used(tmp_path, monkeypatch):
    """Without a frozen config_used.yaml the merge fails fast (extraction never ran)."""
    bids_root = tmp_path / "BIDS"
    bids_root.mkdir()
    derivative_root = get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_foundation_embeddings",
            "--bids_root",
            str(bids_root),
            "--derivative_root",
            str(derivative_root),
        ],
    )
    with pytest.raises(FileNotFoundError, match="config_used.yaml not found"):
        mfe.main()

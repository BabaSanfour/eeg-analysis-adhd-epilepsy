"""All-recordings files of small preproc derivatives, with the cached CBraMod."""

import numpy as np
import pytest
from coco_pipe.io import find_embeddings, load_embeddings, read_json

from eeg_adhd_epilepsy.analysis import extract_foundation_embeddings as efe
from eeg_adhd_epilepsy.analysis import merge_foundation_embeddings as mfe
from tests.analysis.test_extract_foundation_embeddings import _config


def test_merge_writes_one_file_per_label_and_view_and_refuses_a_missing_recording(
    bids_root, tmp_path
):
    config = _config(bids_root)
    scratch, project = tmp_path / "scratch", tmp_path / "project"
    efe.run(config, scratch, model_key="cbramod", weights=[None, 0], subjects=["0001", "0003"])
    options = {"model_key": "cbramod", "descs": ["base"], "weights": [None, 0]}

    # Each label merged from a staged copy, which is removed afterwards.
    stage = tmp_path / "stage"
    written = mfe.run(config, scratch, project, **options, stage_dir=stage, n_jobs=4)
    assert not any(stage.iterdir())

    source, root = scratch / "coco-pipe-cbramod", project / "coco-pipe-cbramod"
    views = ("token", "window", "channel", "patch")
    assert sorted(written) == ["base", "baseRandom0"]
    for label in written:
        assert sorted(written[label]) == sorted(
            f"task-clinical_desc-{label}_{view}embeddings.nc" for view in views
        )
        for view in views:
            parts = load_embeddings(find_embeddings(source, label, view))
            whole = load_embeddings(root / f"task-clinical_desc-{label}_{view}embeddings.nc")
            np.testing.assert_array_equal(whole.X, parts.X)
        manifest = read_json(root / f"desc-{label}_manifest.json")["records"]
        assert manifest == read_json(source / f"desc-{label}_manifest.json")["records"]
    assert (root / "descriptions.tsv").exists()

    # A recording of the manifest without its file stops the run.
    find_embeddings(source, "baseRandom0", "patch")[0].fpath.unlink()
    with pytest.raises(RuntimeError, match="baseRandom0 patch: 1 recording"):
        mfe.run(config, scratch, project, **options)

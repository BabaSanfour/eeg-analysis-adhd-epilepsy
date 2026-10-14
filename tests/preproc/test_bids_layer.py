"""The raw BIDS root and the preproc derivative pass the BIDS validator."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids

from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.preproc import to_bids

VALIDATOR = shutil.which("bids-validator-deno") or str(
    Path(sys.executable).with_name("bids-validator-deno")
)


def _validate(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [VALIDATOR, str(root), "--max-rows", "-1"], capture_output=True, text=True
    )


def test_converted_dataset_and_preproc_derivative_validate(synthetic_raw, tmp_path):
    bids_root = tmp_path / "BIDS"
    for subject in ("0001", "0002"):
        raw = synthetic_raw(seed=int(subject), duration=30.0)
        raw.info["line_freq"] = 60
        write_raw_bids(
            raw,
            BIDSPath(
                root=bids_root,
                subject=subject,
                session="01",
                task="clinical",
                run="01",
                datatype="eeg",
            ),
            format="BrainVision",
            allow_preload=True,
            verbose=False,
        )
    metadata = pd.DataFrame({"study_id": [1, 2, 3], "age": [9.5, None, 30.0], "sex": list("MFF")})
    to_bids._write_participants(bids_root, metadata)
    to_bids._write_dataset_description(bids_root)

    preproc_root = bids_io.get_derivative_root(bids_root, bids_io.DerivativeStage.PREPROC)
    bids_io.write_preproc_dataset_description(preproc_root)
    for vhdr in bids_io.discover_raw_eeg(bids_root):
        raw = mne.io.read_raw_brainvision(vhdr, verbose="ERROR")
        base_path = bids_io.get_stage_output_path(vhdr, preproc_root, "base", create_dir=True)
        raw.save(base_path, verbose="ERROR")
        epochs_path = base_path.with_name(base_path.name.replace("_eeg.fif", "_epo.fif"))
        mne.make_fixed_length_epochs(raw, 10.0, verbose="ERROR").save(epochs_path, verbose="ERROR")
        epochs_path.with_suffix(".json").write_text("{}")
        base_path.with_name(base_path.name.replace("_eeg.fif", "_provenance.json")).write_text("{}")

    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t", keep_default_na=False)
    assert participants["participant_id"].tolist() == ["sub-0001", "sub-0002"]
    for root in (bids_root, preproc_root):
        result = _validate(root)
        assert result.returncode == 0, result.stdout

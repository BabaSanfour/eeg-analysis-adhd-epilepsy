"""The raw BIDS root and the preproc derivative pass the validator; epochs get the labels."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids

from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.metadata import schema
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
    metadata = pd.DataFrame(
        {
            "source_dataset": ["adhd", "drug_resistant", "adhd"],
            "study_id": [1, 2, 3],
            "patient_id": [501, 502, 503],
            "patient_group_id": [0, 1, 2],
            "eeg_date": ["2019-01-02", "2020-03-04", "2021-05-06"],
            "first_eeg": [0, 1, 0],
            "age": [9.5, None, 16.0],
            "age_group": ["9-12", None, "13-18"],
            "sex": ["M", "F", "F"],
            "adhd": [1, 0, 0],
            "autism": [0, 0, 1],
            "epilepsy": [0, 1, 1],
            "combined_diagnosis": ["ADHD", "Epilepsy", "ASD+Epilepsy"],
            "psychostimulant": [1, 0, 0],
            "psychostimulant_category": [
                "Methylphenidate",
                "No Psychostimulant",
                "No Psychostimulant",
            ],
            "asm": [0, 1, 1],
            **{column: [0, 0, 0] for column in schema.EPILEPSY_MED_COLS},
            "other_asm": [0, 1, 1],
            "asm_types": ["No_ASM", "No_ASM", "No_ASM"],
            "meds_summary": ["Psychostim", "ASM", "ASM"],
            "asm_resistant": [0, 1, None],
        }
    )
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

    participants = bids_io.read_participants(bids_root / "participants.tsv")
    assert participants["subject"].tolist() == ["0001", "0002"]
    assert list(participants.columns[1:]) == schema.PARTICIPANTS_COLUMNS
    assert participants["adhd"].tolist() == [1, 0]
    assert not {"patient_id", "eeg_date"} & set(participants.columns)
    for root in (bids_root, preproc_root):
        result = _validate(root)
        assert result.returncode == 0, result.stdout

    # Saved epochs carry the participants.tsv labels of the raw dataset.
    epochs = build_container(bids_root, use_derivatives=True)
    assert set(zip(epochs.coords["subject"], epochs.coords["combined_diagnosis"])) == {
        ("0001", "ADHD"),
        ("0002", "Epilepsy"),
    }

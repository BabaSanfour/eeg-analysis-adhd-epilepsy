"""Generate a synthetic EEG dataset for testing the neurodags pipelines.

Produces BrainVision (.vhdr/.vmrk/.eeg) files under datasets/synthetic_eeg/rawdata/
using neurodags's built-in 1/f noise generator. Each recording gets two BLOCK_*
annotations so condition-level pipelines (step-0c_conditions.yml) can be tested:

  BLOCK_EO   [2–12s]  — eyes open (5 × 2s epochs)
  BLOCK_EC   [16–26s] — eyes closed (5 × 2s epochs)

Layout:
  rawdata/
    sub-0/eeg/sub-0_run-0_eeg.vhdr
    sub-0/eeg/sub-0_run-1_eeg.vhdr
    sub-1/eeg/...
    sub-2/eeg/...

Usage:
    python pipelines/generate_synthetic.py
"""

from pathlib import Path

import mne
import numpy as np

from neurodags.datasets import generate_dummy_dataset

RAW_DIR = Path(__file__).parent.parent / "datasets" / "synthetic_eeg" / "rawdata"

generate_dummy_dataset(
    data_params={
        "PATTERN": "%subject%/eeg/%subject%_run-%run%_eeg",
        "DATASET": "synthetic_eeg",
        "NSUBS": 3,
        "NSESSIONS": 1,
        "NTASKS": 1,
        "NACQS": 1,
        "NRUNS": 2,
        "PREFIXES": {
            "subject": "sub-",
            "session": "ses-",
            "task": "task-",
            "acquisition": "acq-",
            "run": "run-",
        },
        "ROOT": str(RAW_DIR),
    },
    generation_args={
        "NCHANNELS": 8,
        "SFREQ": 256.0,
        "STOP": 30.0,
        "NUMEVENTS": 5,
        "random_state": 42,
    },
)

# Add BLOCK_EO / BLOCK_EC annotations and re-export in-place.
# Recording is 30s; EO occupies 2-12s, EC occupies 16-26s.
EO_ONSET, EO_DUR = 2.0, 10.0
EC_ONSET, EC_DUR = 16.0, 10.0

for vhdr in sorted(RAW_DIR.rglob("*.vhdr")):
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    block_annots = mne.Annotations(
        onset=[EO_ONSET, EC_ONSET],
        duration=[EO_DUR, EC_DUR],
        description=["BLOCK_EO", "BLOCK_EC"],
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + block_annots)
    mne.export.export_raw(str(vhdr), raw, fmt="brainvision", overwrite=True, verbose="ERROR")

print(f"Synthetic dataset generated at: {RAW_DIR}")
print("Files:")
for f in sorted(RAW_DIR.rglob("*.vhdr")):
    print(f"  {f.relative_to(RAW_DIR)}")

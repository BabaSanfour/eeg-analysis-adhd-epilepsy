"""Generate a synthetic EEG dataset for testing the neurodags pipelines.

Produces BrainVision (.vhdr/.vmrk/.eeg) files under datasets/synthetic_eeg/rawdata/
using neurodags's built-in 1/f noise generator. Each recording gets multiple randomly
ordered BLOCK_* annotations mimicking a real paradigm:

  4 blocks total per recording (2 EC + 2 EO), each 6 s, 1 s gap between blocks.
  Block ordering is shuffled per recording (seeded by recording index).
  Example layout: BLOCK_EC [1-7s], BLOCK_EO [8-14s], BLOCK_EC [15-21s], BLOCK_EO [22-28s]

  Each 6 s block yields:
    - 6 × 1 s epochs for AutoReject (>= min_epochs=5)
    - 3 × 2 s epochs for feature extraction

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

# Add randomly ordered BLOCK_EO / BLOCK_EC annotations and re-export in-place.
# 2 EC + 2 EO blocks per recording, each 6 s, 1 s gap, starting at t=1 s.
# Ordering shuffled per recording (seed = recording index).
BLOCK_DURATION = 6.0
INTER_BLOCK_GAP = 1.0
BLOCKS_PER_COND = 2
START_OFFSET = 1.0

_base_conditions = ["EC"] * BLOCKS_PER_COND + ["EO"] * BLOCKS_PER_COND

for i, vhdr in enumerate(sorted(RAW_DIR.rglob("*.vhdr"))):
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")

    rng = np.random.default_rng(seed=i)
    block_order = _base_conditions.copy()
    rng.shuffle(block_order)

    onsets, durations, descriptions = [], [], []
    t = START_OFFSET
    for cond in block_order:
        onsets.append(t)
        durations.append(BLOCK_DURATION)
        descriptions.append(f"BLOCK_{cond}")
        t += BLOCK_DURATION + INTER_BLOCK_GAP

    block_annots = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + block_annots)
    mne.export.export_raw(str(vhdr), raw, fmt="brainvision", overwrite=True, verbose="ERROR")

print(f"Synthetic dataset generated at: {RAW_DIR}")
print("Files:")
for f in sorted(RAW_DIR.rglob("*.vhdr")):
    print(f"  {f.relative_to(RAW_DIR)}")

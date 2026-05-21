"""Generate a synthetic EEG dataset for testing the neurodags pipelines.

Produces BrainVision (.vhdr/.vmrk/.eeg) files under datasets/synthetic_eeg/rawdata/
using neurodags's built-in 1/f noise generator.

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

print(f"Synthetic dataset generated at: {RAW_DIR}")
print("Files:")
for f in sorted(RAW_DIR.rglob("*.vhdr")):
    print(f"  {f.relative_to(RAW_DIR)}")

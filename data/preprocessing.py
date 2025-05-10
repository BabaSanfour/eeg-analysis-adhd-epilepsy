#!/usr/bin/env python3
"""
Process raw EEG data for all subjects and save cleaned derivatives in BIDS format,
keeping only the standard 10–20 channels and limiting each recording to the first
20 minutes.
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import mne
from mne_bids import BIDSPath, write_raw_bids
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Standard 10–20 montage and its channel names
DEFAULT_MONTAGE = mne.channels.make_standard_montage("standard_1020")
SENSORS_1020 = set(DEFAULT_MONTAGE.ch_names)

# Maximum duration in seconds (20 minutes)
MAX_DURATION_SEC = 20 * 60


# -----------------------------------------------------------------------------
# Per‑subject processing
# -----------------------------------------------------------------------------
def process_one_subject(
    subject_id: str,
    bids_root: Path,
    deriv_root: Path,
    overwrite: bool
) -> str:
    """
    Load raw EEG from BIDS, keep only 10–20 channels, limit to 20 min,
    clean, and write a 'cleaned' derivative. Returns the subject_id on success.
    Raises on error.
    """
    # Build source & output BIDSPath
    src = BIDSPath(
        root=str(bids_root),
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
    )
    dst = BIDSPath(
        root=str(deriv_root),
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
        processing="cleaned",
    )

    if dst.fpath.exists() and not overwrite:
        logging.info("Skipping %s (derivative exists)", subject_id)
        return subject_id

    raw = mne.io.read_raw_brainvision(src, preload=True)
    total_duration = raw.times[-1]
    if total_duration > MAX_DURATION_SEC:
        raw.crop(tmin=0, tmax=MAX_DURATION_SEC, include_tmax=False)
        logging.info("Cropped %s to first %d seconds", subject_id, MAX_DURATION_SEC)

    raw.pick_types(eeg=True)
    to_drop = [ch for ch in raw.ch_names if ch not in SENSORS_1020]
    if to_drop:
        raw.drop_channels(to_drop)

    raw.set_eeg_reference("average", verbose=False)
    raw.filter(0.5, 99.5, verbose=False)

    raw.set_montage(DEFAULT_MONTAGE)

    write_raw_bids(
        raw,
        bids_path=dst,
        format="BrainVision",
        overwrite=overwrite,
        allow_preload=True,
        verbose=False,
    )

    return subject_id


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Clean and save EEG derivatives (10–20 channels only, 20 min max)"
    )
    parser.add_argument(
        "--bids-root", type=Path, required=True,
        help="Path to BIDS root containing raw data"
    )
    parser.add_argument(
        "--deriv-root", type=Path, required=True,
        help="Path where cleaned derivatives will be saved"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subjects", type=Path,
        help="CSV file with column 'subject_id' listing subjects to process"
    )
    group.add_argument(
        "--n-subjects", type=int,
        help="If no CSV, assume subjects numbered 1…N (zero‑padded to two digits)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing cleaned derivatives"
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel workers (default=1 = sequential)"
    )
    args = parser.parse_args()

    if args.subjects:
        import pandas as pd
        df = pd.read_csv(args.subjects)
        subs: List[str] = df["subject_id"].astype(str).tolist()
    else:
        subs = [f"{i:02d}" for i in range(1, args.n_subjects + 1)]

    args.deriv_root.mkdir(parents=True, exist_ok=True)

    successes, failures = [], []

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as exe:
            futures = {
                exe.submit(
                    process_one_subject,
                    sid,
                    args.bids_root,
                    args.deriv_root,
                    args.overwrite
                ): sid for sid in subs
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Processing", unit="subj"):
                sid = futures[fut]
                try:
                    fut.result()
                    successes.append(sid)
                except Exception as e:
                    logging.error("Subject %s failed: %s", sid, e)
                    failures.append(sid)
    else:
        # Sequential execution
        for sid in tqdm(subs, desc="Processing", unit="subj"):
            try:
                process_one_subject(
                    sid,
                    args.bids_root,
                    args.deriv_root,
                    args.overwrite
                )
                successes.append(sid)
            except Exception as e:
                logging.error("Subject %s failed: %s", sid, e)
                failures.append(sid)

    logging.info("Finished: %d succeeded, %d failed", len(successes), len(failures))
    if failures:
        logging.warning("Failed subjects: %s", failures)


if __name__ == "__main__":
    main()

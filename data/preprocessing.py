#!/usr/bin/env python3
"""
Process raw EEG data for all subjects and save cleaned derivatives in BIDS format,
keeping only the standard 10–20 channels, limiting each recording to the first
20 minutes, and optionally performing notch filtering and z‑score normalization.
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np
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
DEFAULT_MONTAGE = mne.channels.make_standard_montage("standard_1020")
SENSORS_1020 = set(DEFAULT_MONTAGE.ch_names)
MAX_DURATION_SEC = 20 * 60  # 20 minutes


# -----------------------------------------------------------------------------
# Per‑subject processing
# -----------------------------------------------------------------------------
def process_one_subject(
    subject_id: str,
    bids_root: Path,
    deriv_root: Path,
    overwrite: bool,
    notch: bool,
    z_score: bool,
    z_score_axis: int
) -> str:
    """
    Load raw EEG from BIDS, keep only 10–20 channels,
    limit to 20 min, drop out‑of‑range events,
    optionally apply notch filter at line freq (default 60 Hz),
    optionally z-score normalize clipped to ±15 SD,
    then write a 'cleaned' derivative.
    Returns subject_id on success; raises on error.
    """
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

    components = [
        comp for cond, comp in (
            (notch, "notch"),
            (z_score, f"zscore_axis{z_score_axis}")
        ) if cond
    ]
    processing_str = "cleaned_" + ("_".join(components) if components else "raw")
    
    dst = BIDSPath(
        root=str(deriv_root),
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
        processing=processing_str,
    )

    if dst.fpath.exists() and not overwrite:
        logging.info("Skipping %s (derivative exists)", subject_id)
        return subject_id

    raw = mne.io.read_raw_brainvision(src, preload=True)
    total_duration = raw.times[-1]
    if total_duration > MAX_DURATION_SEC:
        raw.crop(tmin=0, tmax=MAX_DURATION_SEC, include_tmax=False)
        logging.info("Cropped %s to first %d seconds", subject_id, MAX_DURATION_SEC)
        if raw.annotations is not None and len(raw.annotations.onset):
            keep = [i for i, on in enumerate(raw.annotations.onset)
                    if on < MAX_DURATION_SEC]
            if len(keep) < len(raw.annotations.onset):
                raw.set_annotations(raw.annotations[keep])
                logging.info("Dropped %d out‑of‑range events for %s",
                            len(raw.annotations.onset) - len(keep), subject_id)

    raw.pick_types(eeg=True)
    to_drop = [ch for ch in raw.ch_names if ch not in SENSORS_1020]
    if to_drop:
        raw.drop_channels(to_drop)

    if notch:
        line_freq = raw.info.get("line_freq")
        freqs = [line_freq] if isinstance(line_freq, (int, float)) and line_freq > 0 else [60]
        raw.notch_filter(freqs=freqs, picks="eeg", verbose=False)
        logging.info("Applied notch filter at %s Hz for %s", freqs, subject_id)

    raw.set_eeg_reference("average", verbose=False)
    raw.filter(0.5, 99.5, verbose=False)

    raw.set_montage(DEFAULT_MONTAGE)

    if z_score:
        data = raw.get_data()  # shape (n_channels, n_times)
        means = data.mean(axis=z_score_axis, keepdims=True)
        stds = data.std(axis=z_score_axis, keepdims=True)
        normed = np.clip((data - means) / stds, -15, 15)
        raw._data = normed
        logging.info("Applied z-score normalization axis=%d for %s", z_score_axis, subject_id)
        
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
        description="Clean and save EEG derivatives (10–20 ch, 20 min max, notch & z‑score optional)"
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
        help="CSV file with column 'subject_id' listing subjects"
    )
    group.add_argument(
        "--n-subjects", type=int,
        help="Assume subjects numbered 1…N (zero‑padded to two digits)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing derivatives"
    )
    parser.add_argument(
        "--notch", action="store_true",
        help="Apply notch filter at line frequency (default 60 Hz)"
    )
    parser.add_argument(
        "--z-score", action="store_true",
        help="Perform z-score normalization clipped to ±15 SD"
    )
    parser.add_argument(
        "--z-score-axis", type=int, choices=[0, 1], default=1,
        help="Axis for z-score: 1=per-channel (default), 0=per-timepoint"
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel workers (default=1 = sequential)"
    )
    args = parser.parse_args()

    # Build list of subject IDs
    if args.subjects:
        import pandas as pd
        subs = pd.read_csv(args.subjects)["subject_id"].astype(str).tolist()
    else:
        subs = [f"{i:02d}" for i in range(1, args.n_subjects + 1)]

    # Ensure output directory exists
    args.deriv_root.mkdir(parents=True, exist_ok=True)

    successes, failures = [], []

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    process_one_subject,
                    sid,
                    args.bids_root,
                    args.deriv_root,
                    args.overwrite,
                    args.notch,
                    args.z_score,
                    args.z_score_axis
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
        for sid in tqdm(subs, desc="Processing", unit="subj"):
            try:
                process_one_subject(
                    sid,
                    args.bids_root,
                    args.deriv_root,
                    args.overwrite,
                    args.notch,
                    args.z_score,
                    args.z_score_axis
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

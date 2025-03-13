#!/usr/bin/env python3
"""
Script to process raw EEG data for all subjects and save the processed 
data as a derivative in BIDS format.
"""

import os
import logging

import mne
from mne_bids import BIDSPath, write_raw_bids

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import derivatives_dir, bids_dir, sensors_to_keep, n_subjects  # noqa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_raw_data(subject_id: str) -> mne.io.BaseRaw:
    """
    Load the raw EEG data for a given subject.
    """
    bids_path = BIDSPath(
        root=bids_dir,
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
    )
    raw = mne.io.read_raw_brainvision(bids_path, preload=True)
    return raw


def process_subject(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Process the raw data by selecting EEG channels, applying the appropriate filter, and setting the montage.
    """
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False)

    # Drop channels that are not in the 1020 montage.
    channels_to_drop = [ch for ch in raw.ch_names if ch not in sensors_to_keep]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    raw.set_eeg_reference("average")
    raw.filter(0.5, 99.5, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw


def save_derivative_raw(raw: mne.io.BaseRaw, subject_id: str) -> None:
    """
    Save the processed raw data as a derivative using BIDS format.
    """
    bids_path = BIDSPath(
        root=derivatives_dir,
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
        processing="cleaned",
    )
    write_raw_bids(
        raw,
        bids_path=bids_path,
        format="BrainVision",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )


def main() -> None:
    """Process raw EEG data for all subjects."""
    for subject in range(1, n_subjects + 1):
        subject_id = str(subject)
        logging.info("Processing subject %s", subject_id)
        try:
            raw = load_raw_data(subject_id)
            raw = process_subject(raw)
            save_derivative_raw(raw, subject_id)
            logging.info("Finished processing subject %s", subject_id)
        except Exception as err:
            logging.error("Error processing subject %s: %s", subject_id, err)


if __name__ == "__main__":
    main()

"""Epoch construction helpers for preprocessed raws."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import mne
import numpy as np
from tqdm import tqdm

from eeg_adhd_epilepsy.io import bids as bids_io

logger = logging.getLogger(__name__)


def build_block_events_by_condition(
    raw: mne.io.BaseRaw,
    segment_duration: float,
    overlap: float = 0.0,
 ) -> dict[str, np.ndarray]:
    """Build fixed-length events grouped by block condition."""
    blocks = [
        block
        for block in bids_io._collect_block_windows(raw)
        if block.duration >= segment_duration
    ]
    events_by_condition: dict[str, np.ndarray] = {}
    for block in blocks:
        block_events = mne.make_fixed_length_events(
            raw,
            id=1,
            start=block.onset,
            stop=block.stop,
            duration=segment_duration,
            overlap=overlap,
            first_samp=True,
        )
        if len(block_events) == 0:
            continue
        if block.name in events_by_condition:
            events_by_condition[block.name] = np.concatenate(
                [events_by_condition[block.name], block_events]
            )
        else:
            events_by_condition[block.name] = block_events

    for condition_name, events in list(events_by_condition.items()):
        events_by_condition[condition_name] = events[events[:, 0].argsort()]
    return events_by_condition


def make_epochs_from_preproc_raw(
    raw: mne.io.BaseRaw,
    segment_duration: float,
    overlap: float = 0.0,
    ignore_annotations: bool = True,
    save_path: Optional[Path] = None,
    overwrite: bool = False,
) -> mne.Epochs:
    """Create fixed-length epochs from all annotated blocks in a preprocessed raw."""
    events_by_condition = build_block_events_by_condition(
        raw,
        segment_duration=segment_duration,
        overlap=overlap,
    )
    if not events_by_condition:
        raise ValueError("No block events could be constructed from the annotated raw.")
    event_id = {
        condition_name: idx
        for idx, condition_name in enumerate(events_by_condition, start=1)
    }
    remapped = []
    for condition_name, condition_events in events_by_condition.items():
        condition_copy = condition_events.copy()
        condition_copy[:, 2] = event_id[condition_name]
        remapped.append(condition_copy)
    events = np.concatenate(remapped)
    events = events[events[:, 0].argsort()]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=segment_duration,
        baseline=None,
        reject=None,
        verbose="ERROR",
        preload=True,
        proj=False,
        reject_by_annotation=not ignore_annotations,
    )
    if save_path is not None:
        epochs.save(save_path, overwrite=overwrite)
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save fixed epochs from all annotated blocks in preprocessed FIF files"
    )
    parser.add_argument("--bids_root", required=True, help="Path to BIDS dataset root")
    parser.add_argument("--desc", default="base", help="Preprocessed raw desc to read")
    parser.add_argument("--segment_duration", type=float, required=True, help="Epoch length in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Epoch overlap in seconds")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process")
    parser.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore BAD_ annotations during epoching",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing epoch files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    bids_root = Path(args.bids_root)
    preproc_root = bids_io.get_preproc_root(bids_root)

    subject_ids = (
        [bids_io.normalize_subject_id(f"{int(subject):04d}") for subject in args.subjects]
        if args.subjects
        else sorted(path.name for path in preproc_root.glob("sub-*") if path.is_dir())
    )

    logger.info(f"Saving epochs for {len(subject_ids)} subjects in {preproc_root}")

    for subject_id in tqdm(subject_ids, desc="Saving Epochs"):
        input_path = bids_io.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=args.desc,
        )
        output_path = input_path.with_name(input_path.name.replace("_eeg.fif", "_epo.fif"))

        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
        epochs = make_epochs_from_preproc_raw(
            raw,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            ignore_annotations=args.ignore_annotations,
            save_path=output_path,
            overwrite=args.overwrite,
        )
        logger.info(f"Saved {len(epochs)} epochs to {output_path}")


if __name__ == "__main__":
    main()

"""Epoch construction helpers for preprocessed raws."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.io.report_paths import default_reports_root
from eeg_adhd_epilepsy.utils import events as events_utils
from eeg_adhd_epilepsy.utils.logs import setup_logging

LOGGER = logging.getLogger(__name__)


def build_block_events_by_condition(
    raw: mne.io.BaseRaw,
    segment_duration: float,
    overlap: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build fixed-length events grouped by block condition."""
    blocks = [
        block
        for block in events_utils.collect_block_windows(raw)
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
        condition_name = (
            block.name.replace("BLOCK_", "") if str(block.name).startswith("BLOCK_") else block.name
        )
        if condition_name in events_by_condition:
            events_by_condition[condition_name] = np.concatenate(
                [events_by_condition[condition_name], block_events]
            )
        else:
            events_by_condition[condition_name] = block_events

    return events_by_condition


def make_epochs_from_preproc_raw(
    raw: mne.io.BaseRaw,
    segment_duration: float,
    overlap: float = 0.0,
    ignore_annotations: bool = True,
    save_path: Path | None = None,
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
        condition_name: idx for idx, condition_name in enumerate(events_by_condition, start=1)
    }
    remapped = []
    for condition_name, condition_events in events_by_condition.items():
        condition_copy = condition_events.copy()
        condition_copy[:, 2] = event_id[condition_name]
        remapped.append(condition_copy)
    events = np.concatenate(remapped)
    events = events[events[:, 0].argsort()]

    epoch_tmax = max(segment_duration - 1.0 / raw.info["sfreq"], 0.0)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_tmax,
        baseline=None,
        reject=None,
        verbose="ERROR",
        preload=True,
        proj=False,
        reject_by_annotation=not ignore_annotations,
        event_repeated="drop",
    )
    if save_path is not None:
        epochs.save(save_path, overwrite=overwrite)
    return epochs


def _write_epoch_provenance(
    output_path: Path,
    epochs: mne.Epochs,
    *,
    source_path: Path,
    desc: str,
    segment_duration: float,
    overlap: float,
    ignore_annotations: bool,
) -> Path:
    """Write a JSON sidecar describing how an epochs file was constructed."""
    condition_counts = {
        name: int((epochs.events[:, 2] == code).sum()) for name, code in epochs.event_id.items()
    }
    provenance = {
        "source_file": source_path.name,
        "desc": desc,
        "segment_duration": segment_duration,
        "overlap": overlap,
        "ignore_annotations": ignore_annotations,
        "n_epochs": len(epochs),
        "condition_counts": condition_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path = output_path.with_name(output_path.name.replace("_epo.fif", "_epo.json"))
    sidecar_path.write_text(json.dumps(provenance, indent=2))
    return sidecar_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save fixed epochs from all annotated blocks in preprocessed FIF files"
    )
    parser.add_argument("--bids_root", required=True, help="Path to BIDS dataset root")
    parser.add_argument("--desc", default="base", help="Preprocessed raw desc to read")
    parser.add_argument(
        "--segment_duration", type=float, required=True, help="Epoch length in seconds"
    )
    parser.add_argument("--overlap", type=float, default=0.0, help="Epoch overlap in seconds")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process")
    parser.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore BAD_ annotations during epoching",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing epoch files")
    parser.add_argument("--reports_root", type=str, default=None, help="Custom root directory for reports (defaults to sibling of bids_root)")
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    preproc_root = bids_io.get_preproc_root(bids_root)
    reports_root = Path(args.reports_root) if args.reports_root else default_reports_root(bids_root)
    log_file = reports_root / "logs" / "epochs.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    LOGGER.info("Discovering preprocessed files in %s", preproc_root)
    found_files = list(preproc_root.rglob(f"*desc-{args.desc}_eeg.fif"))

    if args.subjects:
        valid_sids = set()
        for s in args.subjects:
            if s.startswith("sub-"):
                valid_sids.add(bids_io.study_id_to_bids_subject(s))
            else:
                valid_sids.add(bids_io.study_id_to_bids_subject(int(s)))
        files_to_process = [
            f for f in found_files if bids_io.parse_bids_components(f)["subject"] in valid_sids
        ]
    else:
        files_to_process = found_files

    LOGGER.info("Found %d matching _eeg.fif files to epoch.", len(files_to_process))

    for input_path in tqdm(files_to_process, desc="Saving Epochs"):
        output_path = input_path.with_name(input_path.name.replace("_eeg.fif", "_epo.fif"))

        if not args.overwrite and output_path.exists():
            LOGGER.info(
                "Skipping %s: epoch file already exists. Use --overwrite to overwrite.",
                input_path.name,
            )
            continue

        try:
            raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
            epochs = make_epochs_from_preproc_raw(
                raw,
                segment_duration=args.segment_duration,
                overlap=args.overlap,
                ignore_annotations=args.ignore_annotations,
                save_path=output_path,
                overwrite=args.overwrite,
            )
            _write_epoch_provenance(
                output_path,
                epochs,
                source_path=input_path,
                desc=args.desc,
                segment_duration=args.segment_duration,
                overlap=args.overlap,
                ignore_annotations=args.ignore_annotations,
            )
            LOGGER.info("Saved %d epochs to %s", len(epochs), output_path.name)
        except Exception as exc:
            LOGGER.error("Failed to epoch %s: %s", input_path.name, exc, exc_info=True)


if __name__ == "__main__":
    main()

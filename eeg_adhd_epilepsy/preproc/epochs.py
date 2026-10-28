"""Epoch construction helpers for preprocessed raws."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
from joblib import Parallel, delayed
from mne_bids import get_entities_from_fname
from tqdm import tqdm

from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.io.report_paths import default_reports_root
from eeg_adhd_epilepsy.utils import events as events_utils
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib

LOGGER = logging.getLogger(__name__)
# The epochs left after rejection by annotation are written as <input desc>rejected
# (desc-base -> desc-baserejected), so each variant's name keeps what it was cut from.
REJECTED_SUFFIX = "rejected"


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
    reject_by_annotation: bool = False,
    save_path: Path | None = None,
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
        reject_by_annotation=reject_by_annotation,
        event_repeated="drop",
    )
    if save_path is not None:
        epochs.save(save_path, overwrite=True)
    return epochs


def _write_epoch_provenance(
    output_path: Path,
    epochs: mne.Epochs,
    *,
    source_path: Path,
    segment_duration: float,
    overlap: float,
    reject_by_annotation: bool,
) -> Path:
    """Write a JSON sidecar describing how an epochs file was constructed.

    ``dropped`` counts the dropped epochs by MNE's reason: an epoch shared by two
    overlapping blocks ("DROP DUPLICATE") or, with rejection, the first ``BAD_``
    annotation it overlaps (an epoch AutoReject rejected inside a manual mark is
    counted under the manual mark).
    """
    condition_counts = {
        name: int((epochs.events[:, 2] == code).sum()) for name, code in epochs.event_id.items()
    }
    dropped = Counter(reason for log in epochs.drop_log for reason in log if reason != "IGNORED")
    provenance = {
        "source_file": source_path.name,
        "segment_duration": segment_duration,
        "overlap": overlap,
        "reject_by_annotation": reject_by_annotation,
        "n_epochs": len(epochs),
        "condition_counts": condition_counts,
        "dropped": dict(dropped),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    sidecar_path = output_path.with_name(output_path.name.replace("_epo.fif", "_epo.json"))
    sidecar_path.write_text(json.dumps(provenance, indent=2))
    return sidecar_path


def _epoch_recording(
    input_path: Path,
    output_path: Path,
    segment_duration: float,
    overlap: float,
    reject_by_annotation: bool,
) -> str | None:
    """Epoch one recording; returns the error, or None.

    On an error the recording's outputs are removed, so no earlier run's epochs are left.
    """
    try:
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
        epochs = make_epochs_from_preproc_raw(
            raw,
            segment_duration=segment_duration,
            overlap=overlap,
            reject_by_annotation=reject_by_annotation,
            save_path=output_path,
        )
        _write_epoch_provenance(
            output_path,
            epochs,
            source_path=input_path,
            segment_duration=segment_duration,
            overlap=overlap,
            reject_by_annotation=reject_by_annotation,
        )
    except Exception as exc:  # noqa: BLE001 - reported per recording, the job fails at the end
        output_path.unlink(missing_ok=True)
        output_path.with_name(output_path.name.replace("_epo.fif", "_epo.json")).unlink(
            missing_ok=True
        )
        return f"{type(exc).__name__}: {exc}"
    return None


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
        "--reject_by_annotation",
        action="store_true",
        help=(
            "Drop the epochs overlapping a BAD_ annotation and write them as "
            f"desc-<desc>{REJECTED_SUFFIX}; without it every epoch is written under --desc."
        ),
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Recordings epoched in parallel")
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root)",
    )
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    preproc_root = bids_io.get_derivative_root(bids_root, bids_io.DerivativeStage.PREPROC)
    reports_root = Path(args.reports_root) if args.reports_root else default_reports_root(bids_root)
    setup_logging(reports_root / "logs" / "epochs.log", "INFO")

    bids_io.write_preproc_dataset_description(preproc_root)
    files = list(preproc_root.rglob(f"*desc-{args.desc}_eeg.fif"))
    if args.subjects:
        subjects = {bids_io.bids_subject(s) for s in args.subjects}
        files = [f for f in files if get_entities_from_fname(f)["subject"] in subjects]
    if not files:
        LOGGER.error("No desc-%s recordings under %s.", args.desc, preproc_root)
        sys.exit(1)
    # Largest first: the long recordings start early and the short ones fill the tail.
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    out_desc = f"{args.desc}{REJECTED_SUFFIX}" if args.reject_by_annotation else args.desc
    LOGGER.info(
        "Epoching %d recordings into desc-%s on %d workers.", len(files), out_desc, args.n_jobs
    )

    # Every output is rewritten: an epoch file left from an earlier cleaning is stale.
    outputs = [
        f.with_name(f.name.replace(f"desc-{args.desc}_eeg.fif", f"desc-{out_desc}_epo.fif"))
        for f in files
    ]
    with tqdm_joblib(tqdm(total=len(files), desc="Saving epochs")):
        errors = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_epoch_recording)(
                f, out, args.segment_duration, args.overlap, args.reject_by_annotation
            )
            for f, out in zip(files, outputs, strict=True)
        )

    failed = {f.name: error for f, error in zip(files, errors, strict=True) if error}
    LOGGER.info("Epoched %d recordings; %d failed.", len(files) - len(failed), len(failed))
    for name, error in failed.items():
        LOGGER.error("Failed to epoch %s: %s", name, error)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

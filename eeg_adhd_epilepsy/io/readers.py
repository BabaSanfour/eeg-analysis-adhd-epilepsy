"""Low-level readers for BIDS and preprocessed EEG data."""

import logging
from pathlib import Path
from typing import Any

import mne
from coco_pipe.io import read_json
from mne_bids import BIDSPath, read_raw_bids

from eeg_adhd_epilepsy.io.bids import (
    get_stage_output_path,
    parse_bids_components,
)

logger = logging.getLogger(__name__)


def read_bids_raw(
    bids_root: Path,
    subject: str,
    task: str = "clinical",
    session: str | None = None,
    run: str | None = None,
) -> mne.io.BaseRaw:
    """Load a raw BIDS eeg file for a subject.

    Parameters
    ----------
    bids_root
        Path to the root of the BIDS dataset.
    subject
        BIDS subject label (without 'sub-').
    task
        BIDS task label.
    session
        Optional BIDS session label.
    run
        Optional BIDS run label.

    Returns
    -------
    raw : mne.io.BaseRaw
        The MNE Raw object containing the continuous BIDS data.
    """
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        task=task,
        session=session,
        run=run,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    logger.info(f"Loading raw BIDS file: {bids_path.fpath}")
    return read_raw_bids(bids_path, verbose="ERROR")


def read_preproc_stage(
    study_id: str,
    preproc_root: Path,
    desc: str = "base",
    task: str = "clinical",
    session: str | None = None,
    run: str | None = None,
) -> tuple[mne.io.BaseRaw | None, dict[str, Any], list[str]]:
    """Load continuous preprocessed EEG data and metadata for a specific stage.

    Parameters
    ----------
    study_id
        The identifier for the study/subject.
    preproc_root
        Path to the derivatives/preproc folder.
    desc
        The processing stage descriptor (e.g., 'base', 'denoise').
    task
        The BIDS task name.

    Returns
    -------
    raw : mne.io.BaseRaw | None
        The loaded MNE Raw object, or None if the file was not found/invalid.
    provenance : dict[str, Any]
        The contents of the associated _provenance.json sidecar.
    issues : list[str]
        A list of any warnings or errors encountered during loading.
    """
    issues = []
    eeg_path = get_stage_output_path(
        subject=study_id,
        preproc_root=preproc_root,
        desc=desc,
        task=task,
        session=session,
        run=run,
    )
    if not eeg_path.exists():
        parsed = parse_bids_components(eeg_path)
        issues.append(f"missing_eeg:{parsed.get('subject', study_id)}")
        return None, {}, issues

    try:
        raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose="ERROR")
    except Exception as exc:
        parsed = parse_bids_components(eeg_path)
        issues.append(f"invalid_fif:{parsed.get('subject', study_id)}:{exc}")
        return None, {}, issues

    prov_path = eeg_path.with_name(eeg_path.name.replace("_eeg.fif", "_provenance.json"))
    prov = {}
    if prov_path.exists():
        try:
            prov = read_json(prov_path)
        except Exception as exc:
            parsed = parse_bids_components(prov_path)
            issues.append(f"invalid_prov:{parsed.get('subject', study_id)}:{exc}")

    return raw, prov, issues

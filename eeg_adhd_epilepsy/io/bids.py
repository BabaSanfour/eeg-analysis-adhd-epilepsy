"""BIDS I/O utilities for EEG analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids

def discover_bids_files(
    bids_root: Path,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    acquisition: str | None = None,
    processing: str | None = None,
    suffix: str = "eeg",
    extension: str = ".vhdr",
    subjects_filter: set[str] | None = None,
) -> List[Path]:
    """Use BIDSPath matching to find EEG files under a BIDS root."""
    template = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        processing=processing,
        datatype="eeg",
        suffix=suffix,
        extension=extension,
    )
    matches = template.match()
    files: List[Path] = []
    for match in matches:
        subj = match.subject or ""
        subj_tag = f"sub-{subj}" if subj else ""
        if subjects_filter:
            if subj_tag not in subjects_filter and subj not in subjects_filter:
                continue
        if match.fpath is not None and match.fpath.exists():
            files.append(match.fpath)
    return sorted(files)


def read_subjects_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def parse_bids_components(filepath: Path) -> dict[str, str]:
    """
    Extract BIDS entities (subject, session, task) from filename.
    Returns dict like {"subject": "01", "session": "01", ...}
    """
    entities = {}
    
    # Standard BIDS regex for entities
    # sub-<label>[_ses-<label>][_task-<label>]...
    parts = filepath.stem.split("_")
    for part in parts:
        if "-" in part:
            key, val = part.split("-", 1)
            entities[key] = val
            
    # Fallback/Normalization
    if "sub" not in entities:
        # Try finding anywhere in string if not strictly underscore separated
        match = re.search(r"sub-([A-Za-z0-9]+)", filepath.name)
        if match:
            entities["sub"] = match.group(1)
            
    # Session
    if "ses" not in entities:
         match = re.search(r"ses-([A-Za-z0-9]+)", filepath.name)
         if match:
             entities["ses"] = match.group(1)

    # Normalize keys to full names if preferred, but BIDS standard uses short keys
    # Let's return mapped keys for clarity
    final = {}
    if "sub" in entities:
        final["subject"] = entities["sub"]
    if "ses" in entities:
        final["session"] = entities["ses"]
    if "task" in entities:
        final["task"] = entities["task"]
        
    return final


def parse_subject_id(filepath: Path) -> str:
    """Return subject ID string (e.g. 'sub-01')."""
    comps = parse_bids_components(filepath)
    if "subject" in comps:
        return f"sub-{comps['subject']}"
    # Fallback
    return filepath.stem


def load_bids_raw(
    filepath: Path,
    bids_root: Path,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    acquisition: str | None = None,
    processing: str | None = None,
) -> mne.io.BaseRaw:
    """Load a raw file using BIDS structure."""
    
    # Auto-infer entities if not provided
    comps = parse_bids_components(filepath)
    if not session:
        session = comps.get("session")
    if not task:
        task = comps.get("task")
    
    # If parse_bids_components is limited, we might need a quick check for run/acq etc.
    if not run and "run" not in comps:
        match = re.search(r"run-([A-Za-z0-9]+)", filepath.name)
        if match: run = match.group(1)
    
    if not acquisition and "acq" not in comps:
        match = re.search(r"acq-([A-Za-z0-9]+)", filepath.name)
        if match: acquisition = match.group(1)
        
    if not processing and "proc" not in comps:
        match = re.search(r"proc-([A-Za-z0-9]+)", filepath.name)
        if match: processing = match.group(1)
        
    subject_clean = parse_subject_id(filepath).replace("sub-", "")
    
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject_clean,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        processing=processing,
        datatype="eeg",
        suffix="eeg",
        extension=filepath.suffix,
    )
    return read_raw_bids(bids_path, verbose="ERROR")


def load_meas_datetimes(bids_root: Path) -> pd.Series:
    """Return measurement datetimes from participants.tsv if present."""
    tsv_path = bids_root / "participants.tsv"
    if not tsv_path.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_csv(tsv_path, sep="\t")
    if "meas" not in df:
        return pd.Series(dtype="datetime64[ns]")
    meas_series = pd.to_datetime(df["meas"], errors="coerce", utc=True).dropna()
    if meas_series.empty:
        return pd.Series(dtype="datetime64[ns]")
    try:
        meas_series = meas_series.dt.tz_convert(None)
    except TypeError:
        meas_series = meas_series.dt.tz_localize(None)
    return meas_series


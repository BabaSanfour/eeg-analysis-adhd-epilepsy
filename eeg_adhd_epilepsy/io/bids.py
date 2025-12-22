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


def parse_subject_id(filepath: Path) -> str:
    match = re.search(r"sub-([A-Za-z0-9]+)", filepath.name)
    if match:
        return f"sub-{match.group(1)}"
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


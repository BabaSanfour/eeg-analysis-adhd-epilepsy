"""BIDS I/O utilities for EEG analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids


_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")
_STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_bids_token(value: str, field_name: str) -> str:
    """Normalize any label into a BIDS-safe alphanumeric token."""
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value).strip())
    if not token:
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return token


def normalize_subject_id(subject_id: str) -> str:
    """Normalize subject labels to 'sub-XXX' format."""
    token = str(subject_id).strip()
    if token.startswith("sub-"):
        token = token[4:]
    if not token or not _ALNUM_RE.fullmatch(token):
        raise ValueError(f"Invalid subject_id: {subject_id!r}")
    return f"sub-{token}"


def validate_stage_desc(desc: str, allowed: set[str] | None = None) -> str:
    """Validate/normalize BIDS desc values used in output filenames."""
    desc_token = _normalize_bids_token(desc, "desc")
    if allowed is not None and desc_token not in allowed:
        allowed_vals = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid desc {desc_token!r}. Expected one of: {allowed_vals}"
        )
    return desc_token


def normalize_stage_name(stage: str) -> str:
    """Normalize report stage folder names (e.g., base/correct/denoise/compare)."""
    stage_name = str(stage).strip().lower()
    if not stage_name or not _STAGE_NAME_RE.fullmatch(stage_name):
        raise ValueError(f"Invalid stage name: {stage!r}")
    return stage_name


def get_preproc_root(bids_root: Path, preproc_root: Path | None = None) -> Path:
    """Return derivatives/preproc root (or caller-provided override)."""
    if preproc_root is not None:
        return Path(preproc_root).expanduser()
    return Path(bids_root).expanduser() / "derivatives" / "preproc"


def get_subject_eeg_dir(
    preproc_root: Path, subject_id: str, create: bool = False
) -> Path:
    """Return '<preproc_root>/<subject>/eeg'."""
    sid = normalize_subject_id(subject_id)
    eeg_dir = Path(preproc_root).expanduser() / sid / "eeg"
    if create:
        eeg_dir.mkdir(parents=True, exist_ok=True)
    return eeg_dir


def get_stage_output_path(
    subject_id: str,
    preproc_root: Path,
    desc: str,
    task: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return stage output FIF path using unified naming."""
    sid = normalize_subject_id(subject_id)
    desc_token = validate_stage_desc(desc)
    eeg_dir = get_subject_eeg_dir(preproc_root, sid, create=create_dir)
    if task:
        task_token = _normalize_bids_token(task, "task")
        fname = f"{sid}_task-{task_token}_desc-{desc_token}_eeg.fif"
    else:
        fname = f"{sid}_desc-{desc_token}_eeg.fif"
    return eeg_dir / fname


def get_stage_provenance_path(
    subject_id: str,
    preproc_root: Path,
    desc: str,
    task: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return stage provenance JSON path using unified naming."""
    out_path = get_stage_output_path(
        subject_id=subject_id,
        preproc_root=preproc_root,
        desc=desc,
        task=task,
        create_dir=create_dir,
    )
    return out_path.with_name(out_path.name.replace("_eeg.fif", "_provenance.json"))


def get_reports_root(
    bids_root: Path | None = None,
    reports_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Return unified reports root (outside BIDS by default)."""
    if reports_root is not None:
        return Path(reports_root).expanduser()
    if bids_root is not None:
        # User preference: same level as BIDS
        return Path(bids_root).expanduser().parent / "reports"
    base = Path(project_root).expanduser() if project_root is not None else Path.cwd()
    return base / "results" / "reports" / "preproc"


def get_subject_report_path(
    reports_root: Path, stage: str, subject_id: str, create_dir: bool = False
) -> Path:
    """Return unified subject report path for a given stage."""
    stage_name = normalize_stage_name(stage)
    sid = normalize_subject_id(subject_id)
    subject_dir = Path(reports_root).expanduser() / stage_name / "subjects" / sid
    if create_dir:
        subject_dir.mkdir(parents=True, exist_ok=True)
    return subject_dir / f"{sid}_{stage_name}_report.html"


def get_stage_summary_report_path(
    reports_root: Path, stage: str, create_dir: bool = False
) -> Path:
    """Return unified dataset summary report path for a given stage."""
    stage_name = normalize_stage_name(stage)
    summary_dir = Path(reports_root).expanduser() / stage_name / "summary"
    if create_dir:
        summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{stage_name}_dataset_summary.html"


def get_compare_summary_paths(
    reports_root: Path, create_dir: bool = False
) -> Dict[str, Path]:
    """Return canonical compare summary artifact paths."""
    summary_dir = Path(reports_root).expanduser() / "compare" / "summary"
    if create_dir:
        summary_dir.mkdir(parents=True, exist_ok=True)
    return {
        "report_html": summary_dir / "compare_dataset_summary.html",
        "metrics_csv": summary_dir / "compare_metrics.csv",
        "run_metadata_json": summary_dir / "compare_run_metadata.json",
    }


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
    preload: bool = True,
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
    raw = read_raw_bids(bids_path, verbose="ERROR")
    if preload:
        raw.load_data()
    return raw


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

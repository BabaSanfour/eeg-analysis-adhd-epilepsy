"""BIDS I/O utilities for EEG analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from coco_pipe.io import BIDSConfig, DataContainer, load_data, read_json, read_table
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

from eeg_adhd_epilepsy.utils import constants

logger = logging.getLogger(__name__)


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


def normalize_session_id(session_id: str | None) -> str:
    """Normalize session labels to 'ses-XXX' format."""
    token = str(session_id or "").strip()
    if not token:
        return ""
    if token.startswith("ses-"):
        token = token[4:]
    if not token or not _ALNUM_RE.fullmatch(token):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    return f"ses-{token}"


def validate_stage_desc(desc: str, allowed: set[str] | None = None) -> str:
    """Validate/normalize BIDS desc values used in output filenames."""
    desc_token = _normalize_bids_token(desc, "desc")
    if allowed is not None and desc_token not in allowed:
        allowed_vals = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid desc {desc_token!r}. Expected one of: {allowed_vals}")
    return desc_token


def normalize_stage_name(stage: str) -> str:
    """Normalize report stage folder names (e.g., base/correct/denoise/compare)."""
    stage_name = str(stage).strip().lower()
    if not stage_name or not _STAGE_NAME_RE.fullmatch(stage_name):
        raise ValueError(f"Invalid stage name: {stage!r}")
    return stage_name


def get_preproc_root(bids_root: Path) -> Path:
    """Return the canonical derivatives/preproc root for a BIDS dataset."""
    return Path(bids_root).expanduser() / "derivatives" / "preproc"


def get_subject_eeg_dir(
    preproc_root: Path, subject_id: str, session: str | None = None, create: bool = False
) -> Path:
    """Return '<preproc_root>/<subject>[/ses-<session>]/eeg'."""
    sid = normalize_subject_id(subject_id)
    parts = [sid]
    if session:
        parts.append(normalize_session_id(session))
    eeg_dir = Path(preproc_root).expanduser().joinpath(*parts) / "eeg"
    if create:
        eeg_dir.mkdir(parents=True, exist_ok=True)
    return eeg_dir


def get_stage_output_path(
    subject_id: str,
    preproc_root: Path,
    desc: str,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return stage output FIF path using unified naming."""
    sid = normalize_subject_id(subject_id)
    desc_token = validate_stage_desc(desc)
    eeg_dir = get_subject_eeg_dir(preproc_root, sid, session=session, create=create_dir)
    parts = [sid]
    if session:
        parts.append(normalize_session_id(session))
    if task:
        task_token = _normalize_bids_token(task, "task")
        parts.append(f"task-{task_token}")
    if run:
        run_token = _normalize_bids_token(run, "run")
        parts.append(f"run-{run_token}")
    parts.append(f"desc-{desc_token}")
    parts.append("eeg")
    fname = "_".join(parts) + ".fif"
    return eeg_dir / fname


def get_stage_provenance_path(
    subject_id: str,
    preproc_root: Path,
    desc: str,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return stage provenance JSON path using unified naming."""
    out_path = get_stage_output_path(
        subject_id=subject_id,
        preproc_root=preproc_root,
        desc=desc,
        session=session,
        task=task,
        run=run,
        create_dir=create_dir,
    )
    return out_path.with_name(out_path.name.replace("_eeg.fif", "_provenance.json"))


def get_reports_root(bids_root: Path) -> Path:
    """Return the canonical shared reports root (sibling to BIDS)."""
    return Path(bids_root).expanduser().parent / "reports"


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


def get_subject_session_stage_dir(
    reports_root: Path,
    subject_id: str,
    session_id: str | None,
    stage: str,
    create_dir: bool = False,
) -> Path:
    """Return a BIDS-like subject/session report directory for a stage."""
    sid = normalize_subject_id(subject_id)
    ses = normalize_session_id(session_id)
    stage_name = normalize_stage_name(stage)
    stage_dir = Path(reports_root).expanduser() / sid
    if ses:
        stage_dir = stage_dir / ses
    stage_dir = stage_dir / stage_name
    if create_dir:
        stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def get_subject_session_stage_report_path(
    reports_root: Path,
    subject_id: str,
    session_id: str | None,
    stage: str,
    report_stem: str,
    create_dir: bool = False,
) -> Path:
    """Return a subject/session report file path for a stage under the shared reports root."""
    stage_dir = get_subject_session_stage_dir(
        reports_root=reports_root,
        subject_id=subject_id,
        session_id=session_id,
        stage=stage,
        create_dir=create_dir,
    )
    return stage_dir / f"{report_stem}_{normalize_stage_name(stage)}_report.html"


def get_stage_summary_dir(
    reports_root: Path,
    stage: str,
    create_dir: bool = False,
) -> Path:
    """Return summary artifact directory for a stage under a shared reports root."""
    stage_name = normalize_stage_name(stage)
    summary_dir = Path(reports_root).expanduser() / "summary" / stage_name
    if create_dir:
        summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def get_stage_summary_report_path(reports_root: Path, stage: str, create_dir: bool = False) -> Path:
    """Return unified dataset summary report path for a given stage."""
    stage_name = normalize_stage_name(stage)
    summary_dir = Path(reports_root).expanduser() / stage_name / "summary"
    if create_dir:
        summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{stage_name}_dataset_summary.html"


def get_compare_summary_paths(reports_root: Path, create_dir: bool = False) -> dict[str, Path]:
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
    desc: str | None = None,
    suffix: str = "eeg",
    extension: str = ".vhdr",
    subjects_filter: set[str] | None = None,
) -> list[Path]:
    """Use BIDSPath matching to find EEG files under a BIDS root."""
    template = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acquisition,
        processing=processing,
        description=desc,
        datatype="eeg",
        suffix=suffix,
        extension=extension,
    )
    matches = template.match()
    files: list[Path] = []
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
    """Extract common BIDS entities from a BIDS-style filepath."""
    entities: dict[str, str] = {}
    for part in filepath.stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        if key and value:
            entities[key] = value

    for short_key in ("sub", "ses", "task", "run", "acq", "proc"):
        if short_key in entities:
            continue
        match = re.search(rf"{short_key}-([A-Za-z0-9]+)", filepath.name)
        if match:
            entities[short_key] = match.group(1)

    final: dict[str, str] = {}
    key_map = {
        "sub": "subject",
        "ses": "session",
        "task": "task",
        "run": "run",
        "acq": "acquisition",
        "proc": "processing",
    }
    for short_key, full_key in key_map.items():
        if short_key in entities:
            final[full_key] = entities[short_key]
    return final


def build_bids_report_ids(
    filepath: Path,
) -> dict[str, str | tuple[str | None, str | None] | tuple[str | None, str | None, str | None]]:
    """Build shared run-aware identifiers for reports and aggregation."""
    comps = parse_bids_components(filepath)
    subject = comps.get("subject", "unknown")
    session = comps.get("session")
    run = comps.get("run")
    subject_id = f"sub-{subject}"
    subject_session_prefix = subject_id if not session else f"{subject_id}_ses-{session}"
    run_prefix = subject_session_prefix if not run else f"{subject_session_prefix}_run-{run}"
    return {
        "subject_id": subject_id,
        "session_id": session or "",
        "run_id": run or "",
        "subject_session_prefix": subject_session_prefix,
        "run_prefix": run_prefix,
        "subject_session_key": (subject_id, session or ""),
        "run_key": (subject_id, session or "", run or ""),
    }


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""
    cleaned = sorted((start, stop) for start, stop in intervals if stop > start)
    if not cleaned:
        return []
    merged: list[tuple[float, float]] = [cleaned[0]]
    for start, stop in cleaned[1:]:
        cur_start, cur_stop = merged[-1]
        if start <= cur_stop:
            merged[-1] = (cur_start, max(cur_stop, stop))
        else:
            merged.append((start, stop))
    return merged


@dataclass
class BlockWindow:
    """Represents a continuous block of time defined by an annotation."""

    onset: float
    duration: float
    description: str

    @property
    def stop(self) -> float:
        return self.onset + self.duration

    @property
    def name(self) -> str:
        if self.description.startswith("BLOCK_"):
            return self.description[6:]
        return self.description

    @property
    def family(self) -> str:
        return parse_block_segment_type(self.name)[0]

    @property
    def eye_state(self) -> str:
        return parse_block_segment_type(self.name)[1]


def parse_block_segment_type(segment_type: str) -> tuple[str, str]:
    segment_type = str(segment_type or "")
    if segment_type == "RAW_baseline":
        return "raw_baseline", "unknown"
    if segment_type == "EO_baseline":
        return "baseline", "eo"
    if segment_type == "EC_baseline":
        return "baseline", "ec"
    if segment_type.startswith("HV_"):
        return "hv", segment_type.split("_", 1)[1].lower()
    if segment_type.startswith("PostHV_"):
        return "post_hv", segment_type.split("_", 1)[1].lower()
    if segment_type.startswith("PHOTO_"):
        return "photo", segment_type.split("_", 1)[1].lower()
    return "unknown", "unknown"


def _resolve_segments_csv(raw: mne.io.BaseRaw, segments_file: str | None) -> Path | None:
    """Resolve the path to the segments CSV file."""
    if segments_file:
        segments_path = Path(segments_file).expanduser()
        if segments_path.exists():
            return segments_path
        if raw.filenames and raw.filenames[0]:
            candidate = Path(raw.filenames[0]).parent / segments_path
            if candidate.exists():
                return candidate
        return segments_path

    if not raw.filenames or not raw.filenames[0]:
        return None

    raw_path = Path(raw.filenames[0])
    stem = raw_path.stem
    for suffix in ("_eeg", "_meg", "_ieeg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return raw_path.parent / f"{stem}_segments.csv"


def _collect_block_windows(raw: mne.io.BaseRaw) -> list[BlockWindow]:
    """Parse annotations to collect all BLOCK_* segments."""
    if raw.n_times == 0:
        return []

    max_t = float(raw.times[-1])
    windows: list[BlockWindow] = []
    for annot in raw.annotations:
        desc = str(annot["description"])
        if not desc.startswith("BLOCK_"):
            continue

        onset = float(annot["onset"])
        duration = float(annot["duration"])
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            continue

        start = max(0.0, onset)
        stop = min(max_t, onset + duration)
        if stop <= start:
            continue

        windows.append(BlockWindow(onset=start, duration=stop - start, description=desc))

    windows.sort(key=lambda block: block.onset)
    return windows


def collect_baseline_windows(raw: mne.io.BaseRaw) -> list[tuple[float, float]]:
    """Return block windows whose segment name contains 'baseline'."""
    windows: list[tuple[float, float]] = []
    for block in _collect_block_windows(raw):
        if "baseline" in block.name.lower():
            windows.append((block.onset, block.stop))
    return windows


def segments_from_block_annotations(raw: mne.io.BaseRaw) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for block in _collect_block_windows(raw):
        family, eye_state = parse_block_segment_type(block.name)
        records.append(
            {
                "segment_type": block.name,
                "block_family": family,
                "eye_state": eye_state,
                "t_start": block.onset,
                "t_stop": block.stop,
                "duration": block.duration,
                "freq_hz": np.nan,
            }
        )
    return pd.DataFrame.from_records(records, columns=constants.SEGMENT_COLUMNS)


def load_segments_for_raw(raw: mne.io.BaseRaw, segments_file: str | None = None) -> pd.DataFrame:
    csv_path = _resolve_segments_csv(raw, segments_file)
    if csv_path is not None and csv_path.exists():
        df = read_table(csv_path, sep=None)
    else:
        df = segments_from_block_annotations(raw)

    if "block_family" not in df.columns or "eye_state" not in df.columns:
        parsed = (
            df.get("segment_type", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .map(parse_block_segment_type)
        )
        df["block_family"] = [item[0] for item in parsed]
        df["eye_state"] = [item[1] for item in parsed]

    for column in constants.SEGMENT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[constants.SEGMENT_COLUMNS].copy()


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
    if not run:
        run = comps.get("run")
    if not acquisition:
        acquisition = comps.get("acquisition") or comps.get("acq")
    if not processing:
        processing = comps.get("processing") or comps.get("proc")

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


def load_stage_artifacts(
    subject_id: str,
    preproc_root: Path,
    desc: str,
    task: str | None = None,
) -> tuple[mne.io.BaseRaw | None, dict[str, Any], list[str]]:
    """Load a preprocessed stage FIF and its provenance JSON for one subject.

    This is the read-side complement to the per-stage save logic in
    ``run_base_record`` / ``run_correction_pipeline`` / ``run_denoising_pipeline``.

    Parameters
    ----------
    subject_id:
        Normalised subject ID (e.g. ``"sub-0001"``).
    preproc_root:
        Root directory that holds per-subject stage outputs.
    desc:
        BIDS ``desc-`` entity identifying the stage (e.g. ``"base"``,
        ``"correct"``, ``"denoise"``).
    task:
        Optional BIDS ``task-`` entity to disambiguate when a subject has
        multiple task-specific outputs.

    Returns
    -------
    Tuple of ``(raw_obj, provenance_dict, issues_list)``.
    ``raw_obj`` is ``None`` and ``issues_list`` is non-empty when loading fails.
    """
    issues: list[str] = []
    out_path = get_stage_output_path(
        subject_id=subject_id, preproc_root=preproc_root, desc=desc, task=task
    )
    prov_path = get_stage_provenance_path(
        subject_id=subject_id, preproc_root=preproc_root, desc=desc, task=task
    )

    raw_obj: mne.io.BaseRaw | None = None
    prov: dict[str, Any] = {}

    if not out_path.exists():
        issues.append(f"missing_raw:{out_path}")
    else:
        try:
            raw_obj = mne.io.read_raw_fif(out_path, preload=True, verbose="ERROR")
        except Exception as exc:
            issues.append(f"bad_raw:{out_path}:{exc}")

    if not prov_path.exists():
        issues.append(f"missing_provenance:{prov_path}")
    else:
        try:
            prov = read_json(prov_path)
        except Exception as exc:
            issues.append(f"bad_provenance:{prov_path}:{exc}")

    return raw_obj, prov, issues


def validate_bids_coverage(
    df: pd.DataFrame | None,
    root: Path,
    desc: str | None = None,
    suffix: str | None = None,
    subject_col: str = "study_id",
) -> dict[str, object]:
    """Compare metadata subject ids against subjects present in a BIDS tree."""
    root = Path(root)

    ignore_descriptions = None
    if desc is not None:
        available_descs = set(get_entity_vals(root, "description"))
        ignore_descriptions = sorted(item for item in available_descs if item != desc)

    ignore_suffixes = None
    if suffix == "epo":
        ignore_suffixes = ["eeg"]
    elif suffix == "eeg":
        ignore_suffixes = ["epo"]

    present = sorted(
        set(
            get_entity_vals(
                root,
                "subject",
                ignore_descriptions=ignore_descriptions,
                ignore_suffixes=ignore_suffixes,
            )
        )
    )

    results: dict[str, object] = {
        "present_subjects": present,
        "present_count": len(present),
    }

    logger.info("Found %s subjects in %s", len(present), root)

    if df is None or subject_col not in df.columns:
        return results

    expected = []
    for subject in df[subject_col].dropna().unique():
        subject_num = pd.to_numeric(subject, errors="coerce")
        if pd.isna(subject_num):
            continue
        expected.append(f"{int(subject_num):04d}")

    present_set = set(present)
    missing = [subject for subject in expected if subject not in present_set]

    results["expected_subjects"] = expected
    results["expected_count"] = len(expected)
    results["missing_subjects"] = missing
    results["missing_study_ids"] = [int(subject) for subject in missing]
    results["missing_count"] = len(missing)

    return results


def load_eeg_data(
    bids_root: Path,
    use_derivatives: bool = False,
    subjects: list[str] | None = None,
    task: str = "clinical",
    session: str | list[str] | None = None,
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    metadata_df: pd.DataFrame | None = None,
    subject_col: str = "study_id",
    target_col: str | None = None,
    desc: str = "base",
    condition: str | None = None,
    window_source: str = "auto",
) -> DataContainer:
    """Load raw BIDS data or saved epoch derivatives into a DataContainer.

    ``window_source='re_epoch'`` re-epochs the cleaned continuous ``desc``
    derivative at ``segment_duration`` (see
    :func:`load_cleaned_continuous_container`); otherwise the loader returns the
    saved epoch derivative (``use_derivatives=True``) or raw-acquisition epochs
    (``use_derivatives=False``).
    """
    subjects = subjects or []
    if window_source == "re_epoch":
        return load_cleaned_continuous_container(
            bids_root=bids_root,
            subjects=list(subjects),
            task=task,
            segment_duration=segment_duration,
            overlap=overlap,
            condition=condition,
            metadata_df=metadata_df,
            subject_col=subject_col,
            desc=desc,
            session=session if isinstance(session, str) else "01",
        )
    external_metadata_df = metadata_df.copy() if metadata_df is not None else None
    if external_metadata_df is not None:
        external_metadata_df[subject_col] = (
            external_metadata_df[subject_col].astype(int).map(lambda value: f"{value:04d}")
        )

    if use_derivatives:
        epochs_root = bids_root / "derivatives" / "preproc"
        logger.info(f"Loading saved epochs from {epochs_root}")
        config = BIDSConfig(
            path=epochs_root,
            datatype="eeg",
            suffix="epo",
            loading_mode="load_existing",
            subjects=subjects if subjects else None,
            event_id=condition if condition else None,
            target_col=target_col,
        )
        return load_data(
            config=config,
            subject_metadata_df=external_metadata_df,
            subject_key=subject_col if external_metadata_df is not None else None,
        )

    logger.info(f"Loading data for {len(subjects)} subjects. Task: {task}")
    config = BIDSConfig(
        path=bids_root,
        task=task,
        session=session,
        loading_mode="epochs",
        window_length=segment_duration,
        stride=segment_duration - overlap,
        subjects=subjects if subjects else None,
        datatype="eeg",
        suffix="eeg",
        target_col=target_col,
    )
    container = load_data(
        config=config,
        subject_metadata_df=external_metadata_df,
        subject_key=subject_col if external_metadata_df is not None else None,
    )
    logger.info(f"Initial Container Shape: {container.X.shape}, Dims: {container.dims}")
    return container


def load_cleaned_continuous_container(
    bids_root: Path,
    subjects: list[str],
    task: str,
    segment_duration: float,
    overlap: float,
    condition: str | None,
    metadata_df: pd.DataFrame | None = None,
    subject_col: str = "study_id",
    desc: str = "base",
    session: str = "01",
) -> DataContainer:
    """Re-epoch the cleaned continuous ``desc`` derivative at ``segment_duration``.

    This is the leakage-safe source for foundation models whose pretrained window
    differs from the saved epoch derivative (e.g. LaBraM at 15 s). It reuses the
    same continuous-stage cleaning (filter / ICA / interpolation) as the saved
    epoch derivative but does **not** re-apply per-epoch autoreject rejection;
    that difference is recorded in ``meta['autoreject_applied'] = False`` and the
    embedding sidecar.

    Notes
    -----
    Returned epochs carry MNE's inclusive endpoint (``segment_duration * sfreq +
    1`` samples). Callers should pass the result through coco-pipe's
    ``normalize_inclusive_endpoint`` to drop the extra sample.
    """
    from eeg_adhd_epilepsy.preproc.epochs import make_epochs_from_preproc_raw

    preproc_root = get_preproc_root(bids_root)
    meta_lookup: pd.DataFrame | None = None
    if metadata_df is not None:
        meta_lookup = metadata_df.copy()
        meta_lookup[subject_col] = (
            meta_lookup[subject_col].astype(int).map(lambda value: f"{value:04d}")
        )
        meta_lookup = meta_lookup.set_index(subject_col)

    x_chunks: list[np.ndarray] = []
    obs_rows: list[dict[str, Any]] = []
    ids: list[str] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None
    times: np.ndarray | None = None
    issues: list[str] = []

    for raw_sid in subjects or []:
        study_id = f"{int(raw_sid):04d}" if str(raw_sid).isdigit() else str(raw_sid)
        subject_id = f"sub-{study_id}"
        raw, _prov, load_issues = load_stage_artifacts(
            subject_id, preproc_root, desc=desc, task=task
        )
        if raw is None:
            issues.extend(load_issues)
            continue
        try:
            epochs = make_epochs_from_preproc_raw(
                raw, segment_duration=segment_duration, overlap=overlap
            )
        except ValueError as exc:
            issues.append(f"no_epochs:{subject_id}:{exc}")
            continue
        if condition is None or condition not in epochs.event_id:
            issues.append(f"missing_condition:{subject_id}:{condition}")
            continue
        cond_epochs = epochs[condition]
        data = np.asarray(cond_epochs.get_data(), dtype=np.float32)
        if data.shape[0] == 0:
            continue
        if ch_names is None:
            ch_names = list(cond_epochs.ch_names)
            sfreq = float(cond_epochs.info["sfreq"])
            times = np.asarray(cond_epochs.times)
        recording_id = f"{subject_id}_ses-{session}_run-01"
        meta_row: dict[str, Any] = {}
        if meta_lookup is not None and study_id in meta_lookup.index:
            meta_row = {
                str(column): meta_lookup.loc[study_id, column] for column in meta_lookup.columns
            }
        for epoch_idx in range(data.shape[0]):
            obs_rows.append(
                {
                    subject_col: study_id,
                    "subject": subject_id,
                    "session": session,
                    "run": "01",
                    "condition": condition,
                    "recording_id": recording_id,
                    **meta_row,
                }
            )
            ids.append(f"{recording_id}_epoch-{epoch_idx:03d}")
        x_chunks.append(data)

    if not x_chunks:
        raise RuntimeError(
            f"No cleaned-continuous epochs for condition {condition!r} at "
            f"{segment_duration:g}s. First issues: {issues[:5]}"
        )

    X = np.concatenate(x_chunks, axis=0)
    obs_df = pd.DataFrame(obs_rows)
    coords: dict[str, Any] = {
        "channel": np.asarray(ch_names, dtype=object),
        "time": np.asarray(times),
    }
    for column in obs_df.columns:
        coords[column] = obs_df[column].to_numpy()

    return DataContainer(
        X=X,
        dims=("obs", "channel", "time"),
        coords=coords,
        ids=np.asarray(ids, dtype=object),
        meta={
            "sfreq": sfreq,
            "window_source": "re_epoch_cleaned_continuous",
            "autoreject_applied": False,
            "segment_duration": float(segment_duration),
            "load_issues": issues,
        },
    )

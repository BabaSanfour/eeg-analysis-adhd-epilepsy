"""BIDS I/O utilities for EEG analysis."""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from mne_bids import BIDSPath

if TYPE_CHECKING:
    from coco_pipe.io import DataContainer

logger = logging.getLogger(__name__)


_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")


def _sanitize_bids_token(value: str, field_name: str) -> str:
    """Remove non-alphanumeric characters from a BIDS entity value."""
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value).strip())
    if not token:
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return token


def study_id_to_bids_subject(study_id: int | str) -> str:
    """Convert a metadata ``study_id`` to a bare BIDS subject entity."""
    text = str(study_id).strip()
    if not text.isdigit():
        raise ValueError(f"study_id must be an integer, got {study_id!r}")
    return f"{int(text):04d}"


def _bids_entity_label(entity: str, value: str) -> str:
    """Add an entity prefix to an already-bare, validated BIDS value."""
    token = str(value).strip()
    if not token or not _ALNUM_RE.fullmatch(token):
        raise ValueError(f"Invalid bare BIDS {entity}: {value!r}")
    return f"{entity}-{token}"


def bids_subject_label(subject: str) -> str:
    """Render a bare BIDS subject entity as a path label."""
    return _bids_entity_label("sub", subject)


def bids_session_label(session: str) -> str:
    """Render a bare BIDS session entity as a path label."""
    return _bids_entity_label("ses", session)


def validate_stage_desc(desc: str, allowed: set[str] | None = None) -> str:
    """Validate/normalize BIDS desc values used in output filenames."""
    desc_token = _sanitize_bids_token(desc, "desc")
    if allowed is not None and desc_token not in allowed:
        allowed_vals = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid desc {desc_token!r}. Expected one of: {allowed_vals}")
    return desc_token


class DerivativeStage(StrEnum):
    """Canonical ``derivatives/<...>`` subtrees for a BIDS dataset.

    The value encodes the full sub-path (including nesting), so the location of
    each stage is defined exactly once here instead of at every call site.
    """

    PREPROC = "preproc"
    DESCRIPTORS = "signal_features/descriptors"
    FOUNDATION_EMBEDDINGS = "eeg_foundation_embeddings"
    DIM_REDUCTION = "dim_reduction"
    DECODING = "decoding"


def get_derivative_root(bids_root: Path, stage: DerivativeStage) -> Path:
    """Return the canonical ``derivatives/<stage>`` root for a BIDS dataset.

    One join point with typed stages (mirroring ``ReportStage``/``summary_report_dir``).
    Downstream stages resolve a derivative tree to the same default location as
    the producer.
    """
    if not isinstance(stage, DerivativeStage):
        raise TypeError(f"stage must be a DerivativeStage, got {stage!r}")
    return Path(bids_root).expanduser() / "derivatives" / stage.value


def get_stage_output_path(
    subject: str,
    preproc_root: Path,
    desc: str,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return stage output FIF path using unified naming."""
    sid = bids_subject_label(subject)
    desc_token = validate_stage_desc(desc)
    directory_parts = [sid]
    if session:
        directory_parts.append(bids_session_label(session))
    eeg_dir = Path(preproc_root).expanduser().joinpath(*directory_parts) / "eeg"
    if create_dir:
        eeg_dir.mkdir(parents=True, exist_ok=True)
    parts = [sid]
    if session:
        parts.append(bids_session_label(session))
    if task:
        task_token = _sanitize_bids_token(task, "task")
        parts.append(f"task-{task_token}")
    if run:
        run_token = _sanitize_bids_token(run, "run")
        parts.append(f"run-{run_token}")
    parts.append(f"desc-{desc_token}")
    parts.append("eeg")
    fname = "_".join(parts) + ".fif"
    return eeg_dir / fname


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
        subj_tag = bids_subject_label(subj) if subj else ""
        if subjects_filter:
            if subj_tag not in subjects_filter and subj not in subjects_filter:
                continue
        if match.fpath is not None and match.fpath.exists():
            files.append(match.fpath)
    return sorted(files)


def parse_bids_components(source: Path | str) -> dict[str, str]:
    """Extract common BIDS entities from a BIDS-style filepath or string."""
    text = str(source)
    if isinstance(source, Path):
        text = source.name

    entities: dict[str, str] = {}

    # Simple split-based extraction for cleanly formatted tokens
    for part in text.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        # Only take alpha-numeric value part (strip extensions if any)
        value = value.split(".")[0]
        if key and value:
            entities[key] = value

    # Fallback to robust regex extraction for missing core keys
    for short_key in ("sub", "ses", "task", "run", "acq", "proc"):
        if short_key in entities:
            continue
        match = re.search(rf"(?:^|[_/]){short_key}-([A-Za-z0-9]+)", text)
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


def add_recording_id(
    container: DataContainer,
    subject_col: str = "subject",
) -> DataContainer:
    """Return a container with a composite ``recording_id`` obs coordinate.

    Relies on the ``subject``/``session``/``run`` coordinates that coco-pipe's
    ``BIDSDataset`` already attaches at load time, and delegates the composition
    to :meth:`coco_pipe.io.DataContainer.combine_coords`. The resulting label
    uses coco-pipe's generic ``"<coord>-<value>"`` form
    (e.g. ``"subject-0001_session-01_run-02"``).

    Parameters
    ----------
    container:
        Container whose observations carry ``subject``/``session``/``run``
        coordinates (as produced by ``BIDSDataset``).
    subject_col:
        Fallback coordinate name to use for the subject entity when no canonical
        ``subject`` coordinate is present.
    """
    if "recording_id" in container.coords:
        return container

    subject_key = "subject" if "subject" in container.coords else subject_col
    keys = [subject_key, "session", "run"]
    missing = [key for key in keys if key not in container.coords]
    if missing:
        raise ValueError(
            f"Cannot build recording_id: missing coordinates {missing}. "
            f"Available coordinates: {sorted(container.coords)}."
        )
    return container.combine_coords(keys, "recording_id")

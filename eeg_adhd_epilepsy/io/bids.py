"""BIDS I/O utilities for EEG analysis."""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from pathlib import Path

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_bids_path_from_fname,
    make_dataset_description,
)

from eeg_adhd_epilepsy.utils.artifacts import write_text_atomic

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
    VARIANCE_DIAGNOSTICS = "eeg_variance_diagnostics"
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


# Authors of the raw dataset and of the derivatives this repo writes.
DATASET_AUTHORS = ["Hamza Abdelhedi"]
# What the preproc tree holds that BIDS does not define: FIF is not a BIDS EEG
# format and saved epochs have no standard yet.
PREPROC_BIDSIGNORE = ("*_eeg.fif", "*_epo.fif", "*_epo.json", "*_provenance.json")
# For a tree where BIDS defines nothing but the dataset description.
DESCRIPTION_ONLY_BIDSIGNORE = ("/*", "!/dataset_description.json")


def write_bidsignore(dataset_root: Path, patterns: tuple[str, ...]) -> None:
    """Write the ``.bidsignore`` of a dataset, for the validator to skip ``patterns``."""
    write_text_atomic(Path(dataset_root) / ".bidsignore", "\n".join(patterns) + "\n")


def write_preproc_dataset_description(preproc_root: Path) -> None:
    """Make ``derivatives/preproc`` a valid BIDS derivative dataset."""
    preproc_root = Path(preproc_root)
    preproc_root.mkdir(parents=True, exist_ok=True)
    make_dataset_description(
        path=preproc_root,
        name="Preprocessed EEG",
        authors=DATASET_AUTHORS,
        dataset_type="derivative",
        generated_by=[
            {
                "Name": "eeg-preprocess",
                "Description": "Cleaned continuous EEG (desc-base) and fixed-length epochs",
            }
        ],
        source_datasets=[{"URL": "../../"}],
        overwrite=True,
        verbose=False,
    )
    write_bidsignore(preproc_root, PREPROC_BIDSIGNORE)


def get_bids_derivative_variant_path(
    source_path: Path,
    derivative_root: Path,
    *,
    processing: str,
    suffix: str,
    datatype: str = "eeg",
) -> Path:
    """Return a BIDS derivative path with explicit updated entities."""
    variant = get_bids_path_from_fname(Path(source_path), check=False)
    variant.update(
        root=Path(derivative_root),
        datatype=datatype,
        processing=_sanitize_bids_token(processing, "processing"),
        suffix=_sanitize_bids_token(suffix, "suffix"),
        check=False,
    )
    output_path = variant.fpath
    if output_path is None:
        raise ValueError(f"Could not construct a BIDS derivative path for {source_path}.")
    return output_path


def get_stage_output_path(
    source: Path | BIDSPath,
    preproc_root: Path,
    desc: str,
    *,
    task: str | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the FIF path of the ``desc`` stage for the recording ``source``.

    ``source`` is any file of the recording (raw or another stage), or a
    ``BIDSPath`` holding its entities. ``task`` replaces the task of ``source``.
    """
    if not isinstance(source, BIDSPath):
        source = get_bids_path_from_fname(source, check=False)
    path = BIDSPath(
        root=Path(preproc_root).expanduser(),
        subject=source.subject,
        session=source.session,
        task=_sanitize_bids_token(task, "task") if task else source.task,
        run=source.run,
        description=validate_stage_desc(desc),
        datatype="eeg",
        suffix="eeg",
        extension=".fif",
        check=False,
    )
    if create_dir:
        path.mkdir()
    return path.fpath


def discover_raw_eeg(bids_root: Path) -> list[Path]:
    """Return the raw BrainVision header files of a BIDS dataset."""
    paths = find_matching_paths(
        bids_root, datatypes="eeg", suffixes="eeg", extensions=".vhdr", ignore_nosub=True
    )
    return sorted(path.fpath for path in paths)

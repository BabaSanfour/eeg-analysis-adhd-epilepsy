"""Canonical report directory layout."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from eeg_adhd_epilepsy.io.bids import bids_session_label, bids_subject_label


class ReportStage(StrEnum):
    """Allowed report namespaces."""

    EEG_PRE_BASE = "eeg_pre_base"
    RAW_QC_PRE_BASE = "raw_qc_pre_base"
    BASE_QC = "base_qc"
    CORRECT_QC = "correct_qc"
    DENOISE_QC = "denoise_qc"
    COMPARE = "compare"
    DESCRIPTOR_QC = "descriptor_qc"
    DIM_REDUCTION = "dim_reduction"


def default_reports_root(bids_root: Path) -> Path:
    """Return the default reports directory beside a BIDS dataset."""
    return Path(bids_root).expanduser().parent / "reports"


def subject_report_dir(
    reports_root: Path,
    subject: str,
    session: str,
    stage: ReportStage,
    *,
    create: bool = False,
) -> Path:
    """Return ``reports/subjects/sub-<subject>/ses-<session>/<stage>``."""
    if not isinstance(stage, ReportStage):
        raise TypeError(f"stage must be a ReportStage, got {stage!r}")
    path = (
        Path(reports_root).expanduser()
        / "subjects"
        / bids_subject_label(subject)
        / bids_session_label(session)
        / stage.value
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def summary_report_dir(
    reports_root: Path,
    stage: ReportStage,
    *,
    create: bool = False,
) -> Path:
    """Return ``reports/summary/<stage>``."""
    if not isinstance(stage, ReportStage):
        raise TypeError(f"stage must be a ReportStage, got {stage!r}")
    path = Path(reports_root).expanduser() / "summary" / stage.value
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def descriptor_qc_report_name(subject: str, session: str, condition: str) -> str:
    """Filename for a per-subject descriptor-QC HTML report.

    Shared by the producer (``extract_descriptors`` resume check) and the QC
    writer (``run_descriptor_subject_qc``) so the naming convention lives in
    one place.
    """
    return (
        f"{bids_subject_label(subject)}_{bids_session_label(session)}_"
        f"{condition}_descriptor_qc_report.html"
    )


def build_bids_report_ids(
    filepath: Path,
) -> dict[str, str | tuple[str | None, str | None] | tuple[str | None, str | None, str | None]]:
    """Build shared run-aware identifiers for reports and aggregation."""
    from eeg_adhd_epilepsy.io.bids import parse_bids_components

    comps = parse_bids_components(filepath)
    subject = comps.get("subject", "unknown")
    session = comps.get("session")
    run = comps.get("run")
    subject_id = bids_subject_label(subject)
    subject_session_prefix = (
        subject_id if not session else f"{subject_id}_{bids_session_label(session)}"
    )
    run_prefix = subject_session_prefix if not run else f"{subject_session_prefix}_run-{run}"
    return {
        "subject": subject,
        "session": session or "01",
        "subject_id": subject_id,
        "session_id": session or "",
        "run_id": run or "",
        "subject_session_prefix": subject_session_prefix,
        "run_prefix": run_prefix,
        "subject_session_key": (subject_id, session or ""),
        "run_key": (subject_id, session or "", run or ""),
    }

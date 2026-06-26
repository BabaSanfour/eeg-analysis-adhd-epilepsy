"""Resolve which subjects an analysis extraction entry point should process."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from eeg_adhd_epilepsy.io.bids import study_id_to_bids_subject

LOGGER = logging.getLogger(__name__)


def resolve_metadata_row(
    metadata_df: pd.DataFrame,
    metadata_row: int,
    subject_col: str,
) -> str | None:
    """BIDS subject for a one-based metadata row (a SLURM array index).

    Shared by the extraction entry points so a surplus array task no-ops the same
    way everywhere: returns ``None`` (and warns) when the row is past the end of
    the table.
    """
    position = metadata_row - 1
    if position < 0 or position >= len(metadata_df):
        LOGGER.warning(
            "Metadata row %d is outside the metadata table with %d rows; nothing to do.",
            metadata_row,
            len(metadata_df),
        )
        return None
    return study_id_to_bids_subject(metadata_df.iloc[position][subject_col])


def resolve_cohort_subjects(
    metadata_df: pd.DataFrame | None,
    subject_col: str,
    requested: Sequence[Any] | None,
) -> list[str] | None:
    """Resolve the BIDS subjects to process, with the metadata as the authority.

    The metadata cohort is the gate: ``requested`` ids are converted with
    :func:`study_id_to_bids_subject` and intersected with it, so a typo'd id
    raises and an out-of-cohort id is dropped rather than silently extracted.
    Extraction entry points have no target-label step downstream to cull
    subjects, so this intersection is their only cohort gate.

    With no metadata there is nothing to gate against, so ``requested`` (or
    ``None`` -> the whole BIDS tree) passes through unchanged.
    """
    if metadata_df is None or subject_col not in metadata_df:
        return list(requested) if requested is not None else None
    valid_subjects = {study_id_to_bids_subject(value) for value in metadata_df[subject_col]}
    if requested:
        subjects = [
            subject
            for subject in (study_id_to_bids_subject(value) for value in requested)
            if subject in valid_subjects
        ]
    else:
        subjects = sorted(valid_subjects)
    if not subjects:
        raise ValueError(
            f"No metadata-cohort subjects matched against the {subject_col!r} column."
        )
    return subjects

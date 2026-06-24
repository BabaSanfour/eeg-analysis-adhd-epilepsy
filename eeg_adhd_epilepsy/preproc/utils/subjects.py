"""Subject-selection helpers shared by all pipeline entry points."""

from __future__ import annotations

import random
from collections.abc import Sequence

from eeg_adhd_epilepsy.io import bids


def _normalize_subject_list(subjects: Sequence[str]) -> list[str]:
    """Convert metadata study IDs to sorted bare BIDS subject entities."""
    return sorted({bids.study_id_to_bids_subject(s) for s in subjects})


def select_subjects(
    subjects_found: Sequence[str],
    selected_subjects: Sequence[str] | None = None,
    start_from: str | None = None,
    use_test: bool = False,
    use_random_test: bool = False,
    use_all: bool = False,
) -> list[str]:
    """Return the subjects to process based on standard CLI selection flags.

    Parameters
    ----------
    subjects_found:
        Full sorted list of subjects available in the dataset.
    selected_subjects:
        Explicit allow-list (``--subjects``).  Takes precedence over all
        other flags.
    start_from:
        Resume from this subject ID onwards, alphabetically (``--start-from``).
    use_test:
        Pick the first 5 subjects (``--test``).
    use_random_test:
        Combined with ``use_test``: pick 5 random subjects with seed 42
        (``--random``).
    use_all:
        Process every discovered subject (``--all``).

    Returns
    -------
    List[str]
        Sorted, normalised subject IDs to process.  Empty list when no
        flag is set — the caller is responsible for handling that case.
    """
    found_sorted = sorted(set(subjects_found))

    if selected_subjects:
        return _normalize_subject_list(selected_subjects)

    if start_from:
        start_sid = bids.study_id_to_bids_subject(start_from)
        return [sid for sid in found_sorted if sid >= start_sid]

    if use_test:
        if use_random_test:
            random.seed(42)
            return sorted(random.sample(found_sorted, min(5, len(found_sorted))))
        return found_sorted[:5]

    if use_all:
        return found_sorted

    return []

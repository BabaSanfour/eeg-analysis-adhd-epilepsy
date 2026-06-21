"""Event and annotation processing utilities."""

from __future__ import annotations

from collections import Counter

import mne


def summarize_annotations(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Return a dictionary of unique annotation counts."""
    if raw.annotations is None:
        return {}
    return dict(Counter(raw.annotations.description))


def compute_special_event_counts(raw: mne.io.BaseRaw) -> dict[str, int]:
    """
    Count occurrences of specific events of interest (e.g. eyes open/closed).
    Logic derived from simplified BIDS annotations.
    """
    if raw.annotations is None:
        return {}
    return summarize_annotations(raw)


def crop_raw_to_recording_start(
    raw: mne.io.BaseRaw,
    event_label: str = "recording_start",
    max_onset_seconds: float = 60.0,
) -> mne.io.BaseRaw:
    """
    Crop raw data to start at the specified reference event.
    Only crops if the event is found within the first `max_onset_seconds`.
    """
    if raw.annotations is None:
        return raw
    ref_onset = None
    for onset, _, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if desc == event_label:
            ref_onset = float(onset)
            break

    if ref_onset is not None:
        total_dur = raw.times[-1]
        if ref_onset < max_onset_seconds and ref_onset < total_dur:
            raw.crop(tmin=ref_onset)
    return raw

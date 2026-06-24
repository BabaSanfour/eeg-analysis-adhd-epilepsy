"""Event and annotation processing utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.utils import constants


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


def summarize_event_counts(
    raw: mne.io.BaseRaw,
    segments_df: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, int]:
    """Count report-facing events for a recording.

    Combines condition-block and eye-state counts (from the ``summary`` dict and
    ``segments_df`` produced by the segmenter) with any residual annotation
    descriptions still present on ``raw`` (e.g. clinical or ``BAD_*`` markers),
    skipping the canonical condition labels that are already summarised above.
    """
    raw_counts = summarize_annotations(raw)
    event_counts = {
        "HV Start": int(summary.get("hv_block_count", 0)),
        "HV End": int(summary.get("hv_block_count", 0)),
        "Photo": int(summary.get("photo_block_count", 0)),
        "Post-HV": int(summary.get("post_hv_block_count", 0)),
        "Eyes Open": 0,
        "Eyes Closed": 0,
    }
    if not segments_df.empty:
        eye_states = segments_df["eye_state"].fillna("unknown").astype(str).str.lower()
        event_counts["Eyes Open"] = int(eye_states.eq("eo").sum())
        event_counts["Eyes Closed"] = int(eye_states.eq("ec").sum())
    for desc, count in raw_counts.items():
        clean_desc = str(desc).strip().lower()
        if (
            clean_desc
            in {"eyes_open", "eyes_closed", "hv_start", "hv_end", "post_hv", "recording_start"}
            or clean_desc == "photo"
            or clean_desc.startswith("photo_")
            or str(desc).startswith("BLOCK_")
        ):
            continue
        event_counts[desc] = event_counts.get(desc, 0) + int(count)
    return event_counts


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


def collect_block_windows(raw: mne.io.BaseRaw) -> list[BlockWindow]:
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
    for block in collect_block_windows(raw):
        if "baseline" in block.name.lower():
            windows.append((block.onset, block.stop))
    return windows


def segments_from_block_annotations(raw: mne.io.BaseRaw) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for block in collect_block_windows(raw):
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

"""Condition segment extraction and statistics."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Set

import mne
import numpy as np
import pandas as pd
from eeg_adhd_epilepsy.utils.config import (
    EYES_CLOSED_LABELS,
    EYES_OPEN_LABELS,
    HV_LABELS,
    PHOTO_LABELS,
    POST_HV_LABELS,
)

SEGMENT_COLUMNS = [
    "segment_type",
    "t_start",
    "t_stop",
    "duration",
    "freq_hz",
    "hv_index",
    "post_hv_index",
    "eyes_open_duration",
    "eyes_closed_duration",
]

PHOTO_FREQ_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*hz", flags=re.IGNORECASE)
MAX_PHOTO_DURATION = 120.0  # seconds


@dataclass(frozen=True)
class AnnotationEntry:
    onset: float
    description: str
    clean: str


def normalize_label(desc: str | None) -> str:
    if not desc:
        return ""
    normalized = unicodedata.normalize("NFKD", desc)
    # Remove accents, lowercase, strip
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.lower().strip()


def _contains_label(clean_desc: str, labels: Sequence[str]) -> bool:
    if not clean_desc:
        return False
    # Check for substring match
    if any(label.lower() in clean_desc for label in labels):
        return True
    return False


def is_hv_start(clean_desc: str) -> bool:
    if "start" in clean_desc or "debut" in clean_desc:
        return _contains_label(clean_desc, HV_LABELS)
    return False


def is_hv_end(clean_desc: str) -> bool:
    if "end" in clean_desc or "fin" in clean_desc:
        return _contains_label(clean_desc, HV_LABELS)
    return False


def is_post_hv(clean_desc: str) -> bool:
    return _contains_label(clean_desc, POST_HV_LABELS)


def is_photo(clean_desc: str) -> bool:
    return _contains_label(clean_desc, PHOTO_LABELS)


def get_photo_freq(desc: str | None) -> float | None:
    if not desc:
        return None
    match = PHOTO_FREQ_PATTERN.search(desc)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if math.isfinite(value) and value.is_integer():
        return float(int(value))
    return value if math.isfinite(value) else None


def _empty_segments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEGMENT_COLUMNS)


def _compute_raw_end(raw: mne.io.BaseRaw) -> float:
    if raw.n_times == 0:
        return 0.0
    sfreq = float(raw.info.get("sfreq") or 0.0)
    t_last = float(raw.times[-1])
    if sfreq and sfreq > 0.0:
        return t_last + 1.0 / sfreq
    return t_last


def _prepare_annotation_entries(raw: mne.io.BaseRaw) -> List[AnnotationEntry]:
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return []
    entries: List[AnnotationEntry] = []
    for onset, desc in zip(annotations.onset, annotations.description):
        if desc is None:
            continue
        clean = normalize_label(desc)
        entries.append(AnnotationEntry(onset=float(onset), description=str(desc), clean=clean))
    return entries


def _build_eye_state_intervals(entries: Sequence[AnnotationEntry], raw_end: float) -> List[Tuple[float, float, str]]:
    intervals: List[Tuple[float, float, str]] = []
    current_state: str | None = None
    state_start: float | None = None
    
    first_eye_event_idx = -1
    for i, entry in enumerate(entries):
        if entry.clean == "eyes_open" or entry.clean == "eyes_closed":
            first_eye_event_idx = i
            break
            
    # Initial state logic
    if first_eye_event_idx >= 0:
        first_entry = entries[first_eye_event_idx]
        if first_entry.clean == "eyes_open":
            current_state = "open"
        elif first_entry.clean == "eyes_closed":
            current_state = "closed"
        state_start = first_entry.onset
    else:
        pass

    for i in range(first_eye_event_idx + 1, len(entries)):
        entry = entries[i]
        new_state: str | None = None
        if entry.clean == "eyes_open":
            new_state = "open"
        elif entry.clean == "eyes_closed":
            new_state = "closed"
        
        if new_state is None:
            continue
            
        # If we have a running state, close it
        if current_state is not None and state_start is not None and entry.onset > state_start:
            intervals.append((state_start, entry.onset, current_state))
        
        current_state = new_state
        state_start = entry.onset
    
    # Close final state
    if current_state is not None and state_start is not None and raw_end > state_start:
        intervals.append((state_start, raw_end, current_state))
        
    return intervals


def _find_hv_blocks(entries: Sequence[AnnotationEntry]) -> List[dict]:
    hv_blocks: List[dict] = []
    hv_index = 1
    hv_start: float | None = None
    for entry in entries:
        if is_hv_start(entry.clean):
            hv_start = entry.onset
            continue
        if is_hv_end(entry.clean) and hv_start is not None:
            if entry.onset > hv_start:
                hv_blocks.append({"segment_type": "HV_block", "t_start": hv_start, "t_stop": entry.onset, "hv_index": hv_index})
                hv_index += 1
            hv_start = None
    return hv_blocks


def _find_next_major_event(entries: Sequence[AnnotationEntry], start_idx: int) -> float | None:
    for idx in range(start_idx, len(entries)):
        entry = entries[idx]
        if is_photo(entry.clean) or is_hv_start(entry.clean):
            return entry.onset
    return None


def _compute_post_hv_blocks(
    hv_blocks: List[dict], 
    photo_blocks: List[dict],
    entries: Sequence[AnnotationEntry], 
    raw_end: float
) -> List[dict]:
    """
    Identify Post-HV blocks starting immediately after HV blocks.
    
    Logic:
    1. Start = End of HV block.
    2. End Constraint = Start of next block (Photo or HV) or Raw End.
    3. If explicit "Post HV" markers exist in the gap, extend to cover them or until constraint.
    4. If no markers, default to 25% of HV duration (min 0s), capped by constraint.
    """
    post_blocks: List[dict] = []
    
    # Sort constraints for easy lookup
    # Constraints are starts of other blocks
    constraints = [b["t_start"] for b in hv_blocks] + [b["t_start"] for b in photo_blocks]
    constraints.append(raw_end)
    constraints = sorted([c for c in constraints if c > 0])
    
    for i, hv_block in enumerate(hv_blocks):
        start = hv_block["t_stop"]
        hv_dur = hv_block["t_stop"] - hv_block["t_start"]
        
        # Find nearest constraint > start
        # Use simple search (list is small)
        limit = raw_end
        for c in constraints:
            if c > start + 0.1: # tolerance
                limit = c
                break
        
        # Check for explicit markers in the gap [start, limit]
        # Ignore markers that are basically AT the limit (next block start)
        markers_in_gap = []
        fin_marker_time = None
        
        for entry in entries:
            if entry.onset < start:
                continue
            if entry.onset >= limit:
                break
            
            if is_post_hv(entry.clean):
                markers_in_gap.append(entry)
                # Check if it's a "Fin" marker
                if "fin" in entry.clean or "end" in entry.clean:
                    fin_marker_time = entry.onset
        
        # Determine End
        stop = start
        
        if markers_in_gap:
            # If explicit markers exist, we assume the Post-HV period is active.
            # If we found a "Fin" marker, stop there.
            # Otherwise, stop at the last observed Post-HV marker (e.g. "POST HV 02:00")
            if fin_marker_time:
                stop = fin_marker_time
            else:
                stop = markers_in_gap[-1].onset
        else:
            # No markers: Default to 25% rule
            req_dur = 0.25 * hv_dur
            stop = start + req_dur
            if stop > limit:
                stop = limit
                
        # Validate block
        if stop > start + 1.0: # Minimum 1s duration
            post_blocks.append({
                "segment_type": "PostHV_block",
                "t_start": start,
                "t_stop": stop,
                "post_hv_index": i + 1
            })
            
    return post_blocks


def _find_photo_blocks(entries: Sequence[AnnotationEntry], raw_end: float) -> List[dict]:
    photo_entries: List[Tuple[int, AnnotationEntry]] = [
        (idx, entry) for idx, entry in enumerate(entries) if is_photo(entry.clean)
    ]
    blocks: List[dict] = []
    for pos, (idx, entry) in enumerate(photo_entries):
        next_start = raw_end
        if pos + 1 < len(photo_entries):
            next_start = photo_entries[pos + 1][1].onset
        if next_start <= entry.onset:
            continue
        duration = next_start - entry.onset
        if duration > MAX_PHOTO_DURATION:
             continue 
        blocks.append(
            {
                "segment_type": "PHOTO_block",
                "t_start": entry.onset,
                "t_stop": entry.onset + duration,
                "freq_hz": get_photo_freq(entry.description),
            }
        )
    return blocks


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned = [(float(start), float(stop)) for start, stop in intervals if stop > start]
    if not cleaned:
        return []
    cleaned.sort(key=lambda pair: pair[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_stop = cleaned[0]
    for start, stop in cleaned[1:]:
        if start <= cur_stop:
            cur_stop = max(cur_stop, stop)
            continue
        merged.append((cur_start, cur_stop))
        cur_start, cur_stop = start, stop
    merged.append((cur_start, cur_stop))
    return merged


def _subtract_interval(base_start: float, base_stop: float, exclusions: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not exclusions:
        return [(base_start, base_stop)]
    keep: List[Tuple[float, float]] = []
    cursor = base_start
    for ex_start, ex_stop in exclusions:
        if ex_stop <= cursor:
            continue
        if ex_start >= base_stop:
            break
        if ex_start > cursor:
            keep.append((cursor, min(ex_start, base_stop)))
        cursor = max(cursor, ex_stop)
        if cursor >= base_stop:
            break
    if cursor < base_stop:
        keep.append((cursor, base_stop))
    return [(s, e) for s, e in keep if e > s]


def _segment_eye_states_within_interval(
    start: float,
    stop: float,
    eye_states: Sequence[Tuple[float, float, str]],
    segment_prefix: str,
    hv_index: float | int | None = np.nan,
    post_hv_index: float | int | None = np.nan,
    freq_hz: float | None = np.nan,
) -> List[dict]:
    segments: List[dict] = []
    for state_start, state_stop, state in eye_states:
        if state_stop <= start or state_start >= stop:
            continue
        seg_start = max(start, state_start)
        seg_stop = min(stop, state_stop)
        if seg_stop <= seg_start:
            continue
        duration = seg_stop - seg_start
        seg_type = f"{segment_prefix}_{'EO' if state == 'open' else 'EC'}"
        segments.append(
            {
                "segment_type": seg_type,
                "t_start": seg_start,
                "t_stop": seg_stop,
                "duration": duration,
                "freq_hz": freq_hz if freq_hz is not None else np.nan,
                "hv_index": hv_index,
                "post_hv_index": post_hv_index,
                "eyes_open_duration": duration if state == "open" else 0.0,
                "eyes_closed_duration": duration if state == "closed" else 0.0,
            }
        )
    total_eye_dur = sum(s["duration"] for s in segments)
    block_dur = stop - start
    if total_eye_dur < block_dur - 1e-3: 
        if not segments:
            segments.append(
                {
                    "segment_type": segment_prefix,
                    "t_start": start,
                    "t_stop": stop,
                    "duration": block_dur,
                    "freq_hz": freq_hz if freq_hz is not None else np.nan,
                    "hv_index": hv_index,
                    "post_hv_index": post_hv_index,
                    "eyes_open_duration": 0.0,
                    "eyes_closed_duration": 0.0,
                }
            )
    return segments


def extract_condition_segments(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Compute EO/EC baseline plus HV/PostHV/PHOTO segments split by eye state."""
    if raw is None:
        raise ValueError("raw must be a valid mne.io.Raw instance")
    # safety check for annotations
    entries = _prepare_annotation_entries(raw)
    
    if not entries:
        return _empty_segments_df()

    raw_end = _compute_raw_end(raw)
    eye_states = _build_eye_state_intervals(entries, raw_end)
    hv_blocks = _find_hv_blocks(entries)
    photo_blocks = _find_photo_blocks(entries, raw_end)

    # Compute Post-HV blocks relative to HV blocks (User request: Start from HV end)
    post_hv_blocks = _compute_post_hv_blocks(hv_blocks, photo_blocks, entries, raw_end)

    exclusion_intervals = _merge_intervals(
        [
            (block["t_start"], block["t_stop"])
            for block in (*hv_blocks, *post_hv_blocks, *photo_blocks)
            if block["t_stop"] > block["t_start"]
        ]
    )

    records: List[dict] = []

    # 1. Raw / Pre-Annotation Baseline Extraction
    # If the first eye state starts significantly after 0, label that period as 'RAW_baseline'
    first_eye_start = eye_states[0][0] if eye_states else raw_end
    if first_eye_start > 0.0:
        # Subtract exclusions (HV/Photo could technically happen before first eye state?)
        remaining = _subtract_interval(0.0, first_eye_start, exclusion_intervals)
        for seg_start, seg_stop in remaining:
            duration = seg_stop - seg_start
            if duration <= 0.0:
                continue
            records.append(
                {
                    "segment_type": "RAW_baseline",
                    "t_start": seg_start,
                    "t_stop": seg_stop,
                    "duration": duration,
                    "freq_hz": np.nan,
                    "hv_index": np.nan,
                    "post_hv_index": np.nan,
                    "eyes_open_duration": 0.0,
                    "eyes_closed_duration": 0.0,
                }
            )

    # 2. EO/EC Baseline Extraction
    for start, stop, state in eye_states:
        remaining = _subtract_interval(start, stop, exclusion_intervals)
        for seg_start, seg_stop in remaining:
            duration = seg_stop - seg_start
            if duration <= 0.0:
                continue
            records.append(
                {
                    "segment_type": "EO_baseline" if state == "open" else "EC_baseline",
                    "t_start": seg_start,
                    "t_stop": seg_stop,
                    "duration": duration,
                    "freq_hz": np.nan,
                    "hv_index": np.nan,
                    "post_hv_index": np.nan,
                    "eyes_open_duration": duration if state == "open" else 0.0,
                    "eyes_closed_duration": duration if state == "closed" else 0.0,
                }
            )

    # 2. Special Blocks Extraction
    for block in hv_blocks:
        records.extend(
            _segment_eye_states_within_interval(
                block["t_start"],
                block["t_stop"],
                eye_states,
                segment_prefix="HV",
                hv_index=block["hv_index"],
                post_hv_index=np.nan,
                freq_hz=np.nan,
            )
        )

    for block in post_hv_blocks:
        records.extend(
            _segment_eye_states_within_interval(
                block["t_start"],
                block["t_stop"],
                eye_states,
                segment_prefix="PostHV",
                hv_index=np.nan,
                post_hv_index=block["post_hv_index"],
                freq_hz=np.nan,
            )
        )

    for block in photo_blocks:
        freq = block.get("freq_hz")
        records.extend(
            _segment_eye_states_within_interval(
                block["t_start"],
                block["t_stop"],
                eye_states,
                segment_prefix="PHOTO",
                hv_index=np.nan,
                post_hv_index=np.nan,
                freq_hz=freq if freq is not None else np.nan,
            )
        )

    if not records:
        return _empty_segments_df()
    df = pd.DataFrame.from_records(records, columns=SEGMENT_COLUMNS)
    return df.sort_values(by=["t_start", "segment_type"]).reset_index(drop=True)


def format_duration_hms(seconds: float | None) -> str:
    """Format seconds into a human-readable string."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "0s"
    if not math.isfinite(value):
        return "0s"
    value = max(0.0, value)
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    value -= minutes * 60
    seconds_str = f"{value:.2f}".rstrip("0").rstrip(".")
    if not seconds_str:
        seconds_str = "0"
    sec_component = f"{seconds_str}s"
    if hours > 0:
        return f"{hours}h {minutes}m {sec_component}"
    if minutes > 0:
        return f"{minutes}m {sec_component}"
    return sec_component


def summarize_condition_segments(df: pd.DataFrame) -> Dict[str, object]:
    """Return summary statistics for condition segments."""
    if df is None or df.empty:
        return {
            "total_duration": 0.0,
            "unique_duration": 0.0,
            "n_segments": 0,
            "segment_type_counts": {},
            "segment_type_durations": {},
            "total_eyes_open_duration": 0.0,
            "total_eyes_closed_duration": 0.0,
            "hv_block_count": 0,
            "post_hv_block_count": 0,
            "photo_block_count": 0,
            "photo_frequency_durations": {},
            "total_duration_readable": "0s",
            "total_eyes_open_duration_readable": "0s",
            "total_eyes_closed_duration_readable": "0s",
        }
        
    total_duration = float(pd.to_numeric(df["duration"], errors="coerce").sum())
    
    intervals = []
    for _, row in df.iterrows():
        s, e = float(row["t_start"]), float(row["t_stop"])
        if e > s:
            intervals.append((s, e))
    merged_intervals = _merge_intervals(intervals)
    unique_duration = sum(end - start for start, end in merged_intervals)
    
    # Calculate unique durations for EO and EC to handle any overlaps
    eo_intervals = []
    ec_intervals = []
    baseline_eo_intervals = []
    baseline_ec_intervals = []
    
    for _, row in df.iterrows():
        s, e = float(row["t_start"]), float(row["t_stop"])
        if e <= s:
            continue
        
        # Check EO/EC status
        is_eo = float(row.get("eyes_open_duration", 0)) > 0
        is_ec = float(row.get("eyes_closed_duration", 0)) > 0
        
        if is_eo:
            eo_intervals.append((s, e))
        if is_ec:
            ec_intervals.append((s, e))
            
        # Breakdown checks
        stype = str(row.get("segment_type", ""))
        if stype == "EO_baseline":
            baseline_eo_intervals.append((s, e))
        elif stype == "EC_baseline":
            baseline_ec_intervals.append((s, e))

    unique_eo = sum(end - start for start, end in _merge_intervals(eo_intervals))
    unique_ec = sum(end - start for start, end in _merge_intervals(ec_intervals))
    unique_baseline_eo = sum(end - start for start, end in _merge_intervals(baseline_eo_intervals))
    unique_baseline_ec = sum(end - start for start, end in _merge_intervals(baseline_ec_intervals))

    summary = {
        "total_duration": unique_duration,
        "sum_of_segments_duration": total_duration,
        "n_segments": int(len(df)),
        "segment_type_counts": {
            str(k): int(v)
            for k, v in df["segment_type"].fillna("Unknown").value_counts().items()
        },
        "segment_type_durations": {
            str(k): float(v)
            for k, v in df.groupby("segment_type", dropna=False)["duration"].sum().items()
        },
        # Use unique sums
        "total_eyes_open_duration": unique_eo,
        "total_eyes_closed_duration": unique_ec,
        "total_baseline_eyes_open_duration": unique_baseline_eo,
        "total_baseline_eyes_closed_duration": unique_baseline_ec,
    }
    segment_types = df["segment_type"].fillna("Unknown").astype(str)
    hv_mask = segment_types.str.startswith("HV_") | (segment_types == "HV_block")
    post_hv_mask = segment_types.str.startswith("PostHV_") | (segment_types == "PostHV_block")
    photo_mask = segment_types.str.startswith("PHOTO_") | (segment_types == "PHOTO_block")

    hv_indices = pd.to_numeric(df.loc[hv_mask, "hv_index"], errors="coerce")
    summary["hv_block_count"] = int(hv_indices.dropna().nunique() or df.loc[hv_mask, "t_start"].nunique())

    post_hv_indices = pd.to_numeric(df.loc[post_hv_mask, "post_hv_index"], errors="coerce")
    summary["post_hv_block_count"] = int(post_hv_indices.dropna().nunique() or df.loc[post_hv_mask, "t_start"].nunique())

    summary["photo_block_count"] = int(df.loc[photo_mask, "t_start"].nunique())

    photo = df.loc[photo_mask]
    if not photo.empty and "freq_hz" in photo:
        freq_summary = (
            photo.dropna(subset=["freq_hz"])
            .groupby("freq_hz")["duration"]
            .sum()
            .sort_index()
        )
        summary["photo_frequency_durations"] = {
            float(freq): float(duration) for freq, duration in freq_summary.items()
        }
    else:
        summary["photo_frequency_durations"] = {}
        
    summary["total_duration_readable"] = format_duration_hms(summary["total_duration"])
    summary["total_eyes_open_duration_readable"] = format_duration_hms(summary["total_eyes_open_duration"])
    summary["total_eyes_closed_duration_readable"] = format_duration_hms(summary["total_eyes_closed_duration"])
    summary["total_baseline_eyes_open_duration_readable"] = format_duration_hms(summary["total_baseline_eyes_open_duration"])
    summary["total_baseline_eyes_closed_duration_readable"] = format_duration_hms(summary["total_baseline_eyes_closed_duration"])
    return summary

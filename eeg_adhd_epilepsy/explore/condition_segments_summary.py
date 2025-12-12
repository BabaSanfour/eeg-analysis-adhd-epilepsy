"""Condition segment extraction, visualization, and reporting CLI."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm

from eeg_adhd_epilepsy.utils.qc_config import (
    EYES_CLOSED_LABELS,
    EYES_OPEN_LABELS,
    HV_LABELS,
    PHOTO_LABELS,
    POST_HV_LABELS,
)

# ---------------------------------------------------------------------------
# Condition segment extraction helpers
# ---------------------------------------------------------------------------

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
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.lower().strip()


def _contains_label(clean_desc: str, labels: Sequence[str]) -> bool:
    if not clean_desc:
        return False
    return any(label in clean_desc for label in labels)


def is_eyes_open(clean_desc: str) -> bool:
    return _contains_label(clean_desc, EYES_OPEN_LABELS)


def is_eyes_closed(clean_desc: str) -> bool:
    return _contains_label(clean_desc, EYES_CLOSED_LABELS)


def is_hv_start(clean_desc: str) -> bool:
    return _contains_label(clean_desc, HV_LABELS) and "start" in clean_desc


def is_hv_end(clean_desc: str) -> bool:
    return _contains_label(clean_desc, HV_LABELS) and "end" in clean_desc


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
    for entry in entries:
        new_state: str | None = None
        if is_eyes_open(entry.clean):
            new_state = "open"
        elif is_eyes_closed(entry.clean):
            new_state = "closed"
        if new_state is None:
            continue
        if current_state is not None and state_start is not None and entry.onset > state_start:
            intervals.append((state_start, entry.onset, current_state))
        current_state = new_state
        state_start = entry.onset
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


def _find_post_hv_blocks(entries: Sequence[AnnotationEntry]) -> List[dict]:
    blocks: List[dict] = []
    post_index = 1
    current_group: List[int] = []
    for entry_idx, entry in enumerate(entries):
        if is_post_hv(entry.clean):
            current_group.append(entry_idx)
            continue
        if current_group:
            block = _finalize_post_hv_group(entries, current_group)
            current_group = []
            if block is None:
                continue
            block["post_hv_index"] = post_index
            blocks.append(block)
            post_index += 1
    if current_group:
        block = _finalize_post_hv_group(entries, current_group)
        if block is not None:
            block["post_hv_index"] = post_index
            blocks.append(block)
    return blocks


def _finalize_post_hv_group(entries: Sequence[AnnotationEntry], group: Sequence[int]) -> dict | None:
    if not group:
        return None
    start_entry = entries[group[0]]
    end_entry = entries[group[-1]]
    start = start_entry.onset
    stop = end_entry.onset
    if stop <= start:
        next_event = _find_next_major_event(entries, group[-1] + 1)
        if next_event is None or next_event <= start:
            return None
        stop = next_event
    return {
        "segment_type": "PostHV_block",
        "t_start": start,
        "t_stop": stop,
    }


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


def _compute_eye_state_durations(
    start: float,
    stop: float,
    eye_states: Sequence[Tuple[float, float, str]],
) -> Tuple[float, float]:
    open_duration = 0.0
    closed_duration = 0.0
    for state_start, state_stop, state in eye_states:
        if state_stop <= start:
            continue
        if state_start >= stop:
            break
        overlap_start = max(start, state_start)
        overlap_stop = min(stop, state_stop)
        if overlap_stop <= overlap_start:
            continue
        duration = overlap_stop - overlap_start
        if state == "open":
            open_duration += duration
        else:
            closed_duration += duration
    return open_duration, closed_duration


def _segment_eye_states_within_interval(
    start: float,
    stop: float,
    eye_states: Sequence[Tuple[float, float, str]],
    segment_prefix: str,
    hv_index: float | int | None = np.nan,
    post_hv_index: float | int | None = np.nan,
    freq_hz: float | None = np.nan,
) -> List[dict]:
    """Split a block interval by eye-state annotations and return segment rows."""
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
    if not segments and stop > start:
        segments.append(
            {
                "segment_type": f"{segment_prefix}_UNKNOWN",
                "t_start": start,
                "t_stop": stop,
                "duration": stop - start,
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
    entries = _prepare_annotation_entries(raw)
    if not entries:
        return _empty_segments_df()

    raw_end = _compute_raw_end(raw)
    eye_states = _build_eye_state_intervals(entries, raw_end)
    hv_blocks = _find_hv_blocks(entries)
    post_hv_blocks = _find_post_hv_blocks(entries)
    photo_blocks = _find_photo_blocks(entries, raw_end)

    exclusion_intervals = _merge_intervals(
        [
            (block["t_start"], block["t_stop"])
            for block in (*hv_blocks, *post_hv_blocks, *photo_blocks)
            if block["t_stop"] > block["t_start"]
        ]
    )

    records: List[dict] = []

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


# ---------------------------------------------------------------------------
# Condition segment visualization/report helpers
# ---------------------------------------------------------------------------

def format_duration_hms(seconds: float | None) -> str:
    """Format seconds into a human-readable string (H M S / M S / S)."""
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


FIGURE_FILENAMES = {
    "segment_duration": "segment_total_duration.png",
    "eye_state_breakdown": "segment_eye_state_breakdown.png",
    "photo_frequency": "photo_frequency_duration.png",
    "hv_blocks": "hv_block_eye_states.png",
    "post_hv_blocks": "post_hv_block_eye_states.png",
    "timeline": "segment_timeline.png",
}


def summarize_condition_segments(df: pd.DataFrame) -> Dict[str, object]:
    """Return summary statistics for condition segments."""
    if df is None or df.empty:
        return {
            "total_duration": 0.0,
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
    summary = {
        "total_duration": total_duration,
        "n_segments": int(len(df)),
        "segment_type_counts": {
            str(k): int(v)
            for k, v in df["segment_type"].fillna("Unknown").value_counts().items()
        },
        "segment_type_durations": {
            str(k): float(v)
            for k, v in df.groupby("segment_type", dropna=False)["duration"].sum().items()
        },
        "total_eyes_open_duration": float(pd.to_numeric(df["eyes_open_duration"], errors="coerce").sum()),
        "total_eyes_closed_duration": float(pd.to_numeric(df["eyes_closed_duration"], errors="coerce").sum()),
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
    return summary


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_total_duration_by_segment(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = (
        df.groupby("segment_type", dropna=False)["duration"]
        .sum()
        .sort_values(ascending=True)
    )
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, len(group) * 0.35)))
    ax.barh(group.index.astype(str), group.values, color="#4C72B0")
    ax.set_xlabel("Total Duration (s)")
    ax.set_title("Total Duration by Segment Type")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["segment_duration"])


def _plot_eye_state_breakdown(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = df.groupby("segment_type")[["eyes_open_duration", "eyes_closed_duration"]].sum()
    if group.empty:
        return None
    group = group.sort_values(by="eyes_open_duration", ascending=False)
    labels = group.index.astype(str)
    open_vals = group["eyes_open_duration"].to_numpy(dtype=float)
    closed_vals = group["eyes_closed_duration"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(3, len(group) * 0.35)))
    positions = np.arange(len(labels))
    ax.barh(positions, open_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, closed_vals, left=open_vals, label="Eyes Closed", color="#C44E52")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Eyes-Open vs Eyes-Closed Duration per Segment Type")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["eye_state_breakdown"])


def _plot_photo_frequency_durations(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    segment_types = df["segment_type"].fillna("").astype(str)
    photo = df[segment_types.str.startswith("PHOTO_") | (segment_types == "PHOTO_block")]
    if photo.empty or "freq_hz" not in photo:
        return None
    group = (
        photo.dropna(subset=["freq_hz"])
        .groupby("freq_hz")["duration"]
        .sum()
        .sort_index()
    )
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(freq) for freq in group.index]
    ax.bar(labels, group.values, color="#8172B2")
    ax.set_xlabel("PHOTO Frequency (Hz)")
    ax.set_ylabel("Total Duration (s)")
    ax.set_title("PHOTO Block Duration by Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["photo_frequency"])


def _plot_block_eye_states(df: pd.DataFrame, block_type: str, fig_dir: Path) -> Path | None:
    if block_type not in {"HV", "PostHV"}:
        raise ValueError("block_type must be 'HV' or 'PostHV'")
    column = "hv_index" if block_type == "HV" else "post_hv_index"
    segment_types = df["segment_type"].fillna("").astype(str)
    legacy_label = f"{block_type}_block"
    mask = segment_types.str.startswith(f"{block_type}_") | (segment_types == legacy_label)
    block_df = df.loc[mask].copy()
    if block_df.empty or column not in block_df:
        return None

    block_df[column] = pd.to_numeric(block_df[column], errors="coerce")
    fallback = pd.Series(np.arange(1, len(block_df) + 1, dtype=int), index=block_df.index)
    block_df[column] = block_df[column].fillna(fallback).astype(int)

    def _label_eye_state(seg_type: str) -> str:
        if "_EO" in seg_type:
            return "EO"
        if "_EC" in seg_type:
            return "EC"
        return "Unknown"

    block_df["eye_state"] = [_label_eye_state(val) for val in block_df["segment_type"]]

    grouped = block_df.groupby([column, "eye_state"])["duration"].sum().unstack(fill_value=0.0)

    legacy_mask = segment_types[mask] == legacy_label
    if legacy_mask.any():
        legacy_df = block_df[legacy_mask]
        legacy_grouped = legacy_df.groupby(column)[["eyes_open_duration", "eyes_closed_duration"]].sum()
        grouped["EO"] = grouped.get("EO", 0.0) + legacy_grouped.get("eyes_open_duration", 0.0)
        grouped["EC"] = grouped.get("EC", 0.0) + legacy_grouped.get("eyes_closed_duration", 0.0)

    if grouped.empty:
        return None

    labels = [f"{block_type} #{idx}" for idx in grouped.index]
    fig, ax = plt.subplots(figsize=(7, max(3, len(grouped) * 0.4)))
    positions = np.arange(len(labels))
    eo_vals = grouped.get("EO", pd.Series(0.0, index=grouped.index))
    ec_vals = grouped.get("EC", pd.Series(0.0, index=grouped.index))
    unknown_vals = grouped.drop(columns=[c for c in grouped.columns if c in {"EO", "EC"}], errors="ignore").sum(axis=1)

    ax.barh(positions, eo_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, ec_vals, left=eo_vals, label="Eyes Closed", color="#C44E52")
    if not (unknown_vals == 0).all():
        ax.barh(
            positions,
            unknown_vals,
            left=eo_vals + ec_vals,
            label="Unknown",
            color="#8172B2",
            alpha=0.7,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title(f"{block_type} Eyes-Open vs Eyes-Closed")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    key = "hv_blocks" if block_type == "HV" else "post_hv_blocks"
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES[key])


def _plot_segment_timeline(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    if df.empty:
        return None
    ordered = df.sort_values("t_start")
    segment_types = list(dict.fromkeys(ordered["segment_type"]))
    if not segment_types:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, len(segment_types) * 0.6)))
    cmap = plt.get_cmap("tab20")
    type_to_y = {seg: idx for idx, seg in enumerate(segment_types)}
    for idx, row in ordered.iterrows():
        seg_type = row["segment_type"]
        y = type_to_y.get(seg_type)
        if y is None:
            continue
        start = float(row["t_start"])
        stop = float(row["t_stop"])
        if not np.isfinite(start) or not np.isfinite(stop):
            continue
        color = cmap(y % cmap.N)
        ax.plot([start, stop], [y, y], linewidth=8, solid_capstyle="butt", color=color)
    ax.set_yticks(list(type_to_y.values()))
    ax.set_yticklabels(segment_types)
    ax.set_xlabel("Time (s)")
    ax.set_title("Condition Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, fig_dir / FIGURE_FILENAMES["timeline"])


def save_condition_segment_figures(df: pd.DataFrame, fig_dir: Path) -> Dict[str, Path]:
    """Create bar plots and other figures summarizing segments."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}
    for key, func in [
        ("segment_duration", _plot_total_duration_by_segment),
        ("eye_state_breakdown", _plot_eye_state_breakdown),
        ("photo_frequency", _plot_photo_frequency_durations),
    ]:
        path = func(df, fig_dir)
        if path:
            figure_paths[key] = path
    for block_type in ("HV", "PostHV"):
        path = _plot_block_eye_states(df, block_type, fig_dir)
        if path:
            figure_paths["hv_blocks" if block_type == "HV" else "post_hv_blocks"] = path
    timeline_path = _plot_segment_timeline(df, fig_dir)
    if timeline_path:
        figure_paths["timeline"] = timeline_path
    return figure_paths


def create_condition_segments_report(
    df: pd.DataFrame,
    summary: Mapping[str, object],
    figure_paths: Mapping[str, Path],
    output_path: Path,
    subject_id: str | None = None,
    raw_duration: float | None = None,
) -> Path:
    """Create an HTML report combining summary stats and generated figures."""
    title = "Condition Segment Summary"
    if subject_id:
        title += f" - {subject_id}"
    report = mne.Report(title=title)
    coverage = ""
    if raw_duration and summary.get("total_duration"):
        pct = (float(summary["total_duration"]) / float(raw_duration)) * 100.0
        coverage = f"<li>Coverage vs raw: {pct:.1f}% of {format_duration_hms(raw_duration)}</li>"
    total_duration_str = summary.get("total_duration_readable", f"{summary.get('total_duration', 0.0):.2f}s")
    eyes_open_str = summary.get(
        "total_eyes_open_duration_readable", f"{summary.get('total_eyes_open_duration', 0.0):.2f}s"
    )
    eyes_closed_str = summary.get(
        "total_eyes_closed_duration_readable", f"{summary.get('total_eyes_closed_duration', 0.0):.2f}s"
    )
    summary_html = f"""
    <h3>Segment Overview</h3>
    <ul>
        <li>Total segments: {summary.get("n_segments", 0)}</li>
        <li>Total duration inside segments: {total_duration_str}</li>
        <li>Total eyes-open duration: {eyes_open_str}</li>
        <li>Total eyes-closed duration: {eyes_closed_str}</li>
        <li>HV blocks: {summary.get("hv_block_count", 0)}</li>
        <li>Post-HV blocks: {summary.get("post_hv_block_count", 0)}</li>
        <li>PHOTO blocks: {summary.get("photo_block_count", 0)}</li>
        {coverage}
    </ul>
    """
    segment_counts = summary.get("segment_type_counts") or {}
    if segment_counts:
        summary_html += "<p>Segment counts:</p><ul>"
        for seg, count in segment_counts.items():
            summary_html += f"<li>{seg}: {count}</li>"
        summary_html += "</ul>"
    photo_summary = summary.get("photo_frequency_durations") or {}
    if photo_summary:
        summary_html += "<p>PHOTO durations (s) by frequency:</p><ul>"
        for freq, duration in photo_summary.items():
            summary_html += f"<li>{freq:g} Hz: {duration:.2f}</li>"
        summary_html += "</ul>"
    report.add_html(summary_html, title="Summary", section="Overview")

    figure_specs = [
        ("Total Duration by Segment Type", figure_paths.get("segment_duration")),
        ("Eyes State Breakdown by Segment Type", figure_paths.get("eye_state_breakdown")),
        ("PHOTO Frequency Durations", figure_paths.get("photo_frequency")),
        ("HV Block Eyes State Breakdown", figure_paths.get("hv_blocks")),
        ("Post-HV Block Eyes State Breakdown", figure_paths.get("post_hv_blocks")),
        ("Condition Timeline", figure_paths.get("timeline")),
    ]
    for caption, path in figure_specs:
        if path and Path(path).exists():
            report.add_image(path, title=caption, section="Figures")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)
    return output_path


def generate_condition_segments_report(
    df: pd.DataFrame,
    output_dir: Path,
    subject_id: str | None = None,
    raw_duration: float | None = None,
    report_filename: str = "condition_segments_report.html",
) -> Dict[str, object]:
    """Convenience function to create figures and HTML report for segments."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    figure_paths = save_condition_segment_figures(df, fig_dir)
    summary = summarize_condition_segments(df)
    report_path = output_dir / report_filename
    create_condition_segments_report(df, summary, figure_paths, report_path, subject_id, raw_duration)
    return {
        "summary": summary,
        "figures": figure_paths,
        "report_path": report_path,
    }


# ---------------------------------------------------------------------------
# Dataset-level CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Condition segments extraction and reporting.")
    parser.add_argument("--input_dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Where to store outputs.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers (-1 for all cores).")
    parser.add_argument("--subjects_list", type=Path, help="Optional file with subject IDs to include.")
    parser.add_argument("--bids_session", default=None, help="BIDS session entity.")
    parser.add_argument("--bids_task", default="RESTING", help="BIDS task entity (default RESTING).")
    parser.add_argument("--bids_run", default=None, help="BIDS run entity.")
    parser.add_argument("--bids_acq", default=None, help="BIDS acquisition label.")
    parser.add_argument("--bids_proc", default=None, help="BIDS processing label.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    parser.add_argument("--skip_reports", action="store_true", help="Disable HTML report generation.")
    return parser.parse_args()


def setup_logging(log_file: Path, level: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("condition_segments")


def read_subjects_list(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def discover_bids_files(bids_root: Path, args: argparse.Namespace, subjects_filter: set[str] | None) -> List[Path]:
    template = BIDSPath(
        root=bids_root,
        subject=None,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    matches = template.match()
    files: List[Path] = []
    for match in matches:
        subject = match.subject or ""
        subj_tag = f"sub-{subject}" if subject else ""
        if subjects_filter and subj_tag not in subjects_filter and subject not in subjects_filter:
            continue
        if match.fpath and match.fpath.exists():
            files.append(match.fpath)
    return sorted(files)


def parse_subject_id(filepath: Path) -> str:
    name = filepath.name
    if "sub-" in name:
        start = name.find("sub-")
        chunk = name[start:].split("_", 1)[0]
        return chunk
    return filepath.stem


def load_raw(filepath: Path, bids_root: Path, args: argparse.Namespace):
    subject_clean = parse_subject_id(filepath).replace("sub-", "")
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject_clean,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    return read_raw_bids(bids_path)


def compute_raw_duration(raw) -> float:
    if raw.n_times == 0:
        return 0.0
    sfreq = float(raw.info.get("sfreq") or 0.0)
    last = float(raw.times[-1])
    if sfreq > 0.0:
        return last + 1.0 / sfreq
    return last


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def process_file(
    filepath: Path,
    args: argparse.Namespace,
    output_dirs: Dict[str, Path],
    logger: logging.Logger,
    skip_reports: bool,
) -> Dict[str, object]:
    subject_id = parse_subject_id(filepath)
    subject_dir = output_dirs["subjects"] / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, object] = {
        "subject_id": subject_id,
        "segments_csv": None,
        "summary_csv": None,
        "report_path": None,
        "summary": None,
        "raw_duration": 0.0,
        "error": "",
    }
    try:
        raw = load_raw(filepath, args.input_dir, args)
        segments_df = extract_condition_segments(raw)
        segments_df = segments_df.copy()
        segments_df.insert(0, "subject_id", subject_id)
        segments_csv_path = subject_dir / f"{subject_id}_condition_segments.csv"
        segments_df.to_csv(segments_csv_path, index=False)
        record["segments_csv"] = segments_csv_path

        raw_duration = compute_raw_duration(raw)
        record["raw_duration"] = raw_duration
        summary = summarize_condition_segments(segments_df)
        summary["subject_id"] = subject_id
        summary["raw_duration"] = raw_duration
        summary["coverage_pct"] = (
            (summary["total_duration"] / raw_duration) * 100.0 if raw_duration > 0 else 0.0
        )
        summary["raw_duration_readable"] = format_duration_hms(raw_duration)
        summary_csv_path = subject_dir / f"{subject_id}_condition_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
        record["summary_csv"] = summary_csv_path
        record["summary"] = summary

        if not skip_reports:
            report_info = generate_condition_segments_report(
                segments_df,
                subject_dir,
                subject_id=subject_id,
                raw_duration=raw_duration,
                report_filename=f"{subject_id}_condition_report.html",
            )
            record["report_path"] = report_info["report_path"]
    except Exception as exc:
        record["error"] = str(exc)
        logger.error("Failed processing %s: %s", subject_id, exc, exc_info=True)
    return record


def concat_segments(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path and Path(path).exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_dataset_summary(
    segments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    skip_reports: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_segments_csv = output_dir / "condition_segments_all_segments.csv"
    segments_df.to_csv(dataset_segments_csv, index=False)

    dataset_summary = summarize_condition_segments(segments_df)
    total_raw = float(summary_df["raw_duration"].sum()) if not summary_df.empty else 0.0
    dataset_summary["total_raw_duration"] = total_raw
    dataset_summary["coverage_pct"] = (
        (dataset_summary["total_duration"] / total_raw) * 100.0 if total_raw > 0 else 0.0
    )
    dataset_summary["total_raw_duration_readable"] = format_duration_hms(total_raw)

    dataset_summary_json = output_dir / "condition_segments_dataset_summary.json"
    dataset_summary_json.write_text(json.dumps(dataset_summary, indent=2))

    dataset_summary_csv = output_dir / "condition_segments_dataset_summary.csv"
    pd.DataFrame([dataset_summary]).to_csv(dataset_summary_csv, index=False)

    report_path = None
    if not skip_reports and not segments_df.empty:
        report_info = generate_condition_segments_report(
            segments_df,
            output_dir,
            subject_id=None,
            raw_duration=total_raw,
            report_filename="dataset_condition_segments_report.html",
        )
        report_path = report_info["report_path"]

    return {
        "segments_csv": dataset_segments_csv,
        "summary_json": dataset_summary_json,
        "summary_csv": dataset_summary_csv,
        "report_path": report_path,
    }


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.output_dir / "logs" / "condition_segments.log", args.log_level)
    logger.info("Starting condition segment extraction")

    subjects_filter = read_subjects_list(args.subjects_list)
    files = discover_bids_files(args.input_dir, args, subjects_filter)
    if not files:
        logger.error("No EEG files found under %s", args.input_dir)
        sys.exit(1)
    logger.info("Found %d BrainVision files", len(files))

    output_dirs = {
        "subjects": args.output_dir / "subjects",
    }
    output_dirs["subjects"].mkdir(parents=True, exist_ok=True)

    if args.n_jobs == 1:
        results = [
            process_file(f, args, output_dirs, logger, args.skip_reports)
            for f in tqdm(files, desc="Processing subjects")
        ]
    else:
        with tqdm_joblib(tqdm(total=len(files), desc="Processing subjects")):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_file)(f, args, output_dirs, logger, args.skip_reports) for f in files
            )

    summary_records = [rec["summary"] for rec in results if rec.get("summary")]
    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_csv_path = args.output_dir / "condition_segments_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info("Saved per-subject summary CSV to %s", summary_csv_path)
    else:
        logger.warning("No successful subject summaries were produced.")

    segment_paths = [rec["segments_csv"] for rec in results if rec.get("segments_csv")]
    if segment_paths:
        dataset_segments = concat_segments(segment_paths)
        dataset_outputs = save_dataset_summary(dataset_segments, summary_df, args.output_dir, args.skip_reports)
        logger.info("Saved dataset-level segments to %s", dataset_outputs["segments_csv"])
        if dataset_outputs["report_path"]:
            logger.info("Dataset summary report written to %s", dataset_outputs["report_path"])
    else:
        logger.warning("No per-subject segments were saved; skipping dataset summary.")

    n_errors = sum(bool(rec.get("error")) for rec in results)
    if n_errors:
        logger.warning("Completed with %d errors.", n_errors)
    else:
        logger.info("Condition segment extraction finished without errors.")


if __name__ == "__main__":
    main()

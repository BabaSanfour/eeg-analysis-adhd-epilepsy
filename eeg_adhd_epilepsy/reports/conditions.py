"""Condition segment reporting logic."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Any

import mne
import pandas as pd

from eeg_adhd_epilepsy.utils.config import SEGMENT_COLUMNS


# Regex patterns to filter out overlapping/granular events from summary tables
IGNORE_EVENT_PATTERNS = [
    re.compile(r"^HV\s+\d{2}:\d{2}$", re.IGNORECASE),
    re.compile(r"^POST\s*HV\s+\d{2}:\d{2}$", re.IGNORECASE),
    re.compile(r"^PHOTO\s+\d+(?:\.\d+)?(?:Hz)?$", re.IGNORECASE),
    re.compile(r"^\d+\s*min\s+post\s+hv$", re.IGNORECASE),
    re.compile(r"^fin\s+post\s+hv$", re.IGNORECASE),
]


def _is_ignored_event(label: str) -> bool:
    """Check if event label should be excluded from summary reports."""
    return any(pat.search(label) for pat in IGNORE_EVENT_PATTERNS)


def format_duration_hms(seconds: float | None) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "0s"
    if not math.isfinite(value):
        return "0s"
    hours, value = divmod(value, 3600)
    minutes, value = divmod(value, 60)
    sec_component = f"{f'{value:.2f}'.rstrip('0').rstrip('.') or '0'}s"
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {sec_component}"
    if minutes >= 1:
        return f"{int(minutes)}m {sec_component}"
    return sec_component


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        cur_start, cur_stop = merged[-1]
        if start <= cur_stop:
            merged[-1] = (cur_start, max(cur_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def summarize_condition_segments(df: pd.DataFrame) -> dict[str, object]:
    if df is None or df.empty:
        return {
            "total_duration": 0.0,
            "n_segments": 0,
            "segment_type_counts": {},
            "total_eyes_open_duration": 0.0,
            "total_eyes_closed_duration": 0.0,
            "total_baseline_eyes_open_duration": 0.0,
            "total_baseline_eyes_closed_duration": 0.0,
            "hv_block_count": 0,
            "post_hv_block_count": 0,
            "photo_block_count": 0,
            "photo_frequency_durations": {},
            "total_duration_readable": "0s",
            "total_eyes_open_duration_readable": "0s",
            "total_eyes_closed_duration_readable": "0s",
            "total_baseline_eyes_open_duration_readable": "0s",
            "total_baseline_eyes_closed_duration_readable": "0s",
        }

    summary_df = df.reindex(columns=SEGMENT_COLUMNS).copy()
    summary_df["t_start"] = pd.to_numeric(summary_df["t_start"], errors="coerce")
    summary_df["t_stop"] = pd.to_numeric(summary_df["t_stop"], errors="coerce")
    summary_df["duration"] = pd.to_numeric(summary_df["duration"], errors="coerce").fillna(0.0)
    summary_df["segment_type"] = summary_df["segment_type"].fillna("Unknown").astype(str)
    summary_df["block_family"] = summary_df["block_family"].fillna("unknown").astype(str)
    summary_df["eye_state"] = summary_df["eye_state"].fillna("unknown").astype(str).str.lower()
    valid = summary_df.loc[summary_df["t_stop"].gt(summary_df["t_start"])].copy()

    def merged_duration(frame: pd.DataFrame) -> float:
        intervals = sorted(frame[["t_start", "t_stop"]].itertuples(index=False, name=None))
        return sum(stop - start for start, stop in _merge_intervals(intervals))

    summary = {
        "total_duration": merged_duration(valid),
        "n_segments": int(len(df)),
        "segment_type_counts": {
            str(key): int(value) for key, value in summary_df["segment_type"].value_counts().items()
        },
        "total_eyes_open_duration": merged_duration(valid.loc[valid["eye_state"].eq("eo")]),
        "total_eyes_closed_duration": merged_duration(valid.loc[valid["eye_state"].eq("ec")]),
        "total_baseline_eyes_open_duration": merged_duration(
            valid.loc[valid["segment_type"].eq("EO_baseline")]
        ),
        "total_baseline_eyes_closed_duration": merged_duration(
            valid.loc[valid["segment_type"].eq("EC_baseline")]
        ),
    }
    hv_mask = valid["block_family"].eq("hv")
    post_hv_mask = valid["block_family"].eq("post_hv")
    photo_mask = valid["block_family"].eq("photo")
    summary["hv_block_count"] = int(valid.loc[hv_mask, ["t_start", "t_stop"]].drop_duplicates().shape[0])
    summary["post_hv_block_count"] = int(
        valid.loc[post_hv_mask, ["t_start", "t_stop"]].drop_duplicates().shape[0]
    )
    summary["photo_block_count"] = int(valid.loc[photo_mask, ["t_start", "t_stop"]].drop_duplicates().shape[0])
    photo = valid.loc[photo_mask].assign(freq_hz=pd.to_numeric(valid.loc[photo_mask, "freq_hz"], errors="coerce"))
    if not photo.empty:
        freq_summary = photo.dropna(subset=["freq_hz"]).groupby("freq_hz")["duration"].sum().sort_index()
        summary["photo_frequency_durations"] = {
            float(freq): float(duration) for freq, duration in freq_summary.items()
        }
    else:
        summary["photo_frequency_durations"] = {}
    summary["total_duration_readable"] = format_duration_hms(summary["total_duration"])
    summary["total_eyes_open_duration_readable"] = format_duration_hms(summary["total_eyes_open_duration"])
    summary["total_eyes_closed_duration_readable"] = format_duration_hms(summary["total_eyes_closed_duration"])
    summary["total_baseline_eyes_open_duration_readable"] = format_duration_hms(
        summary["total_baseline_eyes_open_duration"]
    )
    summary["total_baseline_eyes_closed_duration_readable"] = format_duration_hms(
        summary["total_baseline_eyes_closed_duration"]
    )
    return summary


def create_condition_segments_report(
    summary: Mapping[str, object],
    figure_paths: Mapping[str, Path],
    output_path: Path,
    subject_id: str | None = None,
    raw_duration: float | None = None,
    event_counts: Mapping[str, int] | None = None,
) -> Path:
    """Create an HTML report combining summary stats, generated figures, and event counts."""
    title = "Condition Segment Summary"
    if subject_id:
        title += f" - {subject_id}"
    report = mne.Report(title=title)
    
    # Overview Section
    coverage = ""
    if raw_duration and summary.get("total_duration") and raw_duration > 0:
        pct = (float(summary["total_duration"]) / float(raw_duration)) * 100.0
        coverage = f"<li>Coverage vs raw: {pct:.1f}% of {format_duration_hms(raw_duration)}</li>"
    total_duration_str = summary.get("total_duration_readable", f"{summary.get('total_duration', 0.0):.2f}s")
    eyes_open_str = summary.get(
        "total_eyes_open_duration_readable", f"{summary.get('total_eyes_open_duration', 0.0):.2f}s"
    )
    eyes_closed_str = summary.get(
        "total_eyes_closed_duration_readable", f"{summary.get('total_eyes_closed_duration', 0.0):.2f}s"
    )
    # Calculate Task Durations for clarity
    total_eo = summary.get('total_eyes_open_duration', 0.0)
    base_eo = summary.get('total_baseline_eyes_open_duration', 0.0)
    task_eo = max(0.0, total_eo - base_eo)
    
    total_ec = summary.get('total_eyes_closed_duration', 0.0)
    base_ec = summary.get('total_baseline_eyes_closed_duration', 0.0)
    task_ec = max(0.0, total_ec - base_ec)

    summary_html = f"""
    <h3>Segment Overview</h3>
    <ul>
        <li>Total analysis duration: {total_duration_str}</li>
        <li><b>Eyes Open:</b> {eyes_open_str}</li>
        <ul>
            <li>Baseline: {summary.get("total_baseline_eyes_open_duration_readable", "0s")}</li>
            <li>Task-related: {format_duration_hms(task_eo)}</li>
        </ul>
        <li><b>Eyes Closed:</b> {eyes_closed_str}</li>
        <ul>
            <li>Baseline: {summary.get("total_baseline_eyes_closed_duration_readable", "0s")}</li>
            <li>Task-related: {format_duration_hms(task_ec)}</li>
        </ul>
        <br>
        <li>Total segments: {summary.get("n_segments", 0)}</li>
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
    
    # Event Counts Section
    if event_counts:
        event_html = "<h3>Raw Event Counts (Filtered)</h3><ul>"
        # Sort for consistency
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        count_shown = 0
        for label, count in sorted_events:
             if count > 0 and not _is_ignored_event(label):
                 event_html += f"<li>{label}: {count}</li>"
                 count_shown += 1
        event_html += "</ul>"
        if count_shown == 0:
            event_html += "<p>No relevant events found (granular timestamp events hidden).</p>"
        report.add_html(event_html, title="Event Counts", section="Event Verification")

    # Figures Section
    for title, key in [
        ("Total Duration by Segment Type", "segment_duration"),
        ("Eyes-Open vs Eyes-Closed Breakdown", "eye_state_breakdown"),
        ("PHOTO Frequency Duration", "photo_frequency"),
        ("HV Block Eye States", "hv_blocks"),
        ("Post-HV Block Eye States", "post_hv_blocks"),
        ("Segment Timeline", "timeline"),
    ]:
         path = figure_paths.get(key)
         if path and path.exists():
             report.add_image(path, title=title, section="Figures")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)
    return output_path


def create_dataset_conditions_report(
    subjects_data: List[Dict[str, Any]],
    output_path: Path,
    figure_paths: Dict[str, Path] | None = None,
) -> None:
    """Create a global dataset report highlighting missing conditions."""
    report = mne.Report(title="Dataset Conditions Analysis")
    
    n_subjects = len(subjects_data)
    
    # 1. Missing Conditions Analysis
    missing_ec = [s["subject_id"] for s in subjects_data if s["summary"].get("total_eyes_closed_duration", 0) == 0]
    missing_eo = [s["subject_id"] for s in subjects_data if s["summary"].get("total_eyes_open_duration", 0) == 0]
    missing_baseline_ec = [s["subject_id"] for s in subjects_data if s["summary"].get("total_baseline_eyes_closed_duration", 0) == 0]
    missing_baseline_eo = [s["subject_id"] for s in subjects_data if s["summary"].get("total_baseline_eyes_open_duration", 0) == 0]
    
    missing_hv = [s["subject_id"] for s in subjects_data if s["summary"].get("hv_block_count", 0) == 0]
    missing_photo = [s["subject_id"] for s in subjects_data if s["summary"].get("photo_block_count", 0) == 0]

    no_conditions = []
    for s in subjects_data:
        summ = s["summary"]
        if (summ.get("total_eyes_open_duration", 0) <= 0 and 
            summ.get("total_eyes_closed_duration", 0) <= 0 and
            summ.get("hv_block_count", 0) == 0 and
            summ.get("photo_block_count", 0) == 0):
             no_conditions.append(s["subject_id"])
    
    summary_html = f"""
    <h3>Dataset Overview</h3>
    <p>Total Subjects: {n_subjects}</p>
    <h4>Condition Missingness</h4>
    <ul>
        <li><b>No Conditions (Only Raw):</b> {len(no_conditions)} subjects ({len(no_conditions)/n_subjects*100:.1f}%)</li>
        <li><b>No Eyes Closed (Any):</b> {len(missing_ec)} subjects ({len(missing_ec)/n_subjects*100:.1f}%)</li>
        <li><b>No Eyes Open (Any):</b> {len(missing_eo)} subjects ({len(missing_eo)/n_subjects*100:.1f}%)</li>
        <li><b>No Baseline Eyes Closed:</b> {len(missing_baseline_ec)} subjects ({len(missing_baseline_ec)/n_subjects*100:.1f}%)</li>
        <li><b>No Baseline Eyes Open:</b> {len(missing_baseline_eo)} subjects ({len(missing_baseline_eo)/n_subjects*100:.1f}%)</li>
        <li><b>No HV Blocks:</b> {len(missing_hv)} subjects ({len(missing_hv)/n_subjects*100:.1f}%)</li>
        <li><b>No PHOTO Blocks:</b> {len(missing_photo)} subjects ({len(missing_photo)/n_subjects*100:.1f}%)</li>
    </ul>
    """
    
    # Detailed lists (Truncated)
    def _list_subjects(subs, limit=50):
        if not subs:
            return ""
        s = ", ".join(subs[:limit])
        if len(subs) > limit:
            s += " ..."
        return s

    if no_conditions:
        summary_html += f"<p>Subjects with NO Conditions (Only Raw): {_list_subjects(no_conditions)}</p>"

    if missing_baseline_ec:
        summary_html += f"<p>Subjects with NO Baseline EC: {_list_subjects(missing_baseline_ec)}</p>"
    if missing_baseline_eo:
        summary_html += f"<p>Subjects with NO Baseline EO: {_list_subjects(missing_baseline_eo)}</p>"
    if missing_hv:
        summary_html += f"<p>Subjects with NO HV: {_list_subjects(missing_hv)}</p>"


    report.add_html(summary_html, title="Overview", section="Dataset Summary")
    
    # 2. Add Dataset Figures
    if figure_paths:
        # Define all potential figures to include
        figures_to_include = [
            ("dataset_duration_hist.png", "Distribution of Total Analysis Duration"),
            ("dataset_eye_states_hist.png", "Distribution of Eye State Durations"),
            ("dataset_event_distributions_conditions.png", "Condition Event Counts (EO/EC/HV/Photo)"),
            ("dataset_event_distributions_clinical.png", "Clinical/Artifact Event Counts (Non-zero)"),
            ("dataset_event_distributions.png", "Top Event Counts (Legacy/Fallback)"),
        ]
        
        for key, title in figures_to_include:
            fpath = None
            if key in figure_paths:
                 fpath = figure_paths[key]
            else:
                 # Fallback search by filename if keys are just names
                 for p in figure_paths.values():
                     if p.name == key:
                         fpath = p
                         break
            if fpath and fpath.exists():
                # Avoid inserting legacy plot if specific ones are already present
                if key == "dataset_event_distributions.png":
                    has_specific = (
                        ("dataset_event_distributions_conditions.png" in figure_paths) or
                        ("dataset_event_distributions_clinical.png" in figure_paths)
                    )
                    if has_specific: 
                         continue
                         
                report.add_image(fpath, title=title, section="Dataset Summary")

    # 3. Event Counts Summary (Filtered)
    all_event_keys = set()
    for s in subjects_data:
        if s.get("event_counts"):
            all_event_keys.update(s["event_counts"].keys())
            
    if all_event_keys:
        filtered_keys = [k for k in all_event_keys if not _is_ignored_event(k)]
        
        event_table_html = "<h3>Event Detection Rates (Filtered)</h3><table><tr><th>Event</th><th>Subjects with > 0</th><th>%</th></tr>"
        for key in sorted(filtered_keys):
             n_with = sum(1 for s in subjects_data if s.get("event_counts", {}).get(key, 0) > 0)
             pct = (n_with / n_subjects) * 100.0
             event_table_html += f"<tr><td>{key}</td><td>{n_with}</td><td>{pct:.1f}%</td></tr>"
        event_table_html += "</table>"
        report.add_html(event_table_html, title="Event Rates", section="Dataset Summary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)

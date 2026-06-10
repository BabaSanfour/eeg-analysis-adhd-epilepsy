"""Pre-base EEG report generation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from coco_pipe.report.core import Report, Section

from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.reports._common import (
    add_images as _add_images,
    add_optional_table as _add_optional_table,
    build_subject_overview_table,
)
from eeg_adhd_epilepsy.utils.config import SEGMENT_COLUMNS
from eeg_adhd_epilepsy.utils.formatting import format_clock_time, format_duration_hms


IGNORE_EVENT_PATTERNS = [
    re.compile(r"^HV\s+\d{2}:\d{2}$", re.IGNORECASE),
    re.compile(r"^POST\s*HV\s+\d{2}:\d{2}$", re.IGNORECASE),
    re.compile(r"^PHOTO\s+\d+(?:\.\d+)?(?:Hz)?$", re.IGNORECASE),
    re.compile(r"^\d+\s*min\s+post\s+hv$", re.IGNORECASE),
    re.compile(r"^fin\s+post\s+hv$", re.IGNORECASE),
]

MULTI_RUN_SUBJECT_THRESHOLD = 2


def _is_ignored_event(label: str) -> bool:
    return any(pat.search(label) for pat in IGNORE_EVENT_PATTERNS)

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
        return sum(stop - start for start, stop in bids_io.merge_intervals(list(intervals)))

    summary = {
        "total_duration": merged_duration(valid),
        "n_segments": int(len(df)),
        "segment_type_counts": {
            str(key): int(value) for key, value in summary_df["segment_type"].value_counts().items()
        },
        "total_eyes_open_duration": merged_duration(valid.loc[valid["eye_state"].eq("eo")]),
        "total_eyes_closed_duration": merged_duration(valid.loc[valid["eye_state"].eq("ec")]),
        "total_baseline_eyes_open_duration": merged_duration(valid.loc[valid["segment_type"].eq("EO_baseline")]),
        "total_baseline_eyes_closed_duration": merged_duration(valid.loc[valid["segment_type"].eq("EC_baseline")]),
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
        summary["photo_frequency_durations"] = {float(freq): float(duration) for freq, duration in freq_summary.items()}
    else:
        summary["photo_frequency_durations"] = {}
    summary["total_duration_readable"] = format_duration_hms(summary["total_duration"])
    summary["total_eyes_open_duration_readable"] = format_duration_hms(summary["total_eyes_open_duration"])
    summary["total_eyes_closed_duration_readable"] = format_duration_hms(summary["total_eyes_closed_duration"])
    summary["total_baseline_eyes_open_duration_readable"] = format_duration_hms(summary["total_baseline_eyes_open_duration"])
    summary["total_baseline_eyes_closed_duration_readable"] = format_duration_hms(summary["total_baseline_eyes_closed_duration"])
    return summary


def build_event_counts_table(event_counts: Mapping[str, int] | None) -> pd.DataFrame:
    if not event_counts:
        return pd.DataFrame(columns=["Event", "Count"])
    rows = [
        {"Event": str(label), "Count": int(count)}
        for label, count in sorted(event_counts.items(), key=lambda item: item[1], reverse=True)
        if int(count) > 0 and not _is_ignored_event(str(label))
    ]
    return pd.DataFrame(rows, columns=["Event", "Count"])


def build_condition_summary_table(summary: Mapping[str, object], raw_duration: float | None = None) -> pd.DataFrame:
    total_duration = float(summary.get("total_duration", 0.0) or 0.0)
    raw_duration = float(raw_duration or 0.0)
    coverage = (total_duration / raw_duration * 100.0) if raw_duration > 0 else float("nan")
    rows = [
        {"Metric": "Total analysis duration", "Value": format_duration_hms(total_duration)},
        {"Metric": "Coverage vs raw", "Value": f"{coverage:.1f}%" if math.isfinite(coverage) else "n/a"},
        {"Metric": "Eyes open duration", "Value": format_duration_hms(summary.get("total_eyes_open_duration", 0.0))},
        {"Metric": "Eyes closed duration", "Value": format_duration_hms(summary.get("total_eyes_closed_duration", 0.0))},
        {"Metric": "Baseline EO duration", "Value": format_duration_hms(summary.get("total_baseline_eyes_open_duration", 0.0))},
        {"Metric": "Baseline EC duration", "Value": format_duration_hms(summary.get("total_baseline_eyes_closed_duration", 0.0))},
        {"Metric": "Total segments", "Value": int(summary.get("n_segments", 0) or 0)},
        {"Metric": "HV blocks", "Value": int(summary.get("hv_block_count", 0) or 0)},
        {"Metric": "Post-HV blocks", "Value": int(summary.get("post_hv_block_count", 0) or 0)},
        {"Metric": "PHOTO blocks", "Value": int(summary.get("photo_block_count", 0) or 0)},
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_segment_counts_table(summary: Mapping[str, object]) -> pd.DataFrame:
    counts = summary.get("segment_type_counts") or {}
    rows = [{"Segment Type": str(key), "Count": int(value)} for key, value in counts.items()]
    return pd.DataFrame(rows, columns=["Segment Type", "Count"]).sort_values("Segment Type") if rows else pd.DataFrame(columns=["Segment Type", "Count"])


def build_photo_frequency_table(summary: Mapping[str, object]) -> pd.DataFrame:
    freq_summary = summary.get("photo_frequency_durations") or {}
    rows = [
        {"Frequency (Hz)": float(freq), "Duration": format_duration_hms(duration)}
        for freq, duration in sorted(freq_summary.items(), key=lambda item: float(item[0]))
    ]
    return pd.DataFrame(rows, columns=["Frequency (Hz)", "Duration"])


def build_dataset_summary_table(runs_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
    total_subjects = runs_df["subject_id"].nunique() if "subject_id" in runs_df else 0
    total_subject_sessions = len(subjects_df)
    total_runs = len(runs_df)
    total_raw_duration = pd.to_numeric(runs_df.get("raw_duration"), errors="coerce").fillna(0.0).sum()
    total_analysis_duration = pd.to_numeric(runs_df.get("total_duration"), errors="coerce").fillna(0.0).sum()
    median_run_duration = pd.to_numeric(runs_df.get("raw_duration"), errors="coerce").median()
    mean_run_duration = pd.to_numeric(runs_df.get("raw_duration"), errors="coerce").mean()
    row = {
        "Total subjects": int(total_subjects),
        "Total subject-sessions": int(total_subject_sessions),
        "Total runs": int(total_runs),
        "Total raw duration": format_duration_hms(total_raw_duration),
        "Total analyzed duration": format_duration_hms(total_analysis_duration),
        "Mean run duration": format_duration_hms(mean_run_duration),
        "Median run duration": format_duration_hms(median_run_duration),
    }
    return pd.DataFrame([row])


def build_multi_run_subjects_table(
    subjects_df: pd.DataFrame,
    min_runs: int = MULTI_RUN_SUBJECT_THRESHOLD,
) -> pd.DataFrame:
    if subjects_df.empty or "n_runs" not in subjects_df.columns:
        return pd.DataFrame()
    filtered = subjects_df.loc[pd.to_numeric(subjects_df["n_runs"], errors="coerce").fillna(0).ge(min_runs)].copy()
    if filtered.empty:
        return pd.DataFrame()
    columns = [
        "subject_id",
        "session_id",
        "n_runs",
        "source_dataset",
        "combined_diagnosis",
        "raw_duration",
    ]
    existing = [column for column in columns if column in filtered.columns]
    table_df = filtered[existing].copy()
    if "raw_duration" in table_df.columns:
        table_df["raw_duration"] = pd.to_numeric(table_df["raw_duration"], errors="coerce").map(format_duration_hms)
        table_df = table_df.rename(columns={"raw_duration": "total_duration"})
    return table_df.sort_values(["n_runs", "subject_id"], ascending=[False, True], na_position="last")


def build_recording_timing_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame(columns=["Metric", "Value"])
    starts = pd.to_datetime(runs_df.get("meas_datetime"), errors="coerce")
    if starts.notna().any():
        seconds = starts.dt.hour * 3600 + starts.dt.minute * 60 + starts.dt.second
        earliest_idx = seconds.idxmin()
        latest_idx = seconds.idxmax()
        earliest = starts.loc[earliest_idx].strftime("%H:%M:%S")
        latest = starts.loc[latest_idx].strftime("%H:%M:%S")
    else:
        earliest = "n/a"
        latest = "n/a"
    rows = [
        {"Metric": "Runs with measurement datetime", "Value": int(starts.notna().sum())},
        {"Metric": "Earliest recording time", "Value": earliest},
        {"Metric": "Latest recording time", "Value": latest},
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _build_condition_check_table(
    records_df: pd.DataFrame,
    label_col: str,
    unit_label: str,
    checks: Sequence[tuple[str, pd.Series]],
) -> pd.DataFrame:
    columns = [label_col, unit_label, "%"]
    if records_df.empty:
        return pd.DataFrame(columns=columns)
    n_records = len(records_df)
    rows = []
    for label, mask in checks:
        count = int(mask.fillna(False).sum())
        rows.append({label_col: label, unit_label: count, "%": f"{(count / n_records) * 100.0:.1f}%" if n_records else "0.0%"})
    return pd.DataFrame(rows, columns=columns)


def build_missingness_table(records_df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=["Metric", unit_label, "%"])
    checks = [
        ("No conditions (only raw)", (records_df["total_eyes_open_duration"] <= 0) & (records_df["total_eyes_closed_duration"] <= 0) & (records_df["hv_block_count"] == 0) & (records_df["photo_block_count"] == 0)),
        ("No eyes open", records_df["total_eyes_open_duration"] <= 0),
        ("No eyes closed", records_df["total_eyes_closed_duration"] <= 0),
        ("No baseline EO", records_df["total_baseline_eyes_open_duration"] <= 0),
        ("No baseline EC", records_df["total_baseline_eyes_closed_duration"] <= 0),
        ("No HV", records_df["hv_block_count"] == 0),
        ("No PHOTO", records_df["photo_block_count"] == 0),
    ]
    return _build_condition_check_table(records_df, "Metric", unit_label, checks)


def build_condition_availability_table(records_df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=["Condition", unit_label, "%"])
    checks = [
        ("Eyes open", records_df["total_eyes_open_duration"] > 0),
        ("Eyes closed", records_df["total_eyes_closed_duration"] > 0),
        ("Baseline EO", records_df["total_baseline_eyes_open_duration"] > 0),
        ("Baseline EC", records_df["total_baseline_eyes_closed_duration"] > 0),
        ("HV", records_df["hv_block_count"] > 0),
        ("Post-HV", records_df["post_hv_block_count"] > 0),
        ("PHOTO", records_df["photo_block_count"] > 0),
    ]
    return _build_condition_check_table(records_df, "Condition", unit_label, checks)


def build_event_rates_table(records: Sequence[Mapping[str, Any]], unit_label: str) -> pd.DataFrame:
    all_event_keys = set()
    for record in records:
        all_event_keys.update((record.get("event_counts") or {}).keys())
    rows = []
    total = len(records)
    for key in sorted(all_event_keys):
        if _is_ignored_event(str(key)):
            continue
        n_with = sum(1 for record in records if int((record.get("event_counts") or {}).get(key, 0)) > 0)
        rows.append({"Event": str(key), f"{unit_label} with > 0": int(n_with), "%": f"{(n_with / total) * 100.0:.1f}%" if total else "0.0%"})
    return pd.DataFrame(rows, columns=["Event", f"{unit_label} with > 0", "%"])


def build_metadata_group_table(runs_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if runs_df.empty or group_col not in runs_df.columns:
        return pd.DataFrame()
    grouped = runs_df.copy()
    grouped[group_col] = grouped[group_col].fillna("Unknown").replace("", "Unknown")
    rows = []
    for group_value, frame in grouped.groupby(group_col, dropna=False):
        n_runs = len(frame)
        rows.append(
            {
                group_col: group_value,
                "n_runs": int(n_runs),
                "n_subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else 0,
                "mean_run_duration": format_duration_hms(pd.to_numeric(frame["raw_duration"], errors="coerce").mean()),
                "mean_analysis_duration": format_duration_hms(pd.to_numeric(frame["total_duration"], errors="coerce").mean()),
                "pct_with_eo": f"{(frame['total_eyes_open_duration'].gt(0).mean() * 100.0):.1f}%",
                "pct_with_ec": f"{(frame['total_eyes_closed_duration'].gt(0).mean() * 100.0):.1f}%",
                "pct_with_hv": f"{(frame['hv_block_count'].gt(0).mean() * 100.0):.1f}%",
                "pct_with_photo": f"{(frame['photo_block_count'].gt(0).mean() * 100.0):.1f}%",
            }
        )
    return pd.DataFrame(rows).sort_values(group_col)


def build_dataset_report_tables(
    runs_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    run_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_summary_df": build_dataset_summary_table(runs_df, subjects_df),
        "multi_run_subjects_df": build_multi_run_subjects_table(subjects_df),
        "timing_df": build_recording_timing_table(runs_df),
        "missingness_df": build_missingness_table(runs_df, "Runs"),
        "availability_df": build_condition_availability_table(runs_df, "Runs"),
        "event_rates_df": build_event_rates_table(run_records, "Runs"),
        "source_dataset_df": build_metadata_group_table(runs_df, "source_dataset"),
        "combined_diagnosis_df": build_metadata_group_table(runs_df, "combined_diagnosis"),
        "sex_df": build_metadata_group_table(runs_df, "sex"),
        "age_group_df": build_metadata_group_table(runs_df, "age_group"),
    }


def generate_eeg_subject_report(
    record: Mapping[str, Any],
    run_inventory_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    figure_paths: Mapping[str, Path],
    output_path: Path,
) -> Path:
    report = Report(title=f"EEG Report - {record.get('subject_session_prefix', record.get('subject_id', 'unknown'))}")

    overview = Section("Recording Overview", icon="🎛️")
    overview.add_markdown("Aggregated pre-base EEG summary across all runs for this subject-session.")
    _add_optional_table(overview, build_subject_overview_table(record), "Subject Overview")
    report.add_section(overview)

    if run_inventory_df is not None and not run_inventory_df.empty:
        runs = Section("Run Inventory", icon="📚")
        _add_optional_table(runs, run_inventory_df, "Runs Included")
        report.add_section(runs)

    condition = Section("Condition Summary", icon="🧠")
    _add_optional_table(condition, build_condition_summary_table(record["summary"], raw_duration=record.get("raw_duration")), "Condition Summary")
    _add_optional_table(condition, build_segment_counts_table(record["summary"]), "Segment Counts")
    _add_optional_table(condition, build_photo_frequency_table(record["summary"]), "PHOTO Frequencies")
    report.add_section(condition)

    annotations = Section("Annotations", icon="📝")
    _add_optional_table(annotations, build_event_counts_table(record.get("event_counts")), "Filtered Event Counts")
    report.add_section(annotations)

    if run_summary_df is not None and not run_summary_df.empty:
        run_summaries = Section("Run Summaries", icon="🗂️")
        _add_optional_table(run_summaries, run_summary_df, "Per-Run Summary")
        report.add_section(run_summaries)

    figures = Section("Figures", icon="📈")
    _add_images(figures, figure_paths, ("segment_duration", "eye_state_breakdown", "photo_frequency", "hv_blocks", "post_hv_blocks", "timeline"))
    report.add_section(figures)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path


def generate_eeg_dataset_report(
    tables: Mapping[str, Any],
    figure_paths: Mapping[str, Path],
    output_path: Path,
) -> Path:
    report = Report(title="EEG Dataset Report")

    definition = Section("EEG Definition", icon="🎯")
    definition.add_markdown(
        "Pre-base EEG report built directly from BIDS recordings, canonical `_segments.csv` files, and canonical metadata joined by `study_id`."
    )
    _add_optional_table(definition, tables.get("dataset_summary_df", pd.DataFrame()), "Dataset Summary")
    report.add_section(definition)

    inventory = Section("Recording Inventory", icon="🗂️")
    _add_optional_table(inventory, tables.get("multi_run_subjects_df", pd.DataFrame()), "Subjects With More Than 2 Runs")
    _add_images(inventory, figure_paths, ("runs_per_subject", "run_duration_distribution"))
    report.add_section(inventory)

    timing = Section("Recording Structure and Timing", icon="⏰")
    _add_optional_table(timing, tables.get("timing_df", pd.DataFrame()), "Timing Summary")
    _add_images(timing, figure_paths, ("recording_start_hour_distribution",))
    report.add_section(timing)

    availability = Section("Condition Availability", icon="🧠")
    _add_optional_table(availability, tables.get("missingness_df", pd.DataFrame()), "Condition Missingness")
    _add_optional_table(availability, tables.get("availability_df", pd.DataFrame()), "Condition Availability")
    _add_images(availability, figure_paths, ("dataset_event_distributions_conditions",))
    report.add_section(availability)

    annotations = Section("Annotations and Clinical Content", icon="📝")
    _add_optional_table(annotations, tables.get("event_rates_df", pd.DataFrame()), "Filtered Event Rates")
    _add_images(annotations, figure_paths, ("dataset_event_distributions_clinical",))
    report.add_section(annotations)

    metadata = Section("Metadata-Linked EEG Views", icon="🔗")
    _add_optional_table(metadata, tables.get("source_dataset_df", pd.DataFrame()), "By Source Dataset")
    _add_optional_table(metadata, tables.get("combined_diagnosis_df", pd.DataFrame()), "By Combined Diagnosis")
    _add_optional_table(metadata, tables.get("sex_df", pd.DataFrame()), "By Sex")
    _add_optional_table(metadata, tables.get("age_group_df", pd.DataFrame()), "By Age Group")
    _add_images(
        metadata,
        figure_paths,
        (
            "availability_by_source_dataset",
            "availability_by_combined_diagnosis",
            "duration_by_source_dataset",
            "duration_by_combined_diagnosis",
        ),
    )
    report.add_section(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path

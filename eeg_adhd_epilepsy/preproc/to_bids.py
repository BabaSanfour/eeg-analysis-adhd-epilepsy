"""
Convert raw EEG + canonical metadata to BIDS with standardized annotations.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from eeg_adhd_epilepsy.io import ingest
from eeg_adhd_epilepsy.io import bids as bids_io
import eeg_adhd_epilepsy.qc.raw_metrics as qc_raw
import eeg_adhd_epilepsy.reports.eeg_report as report_eeg
from eeg_adhd_epilepsy.utils import config
import eeg_adhd_epilepsy.utils.events as utils_events
from eeg_adhd_epilepsy.utils.formatting import format_clock_time, format_duration_hms
from eeg_adhd_epilepsy.utils.logs import tqdm_joblib
import eeg_adhd_epilepsy.viz.eeg_report as viz_eeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
SEGMENT_COLUMNS = list(config.SEGMENT_COLUMNS)


def _slug_label(label: str) -> str:
    return label.lower().replace(" - ", "_").replace("/", "_").replace(" ", "_").replace("-", "_")


IGNORE_LABEL_PATTERNS = (*config.IGNORE_PATTERNS, *config.IGNORED_LABELS)
SENSOR_LABEL_PATTERNS = (*config.SENSOR_ARTEFACT_KEYWORDS, *config.SENSOR_ACTION_KEYWORDS)
BAD_INTEREST_LABELS = tuple(
    (pattern, f"BAD_{_slug_label(category)}")
    for category, patterns in config.ANNOTATION_INTEREST_MAP.items()
    if _slug_label(category) not in {"eyes_open", "eyes_closed", "hv", "post_hv", "photo"}
    for pattern in patterns
    if pattern
)
CLINICAL_LABELS = tuple(
    (pattern, _slug_label(category))
    for category, patterns in config.CLINICAL_COMMENT_LABELS.items()
    for pattern in patterns
    if pattern
)


def _matches_pattern(normalized: str, pattern: str) -> bool:
    pattern = str(pattern).lower().strip()
    if not pattern:
        return False
    if pattern == "*":
        return normalized == "*"
    if "*" in pattern:
        return fnmatch.fnmatch(normalized, pattern)
    return normalized == pattern or bool(re.search(r"\b" + re.escape(pattern) + r"\b", normalized))


def _matches_any(normalized: str, patterns: Sequence[str]) -> bool:
    return any(_matches_pattern(normalized, pattern) for pattern in patterns)


def canonicalize_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Rewrite annotations in place to the canonical BIDS-facing vocabulary."""
    new_onset = []
    new_duration = []
    new_descs = []

    for annot in raw.annotations:
        original = str(annot["description"])
        normalized = unicodedata.normalize("NFKD", original)
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = re.sub(r"\s+", " ", normalized.lower().strip())
        if not normalized:
            continue
        elif _matches_any(normalized, IGNORE_LABEL_PATTERNS):
            continue
        elif normalized == config.RECORDING_START_LABEL:
            standardized = "recording_start"
        elif _matches_any(normalized, config.BASIC_1020_CHANNELS) and _matches_any(normalized, SENSOR_LABEL_PATTERNS):
            standardized = "BAD_sensor_artefact"
        elif _matches_any(normalized, config.EYES_OPEN_LABELS):
            standardized = "eyes_open"
        elif _matches_any(normalized, config.EYES_CLOSED_LABELS):
            standardized = "eyes_closed"
        elif _matches_any(normalized, config.POST_HV_LABELS):
            standardized = "post_hv"
        elif _matches_any(normalized, config.HV_LABELS):
            if "start" in normalized or "debut" in normalized:
                standardized = "hv_start"
            elif "end" in normalized or re.search(r"\bfin\b", normalized):
                standardized = "hv_end"
            else:
                standardized = None
        elif _matches_any(normalized, config.PHOTO_LABELS):
            freq = get_photo_freq(original)
            standardized = "photo" if freq is None else f"photo_{freq}hz"
        else:
            standardized = None
            for pattern, label in BAD_INTEREST_LABELS:
                if _matches_pattern(normalized, pattern):
                    standardized = label
                    break
            if standardized is None:
                for pattern, label in CLINICAL_LABELS:
                    if _matches_pattern(normalized, pattern):
                        standardized = label
                        break

        if standardized is None:
            continue

        new_onset.append(annot["onset"])
        new_duration.append(annot["duration"])
        new_descs.append(standardized)

    raw.set_annotations(
        mne.Annotations(
            onset=new_onset,
            duration=new_duration,
            description=new_descs,
            orig_time=raw.annotations.orig_time,
        )
    )
    return raw


def get_photo_freq(desc: str | None) -> int | None:
    if not desc:
        return None
    match = config.PHOTO_FREQ_PATTERN.search(desc)
    if not match:
        return None
    try:
        return int(float(match.group(1)))
    except (TypeError, ValueError):
        return None


def _build_segment_record(
    segment_type: str,
    block_family: str,
    eye_state: str,
    start: float,
    stop: float,
    freq_hz: float | None = np.nan,
) -> dict[str, object]:
    return {
        "segment_type": segment_type,
        "block_family": block_family,
        "eye_state": eye_state,
        "t_start": start,
        "t_stop": stop,
        "duration": stop - start,
        "freq_hz": freq_hz if freq_hz is not None else np.nan,
    }


def _build_eye_state_intervals(
    entries: Sequence[dict[str, object]], raw_end: float
) -> list[tuple[float, float, str]]:
    eye_events = [
        (float(entry["onset"]), "eo" if str(entry["description"]) == "eyes_open" else "ec")
        for entry in entries
        if str(entry["description"]) in {"eyes_open", "eyes_closed"}
    ]
    if not eye_events:
        return []
    intervals: list[tuple[float, float, str]] = []
    for (start, state), (next_start, _) in zip(eye_events, eye_events[1:]):
        if next_start > start:
            intervals.append((start, next_start, state))
    last_start, last_state = eye_events[-1]
    if raw_end > last_start:
        intervals.append((last_start, raw_end, last_state))
    return intervals


def _find_hv_blocks(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    hv_blocks: list[dict[str, object]] = []
    hv_start: float | None = None
    for entry in entries:
        clean = str(entry["description"])
        onset = float(entry["onset"])
        if clean == "hv_start":
            hv_start = onset
        elif clean == "hv_end" and hv_start is not None and onset > hv_start:
            hv_blocks.append(
                {
                    "t_start": hv_start,
                    "t_stop": onset,
                }
            )
            hv_start = None
    return hv_blocks


def _find_photo_blocks(entries: Sequence[dict[str, object]], raw_end: float) -> list[dict[str, object]]:
    photo_entries = [
        entry
        for entry in entries
        if str(entry["description"]) == "photo" or str(entry["description"]).startswith("photo_")
    ]
    blocks: list[dict[str, object]] = []
    for pos, entry in enumerate(photo_entries):
        onset = float(entry["onset"])
        description = str(entry["description"])
        next_start = float(photo_entries[pos + 1]["onset"]) if pos + 1 < len(photo_entries) else raw_end
        if next_start <= onset:
            continue
        blocks.append(
            {
                "t_start": onset,
                "t_stop": next_start,
                "freq_hz": get_photo_freq(description),
            }
        )
    return blocks


def _compute_post_hv_blocks(
    hv_blocks: list[dict[str, object]],
    photo_blocks: list[dict[str, object]],
    entries: Sequence[dict[str, object]],
    raw_end: float,
) -> list[dict[str, object]]:
    post_hv_markers = [float(entry["onset"]) for entry in entries if str(entry["description"]) == "post_hv"]
    post_blocks: list[dict[str, object]] = []
    constraints = sorted(
        [float(block["t_start"]) for block in hv_blocks + photo_blocks if float(block["t_start"]) > 0.0]
        + [raw_end]
    )
    for hv_block in hv_blocks:
        start = float(hv_block["t_stop"])
        hv_duration = float(hv_block["t_stop"]) - float(hv_block["t_start"])
        limit = next((value for value in constraints if value > start + 0.1), raw_end)
        markers = [onset for onset in post_hv_markers if start <= onset < limit]
        if markers:
            stop = markers[-1]
        else:
            stop = min(start + 0.25 * hv_duration, limit)
        if stop > start + 1.0:
            post_blocks.append(
                {
                    "t_start": start,
                    "t_stop": stop,
                }
            )
    return post_blocks


def _subtract_interval(
    base_start: float, base_stop: float, exclusions: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    if base_stop <= base_start:
        return []
    keep: list[tuple[float, float]] = []
    cursor = base_start
    for ex_start, ex_stop in exclusions:
        if ex_stop <= cursor:
            continue
        if ex_start >= base_stop:
            break
        if ex_start > cursor:
            keep.append((cursor, ex_start))
        cursor = max(cursor, ex_stop)
        if cursor >= base_stop:
            break
    if cursor < base_stop:
        keep.append((cursor, base_stop))
    return keep


def _segment_eye_states_within_interval(
    start: float,
    stop: float,
    eye_states: Sequence[tuple[float, float, str]],
    segment_prefix: str,
    block_family: str,
    freq_hz: float | None = np.nan,
) -> list[dict[str, object]]:
    segments = [
        _build_segment_record(
            segment_type=f"{segment_prefix}_{state.upper()}",
            block_family=block_family,
            eye_state=state,
            start=max(start, state_start),
            stop=min(stop, state_stop),
            freq_hz=freq_hz,
        )
        for state_start, state_stop, state in eye_states
        if state_stop > start and state_start < stop and min(stop, state_stop) > max(start, state_start)
    ]
    if segments or stop <= start:
        return segments
    return [
        _build_segment_record(
            segment_type=f"{segment_prefix}_UNKNOWN",
            block_family=block_family,
            eye_state="unknown",
            start=start,
            stop=stop,
            freq_hz=freq_hz,
        )
    ]


def extract_condition_segments(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Compute EO/EC baseline plus HV/PostHV/PHOTO segments split by eye state."""
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    entries = [
        {"onset": float(onset), "description": str(desc)}
        for onset, desc in zip(annotations.onset, annotations.description)
        if desc is not None
    ]

    raw_end = 0.0
    if raw.n_times:
        raw_end = float(raw.times[-1])
        sfreq = float(raw.info.get("sfreq") or 0.0)
        if sfreq > 0.0:
            raw_end += 1.0 / sfreq
    eye_states = _build_eye_state_intervals(entries, raw_end)
    hv_blocks = _find_hv_blocks(entries)
    photo_blocks = _find_photo_blocks(entries, raw_end)
    post_hv_blocks = _compute_post_hv_blocks(hv_blocks, photo_blocks, entries, raw_end)
    exclusion_intervals = bids_io.merge_intervals([
        (block["t_start"], block["t_stop"])
        for block in (*hv_blocks, *post_hv_blocks, *photo_blocks)
        if block["t_stop"] > block["t_start"]
    ])
    block_specs = (
        ("HV", "hv", hv_blocks),
        ("PostHV", "post_hv", post_hv_blocks),
        ("PHOTO", "photo", photo_blocks),
    )

    records: list[dict[str, object]] = []
    first_eye_start = eye_states[0][0] if eye_states else raw_end
    if first_eye_start > 0.0:
        records.extend(
            _build_segment_record(
                segment_type="RAW_baseline",
                block_family="raw_baseline",
                eye_state="unknown",
                start=seg_start,
                stop=seg_stop,
            )
            for seg_start, seg_stop in _subtract_interval(0.0, first_eye_start, exclusion_intervals)
        )

    for start, stop, state in eye_states:
        records.extend(
            _build_segment_record(
                segment_type="EO_baseline" if state == "eo" else "EC_baseline",
                block_family="baseline",
                eye_state=state,
                start=seg_start,
                stop=seg_stop,
            )
            for seg_start, seg_stop in _subtract_interval(start, stop, exclusion_intervals)
        )

    for segment_prefix, block_family, blocks in block_specs:
        for block in blocks:
            records.extend(
                _segment_eye_states_within_interval(
                    block["t_start"],
                    block["t_stop"],
                    eye_states,
                    segment_prefix=segment_prefix,
                    block_family=block_family,
                    freq_hz=block.get("freq_hz", np.nan),
                )
            )

    return pd.DataFrame.from_records(records, columns=SEGMENT_COLUMNS).sort_values(
        by=["t_start", "segment_type"]
    ).reset_index(drop=True)


def _eeg_event_counts(
    raw: mne.io.BaseRaw,
    segments_df: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, int]:
    raw_counts = utils_events.summarize_annotations(raw)
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
            clean_desc in {"eyes_open", "eyes_closed", "hv_start", "hv_end", "post_hv", "recording_start"}
            or clean_desc == "photo"
            or clean_desc.startswith("photo_")
            or str(desc).startswith("BLOCK_")
        ):
            continue
        event_counts[desc] = event_counts.get(desc, 0) + int(count)
    return event_counts


def _clean_scalar(value: object) -> object:
    return None if pd.isna(value) else value


def _record_value(record: object, key: str) -> object:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key)


def _build_run_summary_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "subject_id": record["subject_id"],
        "session_id": record["session_id"],
        "run_id": record["run_id"],
        "subject_session_prefix": record["subject_session_prefix"],
        "run_prefix": record["run_prefix"],
        "study_id": record["study_id"],
        "source_dataset": record["source_dataset"],
        "record_date": record["record_date"],
        "meas_datetime": record["meas_datetime"],
        "filepath": record["filepath"],
        "raw_duration": float(record["raw_duration"]),
        "n_channels": int(record["n_channels"]),
        "age_group": record["age_group"],
        "sex": record["sex"],
        "combined_diagnosis": record["combined_diagnosis"],
        **record["summary"],
    }


def _build_subject_summary_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "subject_id": record["subject_id"],
        "session_id": record["session_id"],
        "subject_session_prefix": record["subject_session_prefix"],
        "study_id": record["study_id"],
        "source_dataset": record["source_dataset"],
        "raw_duration": float(record["raw_duration"]),
        "n_runs": int(record["n_runs"]),
        "age_group": record["age_group"],
        "sex": record["sex"],
        "combined_diagnosis": record["combined_diagnosis"],
        **record["summary"],
    }


def _build_eeg_report_record(
    *,
    ids: dict[str, object],
    record,
    metadata: dict[str, object] | None,
    raw: mne.io.BaseRaw,
    segments_df: pd.DataFrame,
    summary: dict[str, object],
    event_counts: dict[str, int],
    raw_duration: float,
) -> dict[str, object]:
    metadata = metadata or {}
    eeg_record = {
        **ids,
        "study_id": int(_record_value(record, "study_id")),
        "source_dataset": _clean_scalar(metadata.get("source_dataset")) or _clean_scalar(_record_value(record, "source_dataset")),
        "record_date": _clean_scalar(_record_value(record, "record_date")),
        "meas_datetime": _clean_scalar(_record_value(record, "meas_datetime")),
        "filepath": str(ids.get("filepath") or ""),
        "raw_duration": float(raw_duration),
        "n_channels": int(sum(channel not in {"A1", "A2"} for channel in raw.ch_names)),
        "age_group": _clean_scalar(metadata.get("age_group")),
        "sex": _clean_scalar(metadata.get("sex")),
        "combined_diagnosis": _clean_scalar(metadata.get("combined_diagnosis")),
        "segments_df": segments_df,
        "summary": summary,
        "event_counts": event_counts,
    }
    eeg_record["summary_row"] = _build_run_summary_row(eeg_record)
    return eeg_record


def _collect_existing_eeg_report_record(
    bids_path: BIDSPath,
    bids_root: Path,
    record,
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    try:
        raw = bids_io.load_bids_raw(filepath=bids_path.fpath, bids_root=bids_root)
        segments_df = bids_io.load_segments_for_raw(raw)
        ids = bids_io.build_bids_report_ids(bids_path.fpath)
        ids["filepath"] = str(bids_path.fpath)
        summary = report_eeg.summarize_condition_segments(segments_df)
        event_counts = _eeg_event_counts(raw, segments_df, summary)
        raw_duration = raw.times[-1] if raw.n_times > 0 else 0.0
        return _build_eeg_report_record(
            ids=ids,
            record=record,
            metadata=metadata,
            raw=raw,
            segments_df=segments_df,
            summary=summary,
            event_counts=event_counts,
            raw_duration=raw_duration,
        )
    except Exception as exc:
        LOGGER.error("Failed collecting existing EEG report payload for %s: %s", bids_path.fpath, exc)
        return None


def _missingness_payload(records: list[dict[str, object]], label_key: str) -> dict[str, list[str]]:
    payload = {
        "no_conditions": [],
        "no_eyes_open": [],
        "no_eyes_closed": [],
        "no_hv": [],
        "no_photo": [],
    }
    for record in records:
        label = str(record[label_key])
        summary = record["summary"]
        if (
            summary.get("total_eyes_open_duration", 0) <= 0
            and summary.get("total_eyes_closed_duration", 0) <= 0
            and summary.get("hv_block_count", 0) == 0
            and summary.get("photo_block_count", 0) == 0
        ):
            payload["no_conditions"].append(label)
        if summary.get("total_eyes_open_duration", 0) <= 0:
            payload["no_eyes_open"].append(label)
        if summary.get("total_eyes_closed_duration", 0) <= 0:
            payload["no_eyes_closed"].append(label)
        if summary.get("hv_block_count", 0) == 0:
            payload["no_hv"].append(label)
        if summary.get("photo_block_count", 0) == 0:
            payload["no_photo"].append(label)
    return payload


def _write_subject_eeg_report(
    reports_root: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    ids = records[0]
    subject_prefix = str(ids["subject_session_prefix"])
    subject_dir = bids_io.get_subject_session_stage_dir(
        reports_root,
        str(ids["subject_id"]),
        str(ids["session_id"]),
        "eeg_pre_base",
        create_dir=True,
    )
    fig_dir = subject_dir / "figures"
    segments_df = pd.concat(
        [record["segments_df"].assign(run_id=record["run_id"]) for record in records],
        ignore_index=True,
    )
    summary = report_eeg.summarize_condition_segments(segments_df)
    event_counter: Counter = Counter()
    for record in records:
        event_counter.update(record["event_counts"] or {})
    event_counts = dict(event_counter)
    raw_duration = float(sum(float(record["raw_duration"]) for record in records))
    figure_paths = viz_eeg.save_eeg_report_figures(segments_df, fig_dir)
    report_path = bids_io.get_subject_session_stage_report_path(
        reports_root=reports_root,
        subject_id=str(ids["subject_id"]),
        session_id=str(ids["session_id"]),
        stage="eeg_pre_base",
        report_stem=subject_prefix,
        create_dir=True,
    )
    subject_record = {
        "subject_id": str(ids["subject_id"]),
        "session_id": str(ids["session_id"]),
        "subject_session_prefix": subject_prefix,
        "study_id": ids["study_id"],
        "source_dataset": ids["source_dataset"],
        "summary": summary,
        "event_counts": event_counts,
        "raw_duration": raw_duration,
        "n_runs": len(records),
        "age_group": ids["age_group"],
        "sex": ids["sex"],
        "combined_diagnosis": ids["combined_diagnosis"],
    }
    run_inventory_df = pd.DataFrame(
        [
            {
                "Run": record["run_id"],
                "Recording Date": record["record_date"],
                "Recording Time": format_clock_time(record["meas_datetime"]),
                "Duration": format_duration_hms(record["raw_duration"]),
                "EEG Channels": int(record["n_channels"]),
            }
            for record in records
        ]
    )
    run_summary_df = pd.DataFrame(
        [
            {
                "Run": record["run_id"],
                "Analysis Duration": format_duration_hms(record["summary"]["total_duration"]),
                "EO": format_duration_hms(record["summary"]["total_eyes_open_duration"]),
                "EC": format_duration_hms(record["summary"]["total_eyes_closed_duration"]),
                "HV Blocks": int(record["summary"]["hv_block_count"]),
                "PHOTO Blocks": int(record["summary"]["photo_block_count"]),
            }
            for record in records
        ]
    )
    report_eeg.generate_eeg_subject_report(
        record=subject_record,
        run_inventory_df=run_inventory_df.sort_values("Run") if len(records) > 1 else pd.DataFrame(),
        run_summary_df=run_summary_df.sort_values("Run") if len(records) > 1 else pd.DataFrame(),
        figure_paths=figure_paths,
        output_path=report_path,
    )
    return subject_record


def _write_eeg_aggregate_reports(
    reports_root: Path,
    run_records: list[dict[str, object]],
) -> None:
    if not run_records:
        return

    summary_dir = bids_io.get_stage_summary_dir(reports_root, "eeg_pre_base", create_dir=True)

    runs_df = pd.DataFrame([record["summary_row"] for record in run_records]).sort_values(
        ["subject_id", "session_id", "run_id", "filepath"],
        na_position="last",
    )
    runs_df.to_csv(summary_dir / "eeg_runs.csv", index=False)

    subject_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in run_records:
        subject_groups[record["subject_session_key"]].append(record)

    subject_records: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for (_subject_id, _session_id), records in sorted(subject_groups.items()):
        subject_record = _write_subject_eeg_report(reports_root, records)
        subject_records.append(subject_record)
        subject_rows.append(_build_subject_summary_row(subject_record))

    subjects_df = pd.DataFrame(subject_rows).sort_values(
        ["subject_id", "session_id"],
        na_position="last",
    )
    subjects_df.to_csv(summary_dir / "eeg_subjects.csv", index=False)

    dataset_tables = report_eeg.build_dataset_report_tables(runs_df, subjects_df, run_records)
    dataset_tables["dataset_summary_df"].to_csv(summary_dir / "eeg_dataset_summary.csv", index=False)
    figure_paths = viz_eeg.save_dataset_eeg_figures(
        runs_df,
        [record["event_counts"] for record in run_records],
        summary_dir,
    )
    report_eeg.generate_eeg_dataset_report(
        tables=dataset_tables,
        figure_paths=figure_paths,
        output_path=summary_dir / "eeg_pre_base_dataset_report.html",
    )

    missing_export = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_runs_processed": len(run_records),
            "total_subject_sessions": len(subject_records),
            "total_subjects": int(runs_df["subject_id"].nunique()) if not runs_df.empty else 0,
        },
        "runs": _missingness_payload(run_records, "run_prefix"),
        "subjects": _missingness_payload(subject_records, "subject_session_prefix"),
    }
    with open(summary_dir / "eeg_missingness.json", "w") as f:
        json.dump(missing_export, f, indent=2)

def process_record(
    record,
    bids_root: Path,
    overwrite: bool = False,
    metadata: dict[str, object] | None = None,
    eeg_reports_dir: Path | None = None,
    raw_qc_reports_dir: Path | None = None,
    raw_qc_analysis_level: str = "both",
) -> dict[str, object]:
    """Read, standardize, and export one selected recording to BIDS."""
    eeg_path = Path(_record_value(record, "eeg_path"))
    study_id = int(_record_value(record, "study_id"))
    run = str(_record_value(record, "run"))
    subject_id = bids_io.normalize_subject_id(f"{study_id:04d}")
    bids_path = BIDSPath(
        root=str(bids_root),
        subject=subject_id[4:],
        session="01",
        task="clinical",
        run=run,
        suffix="eeg",
        extension=".vhdr",
    )
    result = {
        "study_id": study_id,
        "success": False,
        "eeg_report_record": None,
        "raw_qc_record": None,
    }

    if not overwrite and bids_path.fpath.exists():
        LOGGER.info("Skipping %s run-%s (exists)", subject_id, run)
        if eeg_reports_dir is not None:
            result["eeg_report_record"] = _collect_existing_eeg_report_record(
                bids_path=bids_path,
                bids_root=bids_root,
                record=record,
                metadata=metadata,
            )
            if result["eeg_report_record"] is None:
                return result
        if raw_qc_reports_dir is not None:
            try:
                result["raw_qc_record"] = qc_raw.collect_existing_raw_qc_record(
                    bids_path=bids_path,
                    bids_root=bids_root,
                    metadata=metadata,
                    analysis_level=raw_qc_analysis_level,
                )
            except Exception as exc:
                LOGGER.error("Failed collecting existing raw QC payload for %s: %s", bids_path.fpath, exc)
                return result
        result["success"] = True
        return result

    try:
        raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    except Exception as exc:
        LOGGER.error("Failed to read EEG %s: %s", eeg_path, exc)
        return result

    raw.info["line_freq"] = 60
    meas_datetime = pd.to_datetime(_record_value(record, "meas_datetime"), errors="coerce")
    if pd.notna(meas_datetime):
        raw.set_meas_date(meas_datetime.to_pydatetime())

    raw = canonicalize_annotations(raw)
    segments_df = extract_condition_segments(raw)
    blocks = segments_df.loc[
        segments_df["segment_type"].notna() & segments_df["t_stop"].gt(segments_df["t_start"])
    ]
    if not blocks.empty:
        raw.set_annotations(
            raw.annotations
            + mne.Annotations(
                onset=blocks["t_start"].tolist(),
                duration=(blocks["t_stop"] - blocks["t_start"]).tolist(),
                description=("BLOCK_" + blocks["segment_type"].astype(str)).tolist(),
                orig_time=raw.annotations.orig_time,
            )
        )

    available_targets = [channel for channel in config.BASIC_1020_CHANNELS if channel in raw.ch_names]
    if available_targets:
        raw.pick(available_targets)
    try:
        write_raw_bids(
            raw,
            bids_path=bids_path,
            format="BrainVision",
            overwrite=overwrite,
            allow_preload=False,
            verbose=False,
        )
    except Exception as exc:
        LOGGER.error("Failed writing %s run-%s: %s", subject_id, run, exc)
        return result

    stem = bids_path.fpath.stem[:-4] if bids_path.fpath.stem.endswith("_eeg") else bids_path.fpath.stem
    segments_path = bids_path.fpath.parent / f"{stem}_segments.csv"
    segments_df.to_csv(segments_path, index=False)
    LOGGER.info("Wrote condition segments for %s run-%s to %s", subject_id, run, segments_path)
    if eeg_reports_dir is not None:
        try:
            ids = bids_io.build_bids_report_ids(bids_path.fpath)
            ids["filepath"] = str(bids_path.fpath)
            summary = report_eeg.summarize_condition_segments(segments_df)
            event_counts = _eeg_event_counts(raw, segments_df, summary)
            raw_duration = raw.times[-1] if raw.n_times > 0 else 0.0
            result["eeg_report_record"] = _build_eeg_report_record(
                ids=ids,
                record=record,
                metadata=metadata,
                raw=raw,
                segments_df=segments_df,
                summary=summary,
                event_counts=event_counts,
                raw_duration=raw_duration,
            )
        except Exception as exc:
            LOGGER.error("Failed generating EEG report for %s run-%s: %s", subject_id, run, exc)
            return result
    if raw_qc_reports_dir is not None:
        try:
            summary = report_eeg.summarize_condition_segments(segments_df)
            result["raw_qc_record"] = qc_raw.build_raw_qc_run_record(
                raw=raw,
                bids_path=bids_path,
                condition_segments_df=segments_df,
                condition_summary=summary,
                metadata=metadata,
                analysis_level=raw_qc_analysis_level,
            )
        except Exception as exc:
            LOGGER.error("Failed generating raw QC payload for %s run-%s: %s", subject_id, run, exc)
            return result
    result["success"] = True
    LOGGER.info("Converted %s run-%s", subject_id, run)
    return result


def _consume_record_result(
    record: dict[str, object],
    record_result: dict[str, object],
    *,
    failed_ids: set[int],
    successful_ids: set[int],
    eeg_run_records: list[dict[str, object]],
    raw_qc_run_records: list[dict[str, object]],
) -> None:
    study_id = int(record["study_id"])
    if not record_result["success"]:
        failed_ids.add(study_id)
        return

    successful_ids.add(study_id)
    if record_result["eeg_report_record"] is not None:
        eeg_run_records.append(record_result["eeg_report_record"])
    if record_result["raw_qc_record"] is not None:
        raw_qc_run_records.append(record_result["raw_qc_record"])


def _resolve_n_jobs(n_jobs: int) -> int:
    """Normalize CLI worker count while preserving joblib's ``-1`` semantics."""
    if n_jobs == -1:
        return cpu_count()
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("--n_jobs must be -1 or a positive integer")
    return max(1, int(n_jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG -> BIDS converter")
    parser.add_argument("--raw_root", type=Path, required=True, help="Root directory containing raw_data")
    parser.add_argument("--bids_root", type=Path, required=True, help="BIDS root directory")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Canonical metadata CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS subject folders")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for bidsification.")
    parser.add_argument(
        "--with_eeg_reports",
        action="store_true",
        help="Generate EEG reports from the written segment CSVs after bidsification",
    )
    parser.add_argument(
        "--with_raw_qc",
        action="store_true",
        help="Generate raw QC reports from the written BIDS runs after bidsification",
    )
    parser.add_argument(
        "--raw_qc_analysis_level",
        choices=["whole", "segments", "both"],
        default="both",
        help="Raw QC analysis level when --with_raw_qc is used.",
    )
    args = parser.parse_args()
    reports_root = bids_io.get_reports_root(bids_root=args.bids_root)
    eeg_reports_dir = reports_root if args.with_eeg_reports else None
    raw_qc_reports_dir = reports_root if args.with_raw_qc else None

    metadata_df = pd.read_csv(args.metadata_csv)
    metadata_df["study_id"] = pd.to_numeric(metadata_df["study_id"], errors="coerce").astype("Int64")
    LOGGER.info("Loaded metadata CSV with %d rows", len(metadata_df))
    metadata_lookup = {
        int(row.study_id): {
            "source_dataset": _clean_scalar(getattr(row, "source_dataset", None)),
            "age_group": _clean_scalar(getattr(row, "age_group", None)),
            "sex": _clean_scalar(getattr(row, "sex", None)),
            "combined_diagnosis": _clean_scalar(getattr(row, "combined_diagnosis", None)),
        }
        for row in metadata_df[
            ["study_id", "source_dataset", "age_group", "sex", "combined_diagnosis"]
        ].dropna(subset=["study_id"]).drop_duplicates("study_id").itertuples(index=False)
    }

    inventory_df = pd.DataFrame.from_records(
        ingest.discover_raw_records(args.raw_root, metadata_df),
        columns=[
            "source_dataset",
            "study_id",
            "patient_id",
            "resolved_by",
            "record_stem",
            "pnt_path",
            "eeg_path",
            "meas_datetime",
            "record_date",
        ],
    )
    inventory_df["study_id"] = pd.to_numeric(inventory_df["study_id"], errors="coerce").astype("Int64")
    inventory_df["run"] = pd.Series([None] * len(inventory_df), dtype=object)

    selected_rows = inventory_df.loc[
        inventory_df["study_id"].isin(metadata_df["study_id"]) & inventory_df["eeg_path"].notna()
    ].copy()
    if not selected_rows.empty:
        selected_rows = selected_rows.assign(
            record_date_dt=pd.to_datetime(selected_rows["record_date"], errors="coerce"),
            meas_datetime_dt=pd.to_datetime(selected_rows["meas_datetime"], errors="coerce"),
        )
        selected_rows = selected_rows.sort_values(
            ["study_id", "record_date_dt", "meas_datetime_dt", "record_stem"],
            na_position="last",
        )
        inventory_df.loc[selected_rows.index, "run"] = (
            selected_rows.groupby("study_id").cumcount().add(1).map(lambda value: f"{value:02d}")
        )

    args.bids_root.mkdir(parents=True, exist_ok=True)
    inventory_path = args.bids_root / "raw_record_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    LOGGER.info("Wrote inventory to %s", inventory_path)

    if args.overwrite:
        for study_id in sorted(inventory_df.loc[inventory_df["run"].notna(), "study_id"].dropna().astype(int).unique()):
            subject_id = bids_io.normalize_subject_id(f"{int(study_id):04d}")
            sub_dir = args.bids_root / subject_id
            if sub_dir.exists():
                LOGGER.info("Overwriting %s", subject_id)
                shutil.rmtree(sub_dir)

    failed_ids: set[int] = set()
    successful_ids: set[int] = set()
    eeg_run_records: list[dict[str, object]] = []
    raw_qc_run_records: list[dict[str, object]] = []
    selected_records = inventory_df.loc[inventory_df["run"].notna()].to_dict("records")
    n_jobs = _resolve_n_jobs(args.n_jobs)
    LOGGER.info("Using %d worker(s) for BIDS conversion", n_jobs)
    if n_jobs == 1:
        for record in tqdm(selected_records, total=len(selected_records), desc="Converting records"):
            record_result = process_record(
                record,
                args.bids_root,
                overwrite=args.overwrite,
                metadata=metadata_lookup.get(int(record["study_id"]), {}),
                eeg_reports_dir=eeg_reports_dir,
                raw_qc_reports_dir=raw_qc_reports_dir,
                raw_qc_analysis_level=args.raw_qc_analysis_level,
            )
            _consume_record_result(
                record,
                record_result,
                failed_ids=failed_ids,
                successful_ids=successful_ids,
                eeg_run_records=eeg_run_records,
                raw_qc_run_records=raw_qc_run_records,
            )
    else:
        with tqdm_joblib(tqdm(total=len(selected_records), desc="Converting records")):
            record_results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=1,
                pre_dispatch=n_jobs,
            )(
                delayed(process_record)(
                    record,
                    args.bids_root,
                    args.overwrite,
                    metadata_lookup.get(int(record["study_id"]), {}),
                    eeg_reports_dir,
                    raw_qc_reports_dir,
                    args.raw_qc_analysis_level,
                )
                for record in selected_records
            )
        for record, record_result in zip(selected_records, record_results):
            if record_result is None:
                LOGGER.error("Failed processing study_id %s: no result returned", record["study_id"])
                failed_ids.add(int(record["study_id"]))
                continue
            try:
                _consume_record_result(
                    record,
                    record_result,
                    failed_ids=failed_ids,
                    successful_ids=successful_ids,
                    eeg_run_records=eeg_run_records,
                    raw_qc_run_records=raw_qc_run_records,
                )
            except Exception as exc:
                LOGGER.error("Failed consuming result for study_id %s: %s", record["study_id"], exc)
                failed_ids.add(int(record["study_id"]))

    if failed_ids:
        LOGGER.warning("Failed study_ids: %s", sorted(failed_ids))
    else:
        LOGGER.info("All selected recordings converted successfully.")

    if successful_ids:
        converted_meta = metadata_df[metadata_df["study_id"].isin(sorted(successful_ids))].copy()
        participants_df = converted_meta[["study_id", "age", "sex"]].dropna(subset=["study_id"]).copy()
        participants_df["participant_id"] = participants_df["study_id"].apply(
            lambda value: bids_io.normalize_subject_id(f"{int(value):04d}")
        )
        participants_df = (
            participants_df[["participant_id", "age", "sex"]]
            .drop_duplicates("participant_id")
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        participants_df.to_csv(args.bids_root / "participants.tsv", sep="\t", index=False)
        if args.with_eeg_reports:
            LOGGER.info("Generating EEG aggregate reports in %s", eeg_reports_dir)
            _write_eeg_aggregate_reports(eeg_reports_dir, eeg_run_records)
        if args.with_raw_qc:
            LOGGER.info("Generating raw QC aggregate reports in %s", raw_qc_reports_dir)
            qc_raw.write_raw_qc_aggregate_reports(raw_qc_reports_dir, raw_qc_run_records)


if __name__ == "__main__":
    main()

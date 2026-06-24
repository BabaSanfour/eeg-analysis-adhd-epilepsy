"""
Convert raw EEG + canonical metadata to BIDS with standardized annotations.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import re
import shutil
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from mne_bids import BIDSPath, write_raw_bids
from tqdm import tqdm

import eeg_adhd_epilepsy.qc.raw_qc as qc_raw
import eeg_adhd_epilepsy.reports.eeg_report as report_eeg
from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.io import ingest, readers, report_paths
from eeg_adhd_epilepsy.reports._common import clean_scalar
from eeg_adhd_epilepsy.utils import constants, events
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib

LOGGER = logging.getLogger(__name__)
SEGMENT_COLUMNS = list(constants.SEGMENT_COLUMNS)


def _slug_label(label: str) -> str:
    return label.lower().replace(" - ", "_").replace("/", "_").replace(" ", "_").replace("-", "_")


IGNORE_LABEL_PATTERNS = (*constants.IGNORE_PATTERNS, *constants.IGNORED_LABELS)
SENSOR_LABEL_PATTERNS = (*constants.SENSOR_ARTEFACT_KEYWORDS, *constants.SENSOR_ACTION_KEYWORDS)
BAD_INTEREST_LABELS = tuple(
    (pattern, f"BAD_{_slug_label(category)}")
    for category, patterns in constants.ANNOTATION_INTEREST_MAP.items()
    if _slug_label(category) not in {"eyes_open", "eyes_closed", "hv", "post_hv", "photo"}
    for pattern in patterns
    if pattern
)
CLINICAL_LABELS = tuple(
    (pattern, _slug_label(category))
    for category, patterns in constants.CLINICAL_COMMENT_LABELS.items()
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
        elif normalized == constants.RECORDING_START_LABEL:
            standardized = "recording_start"
        elif _matches_any(normalized, constants.BASIC_1020_CHANNELS) and _matches_any(
            normalized, SENSOR_LABEL_PATTERNS
        ):
            standardized = "BAD_sensor_artefact"
        elif _matches_any(normalized, constants.EYES_OPEN_LABELS):
            standardized = "eyes_open"
        elif _matches_any(normalized, constants.EYES_CLOSED_LABELS):
            standardized = "eyes_closed"
        elif _matches_any(normalized, constants.POST_HV_LABELS):
            standardized = "post_hv"
        elif _matches_any(normalized, constants.HV_LABELS):
            if "start" in normalized or "debut" in normalized:
                standardized = "hv_start"
            elif "end" in normalized or re.search(r"\bfin\b", normalized):
                standardized = "hv_end"
            else:
                standardized = None
        elif _matches_any(normalized, constants.PHOTO_LABELS):
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
    match = constants.PHOTO_FREQ_PATTERN.search(desc)
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
    freq_hz: float | None = None,
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


def _find_photo_blocks(
    entries: Sequence[dict[str, object]], raw_end: float
) -> list[dict[str, object]]:
    photo_entries = [
        entry
        for entry in entries
        if str(entry["description"]) == "photo" or str(entry["description"]).startswith("photo_")
    ]
    blocks: list[dict[str, object]] = []
    for pos, entry in enumerate(photo_entries):
        onset = float(entry["onset"])
        description = str(entry["description"])
        next_start = (
            float(photo_entries[pos + 1]["onset"]) if pos + 1 < len(photo_entries) else raw_end
        )
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
    post_hv_markers = [
        float(entry["onset"]) for entry in entries if str(entry["description"]) == "post_hv"
    ]
    post_blocks: list[dict[str, object]] = []
    constraints = sorted(
        [
            float(block["t_start"])
            for block in hv_blocks + photo_blocks
            if float(block["t_start"]) > 0.0
        ]
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
        if state_stop > start
        and state_start < stop
        and min(stop, state_stop) > max(start, state_start)
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
    exclusion_intervals = events.merge_intervals(
        [
            (block["t_start"], block["t_stop"])
            for block in (*hv_blocks, *post_hv_blocks, *photo_blocks)
            if block["t_stop"] > block["t_start"]
        ]
    )
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

    return (
        pd.DataFrame.from_records(records, columns=SEGMENT_COLUMNS)
        .sort_values(by=["t_start", "segment_type"])
        .reset_index(drop=True)
    )


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
    eeg_path = Path(record["eeg_path"])
    study_id = int(record["study_id"])
    run = str(record["run"])
    subject = bids_io.study_id_to_bids_subject(study_id)
    subject_id = bids_io.bids_subject_label(subject)
    bids_path = BIDSPath(
        root=str(bids_root),
        subject=subject,
        session="01",
        task="clinical",
        run=run,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    result = {
        "study_id": study_id,
        "success": False,
        "skipped": False,
        "eeg_report_record": None,
        "raw_qc_record": None,
    }

    if not overwrite and bids_path.fpath.exists():
        LOGGER.info("Skipping %s run-%s (exists)", subject_id, run)
        result["skipped"] = True
        result["success"] = True
        if eeg_reports_dir is None and raw_qc_reports_dir is None:
            return result
        try:
            existing_raw = readers.read_bids_raw(
                bids_root=bids_root,
                subject=bids_path.subject,
                task=bids_path.task,
                session=bids_path.session,
                run=bids_path.run,
            )
            existing_segments = events.segments_from_block_annotations(existing_raw)
        except Exception as exc:
            LOGGER.warning(
                "Could not load existing BIDS run for %s: %s", subject_id, exc, exc_info=True
            )
            return result
        if eeg_reports_dir is not None:
            result["eeg_report_record"] = report_eeg.build_eeg_run_record(
                raw=existing_raw,
                bids_path=bids_path,
                segments_df=existing_segments,
                record=record,
                metadata=metadata,
            )
        if raw_qc_reports_dir is not None:
            existing_summary = report_eeg.summarize_condition_segments(existing_segments)
            result["raw_qc_record"] = qc_raw.build_raw_qc_run_record(
                raw=existing_raw,
                bids_path=bids_path,
                condition_segments_df=existing_segments,
                condition_summary=existing_summary,
                metadata=metadata,
                analysis_level=raw_qc_analysis_level,
            )
        return result

    try:
        raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    except Exception as exc:
        LOGGER.error("Failed to read EEG %s: %s", eeg_path, exc, exc_info=True)
        return result

    raw.info["line_freq"] = 60
    meas_datetime = pd.to_datetime(record["meas_datetime"], errors="coerce")
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

    available_targets = [
        channel for channel in constants.BASIC_1020_CHANNELS if channel in raw.ch_names
    ]
    if len(available_targets) < len(constants.BASIC_1020_CHANNELS):
        missing_targets = [
            channel for channel in constants.BASIC_1020_CHANNELS if channel not in raw.ch_names
        ]
        LOGGER.warning(
            "Skipping %s run-%s: expected %d canonical channels, found %d. Missing: %s",
            subject_id,
            run,
            len(constants.BASIC_1020_CHANNELS),
            len(available_targets),
            missing_targets,
        )
        result["skipped"] = True
        result["success"] = True
        return result
    if available_targets:
        raw.pick(available_targets)
    if raw.ch_names:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="raise")
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
        LOGGER.error("Failed writing %s run-%s: %s", subject_id, run, exc, exc_info=True)
        return result

    if eeg_reports_dir is not None:
        result["eeg_report_record"] = report_eeg.build_eeg_run_record(
            raw=raw,
            bids_path=bids_path,
            segments_df=segments_df,
            record=record,
            metadata=metadata,
        )
    if raw_qc_reports_dir is not None:
        summary = report_eeg.summarize_condition_segments(segments_df)
        result["raw_qc_record"] = qc_raw.build_raw_qc_run_record(
            raw=raw,
            bids_path=bids_path,
            condition_segments_df=segments_df,
            condition_summary=summary,
            metadata=metadata,
            analysis_level=raw_qc_analysis_level,
        )
    result["success"] = True
    LOGGER.info("Converted %s run-%s", subject_id, run)
    return result


def _consume_record_result(
    record: dict[str, object],
    record_result: dict[str, object],
    failed_ids: set[int],
    successful_ids: set[int],
    skipped_ids: set[int],
    eeg_run_records: list[dict[str, object]],
    raw_qc_run_records: list[dict[str, object]],
) -> None:
    study_id = int(record["study_id"])
    if not record_result["success"]:
        failed_ids.add(study_id)
        return

    if record_result.get("skipped"):
        skipped_ids.add(study_id)
    else:
        successful_ids.add(study_id)
    if record_result["eeg_report_record"] is not None:
        eeg_run_records.append(record_result["eeg_report_record"])
    if record_result["raw_qc_record"] is not None:
        raw_qc_run_records.append(record_result["raw_qc_record"])


def _build_metadata_lookup(metadata_df: pd.DataFrame) -> dict[int, dict[str, object]]:
    """Map ``study_id`` to the canonical metadata fields surfaced in reports."""
    return {
        int(row.study_id): {
            "source_dataset": clean_scalar(getattr(row, "source_dataset", None)),
            "age_group": clean_scalar(getattr(row, "age_group", None)),
            "sex": clean_scalar(getattr(row, "sex", None)),
            "combined_diagnosis": clean_scalar(getattr(row, "combined_diagnosis", None)),
        }
        for row in metadata_df[
            ["study_id", "source_dataset", "age_group", "sex", "combined_diagnosis"]
        ]
        .dropna(subset=["study_id"])
        .drop_duplicates("study_id")
        .itertuples(index=False)
    }


def _build_run_inventory(
    raw_root: Path,
    metadata_df: pd.DataFrame,
    selected_study_ids: set[int] | None,
) -> pd.DataFrame:
    """Discover raw records and assign per-subject, chronological run numbers."""
    inventory_df = pd.DataFrame.from_records(
        ingest.discover_raw_records(raw_root, metadata_df),
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
    inventory_df["study_id"] = pd.to_numeric(inventory_df["study_id"], errors="coerce").astype(
        "Int64"
    )
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

    if selected_study_ids is not None:
        inventory_df = inventory_df.loc[inventory_df["study_id"].isin(selected_study_ids)].copy()
    return inventory_df


def _write_participants_tsv(
    bids_root: Path, metadata_df: pd.DataFrame, successful_ids: set[int]
) -> None:
    """Write ``participants.tsv`` for the successfully converted subjects."""
    converted_meta = metadata_df[metadata_df["study_id"].isin(sorted(successful_ids))].copy()
    participants_df = converted_meta[["study_id", "age", "sex"]].dropna(subset=["study_id"]).copy()
    participants_df["participant_id"] = participants_df["study_id"].apply(
        lambda value: bids_io.bids_subject_label(bids_io.study_id_to_bids_subject(int(value)))
    )
    participants_df = (
        participants_df[["participant_id", "age", "sex"]]
        .drop_duplicates("participant_id")
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    participants_df.to_csv(bids_root / "participants.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG -> BIDS converter")
    parser.add_argument(
        "--raw_root", type=Path, required=True, help="Root directory containing raw_data"
    )
    parser.add_argument("--bids_root", type=Path, required=True, help="BIDS root directory")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Canonical metadata CSV")
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional subject IDs to process (e.g. 0002 0027 or sub-0002 sub-0027)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing BIDS subject folders"
    )
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
    parser.add_argument(
        "--reports_root",
        type=Path,
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root)",
    )
    args = parser.parse_args()
    reports_root = (
        args.reports_root
        if args.reports_root
        else report_paths.default_reports_root(Path(args.bids_root))
    )
    eeg_reports_dir = reports_root if args.with_eeg_reports else None
    raw_qc_reports_dir = reports_root if args.with_raw_qc else None

    log_file = reports_root / "logs" / "to_bids.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    metadata_df = pd.read_csv(args.metadata_csv)
    metadata_df["study_id"] = pd.to_numeric(metadata_df["study_id"], errors="coerce").astype(
        "Int64"
    )
    LOGGER.info("Loaded metadata CSV with %d rows", len(metadata_df))
    metadata_lookup = _build_metadata_lookup(metadata_df)

    selected_study_ids: set[int] | None = None
    if args.subjects:
        selected_study_ids = {
            int(bids_io.study_id_to_bids_subject(subject)) for subject in args.subjects
        }
        LOGGER.info(
            "Filtering to %d selected subject(s): %s",
            len(selected_study_ids),
            sorted(selected_study_ids),
        )

    inventory_df = _build_run_inventory(args.raw_root, metadata_df, selected_study_ids)

    args.bids_root.mkdir(parents=True, exist_ok=True)
    inventory_path = args.bids_root / "raw_record_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    LOGGER.info("Wrote inventory to %s", inventory_path)

    if args.overwrite:
        for study_id in sorted(
            inventory_df.loc[inventory_df["run"].notna(), "study_id"].dropna().astype(int).unique()
        ):
            subject_id = bids_io.bids_subject_label(bids_io.study_id_to_bids_subject(int(study_id)))
            sub_dir = args.bids_root / subject_id
            if sub_dir.exists():
                LOGGER.info("Overwriting %s", subject_id)
                shutil.rmtree(sub_dir)

    failed_ids: set[int] = set()
    successful_ids: set[int] = set()
    skipped_ids: set[int] = set()
    eeg_run_records: list[dict[str, object]] = []
    raw_qc_run_records: list[dict[str, object]] = []
    selected_records = inventory_df.loc[inventory_df["run"].notna()].to_dict("records")
    LOGGER.info("Using %d worker(s) for BIDS conversion", effective_n_jobs(args.n_jobs))

    with tqdm_joblib(tqdm(total=len(selected_records), desc="Converting records")):
        record_results = Parallel(n_jobs=args.n_jobs, backend="loky", batch_size=1)(
            delayed(process_record)(
                record,
                bids_root=args.bids_root,
                overwrite=args.overwrite,
                metadata=metadata_lookup.get(int(record["study_id"]), {}),
                eeg_reports_dir=eeg_reports_dir,
                raw_qc_reports_dir=raw_qc_reports_dir,
                raw_qc_analysis_level=args.raw_qc_analysis_level,
            )
            for record in selected_records
        )

    for record, record_result in zip(selected_records, record_results):
        if record_result is None:
            LOGGER.error("Failed processing study_id %s: no result returned", record["study_id"])
            failed_ids.add(int(record["study_id"]))
            continue
        _consume_record_result(
            record,
            record_result,
            failed_ids=failed_ids,
            successful_ids=successful_ids,
            skipped_ids=skipped_ids,
            eeg_run_records=eeg_run_records,
            raw_qc_run_records=raw_qc_run_records,
        )

    if failed_ids:
        LOGGER.warning("Failed study_ids: %s", sorted(failed_ids))
    if skipped_ids:
        LOGGER.info("Skipped (already converted) study_ids: %s", sorted(skipped_ids))

    if successful_ids:
        LOGGER.info("Successfully converted study_ids: %s", sorted(successful_ids))
    elif not failed_ids:
        LOGGER.info("No new recordings needed conversion.")

    if successful_ids:
        _write_participants_tsv(args.bids_root, metadata_df, successful_ids)
        if args.with_eeg_reports:
            LOGGER.info("Generating EEG aggregate reports in %s", eeg_reports_dir)
            report_eeg.write_eeg_aggregate_reports(eeg_reports_dir, eeg_run_records)
        if args.with_raw_qc:
            LOGGER.info("Generating raw QC aggregate reports in %s", raw_qc_reports_dir)
            qc_raw.write_raw_qc_aggregate_reports(raw_qc_reports_dir, raw_qc_run_records)


if __name__ == "__main__":
    main()

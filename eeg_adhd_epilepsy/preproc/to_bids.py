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
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
from tqdm import tqdm

from eeg_adhd_epilepsy.io import ingest
from eeg_adhd_epilepsy.io import bids as bids_io
from eeg_adhd_epilepsy.utils import config

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


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
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
    exclusion_intervals = _merge_intervals(
        (
            (block["t_start"], block["t_stop"])
            for block in (*hv_blocks, *post_hv_blocks, *photo_blocks)
            if block["t_stop"] > block["t_start"]
        )
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

    return pd.DataFrame.from_records(records, columns=SEGMENT_COLUMNS).sort_values(
        by=["t_start", "segment_type"]
    ).reset_index(drop=True)

def process_record(
    record,
    bids_root: Path,
    overwrite: bool = False,
) -> bool:
    """Read, standardize, and export one selected recording to BIDS."""
    eeg_path = Path(record.eeg_path)
    study_id = int(record.study_id)
    run = str(record.run)
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

    if not overwrite and bids_path.fpath.exists():
        LOGGER.info("Skipping %s run-%s (exists)", subject_id, run)
        return True

    try:
        raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    except Exception as exc:
        LOGGER.error("Failed to read EEG %s: %s", eeg_path, exc)
        return False

    raw.info["line_freq"] = 60
    meas_datetime = pd.to_datetime(record.meas_datetime, errors="coerce")
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

    ch_types = {channel: "misc" for channel in ("A1", "A2") if channel in raw.ch_names}
    if ch_types:
        raw.set_channel_types(ch_types)
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
        return False

    stem = bids_path.fpath.stem[:-4] if bids_path.fpath.stem.endswith("_eeg") else bids_path.fpath.stem
    segments_path = bids_path.fpath.parent / f"{stem}_segments.csv"
    segments_df.to_csv(segments_path, index=False)
    LOGGER.info("Wrote condition segments for %s run-%s to %s", subject_id, run, segments_path)
    LOGGER.info("Converted %s run-%s", subject_id, run)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG -> BIDS converter")
    parser.add_argument("--raw_root", type=Path, required=True, help="Root directory containing raw_data")
    parser.add_argument("--bids_root", type=Path, required=True, help="BIDS root directory")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Canonical metadata CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS subject folders")
    parser.add_argument(
        "--with_condition_reports",
        action="store_true",
        help="Generate condition reports from the written segment CSVs after bidsification",
    )
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)
    metadata_df["study_id"] = pd.to_numeric(metadata_df["study_id"], errors="coerce").astype("Int64")
    LOGGER.info("Loaded metadata CSV with %d rows", len(metadata_df))

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
    selected_rows = inventory_df.loc[inventory_df["run"].notna()]
    for row in tqdm(selected_rows.itertuples(index=False), total=len(selected_rows), desc="Converting records"):
        if not process_record(row, args.bids_root, overwrite=args.overwrite):
            failed_ids.add(int(row.study_id))
        else:
            successful_ids.add(int(row.study_id))

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
        if args.with_condition_reports:
            from eeg_adhd_epilepsy.qc.conditions import generate_condition_reports

            reports_dir = args.bids_root / "condition_reports"
            LOGGER.info("Generating condition reports in %s", reports_dir)
            generate_condition_reports(
                input_dir=args.bids_root,
                output_dir=reports_dir,
                n_jobs=1,
            )


if __name__ == "__main__":
    main()

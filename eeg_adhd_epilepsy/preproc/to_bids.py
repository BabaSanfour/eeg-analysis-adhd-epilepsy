"""
Convert raw EEG + canonical metadata to BIDS with standardized annotations.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import mne
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


def map_annotation_to_category(desc: str) -> Optional[str]:
    """
    Map a raw annotation description to a standardized trial_type category.
    Uses patterns defined in utils.qc_config (loaded from annotations.yaml).
    Returns None if no match is found.
    Returns 'BAD_IGNORE' if it should be dropped.
    """
    normalized = re.sub(r"\s+", " ", desc.lower().strip()) if isinstance(desc, str) else ""
    if not normalized:
        return "BAD_IGNORE"

    def _matches_pattern(pattern: str) -> bool:
        pattern = str(pattern).lower().strip()
        if not pattern:
            return False
        if "*" in pattern:
            return fnmatch.fnmatch(normalized, pattern)
        return normalized == pattern or bool(
            re.search(r"\b" + re.escape(pattern) + r"\b", normalized)
        )

    has_additional_channel = any(
        _matches_pattern(ch) for ch in config.ADDITIONAL_SENSOR_CHANNELS
    )
    has_channel = has_additional_channel or any(
        _matches_pattern(ch) for ch in config.BASIC_1020_CHANNELS
    )
    if any(_matches_pattern(pattern) for pattern in config.IGNORE_PATTERNS + config.IGNORED_LABELS):
        return "BAD_IGNORE"
    if any(_matches_pattern(pattern) for pattern in config.REFERENCE_EVENT_KEYWORDS):
        return "recording_start"
    if has_channel and any(
        _matches_pattern(pattern)
        for pattern in config.SENSOR_ARTEFACT_KEYWORDS + config.SENSOR_ACTION_KEYWORDS
    ):
        return "BAD_IGNORE" if has_additional_channel else "sensor_artefact"

    for category, patterns in config.ANNOTATION_INTEREST_MAP.items():
        cat_slug = category.lower().replace(" ", "_").replace("/", "_")
        for pat in patterns:
            if pat and _matches_pattern(pat):
                return cat_slug

    for category, patterns in config.CLINICAL_COMMENT_LABELS.items():
        cat_slug = category.lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for pat in patterns:
            if _matches_pattern(pat):
                return cat_slug

    return None


def standardize_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Update annotations in place to a cleaner BIDS-facing vocabulary."""
    new_onset = []
    new_duration = []
    new_descs = []

    preserve_original = {"hv", "photo", "post_hv"}
    allow_clean = {"eyes_open", "eyes_closed", "recording_start"}

    for annot in raw.annotations:
        original = annot["description"]
        category = map_annotation_to_category(original)

        if category in {None, "BAD_IGNORE"}:
            continue

        new_onset.append(annot["onset"])
        new_duration.append(annot["duration"])

        if category in preserve_original:
            new_descs.append(original)
        elif category in allow_clean or category.startswith("clinical_"):
            new_descs.append(category)
        else:
            new_descs.append(f"BAD_{category}")

    raw.set_annotations(
        mne.Annotations(
            onset=new_onset,
            duration=new_duration,
            description=new_descs,
            orig_time=raw.annotations.orig_time,
        )
    )
    return raw


def update_participants_tsv(
    bids_dir: Path,
    metadata_df: pd.DataFrame,
) -> None:
    """Rewrite participants.tsv to one row per participant with canonical demographics."""
    tsv_path = bids_dir / "participants.tsv"

    participants_df = metadata_df[["study_id", "age", "sex"]].dropna(subset=["study_id"]).copy()
    participants_df["participant_id"] = participants_df["study_id"].apply(
        lambda value: bids_io.normalize_subject_id(f"{int(value):04d}")
    )
    participants_df = (
        participants_df[["participant_id", "age", "sex"]]
        .drop_duplicates("participant_id")
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    participants_df.to_csv(tsv_path, sep="\t", index=False)


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

    try:
        raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    except Exception as exc:
        LOGGER.error("Failed to read EEG %s: %s", eeg_path, exc)
        return False

    raw.info["line_freq"] = 60
    meas_datetime = pd.to_datetime(record.meas_datetime, errors="coerce")
    if pd.notna(meas_datetime):
        raw.set_meas_date(meas_datetime.to_pydatetime())

    raw = standardize_annotations(raw)

    available_targets = [channel for channel in config.BASIC_1020_CHANNELS if channel in raw.ch_names]
    if available_targets:
        raw.pick(available_targets)

    ch_types = {}
    if "A1" in raw.ch_names:
        ch_types["A1"] = "misc"
    if "A2" in raw.ch_names:
        ch_types["A2"] = "misc"
    if ch_types:
        raw.set_channel_types(ch_types)

    bids_path = BIDSPath(
        root=str(bids_root),
        subject=subject_id[4:],
        session="01",
        task="clinical",
        run=run,
        suffix="eeg",
        extension=".vhdr",
    )

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
        message = str(exc).lower()
        if not overwrite and ("already exists" in message or "file exists" in message):
            LOGGER.info("Skipping write for %s run-%s (exists)", subject_id, run)
            return True
        LOGGER.error("Failed writing %s run-%s: %s", subject_id, run, exc)
        return False

    LOGGER.info("Converted %s run-%s", subject_id, run)
    return True
def main() -> None:
    parser = argparse.ArgumentParser(description="EEG -> BIDS converter")
    parser.add_argument("--raw_root", type=Path, required=True, help="Root directory containing raw_data")
    parser.add_argument("--bids_root", type=Path, required=True, help="BIDS root directory")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Canonical metadata CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS subject folders")
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)
    LOGGER.info("Loaded metadata CSV with %d rows", len(metadata_df))

    records = ingest.discover_raw_records(args.raw_root, metadata_df)
    inventory_df = pd.DataFrame(records)
    if inventory_df.empty:
        inventory_df = pd.DataFrame(
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
                "run",
            ]
        )
    inventory_df["run"] = pd.Series([None] * len(inventory_df), dtype=object)
    if "study_id" in inventory_df.columns:
        inventory_df["study_id"] = pd.to_numeric(inventory_df["study_id"], errors="coerce").astype("Int64")

    run_rows = inventory_df.loc[
        inventory_df["study_id"].isin(metadata_df["study_id"]) & inventory_df["eeg_path"].notna()
    ].copy()
    if not run_rows.empty:
        run_rows["record_date_dt"] = pd.to_datetime(run_rows["record_date"], errors="coerce").dt.date
        run_rows["meas_datetime_dt"] = pd.to_datetime(run_rows["meas_datetime"], errors="coerce")
        run_rows = run_rows.sort_values(
            ["study_id", "record_date_dt", "meas_datetime_dt", "record_stem"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        run_rows["run"] = run_rows.groupby("study_id").cumcount().add(1).map(lambda value: f"{value:02d}")
        inventory_df.loc[run_rows.index, "run"] = run_rows["run"].values
    inventory_df = inventory_df.drop(
        columns=["record_date_dt", "meas_datetime_dt", "status"], errors="ignore"
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

    failed = []
    successful_ids: list[int] = []
    run_rows = inventory_df.loc[inventory_df["run"].notna()]
    for row in tqdm(run_rows.itertuples(index=False), total=len(run_rows), desc="Converting records"):
        if not process_record(row, args.bids_root, overwrite=args.overwrite):
            failed.append(int(row.study_id))
        else:
            successful_ids.append(int(row.study_id))

    if failed:
        LOGGER.warning("Failed study_ids: %s", sorted(set(failed)))
    else:
        LOGGER.info("All selected recordings converted successfully.")

    converted_ids = sorted(set(successful_ids))
    if converted_ids:
        converted_meta = metadata_df[metadata_df["study_id"].isin(converted_ids)].copy()
        update_participants_tsv(args.bids_root, converted_meta)


if __name__ == "__main__":
    main()

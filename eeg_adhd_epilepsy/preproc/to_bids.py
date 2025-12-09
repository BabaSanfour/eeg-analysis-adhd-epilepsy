#!/usr/bin/env python3
"""
Convert raw EEG + metadata → BIDS.
"""

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import List, Set
from datetime import datetime, timezone

import mne
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURE LOGGER
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def get_subject_ids(source_dir: Path) -> List[str]:
    """
    Scan source_dir and return sorted unique subject IDs.
    Assumes files/folders start with ID plus a dot.
    """
    ids: Set[str] = set()
    for item in source_dir.iterdir():
        if item.name.startswith('.'):
            continue
        m = re.match(r"^([A-Z]{2}\d{5,6}[A-Z]?)\.", item.name)
        if m:
            ids.add(m.group(1))
        else:
            prefix = item.stem
            if prefix:
                ids.add(prefix)
    ids.discard("DskUUID")
    return sorted(ids)


def read_subject_data(
    subject_id: str,
    raw_dir: Path,
    bids_root: Path,
    mapping_df: pd.DataFrame,
    overwrite: bool = False,
):
    """
    Read files for one subject, convert EEG data to BIDS format.

    Returns:
        dict with {"participant_id": str, "meas": str | None} on success,
        or None when subject is skipped (e.g. special control).
    """
    # -- determine new_id and meas_date from .pnt
    pnt = raw_dir / f"{subject_id}.pnt"
    meas_dt = None
    meas_iso = None

    if pnt.exists():
        text = pnt.read_bytes().decode("ISO-8859-1", errors="ignore").replace("\x00", "")

        # --- parse numeric ID
        match = re.search(r"ID(\d+(?:\.\d+)?)", text)
        if match:
            # capture full numeric part, strip any .suffix
            new_id_full = match.group(1)
            new_id = new_id_full.split('.')[0]
            if new_id == "2.2":
                logging.warning("Skipping %s → sub-%s (special control)", subject_id, new_id)
                return None  # skip special control
            # apply mapping if needed
            if new_id.isdigit() and len(new_id) >= 4:
                mapped = mapping_df[mapping_df["ID"] == int(new_id)]
                if not mapped.empty:
                    new_id = str(int(mapped["patient"].iat[0]))
        else:
            logging.warning("Could not parse new ID in %s; skipping subject", subject_id)
            return None  # skip subject if no ID found


        # --- parse Date + Start Time → meas_dt
        date_match = re.search(r"Date(\d{4})/(\d{2})/(\d{2})", text)
        start_match = re.search(r"Start Time(\d{2})(\d{2})(\d{2})", text)
        if date_match and start_match:
            try:
                year, month, day = map(int, date_match.groups())
                sh, sm, ss = map(int, start_match.groups())
                # Recording timestamps need to be UTC-aware for BIDS export
                meas_dt = datetime(year, month, day, sh, sm, ss, tzinfo=timezone.utc)
                meas_iso = meas_dt.isoformat().replace("+00:00", "Z")
            except ValueError:
                logging.warning("Invalid date/time in %s; meas_date not set", pnt)
        else:
            logging.warning("Could not parse recording date/time in %s", pnt)
    else:
        logging.warning("No .pnt for %s; using raw ID and no meas_date", subject_id)
        raise FileNotFoundError(f"No .pnt file for {subject_id}")

    # zero-pad if numeric
    if re.fullmatch(r"\d+", new_id):
        new_id = f"{int(new_id):04d}"

    participant_id = f"sub-{new_id}"

    # -- detect existing BIDS output
    sub_dir = bids_root / participant_id
    if sub_dir.exists():
        if not overwrite:
            logging.info("Skipping %s → %s (already exists)", subject_id, participant_id)
            return {"participant_id": participant_id, "meas": meas_iso}
        logging.info("Overwriting %s → %s", subject_id, participant_id)
        shutil.rmtree(sub_dir)

    # -- locate EEG file
    eeg_files = list(raw_dir.glob(f"{subject_id}.EEG"))
    if not eeg_files:
        raise FileNotFoundError(f"No EEG file for {subject_id}")
    eeg_path = eeg_files[0]

    # -- read EEG
    raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    raw.info["line_freq"] = 60

    # -- inject meas_date into MNE object so BIDS picks it up
    if meas_dt is not None:
        raw.set_meas_date(meas_dt)

    # -- write BIDS, catching existing-file errors when overwrite=False
    bids_path = BIDSPath(
        root=str(bids_root),
        subject=new_id,
        session="01",
        task="RESTING",
        run="01",
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
    except Exception as e:
        msg = str(e).lower()
        if not overwrite and ("already exists" in msg or "file exists" in msg):
            logging.info("Skipping write for %s → %s (exists)", subject_id, participant_id)
            return {"participant_id": participant_id, "meas": meas_iso}
        raise

    logging.info("Converted %s → %s", subject_id, participant_id)
    return {"participant_id": participant_id, "meas": meas_iso}  # NEW


def update_participants_tsv(
    bids_dir: Path,
    subjects_df: pd.DataFrame,
    meas_df: pd.DataFrame,
):
    """
    Merge age, sex, and meas into participants.tsv.
    Also writes a participants.csv copy.
    """
    tsv_path = bids_dir / "participants.tsv"
    participants_df = pd.read_csv(tsv_path, sep="\t")

    # Age/Sex from subjects_df
    subjects_meta = (
        subjects_df.rename(columns={"Study ID": "ID"})[["ID", "Age", "Sex"]]
        .copy()
    )
    subjects_meta["participant_id"] = subjects_meta["ID"].apply(
        lambda i: f"sub-{int(i):04d}"
    )
    subjects_meta = subjects_meta[["participant_id", "Age", "Sex"]]

    merged = participants_df.merge(subjects_meta, on="participant_id", how="left")
    merged = merged.rename(columns={"Age": "age", "Sex": "sex"})

    # merge meas (measurement datetime) from meas_df
    if meas_df is not None and not meas_df.empty:
        merged = merged.merge(meas_df, on="participant_id", how="left")

    # Write back TSV
    merged.to_csv(tsv_path, sep="\t", index=False)
    logging.info("Updated participants.tsv at %s", tsv_path)

    # also write a CSV version
    csv_path = bids_dir / "participants.csv"
    merged.to_csv(csv_path, index=False)
    logging.info("Wrote participants.csv at %s", csv_path)


def find_missing(subjects_ids: List[str], bids_dir) -> List[str]:
    """
    Check for missing BIDS directories and files.
    """
    missing = []
    for sid in subjects_ids:
        sub_label = f"sub-{int(sid):04d}"
        sub_dir = bids_dir / sub_label
        if not sub_dir.exists():
            missing.append(sub_label)
    return missing


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EEG → BIDS converter")
    parser.add_argument(
        "--raw", type=Path, required=True, help="raw data directory"
    )
    parser.add_argument(
        "--bids", type=Path, required=True, help="BIDS root directory"
    )
    parser.add_argument(
        "--map", type=Path, required=True, help="mapping CSV file"
    )
    parser.add_argument(
        "--subs", type=Path, required=True, help="subjects CSV file"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing BIDS files if they exist"
    )
    args = parser.parse_args()

    mapping_df = pd.read_csv(
        args.map, header=None, names=["patient", "ID"], sep=";"
    )
    subjects_df = pd.read_csv(
        args.subs, sep=";", encoding="utf-8", low_memory=False
    )

    subject_ids = get_subject_ids(args.raw)
    logging.info("Found %d subjects in %s", len(subject_ids), args.raw)

    failed = []
    meas_records = []

    for sid in tqdm(subject_ids, desc="Converting subjects"):
        try:
            meta = read_subject_data(
                sid, args.raw, args.bids, mapping_df,
                overwrite=args.overwrite
            )
            if meta is not None:
                meas_records.append(meta)
        except Exception as e:
            logging.error("Failed %s: %s", sid, e)
            failed.append(sid)

    if failed:
        logging.warning("Number of failed subjects: %d", len(failed))
        logging.warning("Failed subjects: %s", failed)
    else:
        logging.info("All subjects converted successfully")

    meas_df = pd.DataFrame(meas_records) if meas_records else pd.DataFrame(columns=["participant_id", "meas"])

    update_participants_tsv(args.bids, subjects_df, meas_df)

    subject_ids = subjects_df["Study ID"].astype(str).tolist()
    missing = find_missing(subject_ids, args.bids)
    if missing:
        logging.warning("Missing BIDS directories or files: %s", missing)
        logging.warning("Number of missing subjects: %d", len(missing))
    else:
        logging.info("No missing BIDS directories or files")
    logging.info("BIDS conversion completed successfully")


if __name__ == "__main__":
    main()

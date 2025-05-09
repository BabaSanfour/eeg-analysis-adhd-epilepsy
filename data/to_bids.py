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
    overwrite: bool = False
):
    """
    Read files for one subject, convert EEG data to BIDS format.
    Skips existing outputs when overwrite=False, without marking failure.
    Raises on real processing errors.
    """
    # -- determine new_id
    pnt = raw_dir / f"{subject_id}.pnt"
    if pnt.exists():
        text = pnt.read_bytes().decode("ISO-8859-1", errors="ignore").replace("\x00", "")
        match = re.search(r"ID(\d+(?:\.\d+)?)", text)
        if match:
            # capture full numeric part, strip any .suffix
            new_id_full = match.group(1)
            new_id = new_id_full.split('.')[0]
            if new_id == "2.2":
                logging.warning("Skipping %s → sub-%s (special control)", subject_id, new_id)
                return  # skip special control
            # apply mapping if needed
            if new_id.isdigit() and len(new_id) >= 4:
                mapped = mapping_df[mapping_df["ID"] == int(new_id)]
                if not mapped.empty:
                    new_id = str(int(mapped["patient"].iat[0]))
        else:
            logging.warning("Could not parse new ID in %s; using raw ID", subject_id)
            new_id = subject_id
    else:
        logging.warning("No .pnt for %s; using raw ID", subject_id)
        raise FileNotFoundError(f"No .pnt file for {subject_id}")

    # zero-pad if numeric
    if re.fullmatch(r"\d+", new_id):
        new_id = f"{int(new_id):04d}"

    # -- detect existing BIDS output
    sub_dir = bids_root / f"sub-{new_id}"
    if sub_dir.exists():
        if not overwrite:
            logging.info("Skipping %s → sub-%s (already exists)", subject_id, new_id)
            return
        logging.info("Overwriting %s → sub-%s", subject_id, new_id)
        shutil.rmtree(sub_dir)

    # -- locate EEG file
    eeg_files = list(raw_dir.glob(f"{subject_id}.EEG"))
    if not eeg_files:
        raise FileNotFoundError(f"No EEG file for {subject_id}")
    eeg_path = eeg_files[0]

    # -- read EEG
    raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    raw.info["line_freq"] = 60

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
            logging.info("Skipping write for %s → sub-%s (exists)", subject_id, new_id)
            return
        raise

    logging.info("Converted %s → sub-%s", subject_id, new_id)


def update_participants_tsv(bids_dir: Path, subjects_df: pd.DataFrame):
    """
    Merge only age & sex into participants.tsv to avoid duplicate columns.
    """
    tsv_path = bids_dir / "participants.tsv"
    participants_df = pd.read_csv(tsv_path, sep="\t")

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
    merged.to_csv(tsv_path, sep="\t", index=False)
    logging.info("Updated participants.tsv at %s", tsv_path)


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
    for sid in tqdm(subject_ids, desc="Converting subjects"):
        try:
            read_subject_data(
                sid, args.raw, args.bids, mapping_df,
                overwrite=args.overwrite
            )
        except Exception as e:
            logging.error("Failed %s: %s", sid, e)
            failed.append(sid)

    if failed:
        logging.warning("Number of failed subjects: %d", len(failed))
        logging.warning("Failed subjects: %s", failed)
    else:
        logging.info("All subjects converted successfully")

    update_participants_tsv(args.bids, subjects_df)
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
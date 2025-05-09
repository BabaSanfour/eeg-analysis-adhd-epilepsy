#!/usr/bin/env python3
"""
Convert raw EEG + metadata to BIDS.
"""

import argparse
import logging
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
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
        m = re.match(r"^([A-Z]{2}\d{5,6}[A-Z]?)\.", item.name)
        if m:
            ids.add(m.group(1))
        else:
            prefix = item.stem
            if prefix:
                ids.add(prefix)
    # drop unwanted
    ids.discard("DskUUID")
    return sorted(ids)


def read_subject_data(subject_id: str,
                      raw_dir: Path,
                      bids_root: Path,
                      mapping_df: pd.DataFrame) -> None:
    """
    Convert one subject’s EEG to BIDS. Raises on failure.
    """
    # -- load .pnt and extract new ID
    pnt = raw_dir / f"{subject_id}.pnt"
    if not pnt.exists():
        raise FileNotFoundError(f".pnt not found for {subject_id}")
    raw_bytes = pnt.read_bytes()
    text = raw_bytes.decode("ISO-8859-1", errors="ignore").replace("\x00", "")
    m = re.search(r"ID(\d{1,8}(?:\.\d)?)[EN]", text)
    if not m:
        raise ValueError(f"Could not parse new ID in {subject_id}")
    new_id = m.group(1).rstrip(".1").rstrip(".2")
    if new_id == "2.2":
        return  # skip special case

    # -- map via CSV if needed
    if len(new_id) > 4:
        mapped = mapping_df[mapping_df["ID"] == int(new_id)]
        if mapped.empty:
            raise KeyError(f"No mapping for subject {new_id}")
        new_id = str(int(mapped["patient"].iat[0]))

    new_id = f"{int(new_id):04d}"

    # -- skip if already exists
    bids_subj = bids_root / f"sub-{new_id}" / "eeg"
    if bids_subj.exists():
        logging.info("Skipping %s (already converted)", subject_id)
        return

    # -- read EEG
    eeg_glob = list(raw_dir.glob(f"{subject_id}.EEG"))
    if not eeg_glob:
        raise FileNotFoundError(f"No .EEG for {subject_id}")
    raw = mne.io.read_raw_nihon(str(eeg_glob[0]), preload=True)
    raw.info["line_freq"] = 60

    # -- write BIDS
    bids_path = BIDSPath(
        root=str(bids_root),
        subject=new_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
    )
    write_raw_bids(raw, bids_path=bids_path, format="BrainVision",
                   overwrite=True, allow_preload=True, verbose=False)
    logging.info("Converted %s → sub-%s", subject_id, new_id)


def update_participants_tsv(bids_root: Path, subjects_df: pd.DataFrame):
    tsv = bids_root / "participants.tsv"
    df = pd.read_csv(tsv, sep="\t")
    subjects_df = subjects_df.rename(columns={"Study ID": "ID"})
    subjects_df["participant_id"] = subjects_df["ID"].apply(lambda i: f"sub-{int(i):04d}")
    merged = df.merge(subjects_df, on="participant_id", how="left")
    merged = merged.drop(columns=["age", "ID", "sex"], errors="ignore")
    merged = merged.rename(columns={"Age": "age", "Sex": "sex"})
    merged.to_csv(tsv, sep="\t", index=False)
    logging.info("Updated %s", tsv)


def find_unconverted(bids_root: Path) -> List[str]:
    missing = []
    for subdir in bids_root.glob("sub-*"):
        if not (subdir / "eeg").exists():
            missing.append(subdir.name)
    return missing


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="EEG → BIDS converter")
    p.add_argument("--raw",    type=Path, required=True, help="raw data dir")
    p.add_argument("--bids",   type=Path, required=True, help="BIDS root")
    p.add_argument("--map",    type=Path, required=True, help="mapping CSV")
    p.add_argument("--subs",   type=Path, required=True, help="subjects CSV")
    p.add_argument("--workers",type=int, default=None, help="# parallel workers")
    p.add_argument("--timeout",type=int, default=300,
                   help="per-subject timeout in seconds")
    args = p.parse_args()

    raw_dir     = args.raw
    bids_root   = args.bids
    mapping_df  = pd.read_csv(args.map, header=None, names=["patient","ID"], sep=";")
    subjects_df = pd.read_csv(args.subs, sep=";", encoding="utf-8", low_memory=False)

    subj_ids = get_subject_ids(raw_dir)
    logging.info("Found %d subjects", len(subj_ids))

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(read_subject_data, sid, raw_dir, bids_root, mapping_df): sid
            for sid in subj_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Subjects"):
            sid = futures[fut]
            try:
                fut.result(timeout=args.timeout)
            except TimeoutError:
                logging.error("Timeout for %s", sid)
                failed.append(sid)
            except Exception as e:
                logging.error("Error %s: %s", sid, e)
                failed.append(sid)

    if failed:
        logging.warning("Failed subjects: %s", failed)
    else:
        logging.info("All subjects processed successfully")

    # participants.tsv
    update_participants_tsv(bids_root, subjects_df)

    # detect any unconverted
    missing = find_unconverted(bids_root)
    if missing:
        logging.warning("Unconverted directories: %s", missing)
    else:
        logging.info("All BIDS directories complete")


if __name__ == "__main__":
    main()
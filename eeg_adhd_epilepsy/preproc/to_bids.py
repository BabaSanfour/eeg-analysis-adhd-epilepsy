"""
Convert raw EEG + metadata → BIDS with standardized annotations.
"""

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, Dict

import mne
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath
from tqdm import tqdm

from eeg_adhd_epilepsy.io import ingest
from eeg_adhd_epilepsy.utils import config

# -----------------------------------------------------------------------------
# CONFIGURE LOGGER
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ANNOTATION STANDARDIZATION
# -----------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    """Normalize label string for matching (lowercase, strip)."""
    if not isinstance(label, str):
        return ""
    return re.sub(r"\s+", " ", label.lower().strip())


def map_annotation_to_category(desc: str) -> Optional[str]:
    """
    Map a raw annotation description to a standardized trial_type category.
    Uses patterns defined in utils.qc_config (loaded from annotations.yaml).
    Returns None if no match found (which will become 'other').
    Returns 'BAD_IGNORE' if it should be dropped.
    """
    normalized = normalize_label(desc)
    if not normalized:
        return "BAD_IGNORE"

    # 1. Check for Ignored Demographics -> Drop
    for pat in config.IGNORED_LABELS:
        if re.search(r'\b' + re.escape(pat.lower()) + r'\b', normalized):
            return "BAD_IGNORE"
            
    # 2. Check Reference Event -> recording_start
    for pat in config.REFERENCE_EVENT_KEYWORDS:
        if re.search(r'\b' + re.escape(pat.lower()) + r'\b', normalized):
            return "recording_start"

    # 3. Sensor Artifact Logic
    # Check for presence of any sensor action verb
    has_verb = False
    for verb in config.SENSOR_ACTION_KEYWORDS:
        if re.search(r'\b' + re.escape(verb.lower()) + r'\b', normalized):
            has_verb = True
            break
            
    if has_verb:
        # Check if any ADDITIONAL (non-10-20) channel is mentioned -> Drop
        for ch in config.ADDITIONAL_SENSOR_CHANNELS:
            # Word boundary check for channel names (e.g. "Oz")
            if re.search(r'\b' + re.escape(ch.lower()) + r'\b', normalized):
                return "BAD_IGNORE"
        
        return "sensor_artefact"

    # 4. Standard Interest Map
    for category, patterns in config.ANNOTATION_INTEREST_MAP.items():
        # Clean category string for BIDS trial_type (e.g. "Eyes Open" -> "eyes_open")
        cat_slug = category.lower().replace(" ", "_").replace("/", "_")
        
        for pat in patterns:
            if not pat:
                continue
            if re.search(r'\b' + re.escape(pat.lower()) + r'\b', normalized):
                return cat_slug
                
    # 4. Clinical Comments -> specific categories
    for category, patterns in config.CLINICAL_COMMENT_LABELS.items():
        # "Clinical - Spikes" -> "clinical_spikes"
        cat_slug = category.lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for pat in patterns:
            if re.search(r'\b' + re.escape(pat.lower()) + r'\b', normalized):
                return cat_slug
                
    return None


def standardize_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    In-place update of raw.annotations to use standardized descriptions.
    
    Rules:
    1. 'hv', 'photo', 'post_hv': Keep original description.
    2. 'eyes_open', 'eyes_closed', 'recording_start': Keep standardized category name.
    3. Clinical categories (starting with 'clinical_'): Keep standardized name.
    4. 'BAD_IGNORE': Drop these annotations entirely.
    5. Everything else: Prefix with 'bad_' (e.g. 'bad_movement').
    """
    new_onset = []
    new_duration = []
    new_descs = []
    
    PRESERVE_ORIGINAL = {"hv", "photo", "post_hv"}
    ALLOW_CLEAN = {"eyes_open", "eyes_closed", "recording_start"}
    
    for annot in raw.annotations:
        orig = annot['description']
        cat = map_annotation_to_category(orig)
        
        # If ignore, skip adding to new list (effectively dropping it)
        if cat == "BAD_IGNORE":
            continue
            
        # Add to new lists
        new_onset.append(annot['onset'])
        new_duration.append(annot['duration'])
        
        if not cat:
            new_descs.append("other") 
            continue
        
        if cat in PRESERVE_ORIGINAL:
            new_descs.append(orig)
        elif cat in ALLOW_CLEAN or cat.startswith("clinical_"):
            new_descs.append(cat)
        else:
            new_descs.append(f"BAD_{cat}")

    # Rebuild annotations
    raw.set_annotations(mne.Annotations(
        onset=new_onset,
        duration=new_duration,
        description=new_descs,
        orig_time=raw.annotations.orig_time
    ))
    return raw


# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def process_subject(
    subject_id: str,
    raw_dir: Path,
    bids_root: Path,
    mapping_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    overwrite: bool = False,
) -> Optional[Dict]:
    """
    Read files for one subject, standardize, convert to BIDS.
    """
    # 1. Parse metadata (pnt)
    pnt_file = raw_dir / f"{subject_id}.pnt"
    meta = ingest.parse_pnt_metadata(pnt_file)
    
    # 2. Map ID
    # Use the ID found in .pnt if available, else use filename subject_id
    raw_id_for_mapping = meta['original_id'] if meta['original_id'] else subject_id
    
    raw_id_str = str(raw_id_for_mapping)
    dup_row = duplicates_df[duplicates_df['Study_ID'].astype(str) == raw_id_str]
    if raw_id_str in ['232', '961', '494', '662', '958', '791', '674', '792', '767', "492"]:
        print("here")
    if not dup_row.empty:
        # Found in duplicates mapping -> Use specific ID and Session
        new_id = str(dup_row['Actual_ID'].values[0])
        session = f"{int(dup_row['Session'].values[0]):02d}"
    else:
        # Not in duplicates -> Use general mapping, default session 01
        new_id = ingest.map_subject_id(raw_id_for_mapping, mapping_df)
        session = "01"
    
    if new_id is None:
        LOGGER.warning("Skipping subject %s (ID mapping exclusion)", subject_id)
        return None
    
    # zero-pad if numeric (standardize subject ID format: sub-0123)
    if new_id.isdigit():
        new_id = f"{int(new_id):04d}"
    
    participant_id = f"sub-{new_id}"
    meas_iso = meta['meas_date'].isoformat().replace("+00:00", "Z") if meta['meas_date'] else None

    # -- detect existing BIDS output
    sub_dir = bids_root / participant_id
    if sub_dir.exists():
        if not overwrite:
            LOGGER.info("Skipping %s -> %s (already exists)", subject_id, participant_id)
            return {"participant_id": participant_id, "meas": meas_iso}
        LOGGER.info("Overwriting %s -> %s", subject_id, participant_id)
        shutil.rmtree(sub_dir)

    # -- locate EEG file
    eeg_path = ingest.find_eeg_file(raw_dir, subject_id)
    if not eeg_path:
        LOGGER.error("No EEG file for %s", subject_id)
        return None
        
    # -- read EEG
    try:
        raw = mne.io.read_raw_nihon(str(eeg_path), preload=False)
    except Exception as e:
        LOGGER.error("Failed to read EEG %s: %s", eeg_path, e)
        return None
        
    raw.info["line_freq"] = 60
    if meta['meas_date']:
        raw.set_meas_date(meta['meas_date'])

    # -- Standardize Annotations (NEW)
    raw = standardize_annotations(raw)
    
    # -- Filter Channels & Set Types
    # Keep only 10-20 channels
    targets = list(config.BASIC_1020_CHANNELS)
    
    available_targets = [ch for ch in targets if ch in raw.ch_names]
    
    if available_targets:
        raw.pick(available_targets)
        
    # Set types for A1/A2 if present
    ch_types = {}
    if "A1" in raw.ch_names:
        ch_types["A1"] = "misc"
    if "A2" in raw.ch_names:
        ch_types["A2"] = "misc"
        
    if ch_types:
        raw.set_channel_types(ch_types)

    # -- write BIDS
    bids_path = BIDSPath(
        root=str(bids_root),
        subject=new_id,
        session=session,
        task="clinical",
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
            LOGGER.info("Skipping write for %s -> %s (exists)", subject_id, participant_id)
            return {"participant_id": participant_id, "meas": meas_iso}
        raise

    LOGGER.info("Converted %s -> %s", subject_id, participant_id)
    return {"participant_id": participant_id, "meas": meas_iso}


def update_participants_tsv(
    bids_dir: Path,
    subjects_df: pd.DataFrame,
    meas_df: pd.DataFrame,
):
    """
    Merge age, sex, and meas into participants.tsv.
    """
    tsv_path = bids_dir / "participants.tsv"
    if not tsv_path.exists():
        return
        
    participants_df = pd.read_csv(tsv_path, sep="\t")

    # Age/Sex from subjects_df
    # Expects columns "Study ID", "Age", "Sex"
    subjects_meta = (
        subjects_df.rename(columns={"Study ID": "ID"})[["ID", "Age", "Sex"]]
        .copy()
    )
    # create participant_id col to match BIDS
    subjects_meta["participant_id"] = subjects_meta["ID"].apply(
        lambda i: f"sub-{int(i):04d}"
    )
    subjects_meta = subjects_meta[["participant_id", "Age", "Sex"]]

    merged = participants_df.merge(subjects_meta, on="participant_id", how="left")
    merged = merged.rename(columns={"Age": "age", "Sex": "sex"})

    # merge meas (measurement datetime)
    if meas_df is not None and not meas_df.empty:
        # ensure columns match
        if "participant_id" in meas_df.columns:
            overlap_cols = [c for c in meas_df.columns if c in merged.columns and c != "participant_id"]
            if overlap_cols:
                merged = merged.drop(columns=overlap_cols)
            merged = merged.merge(meas_df, on="participant_id", how="left")

    # Write back TSV
    merged.to_csv(tsv_path, sep="\t", index=False)
    
def check_missing(subjects_ids: list, bids_dir: Path):
    missing = []
    for sid in subjects_ids:
        sub_label = f"sub-{int(sid):04d}"
        sub_dir = bids_dir / sub_label
        if not sub_dir.exists():
            missing.append(sub_label)
    if missing:
        LOGGER.warning("Missing BIDS directories for %d subjects: %s", len(missing), missing)
    else:
        LOGGER.info("No missing BIDS directories.")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EEG -> BIDS converter")
    parser.add_argument("--raw", type=Path, required=True, help="raw data directory")
    parser.add_argument("--bids", type=Path, required=True, help="BIDS root directory")
    parser.add_argument("--map", type=Path, required=True, help="mapping CSV file")
    parser.add_argument("--duplicates", type=Path, required=True, help="duplicates CSV file")
    parser.add_argument("--subs", type=Path, required=True, help="subjects CSV file")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing BIDS")
    args = parser.parse_args()

    # Load metadata tables
    mapping_df = pd.read_csv(args.map, header=None, names=["patient", "ID"], sep=";")
    subjects_df = pd.read_csv(args.subs, encoding="utf-8", low_memory=False, on_bad_lines="warn")
    # removed unnamed columns if they exist
    subjects_df = subjects_df.loc[:, ~subjects_df.columns.str.contains("unnamed", case=False)]
    LOGGER.info("Loaded subjects CSV with columns: %s", subjects_df.columns.tolist())
    duplicates_df = pd.read_csv(args.duplicates, encoding="utf-8", low_memory=False)

    subject_ids = ingest.get_subject_ids(args.raw)
    LOGGER.info("Found %d raw subjects in %s", len(subject_ids), args.raw)

    meas_records = []
    failed = []

    results = []
    for sid in tqdm(subject_ids, desc="Converting subjects"):
        results.append(process_subject(
            sid, args.raw, args.bids, mapping_df, 
            duplicates_df=duplicates_df,
            overwrite=args.overwrite
        ))

    for res in results:
        if not res:
            continue
        if "error" in res:
            LOGGER.error("Failed processing subject %s: %s", res["sid"], res["error"])
            failed.append(res["sid"])
        else:
            meas_records.append(res)

    if failed:
        LOGGER.warning("Failed subjects: %s", failed)
    else:
        LOGGER.info("All subjects processed successfully.")

    # Update participants.tsv
    meas_df = pd.DataFrame(meas_records) if meas_records else pd.DataFrame()
    update_participants_tsv(args.bids, subjects_df, meas_df)

    # Check completeness
    studied_ids = subjects_df["Study ID"].dropna().astype(int).astype(str).tolist()
    check_missing(studied_ids, args.bids)
    
    if failed:
        if args.overwrite: # Only hard exit if we expected everything to work
             pass 

if __name__ == "__main__":
    main()

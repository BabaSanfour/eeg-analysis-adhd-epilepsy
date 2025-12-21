"""
Raw data ingestion logic for EEGADHD dataset.
Handles scanning directories, parsing .pnt metadata files, and basic ID mapping.
"""

import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Set, Optional, Dict, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_subject_ids(source_dir: Path) -> List[str]:
    """
    Scan source_dir and return sorted unique subject IDs.
    Assumes files/folders start with ID plus a dot.
    """
    ids: Set[str] = set()
    for item in source_dir.iterdir():
        if item.name.startswith('.'):
            continue
        # Pattern matches "XX12345." or similar
        m = re.match(r"^([A-Z]{2}\d{5,6}[A-Z]?)\.", item.name)
        if m:
            ids.add(m.group(1))
        else:
            # Fallback for folder names that might be just the ID
            prefix = item.stem
            if prefix and not prefix.startswith("."):
                ids.add(prefix)
    ids.discard("DskUUID")
    return sorted(list(ids))


def parse_pnt_metadata(pnt_path: Path) -> Dict[str, Union[str, datetime, None]]:
    """
    Parse a brainvision-style .pnt file (text) to extract:
    - 'original_id': The ID string found in the file
    - 'meas_date': A timezone-aware (UTC) datetime of recording start
    """
    if not pnt_path.exists():
        return {"original_id": None, "meas_date": None}

    try:
        text = pnt_path.read_bytes().decode("ISO-8859-1", errors="ignore").replace("\x00", "")
    except Exception as e:
        LOGGER.warning("Failed to read %s: %s", pnt_path, e)
        return {"original_id": None, "meas_date": None}

    # --- parse numeric ID ---
    original_id = None
    match = re.search(r"ID(\d+(?:\.\d+)?)", text)
    if match:
        original_id = match.group(1).split('.')[0]  # strip potential suffix

    # --- parse Date + Start Time ---
    meas_dt = None
    date_match = re.search(r"Date(\d{4})/(\d{2})/(\d{2})", text)
    start_match = re.search(r"Start Time(\d{2})(\d{2})(\d{2})", text)
    if date_match and start_match:
        try:
            year, month, day = map(int, date_match.groups())
            sh, sm, ss = map(int, start_match.groups())
            # Recording timestamps need to be UTC-aware for BIDS export
            meas_dt = datetime(year, month, day, sh, sm, ss, tzinfo=timezone.utc)
        except ValueError:
            pass

    return {
        "original_id": original_id,
        "meas_date": meas_dt
    }


def map_subject_id(raw_id: str, mapping_df: pd.DataFrame) -> Optional[str]:
    """
    Map a raw ID (e.g., '2.2', '1234') to the study patient ID using the mapping DataFrame.
    Returns None if the subject should be skipped.
    """
    if raw_id == "2.2":
        # Specific exclusion rule
        return None
    
    # If ID corresponds to 'ID' column in mapping, return 'patient' column
    if raw_id and raw_id.isdigit():
        try:
            rid_int = int(raw_id)
            mapped = mapping_df[mapping_df["ID"] == rid_int]
            if not mapped.empty:
                return str(int(mapped["patient"].iat[0]))
        except ValueError:
            pass
            
    # Default: if no mapping found, return original (or handle roughly)
    return raw_id


def find_eeg_file(raw_dir: Path, subject_id: str) -> Optional[Path]:
    """Locate the .EEG file for a given subject prefix."""
    candidates = list(raw_dir.glob(f"{subject_id}.EEG"))
    if not candidates:
        return None
    return candidates[0]

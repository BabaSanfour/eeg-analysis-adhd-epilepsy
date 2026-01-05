"""
patients.py - IO and cleaning utilities for patients dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from eeg_adhd_epilepsy.io.csv import load as load_csv
from eeg_adhd_epilepsy.utils.config import MAPPING_PSYCHOSTIMULANT

# Constants
EPILEPSY_MED_COLS = [
    "LEV", "LTG", "LCS", "CLB", "CBZ", "VPA", "ETH", 
    "TPM", "RUF", "BRV", "STP", "OXZ", "CBM"
]

def load_raw_patients_df(filepath: Path) -> pd.DataFrame:
    """Load raw patients CSV/Excel file."""
    if filepath.suffix == '.xlsx':
        # Fallback for Excel if needed, though load_csv handles csv/tsv
        df = pd.read_excel(filepath) 
    else:
        df = load_csv(str(filepath), sep=",") # Default to comma for now, load func handles detection
    
    # Basic cleanup of empty columns
    empty_cols = [c for c in df.columns if str(c).strip() == ""]
    if empty_cols:
        logging.info(f"Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
        
    return df

def _log_potential_entries(df: pd.DataFrame) -> None:
    """Report counts of '0 (potentiel)' per column before dropping."""
    pattern = re.compile(r"^\s*0\s*\(potentiel\)\s*$", flags=re.IGNORECASE)
    cols_with_potential = {}
    check_cols = [c for c in ["TDAH", "ADHD", "Epilepsy", "TSA"] if c in df.columns]
    
    for col in check_cols:
        matches = df[col].astype(str).str.match(pattern)
        count = int(matches.sum())
        if count > 0:
            cols_with_potential[col] = count
            
    if cols_with_potential:
        # User requested exact format: "Columns with '0 (potentiel)': {'TDAH': 10, ...}"
        logging.info(f"Columns with '0 (potentiel)': {cols_with_potential}")

def _study_id_to_folder(study_id) -> str | None:
    """Convert Study ID to BIDS folder name."""
    sid = pd.to_numeric(study_id, errors="coerce")
    if pd.isna(sid):
        return None
    try:
        return f"sub-{int(sid):04d}"
    except (ValueError, TypeError):
        return None

def validate_bids_coverage(df: pd.DataFrame, bids_root: Path) -> Dict[str, object]:
    """Check which Study IDs exist in the BIDS folder structure."""
    results: Dict[str, object] = {}
    if "Study ID" not in df.columns:
        return results

    bids_root = Path(bids_root)
    sid_series = df["Study ID"].dropna().unique()
    expected = []
    
    # Identify expected headers
    for sid in sid_series:
        folder = _study_id_to_folder(sid)
        if folder:
            expected.append(folder)

    present = []
    missing = []
    missing_ids = []

    for folder in expected:
        if (bids_root / folder).exists():
            present.append(folder)
        else:
            missing.append(folder)
            try:
                missing_ids.append(int(folder.split("-")[1]))
            except (IndexError, ValueError):
                pass
    
    results["bids_present"] = present
    results["bids_missing"] = missing
    results["missing_study_ids"] = missing_ids
    results["expected_count"] = len(expected)
    results["missing_count"] = len(missing)
    
    if missing:
        logging.info(f"Missing EEG data for {len(missing)} subjects.")
        
    if missing_ids:
        # Also find the Pt ID for missing Study IDs for logging
        if "Pt ID" in df.columns:
            missing_pairs = []
            for m_sid in missing_ids:
                row = df[df["Study ID"] == m_sid]
                if not row.empty:
                    # Handle if there are duplicates (take first)
                    ptid = row.iloc[0]["Pt ID"]
                    # Format: 'PtID:StudyID'
                    pt_str = str(int(ptid)) if pd.notna(ptid) else '?'
                    sid_str = str(m_sid)
                    missing_pairs.append(f"'{pt_str}:{sid_str}'")
            if missing_pairs:
                # Format: ['2395351:119', '3146650:276']
                formatted_list = "[" + ", ".join(missing_pairs) + "]"
                logging.info(f"- Missing (Pt ID:Study ID): {formatted_list}")

    return results

def _normalize_psychostim_description(desc: str | float | None) -> str:
    """Normalize psychostimulant description labels."""
    if pd.isna(desc):
        return "Missing/NA"
    s = str(desc).strip()
    if s.lower() == "no psychostimulants":
        return "no psychostimulants"
    
    replacements = {
        "Lisdexamfetamine (d/c)": "Lisdexamfetamine",
        "Methylphenidate (d/c)": "Methylphenidate",
        "Methylphenidate, Methylphenidate": "Methylphenidate",
        "Methylphenidate (d/c 2019)": "Methylphenidate",
        "Lisdexamfetamine, Methylphenidate": "Lisdexamfetamine + Methylphenidate",
    }
    return replacements.get(s, s)

def _drop_zero_potential_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any diagnosis column contains '0 (potentiel)'."""
    regex = r"^\s*0\s*\(potentiel\)\s*$"
    
    # Columns to check
    check_cols = [c for c in ["TDAH", "ADHD", "Epilepsy", "TSA"] if c in df.columns]
    
    if not check_cols:
        return df

    mask = df[check_cols].apply(
        lambda col: col.astype(str).str.match(regex, flags=re.IGNORECASE)
    ).any(axis=1)
    
    n_dropped = mask.sum()
    if n_dropped > 0:
        return df[~mask].copy()
    
    return df

def _compute_medication_mismatches(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Return mask of rows to drop due to mismatches and list of ID pairs."""
    drop_mask = pd.Series(False, index=df.index)
    
    # 1. Psychostimulant in Non-ADHD    
    if "has_adhd" in df.columns and "has_psychostimulant" in df.columns:
        # has_adhd is 1/True for ADHD.
        mask = (~df["has_adhd"]) & df["has_psychostimulant"]
        if mask.any():
            pair_strs = []
            if {"Pt ID", "Study ID"}.issubset(df.columns):
                subset = df.loc[mask, ["Pt ID", "Study ID"]]
                for _, r in subset.iterrows():
                     p = int(r['Pt ID']) if pd.notna(r['Pt ID']) else '?'
                     s = int(r['Study ID']) if pd.notna(r['Study ID']) else '?'
                     pair_strs.append(f"'{p}:{s}'")
            
            formatted_pairs = "[" + ", ".join(pair_strs) + "]"
            logging.info(f"- Psychostimulant in non-ADHD patients (Pt ID:Study ID pairs: {formatted_pairs})")
            drop_mask = drop_mask | mask
        else:
             logging.info("- Psychostimulant in non-ADHD patients (Pt ID:Study ID pairs: [])")
    else:
        logging.info("- Psychostimulant in non-ADHD patients (Pt ID:Study ID pairs: [])")

    # 2. ASM in Non-Epilepsy
    if "has_epilepsy" in df.columns and "has_epilepsy_med" in df.columns:
        mask = (~df["has_epilepsy"]) & df["has_epilepsy_med"]
        if mask.any():
            pair_strs = []
            if {"Pt ID", "Study ID"}.issubset(df.columns):
                 subset = df.loc[mask, ["Pt ID", "Study ID"]]
                 for _, r in subset.iterrows():
                     p = int(r['Pt ID']) if pd.notna(r['Pt ID']) else '?'
                     s = int(r['Study ID']) if pd.notna(r['Study ID']) else '?'
                     pair_strs.append(f"'{p}:{s}'")

            formatted_pairs = "[" + ", ".join(pair_strs) + "]"
            logging.info(f"- Anti-seizure meds in non-epilepsy patients: (Pt ID:Study ID pairs: {formatted_pairs})")
            drop_mask = drop_mask | mask
        else:
            logging.info("- Anti-seizure meds in non-epilepsy patients: None (Pt ID:Study ID pairs: [])")
    else:
        logging.info("- Anti-seizure meds in non-epilepsy patients: None (Pt ID:Study ID pairs: [])")

    return drop_mask

def _drop_pt_id_duplicates_keep_smallest_study(df: pd.DataFrame, force_drop_mask: pd.Series) -> pd.DataFrame:
    """
    Deduplicate Pt IDs, keeping the one with the smallest Study ID.
    Also applies the force_drop_mask (mismatches) BEFORE deduplication check.
    """
    
    # Check for duplicates BEFORE dropping mismatches for full transparency report
    if {"Pt ID", "Study ID"}.issubset(df.columns):
        vc = df["Pt ID"].value_counts()
        dups = vc[vc > 1].index
        
        if len(dups) > 0:
            mapping = {}
            subset = df[df["Pt ID"].isin(dups)]
            for pt_id, grp in subset.groupby("Pt ID"):
                 # get list of study IDs
                 sids = sorted(grp["Study ID"].dropna().astype(int).tolist())
                 mapping[int(pt_id)] = sids
            
            logging.info(f"Duplicate Pt IDs with Study IDs: {mapping}")
            logging.info(f"- Total duplicate Pt IDs: {len(dups)}")
            logging.info("- We drop rows with larger 'Study ID' per Pt ID.")
        else:
            logging.info("No duplicate Pt IDs found.")

    logging.info("Medication mismatches:")
    # First, simply drop the forced mismatches (computed outside)
    if force_drop_mask.any():
        n_forced = force_drop_mask.sum()
        # Logging handled by compute function for details
        logging.info(f"- Total mismatches: {n_forced}")
        logging.info("- We drop rows who have mismatches.")
        df = df[~force_drop_mask].copy()
    else:
        # Logging handled above
        logging.info(f"- Total mismatches: 0")
        logging.info("- We drop rows who have mismatches.")

    if "Pt ID" not in df.columns or "Study ID" not in df.columns:
        return df
        
    # Deduplicate: Sort by Pt ID then Study ID (ascending) -> Keep first
    df["_sid_temp"] = pd.to_numeric(df["Study ID"], errors="coerce")
    df = df.sort_values(by=["Pt ID", "_sid_temp"])
    df = df.drop_duplicates(subset=["Pt ID"], keep="first")
    df = df.drop(columns=["_sid_temp"])
    
    return df

def clean_patients_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply standard cleaning, type conversion, and flag creation."""
    
    # 0. Log Raw Stats
    n_initial = len(df)
    logging.info(f"Total Subjects in CSV: {n_initial}")
    
    # Check missing Pt ID
    if "Pt ID" in df.columns:
        missing_pt = df["Pt ID"].isna() | (df["Pt ID"].astype(str).str.strip() == "")
        missing_pt = df["Pt ID"].isna() | (df["Pt ID"].astype(str).str.strip() == "")
        n_missing_pt = missing_pt.sum()
        if n_missing_pt > 0:
            logging.info(f"Rows with missing Pt ID: {n_missing_pt}")
    else:
        n_missing_pt = 0
    
    # 1. Log & Drop "0 (potentiel)"
    # 1. Log & Drop "0 (potentiel)"
    _log_potential_entries(df)
    out = _drop_zero_potential_rows(df)
    n_after_potential = len(out)
    logging.info(f"Subjects after dropping '0 (potentiel)': {n_after_potential}")
    
    # 2. Normalize Sex
    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].astype(str).str.upper().str.strip()

    # 3. Numeric conversion helper
    def to_numeric_safe(col):
        return pd.to_numeric(col, errors="coerce")
    
    # 4. Diagnosis Flags
    # TDAH/ADHD
    tdah_col = "TDAH" if "TDAH" in out.columns else ("ADHD" if "ADHD" in out.columns else None)
    if tdah_col:
        vals = to_numeric_safe(out[tdah_col])
        out["has_adhd"] = vals.isin([1])
        
    if "Epilepsy" in out.columns:
        vals = to_numeric_safe(out["Epilepsy"])
        out["has_epilepsy"] = vals.isin([1])
        
    if "TSA" in out.columns:
        vals = to_numeric_safe(out["TSA"])
        out["has_tsa"] = vals.isin([1])

    # 5. Psychostimulants
    if "psychostimulant_description" in out.columns:
        out["psychostimulant_category"] = out["psychostimulant_description"].map(
            MAPPING_PSYCHOSTIMULANT
        ).fillna(0).astype(int)
        
        # normalize description text for display
        out["psychostimulant_description_clean"] = out["psychostimulant_description"].apply(
            _normalize_psychostim_description
        )

    if "psychostimulant_category" in out.columns:
        out["has_psychostimulant"] = out["psychostimulant_category"] > 0
    else:
        out["has_psychostimulant"] = False

    # 6. ASMs (Epilepsy Meds)
    asm_bool_cols = []
    for col in EPILEPSY_MED_COLS:
        if col in out.columns:
            # force numeric
            vals = to_numeric_safe(out[col]).fillna(0)
            bool_col = f"{col}_bool"
            out[bool_col] = vals == 1
            asm_bool_cols.append(bool_col)
    
    if asm_bool_cols:
        out["n_epilepsy_meds"] = out[asm_bool_cols].sum(axis=1)
        out["has_epilepsy_med"] = out["n_epilepsy_meds"] > 0
        out["has_multiple_epilepsy_meds"] = out["n_epilepsy_meds"] >= 2
    else:
        out["has_epilepsy_med"] = False
        out["n_epilepsy_meds"] = 0

    # 7. Age Groups
    if "Age" in out.columns:
        out["Age"] = to_numeric_safe(out["Age"])
        out["age_group"] = pd.cut(
            out["Age"],
            bins=[5, 9, 13, 19],
            labels=["late_childhood_5_8", "early_ado_9_12", "ado_13_18"],
            right=False
        )
    
    # 8. Mismatches & Deduplication
    mismatch_mask = _compute_medication_mismatches(out)
    
    # Manually drop mismatches
    n_mismatches = 0
    if mismatch_mask.any():
        n_mismatches = mismatch_mask.sum()
        out = out[~mismatch_mask].copy()
        logging.info(f"Dropped {n_mismatches} medication mismatches.")
    else:
        logging.info("No medication mismatches dropped.")
    
    n_after_mismatch = len(out)
    logging.info(f"Subjects after dropping mismatches: {n_after_mismatch}")
    
    # Dedup
    out = _drop_pt_id_duplicates_keep_smallest_study(out, force_drop_mask=pd.Series(False, index=out.index))
    n_final = len(out)
    logging.info(f"Subjects after dropping duplicates: {n_final}")
    
    logging.info(f"Total subjects used: {n_final}")
    
    stats = {
        "n_initial": n_initial,
        "n_missing_ptid": n_missing_pt,
        "n_potential_dropped": n_initial - n_after_potential,
        "n_after_potential": n_after_potential,
        "n_mismatches_dropped": n_mismatches,
        "n_after_mismatch": n_after_mismatch,
        "n_duplicates_dropped": n_after_mismatch - n_final,
        "n_final": n_final
    }

    return out, stats

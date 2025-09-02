"""
merge_features_patients.py

Merges aggregate EEG feature CSV (columns like feature-*.spaces-*) with the
patients metadata CSV to produce a single dataset for ML.

Defaults:
- Features CSV: csv_dir/aggregate@raw.csv (expects a `subject` column)
- Patients CSV: csv_dir/EEG_Psychostimulants_PatientList_08-2025.csv (expects `Study ID`)
- Join keys: features.subject == patients."Study ID"
- Join how: inner (keeps intersection)

Applies patient cleaning consistent with the explorer:
- Drops duplicate Pt IDs keeping the row with the smallest Study ID
- Normalizes values (e.g., '0 (potentiel)' -> 2) and ensures psychostimulant flags

Saves merged output to results/merged/features_patients.csv by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from eeg_adhd_epilepsy_psychostimulant.utils.config import csv_dir, results_dir
from eeg_adhd_epilepsy_psychostimulant.io.csv import load as load_csv

# Reuse cleaning utilities from patients explorer
# Reuse cleaning utilities (defined in patients_data module)
from eeg_adhd_epilepsy_psychostimulant.explore.patients_data import (
    _drop_pt_id_duplicates_keep_smallest_study,
    _normalize_values_and_types,
    _ensure_psychostimulant_flags,
    _ensure_age_groups_numeric,
)

from eeg_adhd_epilepsy_psychostimulant.explore.features import (
    _is_feature_col,
    _feature_base,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def prepare_features_df(path: str, subject_col: str = "subject") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Keep id columns (if present) and all feature columns
    keep_cols: List[str] = [c for c in ("dataset", "id", subject_col, "task") if c in df.columns]
    feat_cols = [c for c in df.columns if _is_feature_col(c)]
    if subject_col not in df.columns:
        raise KeyError(f"'{subject_col}' column not found in features CSV")
    out = df[keep_cols + feat_cols].copy()
    logging.info(f"Features: {out.shape[0]} rows, {out.shape[1]} cols (kept {len(feat_cols)} features)")
    return out


def prepare_patients_df(path: str, study_id_col: str = "Study ID") -> pd.DataFrame:
    df = load_csv(path, sep=",")
    df = _drop_pt_id_duplicates_keep_smallest_study(df)
    df = _normalize_values_and_types(df)
    df = _ensure_psychostimulant_flags(df)
    df = _ensure_age_groups_numeric(df)
    if study_id_col not in df.columns:
        raise KeyError(f"'{study_id_col}' column not found in patients CSV")
    logging.info(f"Patients: {df.shape[0]} rows, {df.shape[1]} cols after cleaning")
    return df


def coerce_merge_keys(left: pd.Series, right: pd.Series, numeric: bool = True) -> tuple[pd.Series, pd.Series]:
    if numeric:
        return (
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        )
    # else stringify and strip
    return (left.astype(str).str.strip(), right.astype(str).str.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge aggregate features and patients data for ML.")
    parser.add_argument("--features_csv", type=str, default=str(Path(csv_dir) / "aggregate@raw.csv"))
    parser.add_argument("--patients_csv", type=str, default=str(Path(csv_dir) / "EEG_Psychostimulants_PatientList_08-2025.csv"))
    parser.add_argument("--features_key", type=str, default="subject", help="Join key in features CSV (default: subject)")
    parser.add_argument("--patients_key", type=str, default="Study ID", help="Join key in patients CSV (default: Study ID)")
    parser.add_argument("--how", type=str, default="inner", choices=["inner", "left", "right", "outer"], help="Join type (default: inner)")
    parser.add_argument("--numeric_keys", action="store_true", default=True, help="Coerce join keys to numeric before merging (default: True)")
    parser.add_argument("--out_csv", type=str, default=str(Path(results_dir) / "merged" / "features_patients.csv"))
    args = parser.parse_args()

    feat = prepare_features_df(args.features_csv, subject_col=args.features_key)
    pats = prepare_patients_df(args.patients_csv, study_id_col=args.patients_key)

    # Prepare keys with coercion
    lkey, rkey = coerce_merge_keys(feat[args.features_key], pats[args.patients_key], numeric=args.numeric_keys)
    feat = feat.assign(__merge_key__=lkey)
    pats = pats.assign(__merge_key__=rkey)

    # Report unmatched before merge
    lset = set(feat["__merge_key__"].dropna().unique())
    rset = set(pats["__merge_key__"].dropna().unique())
    only_feat = lset - rset
    only_pats = rset - lset
    logging.info(f"Subjects only in features (pre-merge): {len(only_feat)}")
    logging.info(f"Subjects only in patients (pre-merge): {len(only_pats)}")

    merged = feat.merge(pats, on="__merge_key__", how=args.how, suffixes=("_feat", "_pat"))
    merged = merged.drop(columns=["__merge_key__"])  # clean helper key

    # Drop redundant/unnecessary columns post-merge
    drop_cols = []
    # Prefer keeping features' subject; drop patients' Study ID if both present
    if "subject" in merged.columns and "Study ID" in merged.columns:
        drop_cols.append("Study ID")
    # Patient ID not needed downstream
    for c in ["Pt ID", "psychostimulant_description", "Psychostimulant (y/n)"]:
        if c in merged.columns:
            drop_cols.append(c)
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
        logging.info(f"Dropped redundant columns: {drop_cols}")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logging.info(f"Merged shape: {merged.shape}; saved to {out_path}")


if __name__ == "__main__":
    main()

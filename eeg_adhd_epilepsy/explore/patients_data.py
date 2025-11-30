"""
patients_csv_explorer.py

Explore the ADHD/Epilepsy patients CSV with psychostimulant + anti-seizure medication columns.
Prints dataset info, value counts, medication summaries, confusion matrices, and
optionally exports CSV/plots plus an HTML report.

Usage:
  python -m eeg_adhd_epilepsy_psychostimulant.explore.patients_data \
    --csv_file <path> [--grouped] [--save] [--html_report]
"""

from __future__ import annotations

import argparse
import logging
import base64
import re
from typing import Optional
from pathlib import Path
from datetime import datetime
import itertools
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_adhd_epilepsy.io.csv import load
from eeg_adhd_epilepsy.utils.config import (
    csv_dir,
    MAPPING_PSYCHOSTIMULANT,
    results_dir,
    data_dir,
    bids_dir as default_bids_dir,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

EPILEPSY_MED_COLS = [
    "LEV",
    "LTG",
    "LCS",
    "CLB",
    "CBZ",
    "VPA",
    "ETH",
    "TPM",
    "RUF",
    "BRV",
    "STP",
    "OXZ",
    "CBM",
]


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnamed/empty columns that can leak from Excel exports."""
    empty_cols = [c for c in df.columns if str(c).strip() == ""]
    if empty_cols:
        logging.info(f"Dropping empty/unnamed columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
    return df


def _log_potential_entries(df: pd.DataFrame) -> dict[str, int]:
    """Report counts of '0 (potentiel)' per column before normalization."""
    pattern = re.compile(r"^\s*0\s*\(potentiel\)\s*$", flags=re.IGNORECASE)
    cols_with_potential = {}
    for col in df.columns:
        matches = df[col].astype(str).str.match(pattern)
        count = int(matches.sum())
        if count > 0:
            cols_with_potential[col] = count
    if cols_with_potential:
        logging.info(
            "Number of subjects with '0 (potentiel)' entries in one of the three diagnosis columns "
            f"({len(cols_with_potential)} cols): {cols_with_potential}"
        )
    else:
        logging.info("No '0 (potentiel)' entries found in the dataset.")
    return cols_with_potential


def _normalize_sex_values(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case the Sex column to keep a consistent categorical domain."""
    out = df.copy()
    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].astype(str).str.upper()
    return out


def _normalize_psychostim_label(desc: str | float | None) -> str:
    """Normalize psychostimulant description labels for reporting."""
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


def _study_id_to_bids_folder(study_id) -> str | None:
    """Convert a Study ID to sub-XXXX BIDS folder name (4-digit zero padded)."""
    sid = pd.to_numeric(study_id, errors="coerce")
    if pd.isna(sid):
        return None
    try:
        sid_int = int(sid)
    except (TypeError, ValueError):
        return None
    return f"sub-{sid_int:04d}"


def check_bids_subject_folders(df: pd.DataFrame, bids_root: Path) -> dict[str, object]:
    """Check that BIDS subject folders exist for Study IDs present in df."""
    results: dict[str, object] = {}
    if "Study ID" not in df.columns:
        return results

    bids_root = Path(bids_root)
    sid_series = df["Study ID"].dropna().unique()
    expected = []
    for sid in sid_series:
        folder = _study_id_to_bids_folder(sid)
        if folder:
            expected.append(folder)

    missing = []
    present = []
    for folder in expected:
        path = bids_root / folder
        if path.exists():
            present.append(folder)
        else:
            missing.append(folder)

    results["bids_present"] = present
    results["bids_missing"] = missing
    results["bids_root"] = str(bids_root)
    results["expected_count"] = len(expected)
    results["missing_count"] = len(missing)
    return results


def _normalize_psychostim_label(desc: str | float | None) -> str:
    """Normalize psychostimulant description labels for reporting."""
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


def _parse_psychostimulant_category_value(value) -> float | np.nan:
    """Extract the first numeric code from the psychostimulant category cell."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        tokens = re.findall(r"\d+", value)
        if tokens:
            try:
                return float(tokens[0])
            except ValueError:
                return np.nan
    return np.nan


def _diagnosis_bool(series: pd.Series, include_potential: bool = True) -> pd.Series:
    """Convert diagnosis columns (0/1/2) to boolean flags."""
    numeric = pd.to_numeric(series, errors="coerce")
    positives = {1, 2} if include_potential else {1}
    return numeric.fillna(0).astype(int).isin(positives)


def _add_diagnosis_flags(df: pd.DataFrame, include_potential: bool = True) -> pd.DataFrame:
    """Add boolean flags for ADHD/TDAH, Epilepsy, and TSA."""
    out = df.copy()
    tdah_col = "TDAH" if "TDAH" in out.columns else ("ADHD" if "ADHD" in out.columns else None)
    if tdah_col:
        out["has_adhd"] = _diagnosis_bool(out[tdah_col], include_potential)
    if "Epilepsy" in out.columns:
        out["has_epilepsy"] = _diagnosis_bool(out["Epilepsy"], include_potential)
    if "TSA" in out.columns:
        out["has_tsa"] = _diagnosis_bool(out["TSA"], include_potential)
    return out


def _add_medication_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate anti-seizure medication flags/counts."""
    out = df.copy()
    asm_bool_cols = []
    for col in EPILEPSY_MED_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            bool_col = f"{col}_bool"
            out[bool_col] = out[col].fillna(0).astype(int) == 1
            asm_bool_cols.append(bool_col)
    if asm_bool_cols:
        asm_matrix = out[asm_bool_cols]
        out["n_epilepsy_meds"] = asm_matrix.sum(axis=1)
        out["has_epilepsy_med"] = out["n_epilepsy_meds"] > 0
        out["has_multiple_epilepsy_meds"] = out["n_epilepsy_meds"] >= 2
    else:
        out["n_epilepsy_meds"] = 0
        out["has_epilepsy_med"] = False
        out["has_multiple_epilepsy_meds"] = False
    return out


def _ensure_psychostimulant_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure presence of usable psychostimulant indicators.

    - Creates ``psychostimulant_category`` from description when missing.
    - Creates boolean ``has_psychostimulant`` only from mapped category.
    """
    out = df.copy()

    if "psychostimulant_description" in out.columns:
        out["psychostimulant_category"] = out["psychostimulant_description"].map(
            MAPPING_PSYCHOSTIMULANT
        )
    # fill np.nan with 0
    out["psychostimulant_category"] = out["psychostimulant_category"].fillna(0)

    # Derive has_psychostimulant strictly from mapped category
    if "psychostimulant_category" in out.columns:
        out["has_psychostimulant"] = (
            pd.to_numeric(out["psychostimulant_category"], errors="coerce").fillna(0) > 0
        )
    else:
        out["has_psychostimulant"] = False
    out["psychostimulant_category"] = out["psychostimulant_category"].astype(int)
    return out


def _compute_medication_mismatches(
    df: pd.DataFrame, include_potential: bool = True
) -> dict[str, object]:
    """Return counts, IDs, and masks for medication/diagnosis inconsistencies."""
    results: dict[str, object] = {}

    def _collect_ids(series: pd.Series) -> list[int]:
        num = pd.to_numeric(series, errors="coerce")
        return num.dropna().astype(int).tolist()

    def _collect_pairs(mask: pd.Series) -> list[str]:
        if not {"Pt ID", "Study ID"}.issubset(df.columns):
            return []
        sub = df.loc[mask, ["Pt ID", "Study ID"]].apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna()
        return [f"{int(r['Pt ID'])}:{int(r['Study ID'])}" for _, r in sub.iterrows()]

    # Non-ADHD on psychostimulant
    tdah_col = "TDAH" if "TDAH" in df.columns else ("ADHD" if "ADHD" in df.columns else None)
    if tdah_col and "has_psychostimulant" in df.columns:
        adhd_flag = _diagnosis_bool(df[tdah_col], include_potential)
        mask = (~adhd_flag) & df["has_psychostimulant"]
        results["psychostim_in_non_adhd_count"] = int(mask.sum())
        results["psychostim_in_non_adhd_mask"] = mask
        if "Pt ID" in df.columns:
            results["psychostim_in_non_adhd_pt_ids"] = _collect_ids(df.loc[mask, "Pt ID"])
        if "Study ID" in df.columns:
            results["psychostim_in_non_adhd_study_ids"] = _collect_ids(df.loc[mask, "Study ID"])
        results["psychostim_in_non_adhd_pairs"] = _collect_pairs(mask)

    # Non-epilepsy on ASM
    if "Epilepsy" in df.columns and "has_epilepsy_med" in df.columns:
        epi_flag = _diagnosis_bool(df["Epilepsy"], include_potential)
        mask = (~epi_flag) & df["has_epilepsy_med"]
        results["asm_in_non_epilepsy_count"] = int(mask.sum())
        results["asm_in_non_epilepsy_mask"] = mask
        if "Pt ID" in df.columns:
            results["asm_in_non_epilepsy_pt_ids"] = _collect_ids(df.loc[mask, "Pt ID"])
        if "Study ID" in df.columns:
            results["asm_in_non_epilepsy_study_ids"] = _collect_ids(df.loc[mask, "Study ID"])
        results["asm_in_non_epilepsy_pairs"] = _collect_pairs(mask)

    return results


def _drop_mismatches_and_duplicates(
    df: pd.DataFrame,
    med_mismatches: dict[str, object],
    data_dir_path: Path,
) -> pd.DataFrame:
    """Drop medication mismatches and resolve duplicate Pt IDs in one step."""
    mismatch_mask = med_mismatches.get(
        "psychostim_in_non_adhd_mask", pd.Series(False, index=df.index)
    )
    mismatch_mask = mismatch_mask | med_mismatches.get("asm_in_non_epilepsy_mask", False)
    mismatch_mask = mismatch_mask.reindex(df.index, fill_value=False)

    force_study_ids = set(
        med_mismatches.get("psychostim_in_non_adhd_study_ids", [])
        + med_mismatches.get("asm_in_non_epilepsy_study_ids", [])
    )

    deduped_df, drop_info = _drop_pt_id_duplicates_keep_smallest_study(
        df,
        force_drop_mask=mismatch_mask,
        force_drop_study_ids=force_study_ids,
    )

    if drop_info.get("forced_drop_rows", 0) > 0:
        mismatch_pickle = Path(data_dir_path) / "medication_mismatches.pickle"
        mismatch_info = {
            "med_mismatches": med_mismatches,
            "dropped_pairs": drop_info.get("forced_pairs", []),
        }
        mismatch_pickle.parent.mkdir(parents=True, exist_ok=True)
        with mismatch_pickle.open("wb") as f:
            pickle.dump(mismatch_info, f)
        logging.info(
            f"Dropped {drop_info['forced_drop_rows']} medication/duplicate conflict rows; "
            f"saved details to {mismatch_pickle}"
        )
    return deduped_df


def _add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" in out.columns:
        # Keep consistent with the analysis_exploration binning
        out["age_group"] = pd.cut(
            out["Age"],
            bins=[0, 12, 19],
            labels=["child", "teen"],
            right=False,
        )
    return out


def _normalize_values_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV values and enforce numeric types.

    - Change '0 (potentiel)' entries into numeric 2 across all columns.
    - Convert all columns to numeric except text ones. Should run after
    """
    out = df.copy()

    # Replace variations of '0 (potentiel)' with 2 (trim spaces, case tolerant)
    out = out.replace(to_replace=r"^\s*0\s*\(potentiel\)\s*$", value=2, regex=True)

    keep_text = {"psychostimulant_description", "Sex"}
    for col in out.columns:
        if col not in keep_text:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _ensure_age_groups_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric age_groups column: 1=child [0,12), 2=teen [12,19)."""
    out = df.copy()
    if "Age" in out.columns:
        bins = [0, 12, 19]
        labels = [1, 2]
        out["age_groups"] = pd.cut(out["Age"], bins=bins, labels=labels, right=False)
        # Cast to numeric ints where possible
        out["age_groups"] = pd.to_numeric(out["age_groups"], errors="coerce")
    return out


def _pct(num: int, denom: int) -> float:
    """Convenience percentage helper with NaN on divide-by-zero."""
    if denom == 0:
        return np.nan
    return (num / denom) * 100


def apply_diagnosis_filter(
    df: pd.DataFrame,
    diag_col: str,
    condition: str,
    include_potential: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame rows by a diagnosis column using normalized numeric codes.

    Expected values: 1=positive, 0=negative, 2=potential (formerly '0 (potentiel)').
    - with: keep 1 (+ 2 if include_potential is True)
    - without: keep 0 (and 2 if include_potential is False)
    - combined: no filtering
    Missing column -> no-op.
    """
    if diag_col not in df.columns:
        return df
    if condition == "combined":
        return df
    # 1 means positive; 2 means potential (treated as positive when include_potential=True)
    positives = {1, 2} if include_potential else {1}
    # For "without", include potentials only when not included in positives
    negatives = {0} if include_potential else {0, 2}
    values = positives if condition == "with" else negatives if condition == "without" else None
    if values is None:
        return df
    col = pd.to_numeric(df[diag_col], errors="coerce")
    return df[col.isin(values)]


diag_filter_options = ["with", "without", "combined"]
diagnosis_columns = ["TDAH", "Epilepsy", "TSA"]


sex_filters = {
    "F": lambda d: d[d["Sex"] == "F"],
    "M": lambda d: d[d["Sex"] == "M"],
    "Combined": lambda d: d,
}


age_filters = {
    1: lambda d: d[d["age_groups"] == 1],
    2: lambda d: d[d["age_groups"] == 2],
    "Combined": lambda d: d,
}


def get_counts_by_med_analysis(
    df: pd.DataFrame,
    sex_key: str,
    age_key,
    diag_filters: dict,
    analysis_type: str,
    include_potential: bool = True,
):
    """Calculate subject counts based on analysis type and filters."""
    filtered_df = sex_filters[sex_key](df)
    filtered_df = age_filters[age_key](filtered_df)
    for diag in diagnosis_columns:
        filtered_df = apply_diagnosis_filter(
            filtered_df,
            diag,
            diag_filters.get(diag, "combined"),
            include_potential,
        )

    # Determine control/medication using normalized fields
    if "psychostimulant_category" in filtered_df.columns:
        is_control = filtered_df["psychostimulant_category"].fillna(0).astype(int).eq(0)
    elif "Psychostimulant (y/n)" in filtered_df.columns:
        is_control = (
            pd.to_numeric(filtered_df["Psychostimulant (y/n)"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(0)
        )
    elif "has_psychostimulant" in filtered_df.columns:
        is_control = ~filtered_df["has_psychostimulant"].fillna(False)
    else:
        is_control = pd.Series(True, index=filtered_df.index)

    if analysis_type == "ctrl_vs_all":
        count_control = int(is_control.sum())
        count_med = int((~is_control).sum())
        return count_control, count_med, "Control", "Med"

    elif analysis_type == "med1_vs_med2":
        med_df = filtered_df[filtered_df["psychostimulant_category"].isin([1, 2])]
        count_med1 = int((med_df["psychostimulant_category"] == 1).sum())
        count_med2 = int((med_df["psychostimulant_category"] == 2).sum())
        return count_med1, count_med2, "Med1", "Med2"

    elif analysis_type == "ctrl_vs_med1":
        sub_df = filtered_df[filtered_df["psychostimulant_category"].isin([0, 1])].copy()
        count_control = int(
            sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0).sum()
        )
        count_med1 = int((sub_df["psychostimulant_category"] == 1).sum())
        return count_control, count_med1, "Control", "Med1"

    elif analysis_type == "ctrl_vs_med2":
        sub_df = filtered_df[filtered_df["psychostimulant_category"].isin([0, 2])].copy()
        count_control = int(
            sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0).sum()
        )
        count_med2 = int((sub_df["psychostimulant_category"] == 2).sum())
        return count_control, count_med2, "Control", "Med2"

    return None, None, None, None


def create_analysis_dataframe(
    df: pd.DataFrame,
    analysis_type: str,
    include_potential: bool = True,
) -> pd.DataFrame:
    """Create a summary DataFrame with counts for each combination of filters."""
    results = []
    for sex_key in sex_filters.keys():
        for age_key in age_filters.keys():
            for diag_combo in itertools.product(diag_filter_options, repeat=3):
                diag_filters = dict(zip(diagnosis_columns, diag_combo))
                count1, count2, label1, label2 = get_counts_by_med_analysis(
                    df,
                    sex_key,
                    age_key,
                    diag_filters,
                    analysis_type,
                    include_potential,
                )

                sub_df = df.copy()
                sub_df = sex_filters[sex_key](sub_df)
                sub_df = age_filters[age_key](sub_df)
                for diag in diagnosis_columns:
                    sub_df = apply_diagnosis_filter(
                        sub_df,
                        diag,
                        diag_filters.get(diag, "combined"),
                        include_potential,
                    )

                overall_age_mean = sub_df["Age"].mean() if not sub_df.empty else np.nan
                overall_age_std = sub_df["Age"].std() if not sub_df.empty else np.nan

                female_subset = sub_df[sub_df["Sex"] == "F"]
                female_age_mean = female_subset["Age"].mean() if not female_subset.empty else np.nan
                female_age_std = female_subset["Age"].std() if not female_subset.empty else np.nan

                male_subset = sub_df[sub_df["Sex"] == "M"]
                male_age_mean = male_subset["Age"].mean() if not male_subset.empty else np.nan
                male_age_std = male_subset["Age"].std() if not male_subset.empty else np.nan

                if analysis_type == "ctrl_vs_all":
                    if "psychostimulant_category" in sub_df.columns:
                        is_control = sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0)
                    else:
                        is_control = (
                            pd.to_numeric(sub_df["Psychostimulant (y/n)"], errors="coerce")
                            .fillna(0)
                            .astype(int)
                            .eq(0)
                        )
                    group1_df = sub_df[is_control]
                    group2_df = sub_df[~is_control]
                elif analysis_type == "med1_vs_med2":
                    med_df = sub_df[sub_df["psychostimulant_category"].isin([1, 2])]
                    group1_df = med_df[med_df["psychostimulant_category"] == 1]
                    group2_df = med_df[med_df["psychostimulant_category"] == 2]
                elif analysis_type == "ctrl_vs_med1":
                    sub_sub_df = sub_df[sub_df["psychostimulant_category"].isin([0, 1])]
                    group1_df = sub_sub_df[
                        sub_sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0)
                    ]
                    group2_df = sub_sub_df[sub_sub_df["psychostimulant_category"] == 1]
                elif analysis_type == "ctrl_vs_med2":
                    sub_sub_df = sub_df[sub_df["psychostimulant_category"].isin([0, 2])]
                    group1_df = sub_sub_df[
                        sub_sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0)
                    ]
                    group2_df = sub_sub_df[sub_sub_df["psychostimulant_category"] == 2]
                else:
                    group1_df = pd.DataFrame()
                    group2_df = pd.DataFrame()

                M_count_group1 = int((group1_df["Sex"] == "M").sum()) if not group1_df.empty else 0
                F_count_group1 = int((group1_df["Sex"] == "F").sum()) if not group1_df.empty else 0
                M_count_group2 = int((group2_df["Sex"] == "M").sum()) if not group2_df.empty else 0
                F_count_group2 = int((group2_df["Sex"] == "F").sum()) if not group2_df.empty else 0
                M_count = M_count_group1 + M_count_group2
                F_count = F_count_group1 + F_count_group2

                age_mean_group1 = group1_df["Age"].mean() if not group1_df.empty else np.nan
                age_std_group1 = group1_df["Age"].std() if not group1_df.empty else np.nan
                age_mean_group2 = group2_df["Age"].mean() if not group2_df.empty else np.nan
                age_std_group2 = group2_df["Age"].std() if not group2_df.empty else np.nan

                results.append({
                    "med_analysis": analysis_type,
                    "sex": sex_key,
                    "age_group": age_key,
                    "TDAH_filter": diag_filters["TDAH"],
                    "Epilepsy_filter": diag_filters["Epilepsy"],
                    "TSA_filter": diag_filters["TSA"],
                    "M_count": M_count,
                    "F_count": F_count,
                    "age_mean_overall": overall_age_mean,
                    "age_std_overall": overall_age_std,
                    "age_mean_female": female_age_mean,
                    "age_std_female": female_age_std,
                    "age_mean_male": male_age_mean,
                    "age_std_male": male_age_std,
                    "age_mean_group1": age_mean_group1,
                    "age_std_group1": age_std_group1,
                    "age_mean_group2": age_mean_group2,
                    "age_std_group2": age_std_group2,
                    label1: count1,
                    label2: count2,
                })
    return pd.DataFrame(results)


def remove_small_count_rows(
    df: pd.DataFrame,
    min_count: int = 20,
    count_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter out rows where either of the count columns is < min_count.

    Args:
        df: DataFrame produced by create_analysis_dataframe.
        min_count: Minimum required for each group count.
        count_cols: Optional explicit list of two count column names. If None,
            attempts to infer from common labels (Control, Med, Med1, Med2). As a
            fallback, uses the last two columns.
    """
    if df.empty:
        return df
    if not count_cols:
        candidates = [c for c in ("Control", "Med", "Med1", "Med2") if c in df.columns]
        if len(candidates) >= 2:
            count_cols = candidates[-2:]
        else:
            count_cols = list(df.columns[-2:])
    m = (df[count_cols[0]] >= min_count) & (df[count_cols[1]] >= min_count)
    return df[m]


def generate_med_analysis_datasets(
    df: pd.DataFrame,
    out_dir: Path,
    include_potential: bool = True,
    min_count: int = 20,
) -> None:
    """Generate analysis datasets similar to analysis_exploration and save CSVs.

    Produces both full and filtered variants for each analysis type.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    med_analyses = ["ctrl_vs_all", "med1_vs_med2", "ctrl_vs_med1", "ctrl_vs_med2"]
    for analysis in med_analyses:
        res = create_analysis_dataframe(df.copy(), analysis, include_potential=include_potential)
        # Determine count columns robustly
        count_cols = [c for c in ("Control", "Med", "Med1", "Med2") if c in res.columns]
        if len(count_cols) < 2:
            count_cols = list(res.columns[-2:])
        full_path = out_dir / f"{analysis}_results.csv"
        res.to_csv(full_path, index=False)
        filtered = remove_small_count_rows(res, min_count=min_count, count_cols=count_cols)
        fil_path = out_dir / f"{analysis}_filtered_results.csv"
        filtered.to_csv(fil_path, index=False)
        logging.info(f"Saved analysis CSVs: {full_path}, {fil_path}")


def build_diagnosis_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnosis prevalence using boolean flags."""
    total = len(df)
    rows = []
    diag_map = [
        ("has_adhd", "ADHD/TDAH"),
        ("has_epilepsy", "Epilepsy"),
        ("has_tsa", "TSA"),
    ]
    for col, label in diag_map:
        if col in df.columns:
            pos = int(df[col].sum())
            rows.append(
                {
                    "diagnosis": label,
                    "positive": pos,
                    "negative": total - pos,
                    "% positive": _pct(pos, total),
                }
            )
    return pd.DataFrame(rows)


def build_medication_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize psychostimulant and anti-seizure medication prevalence."""
    total = len(df)
    epi_total = int(df["has_epilepsy"].sum()) if "has_epilepsy" in df.columns else 0
    adhd_total = int(df["has_adhd"].sum()) if "has_adhd" in df.columns else 0
    rows = []

    if "has_psychostimulant" in df.columns:
        psy = int(df["has_psychostimulant"].sum())
        rows.append(
            {
                "medication": "Any psychostimulant",
                "count": psy,
                "% total": _pct(psy, total),
                "% in_ADHD": _pct(psy, adhd_total),
                "% in_epilepsy": _pct(psy, epi_total),
            }
        )
        if "psychostimulant_description" in df.columns:
            labels = df["psychostimulant_description"].apply(_normalize_psychostim_label)
            df = df.assign(_psy_label_=labels)
            label_counts = labels.value_counts(dropna=False)
            for label, count in label_counts.items():
                if label in {"no psychostimulants", "Missing/NA"}:
                    continue
                mask = df["_psy_label_"] == label
                count = int(mask.sum())
                adhd_count = (
                    int((mask & df["has_adhd"]).sum()) if "has_adhd" in df.columns else 0
                )
                epi_count = (
                    int((mask & df["has_epilepsy"]).sum()) if "has_epilepsy" in df.columns else 0
                )
                rows.append(
                    {
                        "medication": f"{label}",
                        "count": count,
                        "% total": _pct(count, total),
                        "% in_ADHD": _pct(adhd_count, adhd_total),
                        "% in_epilepsy": _pct(epi_count, epi_total),
                    }
                )
            df = df.drop(columns=["_psy_label_"])

    if "has_epilepsy_med" in df.columns:
        asm = int(df["has_epilepsy_med"].sum())
        rows.append(
            {
                "medication": "Any anti-seizure med",
                "count": asm,
                "% total": _pct(asm, total),
                "% in_epilepsy": _pct(asm, epi_total),
                "% in_ADHD": _pct(asm, adhd_total),
            }
        )
    if "has_multiple_epilepsy_meds" in df.columns:
        multi = int(df["has_multiple_epilepsy_meds"].sum())
        rows.append(
            {
                "medication": ">=2 anti-seizure meds",
                "count": multi,
                "% total": _pct(multi, total),
                "% in_epilepsy": _pct(multi, epi_total),
                "% in_ADHD": _pct(multi, adhd_total),
            }
        )

    asm_rows = []
    for col in EPILEPSY_MED_COLS:
        if col in df.columns:
            bool_col = f"{col}_bool" if f"{col}_bool" in df.columns else col
            series = df[bool_col]
            count = int(series.fillna(0).astype(int).sum())
            asm_rows.append(
                {
                    "medication": col,
                    "count": count,
                    "% total": _pct(count, total),
                    "% in_epilepsy": _pct(count, epi_total),
                    "% in_ADHD": _pct(count, adhd_total),
                }
            )
    asm_rows = sorted(asm_rows, key=lambda r: r["count"], reverse=True)
    rows.extend(asm_rows)

    return pd.DataFrame(rows)


def _comorbidity_label(row: pd.Series) -> str:
    labels = []
    if bool(row.get("has_adhd", False)):
        labels.append("ADHD")
    if bool(row.get("has_epilepsy", False)):
        labels.append("Epilepsy")
    if bool(row.get("has_tsa", False)):
        labels.append("TSA")
    return "+".join(labels) if labels else "None"


def build_comorbidity_medication_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summaries of ADHD/Epilepsy/TSA comorbidities vs medications."""
    required = {"has_adhd", "has_epilepsy", "has_tsa", "has_psychostimulant", "has_epilepsy_med"}
    if not required.intersection(df.columns):
        return pd.DataFrame()
    tmp = df.copy()
    tmp["comorbidity"] = tmp.apply(_comorbidity_label, axis=1)
    total = len(tmp)
    rows = []
    for label, group in tmp.groupby("comorbidity"):
        count = len(group)
        psy = int(group["has_psychostimulant"].sum()) if "has_psychostimulant" in group else 0
        asm = int(group["has_epilepsy_med"].sum()) if "has_epilepsy_med" in group else 0
        both = 0
        if {"has_psychostimulant", "has_epilepsy_med"}.issubset(group.columns):
            both = int((group["has_psychostimulant"] & group["has_epilepsy_med"]).sum())
        rows.append(
            {
                "comorbidity": label,
                "count": count,
                "% total": _pct(count, total),
                "psychostimulant_n": psy,
                "psychostimulant %": _pct(psy, count),
                "epilepsy_med_n": asm,
                "epilepsy_med %": _pct(asm, count),
                "both_psychostimulant_and_epilepsy_med_n": both,
                "both_psychostimulant_and_epilepsy_med %": _pct(both, count),
            }
        )
    return pd.DataFrame(rows).sort_values(by="count", ascending=False).reset_index(drop=True)


def build_condition_grid(df: pd.DataFrame) -> pd.DataFrame:
    """2x2 ADHD x Epilepsy grid with medication usage."""
    if not {"has_adhd", "has_epilepsy"}.issubset(df.columns):
        return pd.DataFrame()
    total = len(df)
    rows = []
    for adhd_flag in (True, False):
        for epi_flag in (True, False):
            subset = df[(df["has_adhd"] == adhd_flag) & (df["has_epilepsy"] == epi_flag)]
            n = len(subset)
            psy = int(subset["has_psychostimulant"].sum()) if "has_psychostimulant" in subset else 0
            asm = int(subset["has_epilepsy_med"].sum()) if "has_epilepsy_med" in subset else 0
            both_n = 0
            if {"has_psychostimulant", "has_epilepsy_med"}.issubset(subset.columns):
                both_n = int((subset["has_psychostimulant"] & subset["has_epilepsy_med"]).sum())
            rows.append(
                {
                    "ADHD": adhd_flag,
                    "Epilepsy": epi_flag,
                    "count": n,
                    "% total": _pct(n, total),
                    "psychostimulant_n": psy,
                    "psychostimulant %": _pct(psy, n),
                    "epilepsy_med_n": asm,
                    "epilepsy_med %": _pct(asm, n),
                    "both_medications_n": both_n,
                    "both_medications %": _pct(both_n, n),
                    "age_mean": subset["Age"].mean() if "Age" in subset and n else np.nan,
                    "age_std": subset["Age"].std() if "Age" in subset and n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_condition_grid_tsa(df: pd.DataFrame) -> pd.DataFrame:
    """3-way ADHD x Epilepsy x TSA grid with medication usage."""
    required = {"has_adhd", "has_epilepsy", "has_tsa"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    total = len(df)
    rows = []
    for adhd_flag in (True, False):
        for epi_flag in (True, False):
            for tsa_flag in (True, False):
                subset = df[
                    (df["has_adhd"] == adhd_flag)
                    & (df["has_epilepsy"] == epi_flag)
                    & (df["has_tsa"] == tsa_flag)
                ]
                n = len(subset)
                psy = int(subset["has_psychostimulant"].sum()) if "has_psychostimulant" in subset else 0
                asm = int(subset["has_epilepsy_med"].sum()) if "has_epilepsy_med" in subset else 0
                both = 0
                if {"has_psychostimulant", "has_epilepsy_med"}.issubset(subset.columns):
                    both = int((subset["has_psychostimulant"] & subset["has_epilepsy_med"]).sum())
                rows.append(
                    {
                        "ADHD": adhd_flag,
                        "Epilepsy": epi_flag,
                        "TSA": tsa_flag,
                        "count": n,
                        "pct_total": _pct(n, total),
                        "psychostimulant_n": psy,
                        "psychostimulant_pct": _pct(psy, n),
                        "epilepsy_med_n": asm,
                        "epilepsy_med_pct": _pct(asm, n),
                        "both_medications_n": both,
                        "both_medications_pct": _pct(both, n),
                        "age_mean": subset["Age"].mean() if "Age" in subset and n else np.nan,
                        "age_std": subset["Age"].std() if "Age" in subset and n else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_psychostim_epilepsy_med_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Crosstab of psychostimulant vs anti-seizure medication exposure."""
    if not {"has_psychostimulant", "has_epilepsy_med"}.issubset(df.columns):
        return pd.DataFrame()
    total = len(df)
    ct = pd.crosstab(df["has_psychostimulant"], df["has_epilepsy_med"])
    rows = []
    for psy_val in ct.index:
        for asm_val in ct.columns:
            count = int(ct.loc[psy_val, asm_val])
            rows.append(
                {
                    "psychostimulant": bool(psy_val),
                    "epilepsy_med": bool(asm_val),
                    "count": count,
                    "% total": _pct(count, total),
                }
            )
    return pd.DataFrame(rows)


def build_psychostim_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of psychostimulant categories with a fallback to description mapping."""
    if "psychostimulant_category" in df.columns:
        vc = df["psychostimulant_category"].value_counts(dropna=False)
        if not vc.empty:
            return vc.reset_index()
    if "psychostimulant_description" in df.columns:
        mapped = df["psychostimulant_description"].map(MAPPING_PSYCHOSTIMULANT)
        vc = mapped.value_counts(dropna=False)
        if not vc.empty:
            return vc.reset_index()
    return pd.DataFrame(columns=["psychostimulant_category", "count"])


def build_psychostim_mapping_table(category_counts: pd.DataFrame) -> pd.DataFrame:
    """Return a reference table of psychostimulant categories with descriptions and counts."""
    cat_to_desc: dict[float, list[str]] = {}
    for desc, code in MAPPING_PSYCHOSTIMULANT.items():
        cat_to_desc.setdefault(int(code), []).append(desc)

    count_map = {}
    if not category_counts.empty and "psychostimulant_category" in category_counts.columns:
        counts_copy = category_counts.copy()
        counts_copy["psychostimulant_category"] = pd.to_numeric(
            counts_copy["psychostimulant_category"], errors="coerce"
        )
        count_map = counts_copy.set_index("psychostimulant_category")["count"].to_dict()

    rows = []
    seen_cats = set()
    for cat, descs in sorted(cat_to_desc.items(), key=lambda x: x[0]):
        seen_cats.add(cat)
        rows.append(
            {
                "psychostimulant_category": cat,
                "descriptions": ", ".join(sorted(descs)),
                "count": int(count_map.get(cat, 0)),
            }
        )

    # Include any categories present in the data but not in the static mapping
    for cat, cnt in count_map.items():
        if cat not in seen_cats and not pd.isna(cat):
            rows.append(
                {
                    "psychostimulant_category": cat,
                    "descriptions": "Unmapped/Other",
                    "count": int(cnt),
                }
            )

    # Missing/NaN categories
    na_count = count_map.get(np.nan)
    if na_count:
        rows.append(
            {
                "psychostimulant_category": np.nan,
                "descriptions": "Missing/NA",
                "count": int(na_count),
            }
        )

    return pd.DataFrame(rows)


def build_stratified_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence of diagnoses/meds stratified by Sex and age_group."""
    group_cols = [c for c in ("Sex", "age_group") if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    metrics = [
        ("has_adhd", "ADHD/TDAH"),
        ("has_epilepsy", "Epilepsy"),
        ("has_tsa", "TSA"),
        ("has_psychostimulant", "Psychostimulant"),
        ("has_epilepsy_med", "Any ASM"),
    ]
    rows = []
    for col, label in metrics:
        if col not in df.columns:
            continue
        for values, group in df.groupby(group_cols):
            group_size = len(group)
            values = values if isinstance(values, tuple) else (values,)
            rows.append(
                {
                    **dict(zip(group_cols, values)),
                    "metric": label,
                    "count": int(group[col].sum()),
                    "group_size": group_size,
                    "% in_group": _pct(int(group[col].sum()), group_size),
                }
            )
    return pd.DataFrame(rows)


def build_psychostim_category_vs_asm(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of each ASM within each psychostimulant description (normalized)."""
    if "psychostimulant_description" in df.columns:
        labels = df["psychostimulant_description"].apply(_normalize_psychostim_label)
        label_col = "psychostimulant_description"
    elif "psychostimulant_category" in df.columns:
        labels = df["psychostimulant_category"]
        label_col = "psychostimulant_category"
    else:
        return pd.DataFrame()
    bool_cols = [f"{c}_bool" for c in EPILEPSY_MED_COLS if f"{c}_bool" in df.columns]
    if not bool_cols:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_psy_label_"] = labels
    tmp = tmp[tmp["_psy_label_"] != "Missing/NA"]
    rows = []
    for label_val, group in tmp.groupby("_psy_label_"):
        cat_size = len(group)
        for col in bool_cols:
            med_name = col.replace("_bool", "")
            count = int(group[col].sum())
            rows.append(
                {
                    label_col: label_val,
                    "asm": med_name,
                    "count": count,
                    "% in_category": _pct(count, cat_size),
                }
            )
    res = pd.DataFrame(rows)
    if "count" in res.columns:
        res = res[res["count"] > 0]
    return res.reset_index(drop=True)


def build_asm_combination_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top combinations of ASMs taken together."""
    bool_cols = [f"{c}_bool" for c in EPILEPSY_MED_COLS if f"{c}_bool" in df.columns]
    if not bool_cols:
        return pd.DataFrame()

    def combo_label(row: pd.Series) -> str:
        meds = [col.replace("_bool", "") for col in bool_cols if bool(row.get(col, False))]
        return "+".join(sorted(meds)) if meds else "None"

    combos = df.apply(combo_label, axis=1)
    counts = combos.value_counts()
    if "None" in counts.index:
        counts = counts.drop(labels=["None"])
    counts = counts.head(top_n)
    total = len(df)
    return pd.DataFrame(
        {
            "asm_combo": counts.index,
            "count": counts.values,
            "% total": [_pct(c, total) for c in counts.values],
        }
    )


def build_age_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Age distribution per comorbidity and medication exposure."""
    if "Age" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["comorbidity"] = tmp.apply(_comorbidity_label, axis=1)
    group_cols = ["comorbidity"]
    for col in ("has_psychostimulant", "has_epilepsy_med"):
        if col in tmp.columns:
            group_cols.append(col)
    rows = []
    for values, group in tmp.groupby(group_cols):
        group_size = len(group)
        values = values if isinstance(values, tuple) else (values,)
        rows.append(
            {
                **dict(zip(group_cols, values)),
                "count": group_size,
                "age_mean": group["Age"].mean(),
                "age_std": group["Age"].std(),
                "age_median": group["Age"].median(),
                "age_q25": group["Age"].quantile(0.25),
                "age_q75": group["Age"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def save_summary_tables(
    med_prevalence: pd.DataFrame,
    comorbidity_summary: pd.DataFrame,
    condition_grid: pd.DataFrame,
    condition_grid_tsa: pd.DataFrame,
    cross_tab: pd.DataFrame,
    diagnosis_overview: pd.DataFrame,
    stratified_prevalence: pd.DataFrame,
    psychostim_vs_asm: pd.DataFrame,
    asm_combinations: pd.DataFrame,
    age_distribution: pd.DataFrame,
    grouping_tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    med_prevalence.to_csv(out_dir / "medication_prevalence.csv", index=False)
    comorbidity_summary.to_csv(out_dir / "comorbidity_medications.csv", index=False)
    condition_grid.to_csv(out_dir / "adhd_epilepsy_grid.csv", index=False)
    condition_grid_tsa.to_csv(out_dir / "adhd_epilepsy_tsa_grid.csv", index=False)
    cross_tab.to_csv(out_dir / "psychostimulant_vs_epilepsy_meds.csv", index=False)
    diagnosis_overview.to_csv(out_dir / "diagnosis_overview.csv", index=False)
    stratified_prevalence.to_csv(out_dir / "stratified_prevalence.csv", index=False)
    psychostim_vs_asm.to_csv(out_dir / "psychostimulant_category_vs_asm.csv", index=False)
    asm_combinations.to_csv(out_dir / "asm_combinations.csv", index=False)
    age_distribution.to_csv(out_dir / "age_distribution_by_comorbidity_med.csv", index=False)
    for name, table in grouping_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    logging.info(f"Saved medication/comorbidity summaries to {out_dir}")


def _drop_study_id_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows that share the same 'Study ID'.

    Keeps the first patient per Study ID and drops subsequent ones.
    If 'Pt ID' exists, sort so the smallest Pt ID is retained deterministically.
    """
    if "Study ID" not in df.columns:
        return df
    out = df.copy()
    valid = out["Study ID"].notna()
    if valid.sum() == 0:
        return out

    sort_cols = ["Study ID"] + (["Pt ID"] if "Pt ID" in out.columns else [])
    tmp = out.loc[valid].sort_values(by=sort_cols)
    before = len(tmp)
    tmp = tmp.drop_duplicates(subset=["Study ID"], keep="first")
    dropped = before - len(tmp)
    if dropped > 0:
        logging.info(f"Dropped {dropped} duplicate rows by 'Study ID' (kept first per ID).")
    out = pd.concat([tmp, out.loc[~valid]], axis=0, ignore_index=True)
    return out


def _drop_pt_id_duplicates_keep_smallest_study(
    df: pd.DataFrame,
    force_drop_mask: Optional[pd.Series] = None,
    force_drop_study_ids: Optional[set[int]] = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """For duplicate 'Pt ID's, drop rows with larger 'Study ID', with forced drops.

    Keeps exactly one row per Pt ID: the one with the smallest Study ID,
    unless a row is explicitly marked for dropping (e.g., medication mismatches)
    via ``force_drop_mask`` or ``force_drop_study_ids``.
    """
    dropped_info: dict[str, object] = {"forced_drop_rows": 0, "forced_pairs": []}
    # Report Pt ID duplicates across Study IDs if fields exist
    if {"Pt ID", "Study ID"}.issubset(df.columns):
        vc = df["Pt ID"].value_counts()
        dups = vc[vc > 1].index
        if len(dups) > 0:
            mapping = (
                df[df["Pt ID"].isin(dups)][["Pt ID", "Study ID"]]
                .groupby("Pt ID")["Study ID"].apply(list)
                .to_dict()
            )
            logging.info(f"Duplicate Pt IDs across Study IDs: {mapping}")
        else:
            logging.info("No duplicate Pt IDs found.")

    if not {"Pt ID", "Study ID"}.issubset(df.columns):
        return df, dropped_info

    out = df.copy()

    # Forced drops (medication mismatches or specified Study IDs)
    if force_drop_mask is not None:
        forced = out.index[force_drop_mask.reindex(out.index, fill_value=False)]
        if len(forced) > 0:
            pairs = (
                out.loc[forced, ["Pt ID", "Study ID"]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            dropped_info["forced_pairs"] = [
                f"{int(r['Pt ID'])}:{int(r['Study ID'])}" for _, r in pairs.iterrows()
            ]
            out = out.drop(index=forced)
            dropped_info["forced_drop_rows"] += len(forced)

    if force_drop_study_ids and "Study ID" in out.columns:
        study_num = pd.to_numeric(out["Study ID"], errors="coerce")
        mask_study = study_num.isin(force_drop_study_ids)
        if mask_study.any():
            pairs = (
                out.loc[mask_study, ["Pt ID", "Study ID"]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            dropped_info.setdefault("forced_pairs", []).extend(
                [f"{int(r['Pt ID'])}:{int(r['Study ID'])}" for _, r in pairs.iterrows()]
            )
            out = out.loc[~mask_study]
            dropped_info["forced_drop_rows"] += int(mask_study.sum())

    valid_pt = out["Pt ID"].notna()
    if valid_pt.sum() == 0:
        return out, dropped_info

    tmp = out.loc[valid_pt].copy()
    tmp["__sid_num__"] = pd.to_numeric(tmp["Study ID"], errors="coerce")
    before = len(tmp)
    # Sort so the smallest numeric Study ID comes first; NaNs go last
    tmp = tmp.sort_values(by=["Pt ID", "__sid_num__", "Study ID"], na_position="last")
    tmp = tmp.drop_duplicates(subset=["Pt ID"], keep="first")
    dropped = before - len(tmp)
    if dropped > 0:
        logging.info(
            f"Dropped {dropped} rows by duplicate 'Pt ID' (kept smallest 'Study ID' per Pt)."
        )
    tmp = tmp.drop(columns=["__sid_num__"], errors="ignore")

    # Append rows with missing Pt ID untouched
    out = pd.concat([tmp, out.loc[~valid_pt]], axis=0, ignore_index=True)
    return out, dropped_info


def log_pt_id_issues(df: pd.DataFrame) -> dict[str, object]:
    """Log missing Pt IDs and duplicates with their Study IDs and return details."""
    if "Pt ID" not in df.columns:
        logging.info("No 'Pt ID' column present.")
        return {}
    missing = df["Pt ID"].isna() | (df["Pt ID"].astype(str).str.strip() == "")
    missing_count = int(missing.sum())
    logging.info(f"Rows with missing Pt ID: {missing_count}")

    dup_mapping = {}
    if "Study ID" in df.columns:
        dup_pt = df[df["Pt ID"].notna()]["Pt ID"].value_counts()
        dup_pt = dup_pt[dup_pt > 1].index
        if len(dup_pt) > 0:
            dup_mapping = {}
            grouped = (
                df[df["Pt ID"].isin(dup_pt)][["Pt ID", "Study ID"]]
                .groupby("Pt ID")["Study ID"]
                .apply(list)
            )
            for pt, studies in grouped.items():
                pt_int = pd.to_numeric(pt, errors="coerce")
                pt_key = int(pt_int) if not pd.isna(pt_int) else pt
                dup_mapping[pt_key] = [
                    int(x) if not pd.isna(pd.to_numeric(x, errors="coerce")) else x for x in studies
                ]
            logging.info(f"Duplicate Pt IDs with Study IDs: {dup_mapping}")
        else:
            logging.info("No duplicate Pt IDs found.")
    return {"missing_pt_id": missing_count, "duplicate_pt_id": dup_mapping}

def print_basic_counts(df: pd.DataFrame) -> None:
    # Support both French/English naming variants where possible
    col_variants = {
        "TDAH": ["TDAH", "ADHD"],
        "Epilepsy": ["Epilepsy"],
        "TSA": ["TSA", "ASD"],
    }
    cols = []
    for key, variants in col_variants.items():
        for v in variants:
            if v in df.columns:
                cols.append(v)
                break
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        logging.info(f"Value counts — {c}:\n{vc}")

    if "has_psychostimulant" in df.columns:
        vc = df["has_psychostimulant"].value_counts(dropna=False)
        logging.info(f"Value counts — Psychostimulant (bool):\n{vc}")
    elif "Psychostimulant (y/n)" in df.columns:
        vc = df["Psychostimulant (y/n)"].value_counts(dropna=False)
        logging.info(f"Value counts — Psychostimulant (y/n):\n{vc}")
    if "psychostimulant_category" in df.columns:
        vc = df["psychostimulant_category"].value_counts(dropna=False)
        logging.info(f"Value counts — Psychostimulant category:\n{vc}")
    if "psychostimulant_description" in df.columns:
        vc = df["psychostimulant_description"].value_counts(dropna=False)
        logging.info(f"Value counts — Psychostimulant description:\n{vc}")
    if "has_epilepsy_med" in df.columns:
        vc = df["has_epilepsy_med"].value_counts(dropna=False)
        logging.info(f"Value counts — Any anti-seizure med (bool):\n{vc}")
    if "n_epilepsy_meds" in df.columns:
        vc = df["n_epilepsy_meds"].value_counts(dropna=False)
        logging.info(f"Value counts — # anti-seizure meds:\n{vc}")


def print_grouped_summaries(df: pd.DataFrame) -> None:
    if not {"Sex", "age_group"}.issubset(df.columns):
        logging.info("Grouped summaries skipped (Sex/age_group not available).")
        return

    group_cols = ["Sex", "age_group"]
    logging.info(f"Grouped summaries by: {group_cols}")

    # Psychostimulant counts
    psy = (
        df.groupby(group_cols)["has_psychostimulant"].value_counts(dropna=False)
        .rename("count")
        .reset_index()
    )
    logging.info(f"Counts — Psychostimulant by {group_cols}:\n{psy}")

    # TDAH counts
    if "TDAH" in df.columns:
        td = (
            df.groupby(group_cols)["TDAH"].value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        logging.info(f"Counts — TDAH by {group_cols}:\n{td}")

    # Epilepsy counts
    if "Epilepsy" in df.columns:
        ep = (
            df.groupby(group_cols)["Epilepsy"].value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        logging.info(f"Counts — Epilepsy by {group_cols}:\n{ep}")
    if "has_epilepsy_med" in df.columns:
        med = (
            df.groupby(group_cols)["has_epilepsy_med"].value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        logging.info(f"Counts — Anti-seizure meds by {group_cols}:\n{med}")


def plot_confusion(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Build and save a confusion heatmap for two columns using a single helper."""
    ct = pd.crosstab(
        df[row_col], df[col_col],
        rownames=[row_col], colnames=[col_col], dropna=False,
    )
    # Order boolean columns as False, True if present; keep others after
    cols = list(ct.columns)
    ordered = [c for c in (False, True) if c in cols]
    others = [c for c in cols if c not in ordered]
    if ordered:
        ct = ct.reindex(columns=ordered + others)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ct.values, cmap="Blues")
    ax.set_title(title or f"{row_col} vs {col_col}")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    ax.set_xticklabels([str(c) for c in ct.columns])
    ax.set_yticklabels([str(i) for i in ct.index])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(ct.iat[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_confusion_figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Create and save confusion matrix figures under out_dir using one plotter.

    Generated PNGs when columns are present:
      - confusion_TDAH_x_Psychostimulant.png and/or confusion_ADHD_x_Psychostimulant.png
    """
    # ADHD/TDAH vs Psychostimulant
    for tdah_col in [c for c in ("TDAH", "ADHD") if c in df.columns]:
        if "has_psychostimulant" in df.columns:
            path = out_dir / f"confusion_{tdah_col}_x_Psychostimulant.png"
            plot_confusion(df, tdah_col, "has_psychostimulant", path)
            logging.info(f"Saved figure: {path}")


def print_exclusive_diagnosis_counts(df: pd.DataFrame) -> None:
    """Print counts for patients who have only one diagnosis: ADHD, Epilepsy, or TSA.

    Treat value==1 as positive, and 0/2/NaN as negative.
    """
    tdah_col = "TDAH" if "TDAH" in df.columns else ("ADHD" if "ADHD" in df.columns else None)
    if tdah_col is None:
        logging.info("No TDAH/ADHD column found for exclusive counts.")
        return
    def pos(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).eq(1)
    def neg(s):
        return ~pos(s)
    only_adhd = pos(df[tdah_col]) & neg(df.get("Epilepsy")) & neg(df.get("TSA"))
    only_epi = pos(df.get("Epilepsy")) & neg(df[tdah_col]) & neg(df.get("TSA"))
    only_tsa = pos(df.get("TSA")) & neg(df[tdah_col]) & neg(df.get("Epilepsy"))
    logging.info(f"Only ADHD: {int(only_adhd.sum())}")
    logging.info(f"Only Epilepsy: {int(only_epi.sum())}")
    logging.info(f"Only TSA: {int(only_tsa.sum())}")


def build_grouping_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build grouped tables used for CSV export and reporting."""
    tables: dict[str, pd.DataFrame] = {}
    if {"Sex", "age_group"}.issubset(df.columns):
        tables["grouping_by_Sex_Age"] = (
            df.groupby(["Sex", "age_group"]).size().rename("count").reset_index()
        )

    tdah_col = "ADHD" if "ADHD" in df.columns else ("TDAH" if "TDAH" in df.columns else None)
    if tdah_col and {"Epilepsy", "TSA"}.issubset(df.columns):
        filtered = df.copy()
        for col in ("Epilepsy", "TSA", tdah_col):
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce") != 2]
        tables[f"grouping_{tdah_col}_by_Epilepsy_TSA"] = (
            filtered.groupby(["Epilepsy", "TSA"])[tdah_col]
            .value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )

    if "has_psychostimulant" in df.columns and {"Epilepsy", "TSA"}.issubset(df.columns):
        psy_true = df[df["has_psychostimulant"] == True]
        for col in ("Epilepsy", "TSA"):
            psy_true = psy_true[pd.to_numeric(psy_true[col], errors="coerce") != 2]
        tables["grouping_PsychostimulantTrue_by_Epilepsy_TSA"] = (
            psy_true.groupby(["Epilepsy", "TSA"]).size().rename("count").reset_index()
        )
    return tables


def save_custom_groupings(grouping_tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Save requested grouped CSVs under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in grouping_tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        logging.info(f"Saved: {path}")


def _df_to_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.to_html(index=False, float_format=lambda x: f"{x:.1f}")


def _embed_image(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<div class='figure'>"
        f"<h4>{path.name}</h4>"
        f"<img src='data:image/png;base64,{encoded}' alt='{path.name}'/>"
        "</div>"
    )


def generate_html_report(
    *,
    html_path: Path,
    diagnosis_overview: pd.DataFrame,
    med_prevalence: pd.DataFrame,
    comorbidity_summary: pd.DataFrame,
    condition_grid: pd.DataFrame,
    condition_grid_tsa: pd.DataFrame,
    cross_tab: pd.DataFrame,
    stratified_prevalence: pd.DataFrame,
    psychostim_vs_asm: pd.DataFrame,
    asm_combinations: pd.DataFrame,
    age_distribution: pd.DataFrame,
    grouping_tables: dict[str, pd.DataFrame],
    potential_counts: dict[str, int],
    pt_id_info: dict[str, object],
    med_mismatches: dict[str, object],
    total_subjects: int,
    bids_check: dict[str, object],
    figures: list[Path] | None = None,
) -> None:
    """Lightweight HTML report with the key tables/figures."""
    figures = figures or []
    html_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    img_block = "\n".join([_embed_image(p) for p in figures if p.exists()])
    quality_items = []
    if potential_counts:
        quality_items.append({"text": f"Columns with '0 (potentiel)': {potential_counts}"})
    if pt_id_info.get("missing_pt_id") is not None:
        quality_items.append(
            {"text": f"Rows with missing Pt ID: {pt_id_info.get('missing_pt_id')}"}
        )
    if bids_check:
        missing = bids_check.get("bids_missing", [])
        expected = bids_check.get("expected_count", 0)
        quality_items.append(
            {
                "text": (
                    f"BIDS subject folders at {bids_check.get('bids_root')}: "
                    f"{expected} expected, {len(missing)} missing"
                ),
                "sub": [f"Missing folders: {missing}" if missing else "All folders present"],
            }
        )
    dup_info = pt_id_info.get("duplicate_pt_id")
    if dup_info:
        quality_items.append(
            {
                "text": f"Duplicate Pt IDs with Study IDs: {dup_info}",
                "sub": [
                    f"Total duplicate Pt IDs: {len(dup_info)}",
                    "We drop rows with larger 'Study ID' per Pt ID.",
                ],
            }
        )
    if med_mismatches:
        pa = med_mismatches.get("psychostim_in_non_adhd_count")
        asm = med_mismatches.get("asm_in_non_epilepsy_count")
        pairs_psy = med_mismatches.get("psychostim_in_non_adhd_pairs") or []
        pairs_asm = med_mismatches.get("asm_in_non_epilepsy_pairs") or []
        total_mismatch = (pa or 0) + (asm or 0)
        sub_lines = []
        sub_lines.append(f"Psychostimulant in non-ADHD patients (Pt ID:Study ID pairs: {pairs_psy})")
        sub_lines.append(
            f"Anti-seizure meds in non-epilepsy patients: {asm} "
            f"(Pt ID:Study ID pairs: {pairs_asm})"
        )
        sub_lines.append(f"Total mismatches: {total_mismatch}")
        sub_lines.append("We drop rows who have mismatches.")
        quality_items.append({"text": "Medication mismatches:", "sub": sub_lines})
    if not quality_items:
        quality_items.append({"text": "No data quality warnings."})
    quality_lines = []
    for item in quality_items:
        quality_lines.append(f"<li>{item.get('text')}</li>")
        if item.get("sub"):
            quality_lines.append("<ul>")
            for sub in item["sub"]:
                quality_lines.append(f"<li>- {sub}</li>")
            quality_lines.append("</ul>")
    quality_html = "<ul>" + "".join(quality_lines) + "</ul>"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>EEG Psychostimulant/ASM Patient Overview</title>
  <style>
    body {{
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      margin: 24px;
      background: linear-gradient(135deg, #f7fafc 0%, #eef2ff 100%);
      color: #0f172a;
    }}
    h1 {{ margin-bottom: 4px; font-size: 28px; letter-spacing: 0.2px; }}
    h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #d7dce5; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef1f6; }}
    .figure img {{
      max-width: 520px;
      border: 1px solid #d7dce5;
      padding: 6px;
      background: #fff;
    }}
  </style>
</head>
<body>
  <h1>Patient Demographics & Medication Report</h1>
  <p>Generated {timestamp}</p>

  <h2>Demo and Medication Quality Checks</h2>
  {quality_html}
  <p>Total subjects used: {total_subjects}</p>

  <h2>Diagnosis Overview</h2>
  {_df_to_html(diagnosis_overview)}

  <h2>Medication Prevalence</h2>
  {_df_to_html(med_prevalence)}

  <h2>ADHD × Epilepsy Grid</h2>
  {_df_to_html(condition_grid)}
  
  <details>
    <summary>ADHD × Epilepsy × TSA Grid</summary>
    {_df_to_html(condition_grid_tsa)}
  </details>

  <h2>Comorbidities vs Medications</h2>
  {_df_to_html(comorbidity_summary)}

  <h2>Psychostimulant vs Anti-seizure Meds</h2>
  {_df_to_html(cross_tab)}

  <h2>Stratified Prevalence (Sex × Age group)</h2>
  {_df_to_html(stratified_prevalence)}

  <details>
    <summary>Psychostimulant Category × ASM</summary>
    {_df_to_html(psychostim_vs_asm)}
  </details>

  <h2>Top ASM Combinations</h2>
  {_df_to_html(asm_combinations)}

  <h2>Age Distribution by Comorbidity/Med Exposure</h2>
  {_df_to_html(age_distribution)}

  <h2>Grouped Counts (exports)</h2>
  {"".join(f"<h4>{name}</h4>{_df_to_html(tbl)}" for name, tbl in grouping_tables.items())}

  <h2>Figures</h2>
  {img_block or "<p>No figures available.</p>"}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    logging.info(f"Saved HTML report to {html_path}")



def main():
    parser = argparse.ArgumentParser(description="Explore the patients CSV: counts and confusions.")
    default_csv = Path(csv_dir) / "EEG_Psychostimulants_PatientList_08-2025.csv"
    parser.add_argument(
        "--csv_file",
        type=str,
        default=str(default_csv),
        help=(
            "Path to the CSV file "
            "(defaults to csv_dir/EEG_Psychostimulants_PatientList_08-2025.csv)"
        ),
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Print grouped summaries by Sex and age groups",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save derived tables (confusions, grouped counts) next to input CSV",
    )
    parser.add_argument(
        "--potential_in_with",
        action="store_true",
        default=True,
        help=(
            "Include potential diagnoses (code 2) in the 'with' condition for "
            "analysis datasets (default: True)"
        ),
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=20,
        help="Minimum per-group count threshold when filtering analysis datasets (default: 20)",
    )
    parser.add_argument(
        "--html_report",
        action="store_true",
        help="Generate an HTML summary report with key tables/figures.",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=str(Path(results_dir) / "reports" / "explore"),
        help="Directory to save the HTML report (default: results/reports/explore)",
    )
    parser.add_argument(
        "--bids_dir",
        type=str,
        default=str(default_bids_dir),
        help="Path to the BIDS root containing subject folders (default: env/config BIDS dir)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(data_dir),
        help="Directory for intermediate data outputs (default: data/)",
    )
    args = parser.parse_args()

    df = load(args.csv_file, sep=",")
    df = _drop_empty_columns(df)
    potential_counts = _log_potential_entries(df)
    pt_id_info = log_pt_id_issues(df)
    bids_check = check_bids_subject_folders(df, Path(args.bids_dir))

    # Derive psychostimulant flags using the mapping first, then normalize
    df = _ensure_psychostimulant_flags(df)

    med_mismatches = _compute_medication_mismatches(
        df, include_potential=args.potential_in_with
    )

    df = drop_mismatches_and_duplicates(df, med_mismatches, Path(args.data_dir))

    df = _normalize_values_and_types(df)
    df = _normalize_sex_values(df)
    df = _add_medication_flags(df)
    df = _add_diagnosis_flags(df, include_potential=args.potential_in_with)
    df = _ensure_age_groups_numeric(df)
    df = _add_age_groups(df)

    print_basic_counts(df)
    print_exclusive_diagnosis_counts(df)

    diagnosis_overview = build_diagnosis_overview(df)
    med_prevalence = build_medication_prevalence(df)
    comorbidity_summary = build_comorbidity_medication_summary(df)
    condition_grid = build_condition_grid(df)
    condition_grid_tsa = build_condition_grid_tsa(df)
    cross_tab = build_psychostim_epilepsy_med_crosstab(df)
    stratified_prevalence = build_stratified_prevalence(df)
    psychostim_vs_asm = build_psychostim_category_vs_asm(df)
    asm_combinations = build_asm_combination_summary(df)
    age_distribution = build_age_distribution_summary(df)
    grouping_tables = build_grouping_tables(df)

    logging.info(f"Medication prevalence table:\n{med_prevalence}")
    logging.info(f"ADHD x Epilepsy grid:\n{condition_grid}")
    logging.info(f"Comorbidity vs medications:\n{comorbidity_summary}")
    logging.info(f"Psychostimulant category vs ASM:\n{psychostim_vs_asm.head()}")
    logging.info(f"Top ASM combinations:\n{asm_combinations.head()}")
    logging.info(f"Stratified prevalence (first rows):\n{stratified_prevalence.head()}")

    if args.save or args.html_report:
        plots_dir = Path(results_dir) / "plots" / "explore"
        save_confusion_figures(df, plots_dir)
        # Save requested groupings to results/groupings/explore
        groupings_dir = Path(results_dir) / "groupings" / "explore"
        save_custom_groupings(grouping_tables, groupings_dir)
        # Generate analysis datasets to results/analysis/explore
        analysis_dir = Path(results_dir) / "analysis" / "explore"
        generate_med_analysis_datasets(
            df,
            analysis_dir,
            include_potential=args.potential_in_with,
            min_count=args.min_count,
        )
        summary_dir = Path(results_dir) / "summary" / "explore"
        save_summary_tables(
            med_prevalence,
            comorbidity_summary,
            condition_grid,
            condition_grid_tsa,
            cross_tab,
            diagnosis_overview,
            stratified_prevalence,
            psychostim_vs_asm,
            asm_combinations,
            age_distribution,
            grouping_tables,
            summary_dir,
        )
    else:
        plots_dir = Path(results_dir) / "plots" / "explore"
    if args.html_report:
        report_dir = Path(args.report_dir)
        confusion_paths = sorted(plots_dir.glob("confusion_*")) if plots_dir.exists() else []
        html_path = report_dir / "patients_data_report.html"
        generate_html_report(
            html_path=html_path,
            diagnosis_overview=diagnosis_overview,
            med_prevalence=med_prevalence,
            comorbidity_summary=comorbidity_summary,
            condition_grid=condition_grid,
            condition_grid_tsa=condition_grid_tsa,
            cross_tab=cross_tab,
            stratified_prevalence=stratified_prevalence,
            psychostim_vs_asm=psychostim_vs_asm,
            asm_combinations=asm_combinations,
            age_distribution=age_distribution,
            grouping_tables=grouping_tables,
            potential_counts=potential_counts,
            pt_id_info=pt_id_info,
            med_mismatches=med_mismatches,
            total_subjects=len(df),
            bids_check=bids_check,
            figures=confusion_paths,
        )
    if args.grouped:
        print_grouped_summaries(df)


if __name__ == "__main__":
    main()

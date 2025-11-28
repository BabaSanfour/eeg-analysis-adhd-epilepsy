"""
patients_csv_explorer.py

Quickly explores the patients CSV: prints dataset info, value counts for key
columns (TDAH, Epilepsy, TSA, Psychostimulant), and confusion matrices such as
TDAH x Psychostimulant. Optionally groups by Sex/Age groups and saves outputs.

Usage:
  python -m eeg_adhd_epilepsy_psychostimulant.explore.patients_data \
    --csv_file <path> [--grouped] [--save]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
import itertools
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_adhd_epilepsy_psychostimulant.io.csv import load
from eeg_adhd_epilepsy_psychostimulant.utils.config import (
    csv_dir,
    MAPPING_PSYCHOSTIMULANT,
    results_dir,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _ensure_psychostimulant_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure presence of usable psychostimulant indicators.

    - Creates ``psychostimulant_category`` from description when missing.
    - Creates boolean ``has_psychostimulant`` from y/n or category/description.
    """
    out = df.copy()

    if "psychostimulant_category" not in out.columns and "psychostimulant_description" in out.columns:
        out["psychostimulant_category"] = out["psychostimulant_description"].map(MAPPING_PSYCHOSTIMULANT)

    # Derive has_psychostimulant
    if "Psychostimulant (y/n)" in out.columns:
        yn = pd.to_numeric(out["Psychostimulant (y/n)"], errors="coerce")
        out["has_psychostimulant"] = yn.fillna(0).astype(int) == 1
    elif "psychostimulant_category" in out.columns:
        out["has_psychostimulant"] = out["psychostimulant_category"].fillna(0).astype(float) > 0
    elif "psychostimulant_description" in out.columns:
        out["has_psychostimulant"] = out["psychostimulant_description"].fillna("").str.lower() != "no psychostimulants"
    else:
        out["has_psychostimulant"] = False

    return out


def _add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" in out.columns:
        # Keep consistent with the analysis_exploration binning
        out["age_group"] = pd.cut(out["Age"], bins=[0, 12, 19], labels=["child", "teen"], right=False)
    return out


def _normalize_values_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV values and enforce numeric types.

    - Change '0 (potentiel)' entries into numeric 2 across all columns.
    - Convert all columns to numeric except 'psychostimulant_description' and 'Sex'.
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

def apply_diagnosis_filter(df: pd.DataFrame, diag_col: str, condition: str, include_potential: bool = True) -> pd.DataFrame:
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


def get_counts_by_med_analysis(df: pd.DataFrame, sex_key: str, age_key, diag_filters: dict, analysis_type: str, include_potential: bool = True):
    """Calculate subject counts based on analysis type and filters (shared with analysis_exploration)."""
    filtered_df = sex_filters[sex_key](df)
    filtered_df = age_filters[age_key](filtered_df)
    for diag in diagnosis_columns:
        filtered_df = apply_diagnosis_filter(filtered_df, diag, diag_filters.get(diag, "combined"), include_potential)

    # Determine control/medication using normalized fields
    if "psychostimulant_category" in filtered_df.columns:
        is_control = filtered_df["psychostimulant_category"].fillna(0).astype(int).eq(0)
    elif "Psychostimulant (y/n)" in filtered_df.columns:
        is_control = pd.to_numeric(filtered_df["Psychostimulant (y/n)"], errors="coerce").fillna(0).astype(int).eq(0)
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
        count_control = int(sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0).sum())
        count_med1 = int((sub_df["psychostimulant_category"] == 1).sum())
        return count_control, count_med1, "Control", "Med1"

    elif analysis_type == "ctrl_vs_med2":
        sub_df = filtered_df[filtered_df["psychostimulant_category"].isin([0, 2])].copy()
        count_control = int(sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0).sum())
        count_med2 = int((sub_df["psychostimulant_category"] == 2).sum())
        return count_control, count_med2, "Control", "Med2"

    return None, None, None, None


def create_analysis_dataframe(df: pd.DataFrame, analysis_type: str, include_potential: bool = True) -> pd.DataFrame:
    """Create a summary DataFrame with counts for each combination of filters (shared with analysis_exploration)."""
    results = []
    for sex_key in sex_filters.keys():
        for age_key in age_filters.keys():
            for diag_combo in itertools.product(diag_filter_options, repeat=3):
                diag_filters = dict(zip(diagnosis_columns, diag_combo))
                count1, count2, label1, label2 = get_counts_by_med_analysis(
                    df, sex_key, age_key, diag_filters, analysis_type, include_potential
                )

                sub_df = df.copy()
                sub_df = sex_filters[sex_key](sub_df)
                sub_df = age_filters[age_key](sub_df)
                for diag in diagnosis_columns:
                    sub_df = apply_diagnosis_filter(sub_df, diag, diag_filters.get(diag, "combined"), include_potential)

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
                        is_control = pd.to_numeric(sub_df["Psychostimulant (y/n)"], errors="coerce").fillna(0).astype(int).eq(0)
                    group1_df = sub_df[is_control]
                    group2_df = sub_df[~is_control]
                elif analysis_type == "med1_vs_med2":
                    med_df = sub_df[sub_df["psychostimulant_category"].isin([1, 2])]
                    group1_df = med_df[med_df["psychostimulant_category"] == 1]
                    group2_df = med_df[med_df["psychostimulant_category"] == 2]
                elif analysis_type == "ctrl_vs_med1":
                    sub_sub_df = sub_df[sub_df["psychostimulant_category"].isin([0, 1])]
                    group1_df = sub_sub_df[sub_sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0)]
                    group2_df = sub_sub_df[sub_sub_df["psychostimulant_category"] == 1]
                elif analysis_type == "ctrl_vs_med2":
                    sub_sub_df = sub_df[sub_df["psychostimulant_category"].isin([0, 2])]
                    group1_df = sub_sub_df[sub_sub_df["psychostimulant_category"].fillna(0).astype(int).eq(0)]
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


def remove_small_count_rows(df: pd.DataFrame, min_count: int = 20, count_cols: Optional[list[str]] = None) -> pd.DataFrame:
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


def _drop_pt_id_duplicates_keep_smallest_study(df: pd.DataFrame) -> pd.DataFrame:
    """For duplicate 'Pt ID's, drop rows with larger 'Study ID'.

    Keeps exactly one row per Pt ID: the one with the smallest Study ID.
    If 'Study ID' is non-numeric, we attempt numeric coercion first; NaNs
    are treated as the largest and dropped when a numeric Study ID exists
    for the same Pt ID. Ties fall back to lexicographic order of 'Study ID'.
    """
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
        return df

    out = df.copy()
    valid_pt = out["Pt ID"].notna()
    if valid_pt.sum() == 0:
        return out

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
    return out

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


def plot_confusion(df: pd.DataFrame, row_col: str, col_col: str, save_path: Path, title: Optional[str] = None) -> None:
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


def save_custom_groupings(df: pd.DataFrame, out_dir: Path) -> None:
    """Save requested grouped CSVs under out_dir.

    - grouping_by_Sex_Age.csv: counts by Sex x age_group
    - grouping_{ADHD|TDAH}_by_Epilepsy_TSA.csv: counts of ADHD/TDAH per Epilepsy x TSA
    - grouping_PsychostimulantTrue_by_Epilepsy_TSA.csv: among has_psychostimulant==True, counts per Epilepsy x TSA
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sex x Age
    if {"Sex", "age_group"}.issubset(df.columns):
        sex_age = df.groupby(["Sex", "age_group"]).size().rename("count").reset_index()
        sex_age.to_csv(out_dir / "grouping_by_Sex_Age.csv", index=False)
        logging.info(f"Saved: {out_dir / 'grouping_by_Sex_Age.csv'}")

    # Determine ADHD/TDAH column
    tdah_col = "ADHD" if "ADHD" in df.columns else ("TDAH" if "TDAH" in df.columns else None)
    if tdah_col and {"Epilepsy", "TSA"}.issubset(df.columns):
        adhd_by_epi_tsa = (
            df.groupby(["Epilepsy", "TSA"])[tdah_col]
              .value_counts(dropna=False)
              .rename("count")
              .reset_index()
        )
        adhd_by_epi_tsa.to_csv(out_dir / f"grouping_{tdah_col}_by_Epilepsy_TSA.csv", index=False)
        logging.info(f"Saved: {out_dir / f'grouping_{tdah_col}_by_Epilepsy_TSA.csv'}")

    # Psychostimulant True subset grouped by Epi/TSA
    if "has_psychostimulant" in df.columns and {"Epilepsy", "TSA"}.issubset(df.columns):
        psy_true = df[df["has_psychostimulant"] == True]
        psy_by_epi_tsa = (
            psy_true.groupby(["Epilepsy", "TSA"]).size().rename("count").reset_index()
        )
        psy_by_epi_tsa.to_csv(out_dir / "grouping_PsychostimulantTrue_by_Epilepsy_TSA.csv", index=False)
        logging.info(f"Saved: {out_dir / 'grouping_PsychostimulantTrue_by_Epilepsy_TSA.csv'}")



def main():
    parser = argparse.ArgumentParser(description="Explore the patients CSV: counts and confusions.")
    default_csv = Path(csv_dir) / "EEG_Psychostimulants_PatientList_08-2025.csv"
    parser.add_argument(
        "--csv_file",
        type=str,
        default=str(default_csv),
        help="Path to the CSV file (defaults to csv_dir/EEG_Psychostimulants_PatientList_08-2025.csv)",
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
        help="Include potential diagnoses (code 2) in the 'with' condition for analysis datasets (default: True)",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=20,
        help="Minimum per-group count threshold when filtering analysis datasets (default: 20)",
    )
    args = parser.parse_args()

    df = load(args.csv_file, sep=",")
    # Drop duplicate Pt IDs by keeping the smallest Study ID per patient
    df = _drop_pt_id_duplicates_keep_smallest_study(df)
    # Normalize raw dataset before deriving flags/age groups
    df = _normalize_values_and_types(df)
    df = _ensure_psychostimulant_flags(df)
    df = _ensure_age_groups_numeric(df)

    print_basic_counts(df)
    print_exclusive_diagnosis_counts(df)

    plots_dir = Path(results_dir) / "plots" / "explore"
    save_confusion_figures(df, plots_dir)
    # Save requested groupings to results/groupings/explore
    groupings_dir = Path(results_dir) / "groupings" / "explore"
    save_custom_groupings(df, groupings_dir)
    # Generate analysis datasets to results/analysis/explore
    analysis_dir = Path(results_dir) / "analysis" / "explore"
    generate_med_analysis_datasets(
        df,
        analysis_dir,
        include_potential=args.potential_in_with,
        min_count=args.min_count,
    )
    if args.grouped:
        print_grouped_summaries(df)


if __name__ == "__main__":
    main()

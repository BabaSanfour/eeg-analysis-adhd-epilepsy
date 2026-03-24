"""
qc/patients.py - CLI for patients data quality control and reporting.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.io.bids import validate_bids_coverage
from eeg_adhd_epilepsy.io.csv import load as load_csv
from eeg_adhd_epilepsy.io.patients import clean_patients_df
from eeg_adhd_epilepsy.utils.metadata_schema import EPILEPSY_MED_COLS
from eeg_adhd_epilepsy.viz.patients import (
    plot_diagnosis_prevalence,
    plot_medication_counts,
    plot_condition_heatmap,
    plot_age_stratification,
    plot_condition_venn,
    plot_stratified_prevalence,
    plot_comorbidity_matrix_3x3,
    plot_medication_overlap,
    plot_demographics_heatmap,
    plot_comorbidity_vs_meds,
    plot_age_by_med_exposure,
)
from eeg_adhd_epilepsy.reports.patients import generate_patients_report
from eeg_adhd_epilepsy.utils.config import bids_dir as default_bids_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- Helper Logic for Exports ---

def _pct(num: int, denom: int) -> float:
    if denom == 0: return np.nan
    return (num / denom) * 100

def _save_csv(df: pd.DataFrame, out_dir: Path, filename: str):
    path = out_dir / filename
    df.to_csv(path, index=True) 
    logging.info(f"Saved {filename}")

def build_diagnosis_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnosis prevalence."""
    rows = []
    total = len(df)
    for col, label in [("has_adhd", "ADHD"), ("has_epilepsy", "Epilepsy"), ("has_tsa", "TSA")]:
        if col in df.columns:
            pos = df[col].sum()
            n_pos = int(pos)
            rows.append({
                "diagnosis": label,
                "positive": n_pos,
                "negative": total - n_pos,
                "% positive": _pct(n_pos, total)
            })
    return pd.DataFrame(rows)

def build_medication_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize medication usage."""
    rows = []
    total = len(df)
    
    # Psychostimulants
    if "has_psychostimulant" in df.columns:
        n = df["has_psychostimulant"].sum()
        rows.append({"medication": "Any Psychostimulant", "count": int(n), "% total": _pct(int(n), total)})
        
        # Breakdown by description
        if "psychostimulant_description_clean" in df.columns:
            counts = df["psychostimulant_description_clean"].value_counts()
            for label, count in counts.items():
                if label in ["Missing/NA", "no psychostimulants"]: continue
                rows.append({"medication": str(label), "count": int(count), "% total": _pct(int(count), total)})

    # ASMs
    if "has_epilepsy_med" in df.columns:
        n = df["has_epilepsy_med"].sum()
        rows.append({"medication": "Any ASM", "count": int(n), "% total": _pct(int(n), total)})
        
        # Specific ASM Breakdown (Top 10)
        asm_counts = []
        for asm in EPILEPSY_MED_COLS:
            col = f"{asm}_bool" # Assuming boolean columns from cleaning
            if col in df.columns:
                count = df[col].sum()
                if count > 0:
                    asm_counts.append((asm, count))
        
        # Sort values descending and take top 10
        asm_counts.sort(key=lambda x: x[1], reverse=True)
        for asm, count in asm_counts[:10]:
             rows.append({"medication": f"ASM: {asm}", "count": int(count), "% total": _pct(int(count), total)})

    # Polytherapy
    if "has_multiple_epilepsy_meds" in df.columns:
        n = df["has_multiple_epilepsy_meds"].sum()
        rows.append({"medication": "Polytherapy (>=2 ASMs)", "count": int(n), "% total": _pct(int(n), total)})

    # Combined Psychostim + ASM
    if "has_psychostimulant" in df.columns and "has_epilepsy_med" in df.columns:
        n = (df["has_psychostimulant"] & df["has_epilepsy_med"]).sum()
        rows.append({"medication": "Psychostim + ASM", "count": int(n), "% total": _pct(int(n), total)})

    return pd.DataFrame(rows)

# --- Complex Analysis Logic (Restored) ---

def create_analysis_dataframe(df: pd.DataFrame, min_count: int = 0) -> pd.DataFrame:
    """
    Generate comprehensive analysis opportunities (~700 rows) by iterating through:
    Sex x AgeGroups x Constraints x AnalysisTypes.
    """
    
    # 1. Sex Groups
    sex_groups = [
        ("All", None),
        ("Male", lambda d: d[d["Sex"] == "M"] if "Sex" in d.columns else d),
        ("Female", lambda d: d[d["Sex"] == "F"] if "Sex" in d.columns else d),
    ]

    # 2. Age Groups
    age_groups = [("All", None)]
    if "age_group" in df.columns:
        unique_ages = df["age_group"].dropna().unique()
        for ag in sorted(str(u) for u in unique_ages):
            age_groups.append((f"Age_{ag}", lambda d, x=ag: d[d["age_group"].astype(str) == x]))

    # 3. Constraints
    # We define a constraint as a filter function
    def make_constraint(name: str, fn: Callable[[pd.DataFrame], pd.Series]):
        return (name, fn)
    
    constraints = [
        ("No_Constraint", None),
        make_constraint("No_Epilepsy", lambda d: ~d["has_epilepsy"]),
        make_constraint("No_TSA", lambda d: ~d["has_tsa"]),
        make_constraint("No_ADHD", lambda d: ~d["has_adhd"]),
        make_constraint("No_Comorbidities", lambda d: (~d["has_epilepsy"]) & (~d["has_tsa"])),
        make_constraint("Psychostim_True", lambda d: d["has_psychostimulant"]),
        make_constraint("Psychostim_False", lambda d: ~d["has_psychostimulant"]),
        make_constraint("ASM_True", lambda d: d["has_epilepsy_med"]),
        make_constraint("ASM_False", lambda d: ~d["has_epilepsy_med"]),
    ]
    
    # Generate Combinatorial Constraints (pairs & triplets)
    real_constraints = [c for c in constraints if c[0] != "No_Constraint"]
    combined_constraints = []

    def is_redundant(names):
        set_names = set(names)
        # No_ADHD implies Psychostim_False
        if "No_ADHD" in set_names and "Psychostim_False" in set_names: return True
        # No_Epilepsy implies ASM_False
        if "No_Epilepsy" in set_names and "ASM_False" in set_names: return True
        # ASM_True + ASM_False is impossible
        if "ASM_True" in set_names and "ASM_False" in set_names: return True
        # Psychostim_True + Psychostim_False is impossible
        if "Psychostim_True" in set_names and "Psychostim_False" in set_names: return True
        return False
    
    for r in range(2, 4): # Pairs (2) and Triplets (3)
        for combo in itertools.combinations(real_constraints, r):
            names = [c[0] for c in combo]
            if is_redundant(names): continue
            
            name = "+".join(names)
            funcs = [c[1] for c in combo]
            
            # Capture funcs in default arg to close over loop variable correctly
            def combined_fn(d, _funcs=funcs):
                mask = pd.Series(True, index=d.index)
                for f in _funcs:
                    mask &= f(d)
                return mask
            
            combined_constraints.append((name, combined_fn))
        
    all_constraints = constraints + combined_constraints
    # 4. Analysis Types (The comparisons)
    def analysis_adhd_vs_control(d):
        return (~d["has_adhd"], d["has_adhd"], "No ADHD", "ADHD")

    def analysis_epilepsy_vs_control(d):
        return (~d["has_epilepsy"], d["has_epilepsy"], "No Epilepsy", "Epilepsy")
    
    def analysis_med_vs_unmed_adhd(d):
        # Only valid if we are looking at ADHD patients?
        # Or we return masks for Unmedicated vs Medicated
        return (
            d["has_adhd"] & ~d["has_psychostimulant"],
            d["has_adhd"] & d["has_psychostimulant"],
            "ADHD Unmedicated", "ADHD Medicated"
        )
    def analysis_autism_asm_effect(d):
        # TSA subjects: On ASM vs Off ASM
        return (
            d["has_tsa"] & ~d["has_epilepsy_med"],
            d["has_tsa"] & d["has_epilepsy_med"],
            "TSA No-ASM", "TSA on ASM"
        )
        
    def analysis_psychostim_comparison(d):
        # Methylphenidate vs Lisdexamfetamine
        if "psychostimulant_description_clean" not in d.columns: return None
        
        mask_meth = d["psychostimulant_description_clean"] == "Methylphenidate"
        mask_lisd = d["psychostimulant_description_clean"] == "Lisdexamfetamine"
        
        return (mask_meth, mask_lisd, "Methylphenidate", "Lisdexamfetamine")
    # Dynamic Top 2 ASMs
    top_2_asms = []
    asm_counts = {}
    
    for asm in EPILEPSY_MED_COLS:
        col = f"{asm}_bool"
        if col in df.columns:
            # We assume these cols are boolean or 0/1, sum gives count of TRUE
            asm_counts[asm] = df[col].sum()
    
    sorted_asms = sorted(asm_counts.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_asms) >= 2:
        top_2_asms = [sorted_asms[0][0], sorted_asms[1][0]]
        
    def analysis_top_2_asms(d):
        if len(top_2_asms) < 2: return None
        asm1, asm2 = top_2_asms
        col1, col2 = f"{asm1}_bool", f"{asm2}_bool"
        if col1 not in d.columns or col2 not in d.columns: return None
        
        # Comparison: Has ASM1 vs Has ASM2
        return (d[col1], d[col2], asm1, asm2)

    # List of analysis builders
    analysis_specs = [

        ("ADHD_Status", analysis_adhd_vs_control),
        ("Epilepsy_Status", analysis_epilepsy_vs_control),
        ("Psychostimulant_Effect", analysis_med_vs_unmed_adhd),
        ("Autism_ASM_Effect", analysis_autism_asm_effect),
        ("Psychostim_Comparison", analysis_psychostim_comparison),
    ]
    
    if len(top_2_asms) >= 2:
        analysis_specs.append(("Top2_ASM_Comparison", analysis_top_2_asms))


    rows = []
    
    for sex_label, sex_fn in sex_groups:
        sex_df = df if sex_fn is None else sex_fn(df)
        if sex_df.empty: continue
            
        for age_label, age_fn in age_groups:
            age_df = sex_df if age_fn is None else age_fn(sex_df)
            if age_df.empty: continue
                
            for const_label, const_fn in all_constraints:
                # Apply constraint
                subset = age_df if const_fn is None else age_df[const_fn(age_df)]
                if subset.empty: continue
                    
                for ana_name, ana_builder in analysis_specs:
                    try:
                        res = ana_builder(subset)
                        if res is None: continue
                        mask1, mask2, label1, label2 = res
                        
                        g1 = subset[mask1]
                        g2 = subset[mask2]
                        n1, n2 = len(g1), len(g2)
                        
                        if n1 == 0 or n2 == 0: 
                            continue # Skip empty comparisons
                            
                        # Stats
                        rows.append({
                            "Sex": sex_label,
                            "AgeGroup": age_label,
                            "Constraint": const_label,
                            "Analysis": ana_name,
                            "Group 1": label1,
                            "Group 2": label2,
                            "N1": n1,
                            "N2": n2,
                            "N1_Male": int((g1["Sex"] == "M").sum()) if "Sex" in g1 else 0,
                            "N2_Male": int((g2["Sex"] == "M").sum()) if "Sex" in g2 else 0,
                            "N1_AgeMean": round(g1["Age"].mean(), 1) if "Age" in g1 else np.nan,
                            "N2_AgeMean": round(g2["Age"].mean(), 1) if "Age" in g2 else np.nan,
                        })
                    except Exception:
                        pass # robust

    return pd.DataFrame(rows)

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="Patients Data QC & Reporting")
    parser.add_argument("--csv_file", type=Path, required=True, help="Path to patients CSV/Excel")
    parser.add_argument("--bids_root", type=Path, default=Path(default_bids_dir), help="Path to BIDS root")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for outputs")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load & Clean
    logging.info(f"Loading {args.csv_file}...")
    df_raw = load_csv(str(args.csv_file), sep=None)
    df_clean, cleaning_stats = clean_patients_df(df_raw)
    
    # 2. Validate BIDS
    validation = validate_bids_coverage(df_clean, args.bids_root)
    
    # 3. Exports (Restored)
    # 3.1 Grouping TDAH by Epilepsy & TSA
    if "has_adhd" in df_clean.columns and "has_epilepsy" in df_clean.columns and "has_tsa" in df_clean.columns:
        grp = df_clean.groupby(["has_epilepsy", "has_tsa"])["has_adhd"].value_counts().unstack().fillna(0)
        _save_csv(grp, output_dir, "grouping_TDAH_by_Epilepsy_TSA.csv")
    
    # 3.2 Psychostimulant by Epilepsy & TSA
    if "has_psychostimulant" in df_clean.columns:
        grp = df_clean.groupby(["has_epilepsy", "has_tsa"])["has_psychostimulant"].value_counts().unstack().fillna(0)
        _save_csv(grp, output_dir, "grouping_PsychostimulantTrue_by_Epilepsy_TSA.csv")
    
    # 3.3 Age Stats Export
    if "Age" in df_clean.columns:
        # Simple aggregated stats
        age_stats = df_clean.groupby(["has_adhd", "has_epilepsy"])["Age"].describe()
        _save_csv(age_stats, output_dir, "age_stats_grouped.csv")

    # 4. Generate Summaries & Plots
    diag_summary = build_diagnosis_overview(df_clean)
    med_summary = build_medication_overview(df_clean)
    
    # Analysis DataFrame (The Big Table)
    analysis_df = create_analysis_dataframe(df_clean, min_count=0) # Keep all for overview, exclude 0/0 later?
    analysis_df.to_csv(output_dir / "analysis_opportunities.csv", index=False)
    logging.info(f"Generated {len(analysis_df)} analysis opportunities.")
    
    # Figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize by Section
    diag_figs = {}
    med_figs = {}
    demog_figs = {}
    
    # --- 1. Diagnosis Section ---
    
    # Prevalence Bar
    p = fig_dir / "diagnosis_prevalence.png"
    plot_diagnosis_prevalence(diag_summary, p)
    diag_figs["diagnosis_prevalence"] = p
    
    # 3x3 Comorbidity Matrix (Replaces old heatmap)
    p = fig_dir / "comorbidity_matrix_3x3.png"
    plot_comorbidity_matrix_3x3(df_clean, p)
    diag_figs["comorbidity_matrix"] = p
    
    # Venn / Overlaps
    p = fig_dir / "overlap_counts.png"
    plot_condition_venn(df_clean, p)
    diag_figs["overlap_counts"] = p

    # --- 2. Medication Section ---
    
    # Medication Counts
    p = fig_dir / "medication_counts.png"
    plot_medication_counts(med_summary, p)
    med_figs["medication_counts"] = p
    
    # Medication Overlap (Psychostim x ASM)
    p = fig_dir / "medication_overlap.png"
    plot_medication_overlap(df_clean, p)
    med_figs["medication_overlap"] = p
    
    # NEW: Comorbidity vs Meds Heatmap
    p = fig_dir / "comorbidity_vs_meds.png"
    plot_comorbidity_vs_meds(df_clean, p)
    med_figs["comorbidity_vs_meds"] = p

    # --- 3. Demographics Section ---
    
    # Sex x Age Heatmap
    p = fig_dir / "demographics_dist.png"
    plot_demographics_heatmap(df_clean, p)
    demog_figs["demographics_dist"] = p
    
    # Age Distribution (Violin)
    p = fig_dir / "age_distribution.png"
    plot_age_stratification(df_clean, p)
    demog_figs["age_distribution_violin"] = p
    
    # NEW: Age by Med Exposure (Violin)
    p = fig_dir / "age_by_med_exposure.png"
    plot_age_by_med_exposure(df_clean, p)
    demog_figs["age_by_med_exposure"] = p
    
    # Stratified Prevalence
    p = fig_dir / "stratified_prevalence.png"
    plot_stratified_prevalence(df_clean, p)
    demog_figs["stratified_prevalence"] = p

    # 5. Generate Report
    figures_by_section = {
        "Diagnosis": diag_figs,
        "Medication": med_figs,
        "Demographics": demog_figs
    }
    
    generate_patients_report(
        df_clean=df_clean,
        validation_results=validation,
        cleaning_stats=cleaning_stats,
        figures_by_section=figures_by_section,
        analysis_opportunities=analysis_df,
        output_dir=output_dir
    )
    
    logging.info("Done.")

if __name__ == "__main__":
    main()

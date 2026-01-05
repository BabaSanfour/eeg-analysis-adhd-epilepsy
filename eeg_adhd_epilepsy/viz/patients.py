"""
viz/patients.py - Visualizations for patients data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

EPILEPSY_MED_COLS = [
    "LEV", "LTG", "LCS", "CLB", "CBZ", "VPA", "ETH", 
    "TPM", "RUF", "BRV", "STP", "OXZ", "CBM"
]

def _save_fig(fig: plt.Figure, out_path: Path):
    """Save figure to path with tight layout."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figure: {out_path}")

def plot_diagnosis_prevalence(df_summary: pd.DataFrame, out_path: Path):
    """Plot simple bar chart of diagnosis prevalence."""
    if df_summary.empty or "% positive" not in df_summary.columns:
        return

    # Sort for aesthetics
    df = df_summary.sort_values(by="% positive", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="diagnosis", y="% positive", ax=ax, palette="viridis")
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    ax.set_ylim(0, 100)
    ax.set_title("Prevalence of Diagnoses (Dataset)", fontsize=14)
    ax.set_ylabel("Prevalence (%)")
    ax.set_xlabel("")
    sns.despine()
    _save_fig(fig, out_path)

def plot_medication_counts(df_prevalence: pd.DataFrame, out_path: Path):
    """Plot psychostimulant and ASM counts."""
    if df_prevalence.empty:
        return
    
    df = df_prevalence.copy()
    # Plot top 15
    df = df[df["count"] > 0].sort_values("count", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, y="medication", x="count", ax=ax, palette="mako")
    
    ax.set_title("Medication Usage Counts", fontsize=14)
    ax.set_xlabel("Number of Subjects")
    ax.set_ylabel("")
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{int(width)}', 
                    (width, p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, xytext=(5, 0), 
                    textcoords='offset points')
        
    sns.despine()
    _save_fig(fig, out_path)

def plot_condition_heatmap(df_grid: pd.DataFrame, out_path: Path, title="Comorbidity Heatmap"):
    """Plot 2x2 heatmap of ADHD x Epilepsy counts."""
    if df_grid.empty:
        return
    
    # Expected: ADHD, Epilepsy, count
    # Ensure fully populated grid even if missing combos
    try:
        # Pivot first
        matrix = df_grid.pivot(index="ADHD", columns="Epilepsy", values="count").fillna(0).astype(int)
        
        # Explicit reindex to ensure 2x2 grid [True, False] x [False, True]
        index_order = [True, False]
        col_order = [False, True]
        
        matrix = matrix.reindex(index=index_order, columns=col_order, fill_value=0)
        
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, 
                annot_kws={"size": 16, "weight": "bold"})
    
    ax.set_title(title, fontsize=14)
    ax.set_yticklabels(["ADHD (+)", "ADHD (-)"], rotation=0)
    ax.set_xticklabels(["Epilepsy (-)", "Epilepsy (+)"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    _save_fig(fig, out_path)

def plot_age_stratification(df: pd.DataFrame, out_path: Path):
    """Plot Age distribution by diagnosis group using violin plots."""
    if "Age" not in df.columns: return
    tmp = df.copy()
    
    def classify(row):
        adhd = bool(row.get("has_adhd", False))
        epi = bool(row.get("has_epilepsy", False))
        # tsa = bool(row.get("has_tsa", False)) 
        if adhd and epi: return "ADHD+Epilepsy"
        if adhd and not epi: return "ADHD Only"
        if epi and not adhd: return "Epilepsy Only"
        return "Other/Control"
        
    tmp["Group"] = tmp.apply(classify, axis=1)
    
    order = ["Other/Control", "ADHD Only", "Epilepsy Only", "ADHD+Epilepsy"]
    order = [o for o in order if o in tmp["Group"].unique()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=tmp, x="Group", y="Age", order=order, palette="Set2", ax=ax, inner="stick", density_norm="count")
    
    ax.set_title("Age Distribution by Diagnosis", fontsize=14)
    ax.set_xlabel("")
    sns.despine()
    _save_fig(fig, out_path)

def plot_comorbidity_vs_meds(df: pd.DataFrame, out_path: Path):
    """Plot Heatmap of Diagnosis Rows vs Medication Columns."""
    conditions = ["has_adhd", "has_epilepsy", "has_tsa"]
    cond_labels = ["ADHD", "Epilepsy", "TSA"]
    
    meds = ["has_psychostimulant", "has_epilepsy_med", "has_multiple_epilepsy_meds"]
    med_labels = ["Psychostim", "ASM", "Polytherapy"]
    
    # Check columns
    valid_conds = [c for c in conditions if c in df.columns]
    valid_meds = [m for m in meds if m in df.columns]
    
    if not valid_conds or not valid_meds: return

    # Build Matrix: Rows=Conds, Cols=Meds
    matrix = np.zeros((len(conditions), len(meds)), dtype=int)
    
    for i, c_col in enumerate(conditions):
        if c_col not in df.columns: continue
        for j, m_col in enumerate(meds):
            if m_col not in df.columns: continue
            
            c_mask = df[c_col].fillna(False)
            m_mask = df[m_col].fillna(False)
            matrix[i, j] = (c_mask & m_mask).sum()
            
    df_mat = pd.DataFrame(matrix, index=cond_labels, columns=med_labels)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_mat, annot=True, fmt="d", cmap="PuBu", ax=ax, cbar=False,
                annot_kws={"size": 14})
    
    ax.set_title("Comorbidities vs Medication", fontsize=14)
    ax.set_xlabel("Medication Status")
    ax.set_ylabel("Diagnosis")
    
    _save_fig(fig, out_path)

def plot_age_by_med_exposure(df: pd.DataFrame, out_path: Path):
    """Violin plot of Age by detailed Med Status."""
    if "Age" not in df.columns: return
    tmp = df.copy()
    
    def detailed_classify(row):
        # ADHD
        if row.get("has_adhd"):
            if row.get("has_psychostimulant"): return "ADHD (Med)"
            return "ADHD (Unmed)"
        
        # Epilepsy
        if row.get("has_epilepsy"):
            if row.get("has_epilepsy_med"): return "Epilepsy (ASM)"
            return "Epilepsy (No ASM)"
            
        return "Control/Other"

    tmp["Status"] = tmp.apply(detailed_classify, axis=1)
    
    # Order logic
    order = ["Control/Other", "ADHD (Unmed)", "ADHD (Med)", "Epilepsy (No ASM)", "Epilepsy (ASM)"]
    order = [o for o in order if o in tmp["Status"].unique()]
    
    if not order: return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=tmp, x="Status", y="Age", order=order, palette="Pastel1", ax=ax, inner="stick", density_norm="count")
    
    ax.set_title("Age Distribution by Medication Exposure", fontsize=14)
    ax.set_xlabel("")
    sns.despine()
    _save_fig(fig, out_path)

def plot_condition_venn(df: pd.DataFrame, out_path: Path):
    """Bar chart of exclusive vs comorbid counts."""
    adhd = df["has_adhd"].fillna(False) if "has_adhd" in df else pd.Series(False, index=df.index)
    epi = df["has_epilepsy"].fillna(False) if "has_epilepsy" in df else pd.Series(False, index=df.index)
    tsa = df["has_tsa"].fillna(False) if "has_tsa" in df else pd.Series(False, index=df.index)
    
    counts = {
        "Healthy/Control": (~adhd & ~epi & ~tsa).sum(),
        "ADHD Only": (adhd & ~epi & ~tsa).sum(),
        "Epilepsy Only": (epi & ~adhd & ~tsa).sum(),
        "TSA Only": (tsa & ~adhd & ~epi).sum(),
        "ADHD+Epilepsy": (adhd & epi & ~tsa).sum(),
        "ADHD+TSA": (adhd & tsa & ~epi).sum(),
        "Epilepsy+TSA": (epi & tsa & ~adhd).sum(),
        "All Three": (adhd & epi & tsa).sum()
    }
    
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts: return

    s = pd.Series(counts).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=s.index, y=s.values, ax=ax, palette="rocket")
    ax.set_title("Diagnosis Overlaps (Exclusive Groups)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel("Count")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom')
        
    sns.despine()
    _save_fig(fig, out_path)

def plot_stratified_prevalence(df: pd.DataFrame, out_path: Path):
    """Prevalence of ADHD stratified by Sex & Age Group."""
    if "has_adhd" not in df.columns or "Sex" not in df.columns or "age_group" not in df.columns:
        return
        
    # Aggregate
    # We want % ADHD Positive per (Sex, AgeGroup) bin
    grouped = df.groupby(["Sex", "age_group"], observed=True)["has_adhd"].mean().reset_index()
    grouped["% ADHD"] = grouped["has_adhd"] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=grouped, x="age_group", y="% ADHD", hue="Sex", ax=ax, palette="coolwarm")
    
    ax.set_title("ADHD Prevalence by Sex & Age", fontsize=14)
    ax.set_ylabel("Prevalence (%)")
    ax.set_xlabel("Age Group")
    sns.despine()
    _save_fig(fig, out_path)

def plot_asm_combinations(df: pd.DataFrame, out_path: Path):
    """Bar chart of Top ASM Combinations."""
    # Find active ASM cols
    cols = [c for c in EPILEPSY_MED_COLS if f"{c}_bool" in df.columns]
    
    if not cols: return
    
    combinations = []
    for _, row in df.iterrows():
        meds = [c for c in cols if row[f"{c}_bool"]]
        if meds:
            combinations.append("+".join(sorted(meds)))
        else:
            combinations.append("None")
            
    s = pd.Series(combinations)
    s = s[s != "None"]
    
    if s.empty: return
    
    top_combos = s.value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_combos.values, y=top_combos.index, ax=ax, palette="viridis")
    
    ax.set_title("Top 10 ASM Combinations", fontsize=14)
    ax.set_xlabel("Count")
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{int(width)}', 
                    (width, p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, xytext=(5, 0), 
                    textcoords='offset points')

    sns.despine()
    _save_fig(fig, out_path)

def plot_3way_grid(df: pd.DataFrame, out_path: Path):
    """ADHD x Epilepsy x TSA Grid (Heatmap of ADHD Prevalence or Counts)."""
    # X=Epilepsy, Y=TSA, Value=%ADHD 
    if not {"has_adhd", "has_epilepsy", "has_tsa"}.issubset(df.columns):
        return

    # Counts
    grp = df.groupby(["has_epilepsy", "has_tsa"])["has_adhd"].agg(['mean', 'count']).reset_index()
    grp["% ADHD"] = grp["mean"] * 100
    
    # Pivot for Heatmap (% ADHD)
    try:
        matrix_pct = grp.pivot(index="has_tsa", columns="has_epilepsy", values="% ADHD").fillna(0)
        matrix_count = grp.pivot(index="has_tsa", columns="has_epilepsy", values="count").fillna(0).astype(int)
        
        # Ensure full 2x2 via reindex
        index_order = [True, False]
        col_order = [False, True]
        
        matrix_pct = matrix_pct.reindex(index=index_order, columns=col_order, fill_value=0)
        matrix_count = matrix_count.reindex(index=index_order, columns=col_order, fill_value=0)
        
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    # Annotate with "% (n=N)"
    annot = matrix_pct.map(lambda x: f"{x:.1f}%") + matrix_count.map(lambda x: f"\n(n={x})")
    
    sns.heatmap(matrix_pct, annot=annot, fmt="", cmap="Reds", ax=ax, vmin=0, vmax=100,
                annot_kws={"size": 14})
    
    ax.set_title("ADHD Prevalence by Comorbidity", fontsize=14)
    ax.set_yticklabels(["TSA (+)", "TSA (-)"], rotation=0)
    ax.set_xticklabels(["Epilepsy (-)", "Epilepsy (+)"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    sns.despine()
    _save_fig(fig, out_path)

def plot_comorbidity_matrix_3x3(df: pd.DataFrame, out_path: Path):
    """Plot 3x3 Interaction Matrix for ADHD, Epilepsy, TSA."""
    conditions = ["has_adhd", "has_epilepsy", "has_tsa"]
    labels = ["ADHD", "Epilepsy", "TSA"]
    
    # Check coverage
    if not all(c in df.columns for c in conditions): return

    # Build asymmetric matrix: Row (+) AND Col (-)
    # Cell [i, j] = Count(Condition i is TRUE and Condition j is FALSE)
    n = len(conditions)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                c1 = df[conditions[i]].fillna(False)
                matrix[i, j] = c1.sum()
            else:
                c1 = df[conditions[i]].fillna(False)
                c2 = df[conditions[j]].fillna(False)
                # Row (+), Col (-)
                matrix[i, j] = (c1 & ~c2).sum()

    df_mat = pd.DataFrame(matrix, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_mat, annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title("Row (+) vs Col (-)\n(Diagonal = Total Count)", fontsize=14)
    ax.set_xlabel("Negative Condition (-)")
    ax.set_ylabel("Positive Condition (+)")
    _save_fig(fig, out_path)

def plot_medication_overlap(df: pd.DataFrame, out_path: Path):
    """Plot Psychostimulant x ASM Overlap Matrix."""
    if "has_psychostimulant" not in df.columns or "has_epilepsy_med" not in df.columns:
        return

    # Group by both
    grp = df.groupby(["has_psychostimulant", "has_epilepsy_med"]).size().reset_index(name="count")
    
    # Pivot
    try:
        matrix = grp.pivot(index="has_psychostimulant", columns="has_epilepsy_med", values="count").fillna(0).astype(int)
        # Reindex to ensure 2x2: True/False
        order = [True, False]
        matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", ax=ax, cbar=False,
                annot_kws={"size": 16})
    
    ax.set_title("Medication Overlap", fontsize=14)
    ax.set_yticklabels(["Psychostim (+)", "Psychostim (-)"], rotation=0)
    ax.set_xticklabels(["ASM (+)", "ASM (-)"])
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    _save_fig(fig, out_path)

def plot_demographics_heatmap(df: pd.DataFrame, out_path: Path):
    """Plot Sex x Age Group Heatmap."""
    if "Sex" not in df.columns or "age_group" not in df.columns: return

    grp = df.groupby(["Sex", "age_group"]).size().reset_index(name="count")
    
    try:
        matrix = grp.pivot(index="Sex", columns="age_group", values="count").fillna(0).astype(int)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", ax=ax, cbar=False)
    
    ax.set_title("Demographics Distribution (Sex x Age)", fontsize=14)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Sex")
    
    _save_fig(fig, out_path)

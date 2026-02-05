"""
qc/recruitment_strategy.py

Consolidated script to calculate recruitment targets for EEG study.
Generates an HTML report with:
1. Validated Baseline (BIDS-checked)
2. Milestones (1500, 2000, 3000, 5000)
3. Recruitment Visualizations (Stacked Bars)

Usage:
    python -m eeg_adhd_epilepsy.qc.recruitment_strategy
"""

import sys
import argparse
import logging
import base64
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Project imports
from eeg_adhd_epilepsy.io.patients import load_raw_patients_df, clean_patients_df
from eeg_adhd_epilepsy.utils.config import bids_dir as default_bids_dir

# --- CONFIGURATION ---

TARGETS = {
    1500: {
        "Healthy Controls": 350,
        "Pure ADHD (Total)": 250,
        "Pure Epilepsy (Total)": 350,
        "Epilepsy + ADHD (Comorbid)": 400,
        "TSA (All)": 150
    },
    2000: {
        "Healthy Controls": 500,
        "Pure ADHD (Total)": 400,
        "Pure Epilepsy (Total)": 500,
        "Epilepsy + ADHD (Comorbid)": 450,
        "TSA (All)": 150
    },
    3000: {
        "Healthy Controls": 1000,
        "Pure ADHD (Total)": 600,
        "Pure Epilepsy (Total)": 800,
        "Epilepsy + ADHD (Comorbid)": 500,
        "TSA (All)": 100
    },
    5000: {
        "Healthy Controls": 1650,
        "Pure ADHD (Total)": 1000,
        "Pure Epilepsy (Total)": 1350,
        "Epilepsy + ADHD (Comorbid)": 850,
        "TSA (All)": 150
    }
}

ROW_HIERARCHY = [
    ("Healthy Controls", 0),
    ("Pure ADHD (Total)", 0),
    ("Unmedicated", 1),
    ("Medicated (Any)", 1),
    ("Methylphenidate", 2),
    ("Lisdexamfetamine", 2),
    ("Pure Epilepsy (Total)", 0),
    ("Unmedicated", 1),
    ("Medicated (Any)", 1),
    ("Levetiracetam (Mono)", 2),
    ("Valproic Acid (Mono)", 2),
    ("Epilepsy + ADHD (Comorbid)", 0),
    ("Fully Medicated (Both)", 1),
    ("ASM Only", 1),
    ("Psychostimulant Only", 1),
    ("Unmedicated", 1),
]

# --- UTILS ---

def _get_base64_plot(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=300, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def _generate_datatable_js(df, table_id="recruitmentTable"):
    data_json = df.to_json(orient="records")
    columns = [{"data": c, "title": c} for c in df.columns]
    columns_json = json.dumps(columns)
    
    return f"""
    <div class="table-responsive">
        <table id="{table_id}" class="display table table-striped table-bordered" style="width:100%"></table>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            $('#{table_id}').DataTable({{
                data: {data_json},
                columns: {columns_json},
                pageLength: 25,
                dom: 'Bfrtip',
                ordering: false,
                buttons: ['copy', 'csv', 'excel']
            }});
        }});
    </script>
    """

def get_row_stats(df, label, indent_level):
    n = len(df)
    prefix = "&nbsp;" * (indent_level * 4) + ("↳ " if indent_level > 0 else "")
    full_label = f"{prefix}{label}"
    
    if n == 0:
        return {"Label": full_label, "N": 0, "Male%": "-", "Age": "-", "RawN": 0, "CleanLabel": label}
    
    n_male = df[df["Sex"] == "M"].shape[0] if "Sex" in df.columns else 0
    pct_male = (n_male / n) * 100
    age_mean = df["Age"].mean() if "Age" in df.columns else np.nan
    age_std = df["Age"].std() if "Age" in df.columns else np.nan
    
    return {
        "Label": full_label,
        "N": n,
        "Male%": f"{pct_male:.1f}%",
        "Age": f"{age_mean:.1f} ± {age_std:.1f}",
        "RawN": n,
        "CleanLabel": label
    }

# --- CALCULATION LOGIC ---

def calculate_recruitment_logic(base_rows, milestone_n):
    t = TARGETS[milestone_n]
    result_rows = []
    
    parent_target_map = {}
    
    for r in base_rows:
        label = r['CleanLabel']
        target_n = None
        strategy = ""
        recruit_val = 0
        
        # 1. Top Level
        if label in t:
            target_n = t[label]
            parent_target_map[label] = target_n
            delta = target_n - r['RawN']
            recruit_val = max(0, delta)
            strategy = f"+{recruit_val}" if recruit_val > 0 else "Done"
            
        # 2. Children
        # ADHD
        elif label == "Unmedicated" and "Pure ADHD (Total)" in parent_target_map:
            if milestone_n == 1500: target_n = 100
            elif milestone_n == 2000: target_n = 150
            elif milestone_n >= 3000: target_n = 200
            
        elif label == "Medicated (Any)" and "Pure ADHD (Total)" in parent_target_map:
             total = parent_target_map["Pure ADHD (Total)"]
             unmed_target = 100 if milestone_n==1500 else (150 if milestone_n==2000 else 200)
             target_n = total - unmed_target
             
        elif label in ["Methylphenidate", "Lisdexamfetamine"]:
             if "Pure ADHD (Total)" in parent_target_map:
                 total = parent_target_map["Pure ADHD (Total)"]
                 unmed = 100 if milestone_n==1500 else (150 if milestone_n==2000 else 200)
                 target_n = int((total - unmed) / 2) # Equal split

        # Epilepsy
        elif label == "Unmedicated" and "Pure Epilepsy (Total)" in parent_target_map:
             if milestone_n == 1500: target_n = 100
             elif milestone_n == 2000: target_n = 150
             elif milestone_n >= 3000: target_n = 200
             
        elif label == "Medicated (Any)" and "Pure Epilepsy (Total)" in parent_target_map:
             total = parent_target_map["Pure Epilepsy (Total)"]
             unmed = 100 if milestone_n==1500 else (150 if milestone_n==2000 else 200)
             target_n = total - unmed

        elif "Mono" in label: # LEV/VPA
             if "Pure Epilepsy (Total)" in parent_target_map:
                 total = parent_target_map["Pure Epilepsy (Total)"]
                 unmed = 100 if milestone_n==1500 else (150 if milestone_n==2000 else 200)
                 target_n = int((total - unmed) / 2)
        
        # Comorbid
        elif label == "Fully Medicated (Both)":
             if milestone_n == 1500: target_n = 200
             elif milestone_n == 2000: target_n = 300
             elif milestone_n == 3000: target_n = 350
             elif milestone_n == 5000: target_n = 600
             
        elif label in ["ASM Only", "Psychostimulant Only"]:
             if milestone_n == 1500: target_n = 75
             elif milestone_n == 2000: target_n = 50
             elif milestone_n >= 3000: target_n = 50
             if milestone_n == 5000: target_n = 100 # Boost for 5k?
             
        elif label == "Unmedicated" and "Epilepsy + ADHD (Comorbid)" in parent_target_map:
             target_n = 50
             
        # Calculate Delta
        if target_n is not None:
            delta = target_n - r['RawN']
            recruit_val = max(0, delta)
            recruit_str = f"+{recruit_val}" if recruit_val > 0 else "0"
        else:
            target_n = "-"
            recruit_str = "-"
            recruit_val = 0
            
        result_rows.append({
            "Group": r["Label"],
            "Current": r["RawN"],
            "Target": target_n,
            "Recruit": recruit_str,
            "Strategy": strategy,
            "RecruitN": recruit_val, # Numeric for plotting
            "CleanLabel": label,
            "IsParent": r["Label"].strip().startswith("&") is False
        })
        
    return result_rows

# --- PLOTTING ---

def plot_recruitment_gap(milestone_data, milestone_n):
    """Stacked bar plot: Current vs Recruitment Needed."""
    # Filter for top-level groups only
    df = pd.DataFrame(milestone_data)
    # Filter using CleanLabel to match TARGET keys
    df = df[df["CleanLabel"].isin(TARGETS[milestone_n].keys())]
    
    if df.empty: return None
    
    # Sort by Target size
    # df["TargetNum"] = pd.to_numeric(df["Target"], errors='coerce').fillna(0)
    # df = df.sort_values("TargetNum", ascending=True)
    
    labels = df["CleanLabel"]
    current = df["Current"]
    recruit = df["RecruitN"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Stacked
    ax.barh(labels, current, label="Current", color="#2ecc71")
    ax.barh(labels, recruit, left=current, label="To Recruit", color="#e74c3c")
    
    ax.set_title(f"Recruitment Gap for Milestone N={milestone_n}", fontsize=15)
    ax.set_xlabel("Number of Subjects")
    ax.legend()
    
    # Annotate total target
    for i, (c, r) in enumerate(zip(current, recruit)):
        total = c + r
        ax.text(total + 5, i, f"Target: {total}", va='center', fontweight='bold')

    sns.despine()
    return fig

def plot_milestone_progression(all_milestones, current_total):
    """Line plot showing growth of total dataset."""
    milestones = sorted(TARGETS.keys())
    x = ["Current"] + [f"N={m}" for m in milestones]
    y = [current_total] + milestones
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker='o', linestyle='-', color='#3498db', linewidth=2)
    
    for i, val in enumerate(y):
        ax.annotate(f"{val}", (i, val), xytext=(0, 10), textcoords='offset points', ha='center')
        
    ax.set_title("Dataset Growth Trajectory", fontsize=15)
    ax.set_ylabel("Total Subjects")
    ax.grid(True, linestyle='--', alpha=0.6)
    sns.despine()
    return fig

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Generate Recruitment Strategy Report")
    parser.add_argument("--csv", default="EEG_Psychostimulants_PatientList_08-2025.csv")
    parser.add_argument("--bids-dir", default="BIDS")
    parser.add_argument("--out-dir", default="analysis_output")
    args = parser.parse_args()
    
    # Setup
    csv_path = Path(args.csv)
    bids_path = Path(args.bids_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Data
    logging.info("Loading data...")
    df_raw = load_raw_patients_df(csv_path)
    df_clean, stats = clean_patients_df(df_raw)
    
    # 2. Filter BIDS
    logging.info(f"Filtering BIDS from {bids_path}...")
    valid_indices = []
    for idx, row in df_clean.iterrows():
        sid = row["Study ID"]
        if pd.isna(sid): continue
        if (bids_path / f"sub-{int(sid):04d}").exists():
            valid_indices.append(idx)
            
    df = df_clean.loc[valid_indices].copy()
    current_total = len(df)
    logging.info(f"Analyzable Subjects: {current_total}")
    
    # 3. Calculate Base Rows (Hierarchy)
    base_rows = []
    
    # Helper lambda to filter
    f = lambda q: df.query(q) if q else df
    
    # Controls
    base_rows.append(get_row_stats(
        df[(~df["has_adhd"]) & (~df["has_epilepsy"]) & (~df["has_tsa"])], 
        "Healthy Controls", 0
    ))
    
    # ADHD
    adhd = df[df["has_adhd"] & (~df["has_epilepsy"]) & (~df["has_tsa"])]
    base_rows.append(get_row_stats(adhd, "Pure ADHD (Total)", 0))
    base_rows.append(get_row_stats(adhd[~adhd["has_psychostimulant"]], "Unmedicated", 1))
    med_adhd = adhd[adhd["has_psychostimulant"]]
    base_rows.append(get_row_stats(med_adhd, "Medicated (Any)", 1))
    
    meth = pd.DataFrame() 
    lisd = pd.DataFrame()
    if "psychostimulant_description_clean" in med_adhd.columns:
        meth = med_adhd[med_adhd["psychostimulant_description_clean"] == "Methylphenidate"]
        lisd = med_adhd[med_adhd["psychostimulant_description_clean"] == "Lisdexamfetamine"]
    base_rows.append(get_row_stats(meth, "Methylphenidate", 2))
    base_rows.append(get_row_stats(lisd, "Lisdexamfetamine", 2))
    
    # Epilepsy
    ep = df[df["has_epilepsy"] & (~df["has_adhd"]) & (~df["has_tsa"])]
    base_rows.append(get_row_stats(ep, "Pure Epilepsy (Total)", 0))
    base_rows.append(get_row_stats(ep[~ep["has_epilepsy_med"]], "Unmedicated", 1))
    med_ep = ep[ep["has_epilepsy_med"]]
    base_rows.append(get_row_stats(med_ep, "Medicated (Any)", 1))
    
    lev = pd.DataFrame()
    vpa = pd.DataFrame()
    if "LEV_bool" in med_ep.columns:
        lev = med_ep[med_ep["LEV_bool"] & (med_ep["n_epilepsy_meds"] == 1)]
        vpa = med_ep[med_ep["VPA_bool"] & (med_ep["n_epilepsy_meds"] == 1)]
    base_rows.append(get_row_stats(lev, "Levetiracetam (Mono)", 2))
    base_rows.append(get_row_stats(vpa, "Valproic Acid (Mono)", 2))
    
    # Comorbid
    com = df[df["has_epilepsy"] & df["has_adhd"] & (~df["has_tsa"])]
    base_rows.append(get_row_stats(com, "Epilepsy + ADHD (Comorbid)", 0))
    base_rows.append(get_row_stats(com[com["has_psychostimulant"] & com["has_epilepsy_med"]], "Fully Medicated (Both)", 1))
    base_rows.append(get_row_stats(com[(~com["has_psychostimulant"]) & com["has_epilepsy_med"]], "ASM Only", 1))
    base_rows.append(get_row_stats(com[com["has_psychostimulant"] & (~com["has_epilepsy_med"])], "Psychostimulant Only", 1))
    base_rows.append(get_row_stats(com[(~com["has_psychostimulant"]) & (~com["has_epilepsy_med"])], "Unmedicated", 1))
    
    # 4. Generate Report Content
    
    # CSS
    css = """
    body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f4f6f9; }
    .container { max_width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }
    h2 { color: #34495e; margin-top: 40px; border-left: 5px solid #e74c3c; padding-left: 10px; }
    .stats-box { background: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
    .alert { padding: 15px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; border-radius: 4px; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
    .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    img { max-width: 100%; height: auto; }
    table { width: 100%; margin-bottom: 20px; }
    """
    
    # Section 1: Header & Cleaning Summary
    
    # Calculate drops
    n_init = stats.get('n_initial', 0)
    n_pot = stats.get('n_potential_dropped', 0)
    n_mis = stats.get('n_mismatches_dropped', 0)
    n_dup = stats.get('n_duplicates_dropped', 0)
    n_clean = len(df_clean)
    n_bids_missing = n_clean - current_total
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruitment Strategy Report</title>
        <style>{css}</style>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.dataTables.min.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>EEG Recruitment Strategy Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <div class="alert">
                <strong>Data Inclusion Summary:</strong><br>
                Out of <strong>{n_init}</strong> initial subjects:<br>
                <ul>
                    <li>Dropped <strong>{n_pot}</strong> with "0 (potentiel)" status in TSA, Epilepsy, or ADHD.</li>
                    <li>Dropped <strong>{n_mis}</strong> with medication mismatches and <strong>{n_dup}</strong> duplicate IDs.</li>
                    <li>Excluded <strong>{n_bids_missing}</strong> subjects missing accessible BIDS data (encrypted/missing files).</li>
                </ul>
                <strong>Final Analyzable Dataset: {current_total} subjects.</strong>
            </div>

            <div class="stats-box">
                <h3>Collection Criteria & Priorities</h3>
                <p>This strategy focuses on <strong>balancing sample sizes</strong> to enable robust statistical comparisons (aiming for 1:1 matching where possible). Our key priorities are:</p>
                <ol>
                    <li><strong>Controls vs. Disease:</strong> Substantially increasing Healthy Controls to match the large Epilepsy and Comorbid populations.</li>
                    <li><strong>Unmedicated Gap:</strong> Urgently recruiting <em>Unmedicated</em> ADHD and Epilepsy subjects to distinguish pure disease markers from medication effects.</li>
                    <li><strong>Medication Sub-types:</strong> Ensuring sufficient N to compare specific drugs (e.g., Methylphenidate vs. Lisdexamfetamine; Levetiracetam vs. Valproic Acid).</li>
                    <li><strong>Comorbid Interactions:</strong> Targeting "Partial Medication" groups (e.g., ASM Only) within the Comorbid population to isolate the additive effects of treatments.</li>
                </ol>
            </div>
    """
    
    # Section 2: Baseline
    html += "<h2>1. Analyzable Baseline</h2>"
    df_base = pd.DataFrame(base_rows).drop(columns=["CleanLabel"])
    html += _generate_datatable_js(df_base, "baselineTable")
    
    # Section 2: Milestones & Viz
    milestones = [1500, 2000, 3000, 5000]
    
    # Trajectory Plot
    fig_traj = plot_milestone_progression(milestones, current_total)
    src_traj = _get_base64_plot(fig_traj)
    html += f"""
    <div style="text-align: center; margin: 30px 0;">
        <img src="{src_traj}" style="max-width: 800px; border: 1px solid #ddd; border-radius: 8px;">
    </div>
    """
    
    for m in milestones:
        html += f"<h2>Milestone: Total N = {m}</h2>"
        
        # Calculation
        m_rows = calculate_recruitment_logic(base_rows, m)
        df_m = pd.DataFrame(m_rows)
        
        # Viz
        fig_gap = plot_recruitment_gap(m_rows, m)
        src_gap = _get_base64_plot(fig_gap) if fig_gap else ""
        
        # Table (Drop helper cols for display)
        df_disp = df_m.drop(columns=["RecruitN", "CleanLabel", "IsParent"])
        
        html += f"""
        <div class="grid">
            <div class="card">
                <h3>Recruitment Table</h3>
                {_generate_datatable_js(df_disp, f"table_{m}")}
            </div>
            <div class="card">
                <h3>Strategy Visualization</h3>
                <img src="{src_gap}">
            </div>
        </div>
        """
        
    html += "</div></body></html>"
    
    out_file = out_dir / "recruitment_report.html"
    out_file.write_text(html, encoding="utf-8")
    logging.info(f"Report generated: {out_file}")
    print(f"Report generated: {out_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

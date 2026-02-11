"""Preprocessing HTML Report Generation.

Generates a subject-level HTML report summarizing the preprocessing steps,
including filtering effects (PSD), bad channel detection, and artifact correction.
"""

from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import mne
import matplotlib
import matplotlib.pyplot as plt

from eeg_adhd_epilepsy.viz import qc as viz_qc

LOGGER = logging.getLogger("preproc_reports")
matplotlib.use("Agg")


def _normalize_html_report_path(path: Path, field_name: str) -> Path:
    """Validate report output path and ensure parent directory exists."""
    out_path = Path(path).expanduser()
    if out_path.suffix.lower() != ".html":
        raise ValueError(f"{field_name} must be an .html file path, got: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _save_message_only_dataset_report(summary_report_path: Path, message_html: str) -> None:
    """Save a minimal dataset report when provenance aggregation is unavailable."""
    report = mne.Report(title="Dataset Preprocessing Summary")
    report.add_html(message_html, title="Overview", section="Summary")
    report.save(summary_report_path, overwrite=True, open_browser=False)
    LOGGER.info("Dataset report saved to %s", summary_report_path)


def create_preprocessing_report(
    subject_id: str,
    raw: mne.io.BaseRaw,
    psd_before: Tuple[np.ndarray, np.ndarray],
    psd_after: Tuple[np.ndarray, np.ndarray],
    provenance: Dict,
    subject_report_path: Path,
    figures_dir: Optional[Path] = None,
    zapline_obj: Optional[Any] = None,
    raw_before_zap: Optional[mne.io.BaseRaw] = None,
) -> None:
    """Generate and save an HTML report for the preprocessing pipeline.

    Args:
        subject_id: Unique subject identifier.
        raw: The final preprocessed Raw object.
        psd_before: Tuple of (freqs, psd_data) for the raw data BEFORE processing.
        psd_after: Tuple of (freqs, psd_data) for the raw data AFTER processing.
        provenance: Dictionary containing pipeline provenance (config, stats, bad channels).
        subject_report_path: Full output path for subject HTML report.
        figures_dir: Directory containing saved figures (ZapLine, AutoReject) to include.
        zapline_obj: The fitted ZapLine estimator (optional).
        raw_before_zap: Raw data before ZapLine cleaning (optional).
    """
    out_path = _normalize_html_report_path(subject_report_path, "subject_report_path")
    report = mne.Report(title=f"Preprocessing Report - {subject_id}")
    
    # 1. Pipeline Summary
    # -------------------
    duration_min = raw.times[-1] / 60.0
    n_channels = len(raw.ch_names)
    bads = raw.info['bads']
    n_bads = len(bads)
    pct_bad = (n_bads / n_channels) * 100.0 if n_channels > 0 else 0
    sfreq = raw.info['sfreq']
    
    # New Stats
    clean_stats = provenance.get("integrity_stats", {})
    clean_dur = clean_stats.get("clean_duration_s", 0) / 60.0
    clean_pct = clean_stats.get("clean_fraction", 0) * 100.0
    manual_bad_pct = clean_stats.get("manual_bad_fraction", 0) * 100.0
    ar_bad_pct = clean_stats.get("autoreject_bad_fraction", 0) * 100.0
    
    spectral = provenance.get("spectral_stats", {})
    alpha = spectral.get("alpha_peak", float("nan"))
    slope = spectral.get("aperiodic_slope", float("nan"))
    lsd = spectral.get("lsd", float("nan"))

    # Artifact Stats
    art_stats = provenance.get("artifact_stats", {})
    manual_overlap = art_stats.get("manual_overlap_pct", float("nan"))

    summary_html = f"""
    <h3>Pipeline Summary</h3>
    <ul>
        <li><b>Subject ID:</b> {subject_id}</li>
        <li><b>Total Duration:</b> {duration_min:.2f} min @ {sfreq:.1f} Hz</li>
        <li><b>Clean Data:</b> {clean_dur:.2f} min ({clean_pct:.1f}%)</li>
        <li><b>Manual Bad Data:</b> {manual_bad_pct:.1f}%</li>
        <li><b>AutoReject Bad Data:</b> {ar_bad_pct:.1f}%</li>
        <li><b>Channels:</b> {n_channels} Total / {n_bads} Bad ({pct_bad:.1f}%)</li>
        <li><b>Bad Channels:</b> {', '.join(bads) if bads else 'None'}</li>
        <li><b>Manual Bad Overlap:</b> {manual_overlap:.1f}% (Redundant with AutoReject)</li>
    </ul>
    """
    
    # Spectral Table
    summary_html += f"""
    <h4>Spectral Quality</h4>
    <table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Alpha Peak</td>
                <td>{alpha:.2f} Hz</td>
            </tr>
            <tr>
                <td>1/f Slope</td>
                <td>{slope:.2f}</td>
            </tr>
             <tr>
                <td>LSD (Pre-Post)</td>
                <td>{lsd:.2f} dB</td>
            </tr>
        </tbody>
    </table>
    """

    if 'processing' in provenance.get('config', {}):
        proc_conf = provenance['config']['processing']
        summary_html += f"""
        <h4>Processing Config</h4>
        <ul>
            <li>Highpass: {proc_conf.get('highpass_hz')} Hz</li>
            <li>Lowpass: {proc_conf.get('lowpass_hz')} Hz</li>
            <li>Resample: {proc_conf.get('resample_hz') or 'None'} Hz</li>
        </ul>
        """
        
    # timing
    timings = provenance.get("benchmarks", {}).get("timing", {})
    if timings:
        summary_html += "<h4>Processing Time (sec)</h4><ul>"
        total_time = sum(timings.values())
        for step, dur in timings.items():
            summary_html += f"<li><b>{step}:</b> {dur:.2f}s</li>"
        summary_html += f"<li><b>Total:</b> {total_time:.2f}s</li></ul>"

    report.add_html(summary_html, title="Summary", section="Overview")

    # 2. Spectral Analysis (PSD Overlay)
    # ----------------------------------
    # Use standard EPS to avoid log(0)
    EPS = np.finfo(float).eps
    
    # Defensive unpacking
    try:
        if isinstance(psd_before, tuple) and len(psd_before) == 2:
            freqs_pre, psd_pre = psd_before
        else:
            freqs_pre, psd_pre = np.array([]), np.array([])
            
        if isinstance(psd_after, tuple) and len(psd_after) == 2:
            freqs_post, psd_post = psd_after
        else:
            freqs_post, psd_post = np.array([]), np.array([])
    except (TypeError, ValueError) as e:
        LOGGER.warning(f"Could not unpack PSD data: {e}")
        freqs_pre, psd_pre = np.array([]), np.array([])
        freqs_post, psd_post = np.array([]), np.array([])
    
    # Ensure shapes match for overlay if possible, or plot strictly on freq
    # Simple check: do we have data?
    if psd_pre.size > 0 and psd_post.size > 0:
        fig_overlay = viz_qc.plot_psd_overlay(
            freqs_pre, psd_pre, 
            freqs_post, psd_post, 
            EPS=EPS,
            label_before="Raw (Pre)",
            label_after="Clean (Post)"
        )
        report.add_figure(fig_overlay, title="PSD Overlay (Pre vs Post)", section="Spectral Analysis")
        # Explicitly close to save memory
        plt.close(fig_overlay)
    
    # 3. Pipeline Step Visualizations
    # -------------------------------
    # ZapLine Power
    if zapline_obj is not None and raw_before_zap is not None:
        try:
            from mne_denoise.viz.zapline import plot_cleaning_summary
            
            # Get effective line freq from provenance or default
            line_freq = provenance.get("zapline_stats", {}).get("line_freq", 60.0)
            adaptive = provenance.get("zapline_stats", {}).get("adaptive", False)
            
            # Use auto-detected freq if available
            eff_line_freq = line_freq
            if adaptive and hasattr(zapline_obj, 'adaptive_results_') and zapline_obj.adaptive_results_:
                eff_line_freq = zapline_obj.adaptive_results_.get('line_freq', line_freq)
            
            # Use data from both states
            data_before = raw_before_zap.get_data()
            data_after = raw.get_data()
            sfreq = raw.info['sfreq']
            
            LOGGER.info(f"Generating ZapLine summary plot for {subject_id} report...")
            fig_summary = plot_cleaning_summary(
                data_before, data_after, zapline_obj, sfreq, 
                line_freq=eff_line_freq, show=False
            )
            report.add_figure(fig_summary, title="Line Noise Removal (ZapLine Summary)", section="Line Noise")
            plt.close(fig_summary)
            
        except Exception as e:
            LOGGER.warning(f"Could not generate dynamic ZapLine plot for {subject_id}: {e}")
            
    # Legacy/Fallback: Load from disk if available
    elif figures_dir and figures_dir.exists():
        zapline_path = figures_dir / f"{subject_id}_zapline_summary.png"
        if zapline_path.exists():
            report.add_image(str(zapline_path), title="Line Noise Removal (ZapLine Summary)", section="Line Noise")
            
    # AutoReject Logs
    ar_plots: List[Path] = []
    if figures_dir and figures_dir.exists():
        ar_plots = sorted(list(figures_dir.glob(f"{subject_id}_autoreject_*.png")))
        for ar_plot in ar_plots:
            # e.g. sub-001_autoreject_rest_eyes_open.png
            # Clean name logic: remove prefix
            condition = ar_plot.stem.split("autoreject_")[-1].replace("_", " ").title()
            report.add_image(str(ar_plot), title=f"AutoReject: {condition}", section="Artifact Correction")

    # 4. Final Output
    # ---------------
    report.save(out_path, overwrite=True, open_browser=False)
    LOGGER.info("Report saved to %s", out_path)


def create_dataset_report(
    search_dir: Path, 
    summary_report_path: Path,
    success_subjects: Optional[List[str]] = None,
    failed_subjects: Optional[List[str]] = None,
) -> None:
    """Generate a dataset-level summary report from provenance files.

    Args:
        search_dir: Directory to search for provenance files (e.g. derivatives/preproc).
        summary_report_path: Full output path for dataset summary HTML report.
    """
    search_dir = Path(search_dir)
    summary_report_path = _normalize_html_report_path(
        summary_report_path, "summary_report_path"
    )
    provenance_files = sorted(list(search_dir.rglob("*_desc-base_provenance.json")))
    
    if not provenance_files:
        LOGGER.warning("No provenance files found in %s.", search_dir)
        _save_message_only_dataset_report(
            summary_report_path=summary_report_path,
            message_html=f"""
            <h3>Run Outcome</h3>
            <ul>
                <li><b>Subjects Processed:</b> 0</li>
                <li><b>Succeeded:</b> 0</li>
                <li><b>Failed:</b> 0</li>
            </ul>
            <h3>Dataset Overview</h3>
            <p>No base provenance files were found under <code>{search_dir}</code>.</p>
            """,
        )
        return

    records: List[Dict[str, Any]] = []
    all_bad_channels: List[str] = []
    for p_file in provenance_files:
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                prov = json.load(f)
            
            # Extract basic stats
            subj_id = prov.get("subject_id", p_file.stem.replace("_provenance", ""))
            
            # Bad Channels
            bads = prov.get("bad_channels_global", [])
            n_bads = len(bads)
            all_bad_channels.extend(bads)
            
            # ZapLine
            zap_stats = prov.get("zapline_stats", {})
            n_removed = zap_stats.get("n_removed", 0)
            
            # AutoReject / Artifacts
            art_stats = prov.get("artifact_stats", {})
            bad_epochs = art_stats.get("bad_epochs", 0)
            artifacts_count = art_stats.get("artifacts_count", 0)
            
            # Spectral & Integrity
            spec = prov.get("spectral_stats", {})
            integ = prov.get("integrity_stats", {})
            
            # Timing
            timings = prov.get("benchmarks", {}).get("timing", {})
            total_time = sum(timings.values()) if timings else 0
            
            records.append({
                "Subject": subj_id,
                "Bad Channels": n_bads,
                "ZapLine Removed": n_removed,
                "Bad Epochs": bad_epochs,
                "Artifact Segments": artifacts_count,
                "Alpha Peak (Hz)": spec.get("alpha_peak", float("nan")),
                "1/f Slope": spec.get("aperiodic_slope", float("nan")),
                "LSD (dB)": spec.get("lsd", float("nan")),
                "Clean Data %": integ.get("clean_fraction", 0) * 100,
                "Manual Overlap %": art_stats.get("manual_overlap_pct", float("nan")),
                "Process Time (s)": total_time
            })
            
        except Exception as e:
            LOGGER.warning(f"Failed to read {p_file}: {e}")

    if not records:
        LOGGER.warning("No valid provenance records found in %s.", search_dir)
        _save_message_only_dataset_report(
            summary_report_path=summary_report_path,
            message_html=f"""
            <h3>Run Outcome</h3>
            <ul>
                <li><b>Subjects Processed:</b> 0</li>
                <li><b>Succeeded:</b> 0</li>
                <li><b>Failed:</b> 0</li>
            </ul>
            <h3>Dataset Overview</h3>
            <p>Provenance files were found but none could be parsed successfully.</p>
            """,
        )
        return

    df = pd.DataFrame(records)
    
    # Generate Report
    report = mne.Report(title="Dataset Preprocessing Summary")
    
    # 1. Summary Table
    # ----------------
    success_subjects = sorted(set(success_subjects or []))
    failed_subjects = sorted(set(failed_subjects or []))

    run_status_html = f"""
    <h3>Run Outcome</h3>
    <ul>
        <li><b>Subjects Processed:</b> {len(success_subjects) + len(failed_subjects)}</li>
        <li><b>Succeeded:</b> {len(success_subjects)}</li>
        <li><b>Failed:</b> {len(failed_subjects)}</li>
        <li><b>Succeeded IDs:</b> {', '.join(success_subjects) if success_subjects else 'None'}</li>
        <li><b>Failed IDs:</b> {', '.join(failed_subjects) if failed_subjects else 'None'}</li>
    </ul>
    """

    summary_html = f"""
    <h3>Dataset Overview</h3>
    <ul>
        <li><b>Total Subjects:</b> {len(df)}</li>
        <li><b>Avg Bad Channels:</b> {df["Bad Channels"].mean():.2f} (±{df["Bad Channels"].std():.2f})</li>
        <li><b>Avg Clean Data:</b> {df["Clean Data %"].mean():.1f}% (±{df["Clean Data %"].std():.1f}%)</li>
        <li><b>Avg Manual Overlap:</b> {df["Manual Overlap %"].mean():.1f}% (±{df["Manual Overlap %"].std():.1f}%)</li>
        <li><b>Avg Process Time:</b> {df["Process Time (s)"].mean():.1f}s (±{df["Process Time (s)"].std():.1f}s)</li>
    </ul>
    """
     # Add Spectral Summary
    summary_html += f"""
    <h4>Spectral Summary</h4>
    <ul>
        <li><b>Alpha Peak:</b> {df["Alpha Peak (Hz)"].mean():.2f} Hz (±{df["Alpha Peak (Hz)"].std():.2f})</li>
        <li><b>1/f Slope:</b> {df["1/f Slope"].mean():.2f} (±{df["1/f Slope"].std():.2f})</li>
         <li><b>LSD:</b> {df["LSD (dB)"].mean():.2f} dB (±{df["LSD (dB)"].std():.2f})</li>
    </ul>
    """
    
    report.add_html(run_status_html + summary_html, title="Overview", section="Summary")
    
    # 2. Visualizations (Scalable for large datasets)
    # ------------------------------------------------
    # Run Outcome Counts
    fig_status, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Succeeded", "Failed"],
        [len(success_subjects), len(failed_subjects)],
        color=["mediumseagreen", "indianred"],
        edgecolor="black",
    )
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Batch Run Outcome")
    for i, value in enumerate([len(success_subjects), len(failed_subjects)]):
        ax.text(i, value + 0.1, str(value), ha="center", va="bottom")
    plt.tight_layout()
    report.add_figure(fig_status, title="Run Outcome", section="Summary")
    plt.close(fig_status)

    
    # Bad Channels Distribution (Histogram)
    fig_bads, ax = plt.subplots(figsize=(8, 5))
    max_bads = int(df["Bad Channels"].max()) + 1
    bins = range(0, max_bads + 2)
    ax.hist(df["Bad Channels"], bins=bins, edgecolor="black", color="indianred", align="left")
    ax.set_xlabel("Number of Bad Channels per Subject")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Distribution of Bad Channel Counts (Per Subject)")
    ax.set_xticks(range(0, max_bads + 1))
    plt.tight_layout()
    report.add_figure(fig_bads, title="Bad Channels Count", section="Global Stats")
    plt.close(fig_bads)
    
    if all_bad_channels:
        from collections import Counter
        bad_counts = Counter(all_bad_channels)
        # Sort by count descending
        sorted_bads = sorted(bad_counts.items(), key=lambda x: x[1], reverse=True)
        ch_names = [x[0] for x in sorted_bads]
        counts = [x[1] for x in sorted_bads]
        
        fig_ch, ax = plt.subplots(figsize=(max(10, len(ch_names)*0.3), 6))
        ax.bar(ch_names, counts, color="salmon", edgecolor="black")
        ax.set_ylabel("Count of Subjects")
        ax.set_title("Frequency of Bad Channels (Dataset Level)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        report.add_figure(fig_ch, title="Bad Channels Frequency", section="Global Stats")
        plt.close(fig_ch)

    # NEW: Low Overlap Identification
    low_overlap_df = df[df["Manual Overlap %"] < 80.0]
    if not low_overlap_df.empty:
        low_overlap_html = "<h3>Subjects with Low Manual Overlap (< 80%)</h3><ul>"
        for _, row in low_overlap_df.iterrows():
            low_overlap_html += f"<li><b>{row['Subject']}</b>: {row['Manual Overlap %']:.1f}%</li>"
        low_overlap_html += "</ul>"
        
        # Add a warning section at the top
        report.add_html(low_overlap_html, title="Low Overlap Warning", section="Summary")
    
    # Artifact Stats: Bad Epochs Distribution
    fig_epochs, ax = plt.subplots(figsize=(6, 5))
    df.boxplot(column=["Bad Epochs", "Artifact Segments"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("AutoReject: Epochs Detected as Bad")
    plt.tight_layout()
    report.add_figure(fig_epochs, title="Epoch Artifacts", section="Global Stats")
    plt.close(fig_epochs)
    
    # Add description below the plot
    artifact_desc = """
    <p style="font-size: 0.9em; color: #555;">
        <b>Bad Epochs:</b> Segments where the entire epoch across all channels is marked bad (global artifact).<br>
        <b>Artifact Segments:</b> Channel-specific local bad spans (e.g., one channel has artifact while others are clean).
    </p>
    """
    report.add_html(artifact_desc, title="Artifact Legend", section="Global Stats")
    
    # ZapLine Components Distribution (Histogram)
    fig_zap, ax = plt.subplots(figsize=(8, 5))
    max_zap = int(df["ZapLine Removed"].max()) + 1
    bins = range(0, max_zap + 2)
    ax.hist(df["ZapLine Removed"], bins=bins, edgecolor="black", color="steelblue", align="left")
    ax.set_xlabel("Components Removed")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Distribution of ZapLine Components Removed")
    ax.set_xticks(range(0, max_zap + 1))
    plt.tight_layout()
    report.add_figure(fig_zap, title="ZapLine Dist", section="Global Stats")
    plt.close(fig_zap)
    
    # Spectral Distribution
    fig_spec, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=["Alpha Peak (Hz)", "1/f Slope", "LSD (dB)"], ax=ax)
    ax.set_title("Spectral Metrics Distribution")
    plt.tight_layout()
    report.add_figure(fig_spec, title="Spectral Dist", section="Global Stats")
    plt.close(fig_spec)

    # Clean Data % Distribution (Histogram)
    fig_clean, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Clean Data %"], bins=20, edgecolor="black", color="mediumseagreen")
    ax.set_xlabel("Clean Data (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Distribution of Clean Data Percentage")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    report.add_figure(fig_clean, title="Clean Data Dist", section="Global Stats")
    plt.close(fig_clean)
    
    # Save
    report.save(summary_report_path, overwrite=True, open_browser=False)
    LOGGER.info("Dataset report saved to %s", summary_report_path)

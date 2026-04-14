"""Artifact Correction HTML Report Generation (Stage 1).

Generates a subject-level HTML report summarizing the artifact correction steps (EOG, ECG, EMG).
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

import eeg_adhd_epilepsy.viz.clean_qc as viz_qc

LOGGER = logging.getLogger("correct_reports")
matplotlib.use("Agg")


def _normalize_html_report_path(path: Path, field_name: str) -> Path:
    """Validate report output path and ensure parent directory exists."""
    out_path = Path(path).expanduser()
    if out_path.suffix.lower() != ".html":
        raise ValueError(f"{field_name} must be an .html file path, got: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def create_correction_report(
    subject_id: str,
    raw: mne.io.BaseRaw,
    psd_before: Tuple[np.ndarray, np.ndarray],
    psd_after: Tuple[np.ndarray, np.ndarray],
    provenance: Dict,
    subject_report_path: Path,
    figures_dir: Optional[Path] = None,
) -> None:
    """Generate and save an HTML report for the artifact correction pipeline.

    Args:
        subject_id: Unique subject identifier.
        raw: The final preprocessed Raw object (after correction).
        psd_before: Tuple of (freqs, psd_data) for the data BEFORE correction (output of Base pipeline).
        psd_after: Tuple of (freqs, psd_data) for the data AFTER correction.
        provenance: Dictionary containing correction provenance (stats, fit segments, etc.).
        subject_report_path: Full output path for subject HTML report.
        figures_dir: Directory containing saved figures (optional).
    """
    out_path = _normalize_html_report_path(subject_report_path, "subject_report_path")

    report = mne.Report(title=f"Artifact Correction Report (Stage 1) - {subject_id}")
    
    # 1. Pipeline Summary
    # -------------------
    duration_min = raw.times[-1] / 60.0
    n_channels = len(raw.ch_names)
    bads = raw.info['bads']
    n_bads = len(bads)
    pct_bad = (n_bads / n_channels) * 100.0 if n_channels > 0 else 0
    sfreq = raw.info['sfreq']
    
    condition_name = provenance.get("condition_name")
    fit_segments_used = provenance.get("fit_segments_used", False)
    n_bad_segments_excluded = provenance.get("n_bad_segments", 0)

    summary_html = f"""
    <h3>Correction Summary</h3>
    <ul>
        <li><b>Subject ID:</b> {subject_id}</li>
        <li><b>Condition Processed:</b> {condition_name if condition_name else "All Conditions"}</li>
        <li><b>Training Strategy:</b> {"Separate Fit Segments (e.g. Rest)" if fit_segments_used else "Self-Training (Target Data)"}</li>
        <li><b>Bad Segments Excluded from Fit:</b> {n_bad_segments_excluded}</li>
        <li><b>Total Duration:</b> {duration_min:.2f} min @ {sfreq:.1f} Hz</li>
        <li><b>Channels:</b> {n_channels} Total / {n_bads} Bad ({pct_bad:.1f}%)</li>
    </ul>
    """
    
    # 2. Correction Stats
    # -------------------
    corr_stats = provenance.get("correction_stats", {})
    
    summary_html += "<h4>Artifact Removal Stats</h4><table border='1' class='dataframe'><thead><tr><th>Artifact</th><th>Method</th><th>Details</th></tr></thead><tbody>"
    
    for artifact_type, stats in corr_stats.items():
        if not stats:
            continue
            
        method = stats.get("method", "Unknown")
        details = []
        if "n_components_removed" in stats:
            details.append(f"{stats['n_components_removed']} comps removed")
        if "n_blinks" in stats:
             details.append(f"{stats['n_blinks']} blinks detected")
        if "n_qrs" in stats:
             details.append(f"{stats['n_qrs']} QRS events")
        if "n_components_kept" in stats: # MWF
             details.append(f"{stats['n_components_kept']} comps kept")
        if "skipped" in stats:
             details.append(f"SKIPPED: {stats.get('reason', 'unknown')}")
        if "error" in stats:
             details.append(f"ERROR: {stats['error']}")
             
        summary_html += f"<tr><td>{artifact_type.upper()}</td><td>{method.upper()}</td><td>{', '.join(details)}</td></tr>"
        
    summary_html += "</tbody></table>"
    
    # Timing
    timing_map = provenance.get("benchmarks", {}).get("timing", {})
    if timing_map:
        summary_html += "<h4>Processing Time</h4><ul>"
        total_time = float(sum(float(v) for v in timing_map.values()))
        for step, t in timing_map.items():
            summary_html += f"<li><b>{step}:</b> {float(t):.2f}s</li>"
        summary_html += f"<li><b>total:</b> {total_time:.2f}s</li>"
        summary_html += "</ul>"

    spectral = provenance.get("spectral_stats", {})
    alpha = spectral.get("alpha_peak", float("nan"))
    slope = spectral.get("aperiodic_slope", float("nan"))
    lsd = spectral.get("lsd", float("nan"))

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

    report.add_html(summary_html, title="Summary", section="Overview")
    
    # 3. Artifact Details (Plots)
    # ---------------------------
    for artifact_type, stats in corr_stats.items():
        if not stats or 'plot_paths' not in stats:
            continue
            
        plot_paths = stats['plot_paths']
        if not plot_paths:
            continue
            
        section_title = f"{artifact_type.upper()} Correction Details"
        
        for name, path_str in plot_paths.items():
            if Path(path_str).exists():
                # Clean up title
                img_title = f"{artifact_type.upper()} - {name.replace('_', ' ').title()}"
                try:
                    report.add_image(
                        image=path_str,
                        title=img_title,
                        section=section_title
                    )
                except AttributeError:
                    # Fallback for older MNE versions without add_image? 
                    # Or if image path not supported directly.
                    # Try embedding as HTML
                    import base64
                    with open(path_str, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                    html = f'<img src="data:image/png;base64,{encoded_string}" alt="{img_title}" style="max-width:100%;">'
                    report.add_html(html, title=img_title, section=section_title)
                except Exception as e:
                    LOGGER.warning(f"Failed to add image {path_str}: {e}")

    # 3b. EEG Signal Progression (before/after snapshots)
    # ---------------------------------------------------
    eeg_snapshots = provenance.get("eeg_snapshots", {})
    snapshot_order = [
        ("before_correction", "Before Any Correction"),
        ("after_eog", "After EOG Removal"),
        ("after_ecg", "After ECG Removal"),
        ("after_emg", "After EMG Removal"),
    ]
    
    added_snapshots = False
    for key, title in snapshot_order:
        path_str = eeg_snapshots.get(key)
        if path_str and Path(path_str).exists():
            try:
                report.add_image(
                    image=path_str,
                    title=title,
                    section="EEG Signal Progression"
                )
                added_snapshots = True
            except AttributeError:
                import base64
                with open(path_str, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                html = f'<img src="data:image/png;base64,{encoded_string}" alt="{title}" style="max-width:100%;">'
                report.add_html(html, title=title, section="EEG Signal Progression")
                added_snapshots = True
            except Exception as e:
                LOGGER.warning(f"Failed to add EEG snapshot {key}: {e}")
    
    if not added_snapshots and eeg_snapshots:
        LOGGER.info("No EEG snapshot images found on disk.")

    # 3c. Artifact Removal Comparisons (before/after/removed at peak)
    # ---------------------------------------------------------------
    artifact_comparisons = provenance.get("artifact_comparisons", {})
    comp_order = [
        ("eog", "EOG (Eye) Artifact Removal"),
        ("ecg", "ECG (Heart) Artifact Removal"),
        ("emg", "EMG (Muscle) Artifact Removal"),
    ]
    
    for key, title in comp_order:
        path_str = artifact_comparisons.get(key)
        if path_str and Path(path_str).exists():
            try:
                report.add_image(
                    image=path_str,
                    title=title,
                    section="Artifact Removal Detail"
                )
            except AttributeError:
                import base64
                with open(path_str, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                html = f'<img src="data:image/png;base64,{encoded_string}" alt="{title}" style="max-width:100%;">'
                report.add_html(html, title=title, section="Artifact Removal Detail")
            except Exception as e:
                LOGGER.warning(f"Failed to add artifact comparison {key}: {e}")

    # 4. Spectral Analysis (PSD Overlay)
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
    
    if psd_pre.size > 0 and psd_post.size > 0:
        fig_overlay = viz_qc.plot_psd_overlay(
            freqs_pre, psd_pre, 
            freqs_post, psd_post, 
            EPS=EPS,
            label_before="Base (Stage 0)",
            label_after="Corrected (Stage 1)"
        )
        report.add_figure(fig_overlay, title="PSD Overlay (Base vs Corrected)", section="Spectral Analysis")
        plt.close(fig_overlay)

    # 5. Channel Variance Comparison
    # ------------------------------
    if "variance_comparison" in provenance.get("artifact_comparisons", {}):
        path_str = provenance["artifact_comparisons"]["variance_comparison"]
        if Path(path_str).exists():
            try:
                report.add_image(
                    image=path_str,
                    title="Channel Variance (Before vs After)",
                    section="Correction Quality"
                )
            except Exception as e:
                LOGGER.warning(f"Failed to add variance comparison plot: {e}")

    # 6. Save
    report.save(out_path, overwrite=True, open_browser=False)
    LOGGER.info("Report saved to %s", out_path)


def create_correction_dataset_report(
    search_dir: Path,
    summary_report_path: Path,
    output_desc: str = "correct",
    success_subjects: Optional[List[str]] = None,
    failed_subjects: Optional[List[str]] = None,
) -> None:
    """Aggregate artifact correction results across subjects and generate a summary report.
    
    Searches for all `*desc-{output_desc}_provenance.json` files in the search_dir.
    """
    import glob
    
    # 1. Find all provenance files
    summary_report_path = _normalize_html_report_path(
        summary_report_path, "summary_report_path"
    )
    pattern = str(search_dir / "**" / f"*_desc-{output_desc}_provenance.json")
    provenance_files = glob.glob(pattern, recursive=True)
    
    if not provenance_files:
        LOGGER.warning(f"No correction provenance files found for {output_desc} in {search_dir}")
        return

    records = []
    for p_file in provenance_files:
        try:
            with open(p_file, "r") as f:
                prov = json.load(f)
            
            subject_id = prov.get("subject_id", Path(p_file).stem.split("_")[0])
            stats = prov.get("correction_stats", {})
            timings = prov.get("benchmarks", {}).get("timing", {})
            total_t = float(sum(float(v) for v in timings.values()))
            
            row = {
                "Subject": subject_id,
                "Total Time (s)": total_t,
                "EOG Method": stats.get("eog", {}).get("method", "none"),
                "EOG Comps": stats.get("eog", {}).get("n_components_removed", 0),
                "ECG Method": stats.get("ecg", {}).get("method", "none"),
                "ECG Comps": stats.get("ecg", {}).get("n_components_removed", 0),
                "EMG Method": stats.get("emg", {}).get("method", "none"),
                "EMG Comps": stats.get("emg", {}).get("n_components_removed", 0),
            }
            # Add specific timing if exists
            for k, v in timings.items():
                if k != "total":
                    row[f"Time {k} (s)"] = v
                    
            # Spectral stats
            spec = prov.get("spectral_stats", {})
            row.update({
                "Alpha Peak (Hz)": spec.get("alpha_peak", float("nan")),
                "1/f Slope": spec.get("aperiodic_slope", float("nan")),
                "LSD (dB)": spec.get("lsd", float("nan")),
            })
            
            records.append(row)
        except Exception as e:
            LOGGER.warning(f"Could not process {p_file}: {e}")

    if not records:
        return

    df = pd.DataFrame(records)
    
    # Generate Report
    report = mne.Report(title=f"Dataset Correction Summary ({output_desc})")
    
    # 1. Overview Table
    summary_html = f"""
    <h3>Correction Overview</h3>
    <ul>
        <li><b>Total Subjects:</b> {len(df)}</li>
        <li><b>Avg Processing Time:</b> {df["Total Time (s)"].mean():.1f}s (±{df["Total Time (s)"].std():.1f}s)</li>
    </ul>
    """
    
    # Run Outcome
    if success_subjects is not None or failed_subjects is not None:
        n_succ = len(success_subjects) if success_subjects else 0
        n_fail = len(failed_subjects) if failed_subjects else 0
        summary_html += f"""
        <h4>Run Outcome</h4>
        <ul>
            <li><b>Processed:</b> {n_succ + n_fail}</li>
            <li><b>Succeeded:</b> {n_succ}</li>
            <li><b>Failed:</b> {n_fail}</li>
            <li><b>Failed IDs:</b> {', '.join(failed_subjects) if failed_subjects else 'None'}</li>
        </ul>
        """

     # Spectral Summary
    if "Alpha Peak (Hz)" in df.columns:
        summary_html += f"""
        <h4>Spectral Summary</h4>
        <ul>
            <li><b>Alpha Peak:</b> {df["Alpha Peak (Hz)"].mean():.2f} Hz (±{df["Alpha Peak (Hz)"].std():.2f})</li>
            <li><b>1/f Slope:</b> {df["1/f Slope"].mean():.2f} (±{df["1/f Slope"].std():.2f})</li>
             <li><b>LSD:</b> {df["LSD (dB)"].mean():.2f} dB (±{df["LSD (dB)"].std():.2f})</li>
        </ul>
        """
    
    # Aggregate artifacts
    art_html = "<h4>Artifact Components Removed (Average)</h4><ul>"
    for art in ["EOG", "ECG", "EMG"]:
        if f"{art} Comps" in df:
            val = df[f"{art} Comps"].mean()
            std = df[f"{art} Comps"].std()
            art_html += f"<li><b>{art}:</b> {val:.2f} (±{std:.2f})</li>"
    art_html += "</ul>"
    summary_html += art_html
    
    report.add_html(summary_html, title="Overview", section="Summary")
    
    # 2. Histograms
    # Artifact Components Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, art in enumerate(["EOG", "ECG", "EMG"]):
        col = f"{art} Comps"
        if col in df:
            ax = axes[i]
            max_val = int(df[col].max()) + 1
            bins = range(0, max_val + 2)
            ax.hist(df[col], bins=bins, edgecolor="black", color="skyblue", align="left")
            ax.set_title(f"{art} Components Removed")
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Number of Subjects")
            ax.set_xticks(range(0, max_val + 1))
    plt.tight_layout()
    report.add_figure(fig, title="Component Distributions", section="Global Stats")
    plt.close(fig)
    
    # Processing Time Distribution
    fig_time, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Total Time (s)"], bins=15, color="lightcoral", edgecolor="black")
    ax.set_title("Distribution of Processing Times")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of Subjects")
    plt.tight_layout()
    report.add_figure(fig_time, title="Processing Time Distribution", section="Performance")
    plt.close(fig_time)
    
    # Detailed Table
    table_html = df.to_html(classes="table table-striped", index=False)
    report.add_html(table_html, title="Detailed Statistics", section="Details")
    
    # Save
    report.save(summary_report_path, overwrite=True, open_browser=False)
    LOGGER.info(f"Dataset summary report saved to {summary_report_path}")

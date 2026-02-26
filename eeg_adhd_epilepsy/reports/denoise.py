"""Residual Denoising HTML Report Generation (Stage 2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.viz import qc as viz_qc

LOGGER = logging.getLogger("denoise_reports")
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
    report = mne.Report(title="Dataset Denoising Summary")
    report.add_html(message_html, title="Overview", section="Summary")
    report.save(summary_report_path, overwrite=True, open_browser=False)
    LOGGER.info("Dataset report saved to %s", summary_report_path)


def create_denoising_report(
    subject_id: str,
    raw: mne.io.BaseRaw,
    psd_before: Tuple[np.ndarray, np.ndarray],
    psd_after: Tuple[np.ndarray, np.ndarray],
    provenance: Dict[str, Any],
    subject_report_path: Path,
    figures_dir: Optional[Path] = None,
) -> None:
    """Generate and save a Stage 2 subject report."""
    out_path = _normalize_html_report_path(subject_report_path, "subject_report_path")
    report = mne.Report(title=f"Residual Denoising Report (Stage 2) - {subject_id}")

    transient_stats = provenance.get("transient_stats", {})
    ar_stats = provenance.get("autoreject_stats", {})

    method = str(transient_stats.get("method", "none")).upper()
    n_base_total = int(ar_stats.get("base_total", 0))
    n_corrected = int(ar_stats.get("n_base_corrected", 0))
    correction_rate = (100.0 * n_corrected / n_base_total) if n_base_total > 0 else 0.0
    current_bads = int(
        sum(1 for annot in raw.annotations if str(annot["description"]).startswith("BAD_"))
    )

    summary_html = f"""
    <h3>Denoising Summary</h3>
    <ul>
        <li><b>Subject ID:</b> {subject_id}</li>
        <li><b>Transient Method:</b> {method}</li>
        <li><b>AutoReject Correction Rate:</b> {correction_rate:.1f}% ({n_corrected}/{n_base_total} segments)</li>
        <li><b>Final BAD Segment Count:</b> {current_bads}</li>
    </ul>
    """
    
    # Recovery Details
    if "recovered_seconds" in ar_stats:
        rec_sec = float(ar_stats["recovered_seconds"])
        base_sec = float(ar_stats.get("base_bad_seconds", 0))
        final_sec = float(ar_stats.get("final_bad_seconds", 0))
        rec_rate = float(ar_stats.get("recovery_rate_pct", 0))
        
        summary_html += f"""
        <h4>Recovery Statistics</h4>
        <ul>
            <li><b>Base Bad Duration:</b> {base_sec:.2f}s</li>
            <li><b>Recovered Duration:</b> {rec_sec:.2f}s ({rec_rate:.1f}%)</li>
            <li><b>Final Bad Duration:</b> {final_sec:.2f}s</li>
        </ul>
        """
        
        # Artifact Composition Plot
        try:
            total_dur = raw.times[-1]
            # Simple 3-part composition:
            # 1. Always Good (never bad) = Total - Base_Bad - (New_Bad?) ... hard to track New exactly without intersection
            # Let's simplify: 
            # Final Good = Total - Final Bad
            # Final Bad = Final Bad
            # "Recovered" is part of Final Good.
            
            # Let's show: [Original Good, Recovered, Still Bad]
            # Assuming Final Bad is mostly subset of Base Bad (ignoring new artifacts for simplicity or treating them as 'Bad')
            # If Final Bad > Base Bad - Recovered, then we have New Bad.
            
            # Categories:
            # - Recovered (was bad, now good)
            # - Good (was good, stayed good ... approx Total - Base_Bad)
            # - Bad (Final Bad)
            
            # Check consistency
            # If we assume 'New Bad' is negligible or lumping it into 'Bad'
            orig_good = max(0, total_dur - base_sec)
            recovered = rec_sec
            current_bad = final_sec
            
            # Use 'Rest of Good' to balance total if needed, or just normalize pie
            # Actually: Final Good = Total - Final Bad.
            # Consisting of: Recovered + Original Good (approx)
            
            sizes = [orig_good, recovered, current_bad]
            labels = ["Original Clean", "Recovered", "Final Bad"]
            colors = ["lightgray", "mediumseagreen", "salmon"]
            
            fig_comp, ax = plt.subplots(figsize=(6, 4))
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            ax.set_title("Data Composition (Post-Denoising)")
            report.add_figure(fig_comp, title="Data Composition", section="Overview")
            plt.close(fig_comp)
        except Exception as e:
            LOGGER.warning("Failed to create composition plot: %s", e)

    if method != "NONE":
        summary_html += "<h4>Transient Removal Details</h4><ul>"
        if "n_components" in transient_stats:
            summary_html += f"<li><b>Components:</b> {transient_stats['n_components']}</li>"
        if "cutoff" in transient_stats:
            summary_html += f"<li><b>ASR Cutoff:</b> {transient_stats['cutoff']}</li>"
        summary_html += "</ul>"

    spectral_stats = provenance.get("spectral_stats", {})
    if spectral_stats:
        alpha_pre = float(spectral_stats.get("alpha_peak_pre", float("nan")))
        alpha_post = float(spectral_stats.get("alpha_peak_post", float("nan")))
        alpha_delta = float(spectral_stats.get("alpha_peak_delta", float("nan")))
        lsd_db = float(spectral_stats.get("lsd_db", float("nan")))

        summary_html += "<h4>Spectral Deltas</h4><ul>"
        summary_html += f"<li><b>Alpha Peak:</b> {alpha_pre:.2f} Hz -> {alpha_post:.2f} Hz (delta {alpha_delta:+.2f} Hz)</li>"
        summary_html += f"<li><b>LSD:</b> {lsd_db:.2f} dB</li>"
        summary_html += "</ul>"

        band_delta_pct = spectral_stats.get("band_power_delta_pct", {})
        if isinstance(band_delta_pct, dict) and band_delta_pct:
            summary_html += "<h5>Band Power Delta (%)</h5><ul>"
            for band_name, delta in sorted(band_delta_pct.items()):
                summary_html += f"<li><b>{band_name}:</b> {float(delta):+.2f}%</li>"
            summary_html += "</ul>"

    timing_map = provenance.get("benchmarks", {}).get("timing", {})
    if timing_map:
        summary_html += "<h4>Processing Time</h4><ul>"
        total_time = float(sum(float(v) for v in timing_map.values()))
        for step, t in timing_map.items():
            summary_html += f"<li><b>{step}:</b> {float(t):.2f}s</li>"
        summary_html += f"<li><b>total:</b> {total_time:.2f}s</li>"
        summary_html += "</ul>"

    report.add_html(summary_html, title="Summary", section="Overview")

    eps = np.finfo(float).eps
    try:
        freqs_pre, psd_pre = psd_before
        freqs_post, psd_post = psd_after
        if psd_pre.size > 0 and psd_post.size > 0:
            fig = viz_qc.plot_psd_overlay(
                freqs_pre,
                psd_pre,
                freqs_post,
                psd_post,
                EPS=eps,
                label_before="Corrected (Stage 1)",
                label_after="Denoised (Stage 2)",
            )
            report.add_figure(fig, title="PSD Overlay (Stage 1 vs Stage 2)", section="Spectral Analysis")
            plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Failed to plot PSD overlay: %s", exc)

    for name, path_str in transient_stats.get("plot_paths", {}).items():
        img_path = Path(path_str)
        if img_path.exists():
            report.add_image(
                image=str(img_path),
                title=f"Transient Removal: {name.replace('_', ' ').title()}",
                section="Transient Artifacts",
            )

    if figures_dir and figures_dir.exists():
        for plot_path in sorted(figures_dir.glob(f"{subject_id}_autoreject_*.png")):
            clean_name = plot_path.stem.split("autoreject_")[-1].replace("_", " ").title()
            report.add_image(
                str(plot_path),
                title=f"AutoReject Refinement: {clean_name}",
                section="Final Cleanup",
            )

    report.save(out_path, overwrite=True, open_browser=False)
    LOGGER.info("Report saved to %s", out_path)


def create_denoising_dataset_report(
    search_dir: Path,
    summary_report_path: Path,
    output_desc: str = "denoise",
    success_subjects: Optional[List[str]] = None,
    failed_subjects: Optional[List[str]] = None,
) -> None:
    """Aggregate Stage 2 provenance and save dataset summary report."""
    search_dir = Path(search_dir)
    summary_report_path = _normalize_html_report_path(summary_report_path, "summary_report_path")
    provenance_files = sorted(search_dir.rglob(f"*_desc-{output_desc}_provenance.json"))

    if not provenance_files:
        LOGGER.warning("No denoising provenance files found for %s in %s", output_desc, search_dir)
        _save_message_only_dataset_report(
            summary_report_path,
            f"<h3>Dataset Overview</h3><p>No Stage 2 provenance files found under <code>{search_dir}</code>.</p>",
        )
        return

    records: List[Dict[str, Any]] = []
    for p_file in provenance_files:
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                prov = json.load(f)

            subject_id = prov.get("subject_id", p_file.stem.split("_")[0])
            transient_stats = prov.get("transient_stats", {})
            ar_stats = prov.get("autoreject_stats", {})
            timing_map = prov.get("benchmarks", {}).get("timing", {})
            spectral_stats = prov.get("spectral_stats", {})

            total_time = float(sum(float(v) for v in timing_map.values()))
            n_base_total = int(ar_stats.get("base_total", 0))
            n_corrected = int(ar_stats.get("n_base_corrected", 0))
            correction_rate = (100.0 * n_corrected / n_base_total) if n_base_total > 0 else 0.0

            # Duration stats
            dur_s = float(prov.get("data_duration_s", 0))
            final_bad_s = float(ar_stats.get("final_bad_seconds", 0))
            good_pct = 0.0
            if dur_s > 0:
                good_pct = max(0.0, (dur_s - final_bad_s) / dur_s * 100.0)

            row = {
                "Subject": subject_id,
                "Total Time (s)": total_time,
                "Transient Method": transient_stats.get("method", "none"),
                "Transient Components": transient_stats.get("n_components", 0),
                "Base BAD Segments": n_base_total,
                "Corrected Base BAD Segments": n_corrected,
                "Correction Rate (%)": correction_rate,
                "Final Good Data (%)": good_pct,
                "Alpha Pre (Hz)": float(spectral_stats.get("alpha_peak_pre", float("nan"))),
                "Alpha Post (Hz)": float(spectral_stats.get("alpha_peak_post", float("nan"))),
                "Alpha Delta (Hz)": float(spectral_stats.get("alpha_peak_delta", float("nan"))),
                "LSD (dB)": float(spectral_stats.get("lsd_db", float("nan"))),
            }
            band_delta_pct = spectral_stats.get("band_power_delta_pct", {})
            if isinstance(band_delta_pct, dict):
                for band_name, delta in band_delta_pct.items():
                    row[f"Delta {band_name} (%)"] = float(delta)
            for key, val in timing_map.items():
                row[f"Time {key} (s)"] = float(val)
            records.append(row)
        except Exception as exc:
            LOGGER.warning("Could not parse %s: %s", p_file, exc)

    if not records:
        _save_message_only_dataset_report(
            summary_report_path,
            "<h3>Dataset Overview</h3><p>No valid Stage 2 provenance records could be parsed.</p>",
        )
        return

    df = pd.DataFrame(records)
    report = mne.Report(title=f"Dataset Denoising Summary ({output_desc})")

    summary_html = ""
    
    # Run Outcome
    if success_subjects is not None or failed_subjects is not None:
        n_succ = len(success_subjects) if success_subjects else 0
        n_fail = len(failed_subjects) if failed_subjects else 0
        summary_html += f"""
        <h3>Run Outcome</h3>
        <ul>
            <li><b>Processed:</b> {n_succ + n_fail}</li>
            <li><b>Succeeded:</b> {n_succ}</li>
            <li><b>Failed:</b> {n_fail}</li>
            <li><b>Failed IDs:</b> {', '.join(failed_subjects) if failed_subjects else 'None'}</li>
        </ul>
        """
    
    summary_html += f"""
    <h3>Denoising Overview</h3>
    <ul>
        <li><b>Total Subjects:</b> {len(df)}</li>
        <li><b>Avg Processing Time:</b> {df['Total Time (s)'].mean():.1f}s (±{df['Total Time (s)'].std():.1f}s)</li>
        <li><b>Avg Correction Rate:</b> {df['Correction Rate (%)'].mean():.1f}% (±{df['Correction Rate (%)'].std():.1f}%)</li>
        <li><b>Avg Alpha Delta:</b> {df['Alpha Delta (Hz)'].mean():+.2f} Hz (±{df['Alpha Delta (Hz)'].std():.2f})</li>
        <li><b>Avg LSD:</b> {df['LSD (dB)'].mean():.2f} dB (±{df['LSD (dB)'].std():.2f})</li>
    </ul>
    """

    method_counts = df["Transient Method"].value_counts(dropna=False).rename_axis("Method").reset_index(name="Count")
    summary_html += "<h4>Transient Method Usage</h4>" + method_counts.to_html(index=False)

    band_delta_cols = sorted([c for c in df.columns if c.startswith("Delta ") and c.endswith("(%)")])
    if band_delta_cols:
        band_delta_summary = (
            pd.DataFrame(
                {
                    "Band": [col.replace("Delta ", "").replace(" (%)", "") for col in band_delta_cols],
                    "Mean Delta (%)": [float(df[col].mean()) for col in band_delta_cols],
                    "Std Delta (%)": [float(df[col].std()) for col in band_delta_cols],
                }
            )
            .sort_values("Band")
            .reset_index(drop=True)
        )
        summary_html += "<h4>Band Power Delta Summary</h4>" + band_delta_summary.to_html(index=False)
    report.add_html(summary_html, title="Overview", section="Summary")

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    ax_time.hist(df["Total Time (s)"], bins=15, color="steelblue", edgecolor="black")
    ax_time.set_title("Processing Time Distribution")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Number of Subjects")
    plt.tight_layout()
    report.add_figure(fig_time, title="Processing Time", section="Global Stats")
    plt.close(fig_time)

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    ax_rate.hist(df["Correction Rate (%)"], bins=15, color="seagreen", edgecolor="black")
    ax_rate.set_title("Correction Rate Distribution")
    ax_rate.set_xlabel("Correction Rate (%)")
    ax_rate.set_ylabel("Number of Subjects")
    plt.tight_layout()
    report.add_figure(fig_rate, title="Correction Rate", section="Global Stats")
    plt.close(fig_rate)

    fig_lsd, ax_lsd = plt.subplots(figsize=(8, 5))
    ax_lsd.hist(df["LSD (dB)"].dropna(), bins=15, color="mediumpurple", edgecolor="black")
    ax_lsd.set_title("Log-Spectral Distance (LSD) Distribution")
    ax_lsd.set_xlabel("LSD (dB)")
    ax_lsd.set_ylabel("Number of Subjects")
    plt.tight_layout()
    report.add_figure(fig_lsd, title="Spectral Distance", section="Global Stats")
    plt.close(fig_lsd)

    # Final Good Data %
    if "Final Good Data (%)" in df.columns:
        fig_good, ax_good = plt.subplots(figsize=(8, 5))
        ax_good.hist(df["Final Good Data (%)"], bins=20, range=(0, 100), color="cornflowerblue", edgecolor="black")
        ax_good.set_title("Final Good Data % Distribution")
        ax_good.set_xlabel("Good Data (%)")
        ax_good.set_ylabel("Number of Subjects")
        ax_good.set_xlim(0, 100)
        plt.tight_layout()
        report.add_figure(fig_good, title="Final Data Quality", section="Global Stats")
        plt.close(fig_good)

    # Spectral Boxplots
    spec_cols = [c for c in ["Alpha Delta (Hz)", "LSD (dB)"] if c in df.columns]
    if spec_cols:
        fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
        df.boxplot(column=spec_cols, ax=ax_spec)
        ax_spec.set_title("Spectral Metrics Distribution")
        plt.tight_layout()
        report.add_figure(fig_spec, title="Spectral Metrics", section="Global Stats")
        plt.close(fig_spec)

    report.add_html(df.to_html(index=False), title="Detailed Statistics", section="Details")
    report.save(summary_report_path, overwrite=True, open_browser=False)
    LOGGER.info("Dataset summary report saved to %s", summary_report_path)

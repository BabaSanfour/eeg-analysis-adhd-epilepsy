"""Compare reports for DSS vs ICA pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import mne
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compare_reports")
matplotlib.use("Agg")


def _normalize_html_report_path(path: Path, field_name: str) -> Path:
    """Validate report output path and ensure parent directory exists."""
    out_path = Path(path).expanduser()
    if out_path.suffix.lower() != ".html":
        raise ValueError(f"{field_name} must be an .html file path, got: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _series_to_html_table(series: pd.Series, key_name: str, value_name: str) -> str:
    """Render one series as a simple HTML table."""
    df = series.rename_axis(key_name).reset_index(name=value_name)
    return df.to_html(index=False)


def calculate_comparison_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate normalized scores for DSS vs ICA based on multiple objectives."""
    df = metrics_df.copy()
    
    # 1. Signal Preservation Score (0.4) - Favor high correlation
    df["score_corr"] = df["mean_dss_ica_corr"].map(
        lambda x: 0.4 if x > 0.98 else 0.3 if x > 0.95 else 0.15 if x > 0.9 else 0.0
    )
    
    # 2. Spectral Distortion Score (0.3) - Favor low slope shift
    df["score_slope"] = df["slope_distortion"].map(
        lambda x: 0.3 if x < 0.15 else 0.2 if x < 0.3 else 0.1 if x < 0.5 else 0.0
    )
    
    # 3. Component Sanity (0.2) - Penalize aggressive removal
    total_comps = df.get("eog_components", 0) + df.get("ecg_components", 0) + df.get("emg_components", 0)
    df["score_comps"] = total_comps.map(
        lambda x: 0.2 if x <= 4 else 0.15 if x <= 7 else 0.1 if x <= 10 else 0.0
    )
    
    # 4. Variance Efficiency (0.1) - Favor moderate variance reduction
    df["score_var"] = df["variance_removed_pct"].map(
        lambda x: 0.1 if 5 <= x <= 25 else 0.05 if x < 5 or 25 < x <= 40 else 0.0
    )
    
    df["total_score"] = df["score_corr"] + df["score_slope"] + df["score_comps"] + df["score_var"]
    return df


def create_compare_subject_report(
    subject_id: str,
    metrics_rows: List[Dict[str, Any]],
    correlation_map: Optional[Dict[str, float]],
    plot_paths: Dict[str, str],
    subject_report_path: Path,
    mode: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Create per-subject compare report (Original vs DSS vs ICA)."""
    out_path = _normalize_html_report_path(subject_report_path, "subject_report_path")
    report = mne.Report(title=f"Compare Report - {subject_id} ({mode})")

    summary_html = f"""
    <h3>Subject Compare Summary</h3>
    <ul>
        <li><b>Subject ID:</b> {subject_id}</li>
        <li><b>Compare Mode:</b> {mode}</li>
    </ul>
    """
    if metadata:
        trace_items = []
        if metadata.get("dss_desc"):
            trace_items.append(f"<li><b>DSS desc:</b> {metadata['dss_desc']}</li>")
        if metadata.get("ica_desc"):
            trace_items.append(f"<li><b>ICA desc:</b> {metadata['ica_desc']}</li>")
        if metadata.get("condition"):
            trace_items.append(f"<li><b>Condition:</b> {metadata['condition']}</li>")
        if trace_items:
            summary_html += "<h4>Traceability</h4><ul>" + "".join(trace_items) + "</ul>"

    winner_method = "Unknown"
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        if "slope_distortion" in df.columns:
            scored_df = calculate_comparison_scores(df)
            best_idx = scored_df["total_score"].idxmax()
            winner_method = scored_df.loc[best_idx, "method"].upper()
            winner_score = scored_df.loc[best_idx, "total_score"]
            summary_html += f"""
            <div style="background-color: #e8f5e9; border: 2px solid #2e7d32; padding: 10px; border-radius: 8px; margin: 10px 0;">
                <h4 style="margin-top: 0; color: #1b5e20;">🏆 Recommended Winner: <b>{winner_method}</b></h4>
                <p style="margin-bottom: 0;">Score: <b>{winner_score:.2f} / 1.0</b> (Weighted blend of Correlation, Spectral Preservation, and Component Sanity)</p>
            </div>
            """

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        keep_cols = [
            "method",
            "variance_removed_pct",
            "slope_distortion",
            "alpha_peak_shift",
            "mean_dss_ica_corr",
            "eog_components",
            "ecg_components",
            "emg_components",
            "duration_sec",
        ]
        cols = [col for col in keep_cols if col in df.columns]
        if cols:
            display_df = df[cols].copy()
            if "duration_sec" in display_df:
                display_df["duration_sec"] = display_df["duration_sec"].map(lambda v: f"{float(v):.2f}")
            if "variance_removed_pct" in display_df:
                display_df["variance_removed_pct"] = display_df["variance_removed_pct"].map(lambda v: f"{float(v):.3f}")
            if "mean_dss_ica_corr" in display_df:
                display_df["mean_dss_ica_corr"] = display_df["mean_dss_ica_corr"].map(
                    lambda v: "-" if pd.isna(v) else f"{float(v):.4f}"
                )
            if "slope_distortion" in display_df:
                display_df["slope_distortion"] = display_df["slope_distortion"].map(lambda v: f"{float(v):.4f}")
            if "alpha_peak_shift" in display_df:
                display_df["alpha_peak_shift"] = display_df["alpha_peak_shift"].map(lambda v: f"{float(v):.2f} Hz")
            
            summary_html += "<h4>Per-Method Metrics</h4>" + display_df.to_html(index=False)

    if correlation_map:
        corr_series = pd.Series(correlation_map).sort_values(ascending=False)
        summary_html += "<h4>Channel Correlation (DSS vs ICA)</h4>"
        summary_html += _series_to_html_table(corr_series, "Channel", "Pearson r")

    report.add_html(summary_html, title="Overview", section="Summary")

    for name, path_str in sorted(plot_paths.items()):
        plot_path = Path(path_str)
        if plot_path.exists():
            try:
                report.add_image(
                    image=str(plot_path),
                    title=name.replace("_", " ").title(),
                    section="Plots",
                )
            except Exception as exc:
                LOGGER.warning("Failed to add plot %s: %s", plot_path, exc)

    report.save(out_path, overwrite=True, open_browser=False)
    LOGGER.info("Subject compare report saved to %s", out_path)


def create_compare_dataset_report(
    metrics_df: pd.DataFrame,
    summary_report_path: Path,
    mode: str,
    subject_reports: Dict[str, Path],
    global_plot_paths: Dict[str, str],
    missing_subjects: Optional[List[str]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    metrics_csv_path: Optional[Path] = None,
) -> None:
    """Create dataset compare summary report."""
    out_path = _normalize_html_report_path(summary_report_path, "summary_report_path")
    report = mne.Report(title=f"Dataset Compare Summary ({mode})")

    missing_subjects = sorted(set(missing_subjects or []))
    processed_subjects = sorted(set(subject_reports.keys()))

    summary_html = f"""
    <h3>Compare Overview</h3>
    <ul>
        <li><b>Mode:</b> {mode}</li>
        <li><b>Subjects With Reports:</b> {len(processed_subjects)}</li>
        <li><b>Missing/Skipped Subjects:</b> {len(missing_subjects)}</li>
        <li><b>Missing IDs:</b> {', '.join(missing_subjects) if missing_subjects else 'None'}</li>
    </ul>
    """

    if not metrics_df.empty and "slope_distortion" in metrics_df.columns:
        scored_df = calculate_comparison_scores(metrics_df)
        winners = []
        for sub in processed_subjects:
            sub_df = scored_df[scored_df["subject"] == sub]
            if not sub_df.empty:
                best_method = sub_df.loc[sub_df["total_score"].idxmax(), "method"]
                winners.append(best_method)
        
        counts = pd.Series(winners).value_counts().to_dict()
        dss_wins = counts.get("dss", 0)
        ica_wins = counts.get("ica", 0)
        
        summary_html += f"""
        <div style="display: flex; gap: 20px; margin: 20px 0;">
            <div style="flex: 1; background: #e3f2fd; border: 2px solid #1976d2; padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="margin: 0; color: #1976d2;">{dss_wins}</h2>
                <p style="margin: 5px 0 0 0; font-weight: bold;">DSS Winners</p>
            </div>
            <div style="flex: 1; background: #fff3e0; border: 2px solid #f57c00; padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="margin: 0; color: #f57c00;">{ica_wins}</h2>
                <p style="margin: 5px 0 0 0; font-weight: bold;">ICA Winners</p>
            </div>
        </div>
        """

    if metrics_csv_path is not None:
        summary_html += f"<p><b>Metrics CSV:</b> <code>{metrics_csv_path}</code></p>"

    if run_metadata:
        trace_rows = []
        for key in ["mode", "dss_desc", "ica_desc", "strict_existing", "use_provenance_metrics", "condition", "train_condition"]:
            if key in run_metadata:
                trace_rows.append(f"<tr><td>{key}</td><td>{run_metadata[key]}</td></tr>")
        if trace_rows:
            summary_html += "<h4>Run Traceability</h4>"
            summary_html += "<table border='1'><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>"
            summary_html += "".join(trace_rows)
            summary_html += "</tbody></table>"

    if not metrics_df.empty:
        by_method = metrics_df.groupby("method").agg(
            subjects=("subject", "nunique"),
            avg_time_s=("duration_sec", "mean"),
            avg_var_removed_pct=("variance_removed_pct", "mean"),
            avg_eog_comp=("eog_components", "mean"),
            avg_ecg_comp=("ecg_components", "mean"),
            avg_emg_comp=("emg_components", "mean"),
            avg_corr=("mean_dss_ica_corr", "mean"),
        )
        summary_html += "<h4>Method Aggregates</h4>" + by_method.reset_index().to_html(index=False)

    if processed_subjects:
        rows = []
        for subject_id in processed_subjects:
            report_path = subject_reports[subject_id]
            try:
                rel = report_path.relative_to(out_path.parent)
                href = str(rel)
            except Exception:
                href = str(report_path)
            rows.append({"Subject": subject_id, "Report": f"<a href='{href}'>{report_path.name}</a>"})
        idx_df = pd.DataFrame(rows)
        summary_html += "<h4>Subject Report Index</h4>" + idx_df.to_html(index=False, escape=False)

    report.add_html(summary_html, title="Overview", section="Summary")

    for name, path_str in sorted(global_plot_paths.items()):
        path = Path(path_str)
        if path.exists():
            try:
                report.add_image(
                    image=str(path),
                    title=name.replace("_", " ").title(),
                    section="Global Stats",
                )
            except Exception as exc:
                LOGGER.warning("Failed to add global plot %s: %s", path, exc)

    if not metrics_df.empty:
        report.add_html(metrics_df.to_html(index=False), title="Detailed Metrics", section="Details")

    report.save(out_path, overwrite=True, open_browser=False)
    LOGGER.info("Dataset compare report saved to %s", out_path)

"""Compare reports for DSS vs ICA pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import pandas as pd
from coco_pipe.report.core import ImageElement, Report, Section, StatCardElement, TableElement
from coco_pipe.report.elements import CalloutElement

LOGGER = logging.getLogger("compare_reports")
matplotlib.use("Agg")


def _normalize_html_report_path(path: Path, field_name: str) -> Path:
    """Validate report output path and ensure parent directory exists."""
    out_path = Path(path).expanduser()
    if out_path.suffix.lower() != ".html":
        raise ValueError(f"{field_name} must be an .html file path, got: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def calculate_comparison_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate normalized scores for DSS vs ICA based on multiple objectives."""
    df = metrics_df.copy()

    # 1. Signal Preservation Score (0.4) - Favor high correlation
    corr = pd.to_numeric(df.get("mean_dss_ica_corr"), errors="coerce")
    df["score_corr"] = corr.map(
        lambda x: 0.4 if x > 0.98 else 0.3 if x > 0.95 else 0.15 if x > 0.9 else 0.0
        if pd.notna(x)
        else 0.0
    )

    # 2. Spectral Distortion Score (0.3) - Favor low slope shift
    slope = pd.to_numeric(df.get("slope_distortion"), errors="coerce")
    df["score_slope"] = slope.map(
        lambda x: 0.3 if x < 0.15 else 0.2 if x < 0.3 else 0.1 if x < 0.5 else 0.0
        if pd.notna(x)
        else 0.0
    )

    # 3. Component Sanity (0.2) - Penalize aggressive removal
    total_comps = (
        pd.to_numeric(df.get("eog_components"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("ecg_components"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("emg_components"), errors="coerce").fillna(0)
    )
    df["score_comps"] = total_comps.map(
        lambda x: 0.2 if x <= 4 else 0.15 if x <= 7 else 0.1 if x <= 10 else 0.0
    )

    # 4. Variance Efficiency (0.1) - Favor moderate variance reduction
    variance_removed = pd.to_numeric(df.get("variance_removed_pct"), errors="coerce")
    df["score_var"] = variance_removed.map(
        lambda x: 0.1 if 5 <= x <= 25 else 0.05 if x < 5 or 25 < x <= 40 else 0.0
        if pd.notna(x)
        else 0.0
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
    report = Report(title=f"Compare Report - {subject_id} ({mode})")

    summary = Section("Summary", icon="🆚")
    summary.add_markdown(
        f"**Subject ID:** {subject_id}\n\n"
        f"**Compare Mode:** {mode}"
    )

    if metadata:
        trace_lines = []
        if metadata.get("dss_desc"):
            trace_lines.append(f"- **DSS desc:** {metadata['dss_desc']}")
        if metadata.get("ica_desc"):
            trace_lines.append(f"- **ICA desc:** {metadata['ica_desc']}")
        if metadata.get("condition"):
            trace_lines.append(f"- **Condition:** {metadata['condition']}")
        if trace_lines:
            summary.add_markdown("#### Traceability\n" + "\n".join(trace_lines))

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        if "slope_distortion" in df.columns:
            scored_df = calculate_comparison_scores(df)
            best_idx = scored_df["total_score"].idxmax()
            winner_method = str(scored_df.loc[best_idx, "method"]).upper()
            winner_score = scored_df.loc[best_idx, "total_score"]
            summary.add_element(
                CalloutElement(
                    f"Score: {winner_score:.2f} / 1.0 (weighted blend of "
                    "correlation, spectral preservation, and component sanity).",
                    kind="tip",
                    title=f"\U0001f3c6 Recommended Winner: {winner_method}",
                )
            )

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

            summary.add_element(TableElement(display_df, title="Per-Method Metrics"))

    if correlation_map:
        corr_series = pd.Series(correlation_map).sort_values(ascending=False)
        corr_df = corr_series.rename_axis("Channel").reset_index(name="Pearson r")
        summary.add_element(TableElement(corr_df, title="Channel Correlation (DSS vs ICA)"))

    report.add_section(summary)

    plots = Section("Plots", icon="📈")
    for name, path_str in sorted(plot_paths.items()):
        plot_path = Path(path_str)
        if plot_path.exists():
            try:
                plots.add_element(ImageElement(str(plot_path), caption=name.replace("_", " ").title()))
            except Exception as exc:
                LOGGER.warning("Failed to add plot %s: %s", plot_path, exc)
    if plots.children:
        report.add_section(plots)

    report.save(str(out_path))
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
    report = Report(title=f"Dataset Compare Summary ({mode})")

    missing_subjects = sorted(set(missing_subjects or []))
    processed_subjects = sorted(set(subject_reports.keys()))

    overview = Section("Overview", icon="🆚")
    overview.add_markdown(
        f"**Mode:** {mode}\n\n"
        f"**Subjects With Reports:** {len(processed_subjects)}\n\n"
        f"**Missing/Skipped Subjects:** {len(missing_subjects)}\n\n"
        f"**Missing IDs:** {', '.join(missing_subjects) if missing_subjects else 'None'}"
    )

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

        overview.add_columns(
            [
                StatCardElement("DSS Winners", dss_wins, color="blue"),
                StatCardElement("ICA Winners", ica_wins, color="yellow"),
            ],
            cols=2,
        )

    if metrics_csv_path is not None:
        overview.add_markdown(f"**Metrics CSV:** `{metrics_csv_path}`")

    if run_metadata:
        trace_rows = []
        for key in ["mode", "dss_desc", "ica_desc", "strict_existing", "use_provenance_metrics", "condition", "train_condition"]:
            if key in run_metadata:
                trace_rows.append({"Field": key, "Value": run_metadata[key]})
        if trace_rows:
            overview.add_element(TableElement(pd.DataFrame(trace_rows), title="Run Traceability"))

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
        overview.add_element(TableElement(by_method.reset_index(), title="Method Aggregates"))

    if processed_subjects:
        rows = []
        for subject_id in processed_subjects:
            report_path = subject_reports[subject_id]
            try:
                rel = report_path.relative_to(out_path.parent)
                href = str(rel)
            except ValueError as exc:
                LOGGER.warning(
                    "Subject report %s is not relative to %s: %s; using absolute path",
                    report_path,
                    out_path.parent,
                    exc,
                )
                href = str(report_path)
            rows.append({"Subject": subject_id, "Report": f"<a href='{href}'>{report_path.name}</a>"})
        idx_df = pd.DataFrame(rows)
        overview.add_element(TableElement(idx_df, title="Subject Report Index"))

    report.add_section(overview)

    global_stats = Section("Global Stats", icon="📈")
    for name, path_str in sorted(global_plot_paths.items()):
        path = Path(path_str)
        if path.exists():
            try:
                global_stats.add_element(ImageElement(str(path), caption=name.replace("_", " ").title()))
            except Exception as exc:
                LOGGER.warning("Failed to add global plot %s: %s", path, exc)
    if global_stats.children:
        report.add_section(global_stats)

    if not metrics_df.empty:
        details = Section("Details", icon="📋")
        details.add_element(TableElement(metrics_df, title="Detailed Metrics"))
        report.add_section(details)

    report.save(str(out_path))
    LOGGER.info("Dataset compare report saved to %s", out_path)

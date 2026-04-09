"""Quality Control HTML Report Generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

from eeg_adhd_epilepsy.utils.config import BAND_LIMITS
from eeg_adhd_epilepsy.viz import qc as viz_qc


SEGMENT_REPORT_METRICS = (
    {"column": "segment_duration_sec", "title": "Segment Duration", "ylabel": "Duration (s)", "kind": "bar"},
    {"column": "segment_amplitude_mean_uv", "title": "Mean Amplitude", "ylabel": "Mean amplitude (uV)", "kind": "line"},
    {"column": "segment_pct_bad_channels", "title": "Percent Bad Channels", "ylabel": "Bad channels (%)", "kind": "line"},
    {"column": "segment_line_noise_ratio", "title": "Line Noise Ratio", "ylabel": "Line-noise ratio", "kind": "line"},
    {"column": "segment_hf_lf_ratio", "title": "HF/LF Ratio", "ylabel": "HF/LF ratio", "kind": "line"},
    {"column": "segment_aperiodic_slope", "title": "Aperiodic Slope", "ylabel": "Aperiodic slope", "kind": "line"},
    {"column": "segment_band_power_alpha", "title": "Alpha Band Power", "ylabel": "Alpha power (uV^2)", "kind": "line"},
)

SUBJECT_SEGMENT_REPORT_METRICS = tuple(
    spec for spec in SEGMENT_REPORT_METRICS if spec["column"] != "segment_duration_sec"
)


def format_seconds_hms(seconds: float | None) -> str:
    """Return a human-readable H:M:S string for a seconds value."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "0s"
    if not math.isfinite(value):
        return "0s"
    value = max(0.0, value)
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    value -= minutes * 60
    seconds_str = f"{value:.2f}".rstrip("0").rstrip(".")
    if not seconds_str:
        seconds_str = "0"
    sec_component = f"{seconds_str}s"
    if hours > 0:
        return f"{hours}h {minutes}m {sec_component}"
    if minutes > 0:
        return f"{minutes}m {sec_component}"
    return sec_component


def _compute_flagged_percentages_by_segment(
    segments_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (% flagged segments, % flagged subjects, subject counts, flagged subject counts) per segment_type."""
    if segments_df is None or segments_df.empty or "segment_type" not in segments_df:
        empty = pd.Series(dtype=float)
        empty_int = pd.Series(dtype=int)
        return empty, empty, empty_int, empty_int
    df = segments_df.copy()
    df["segment_type"] = df["segment_type"].fillna("Unknown").astype(str)
    if "segment_flag_bad" in df:
        df["flag_bad_bool"] = pd.to_numeric(df["segment_flag_bad"], errors="coerce").fillna(0).astype(bool)
    else:
        df["flag_bad_bool"] = False

    seg_pct = df.groupby("segment_type")["flag_bad_bool"].mean() * 100.0
    subj_pct = pd.Series(dtype=float)
    subject_counts = pd.Series(dtype=int)
    flagged_subject_counts = pd.Series(dtype=int)
    if "subject_id" in df:
        subject_counts = df.groupby("segment_type")["subject_id"].nunique()
        flagged_subject_counts = (
            df[df["flag_bad_bool"]]
            .groupby("segment_type")["subject_id"]
            .nunique()
        )
        subj_pct = (flagged_subject_counts / subject_counts.replace(0, np.nan) * 100.0).fillna(0.0)

    subject_counts = subject_counts.sort_index()
    flagged_subject_counts = flagged_subject_counts.reindex(subject_counts.index).fillna(0).astype(int)
    return seg_pct.sort_index(), subj_pct.sort_index(), subject_counts, flagged_subject_counts


def create_subject_report(
    raw: mne.io.BaseRaw | mne.Epochs,
    metrics: Dict[str, object],
    subject_id: str,
    output_path: Path,
    fig_paths: Mapping[str, Path | str],
) -> None:
    """Reusable subject HTML report using saved figure paths."""
    report = mne.Report(title=f"EEG QC Report - {subject_id}")
    
    # --- QC Summary (First) ---
    duration_min = metrics.get("duration_min", float("nan"))
    sfreq = metrics.get("sfreq", float("nan"))
    n_channels = metrics.get("segment_n_channels", 0)
    n_1020 = metrics.get("n_channels_1020_match", 0)
    pct_bad = metrics.get("segment_pct_bad_channels", float("nan"))
    amp_mean = metrics.get("segment_amplitude_mean_uv", float("nan"))
    amp_median = metrics.get("segment_amplitude_median_uv", float("nan"))
    amp_max = metrics.get("segment_amplitude_max_uv", float("nan"))
    alpha_peak = metrics.get("segment_alpha_peak_hz", float("nan"))
    start_sec = metrics.get("actual_signal_start_sec", float("nan"))
    end_sec = metrics.get("actual_signal_end_sec", float("nan"))
    empty_start = metrics.get("empty_start_sec", float("nan"))
    empty_end = metrics.get("empty_end_sec", float("nan"))
    n_flat = metrics.get("segment_n_flat_channels", 0)
    n_noisy = metrics.get("segment_n_noisy_channels", 0)
    line_noise_ratio = metrics.get("segment_line_noise_ratio", float("nan"))
    hf_ratio = metrics.get("segment_hf_lf_ratio", float("nan"))
    slope = metrics.get("segment_aperiodic_slope", float("nan"))

    band_power_items = []
    for band in BAND_LIMITS:
        value = metrics.get(f"segment_band_power_{band}", float("nan"))
        if np.isnan(value):
            continue
        band_power_items.append(f"{band.title()}: {value:.2e} uV^2")
    band_str = ", ".join(band_power_items) if band_power_items else "Unavailable"

    qc_summary_html = "<ul>"
    qc_summary_html += f"<li>Duration: {duration_min:.2f} min @ {sfreq:.1f} Hz</li>"
    qc_summary_html += f"<li>Channels: {n_channels} total / {n_1020} (10-20 match)</li>"
    qc_summary_html += f"<li>Bad channels: {pct_bad:.1f}% (flat={n_flat}, noisy={n_noisy})</li>"
    qc_summary_html += (
        f"<li>Signal activity: start {start_sec:.1f}s (empty {empty_start:.1f}s), "
        f"end {end_sec:.1f}s (empty tail {empty_end:.1f}s)</li>"
    )
    qc_summary_html += (
        f"<li>Amplitude (uV): mean {amp_mean:.1f}, median {amp_median:.1f}, max {amp_max:.1f}</li>"
    )
    qc_summary_html += f"<li>Alpha peak: {alpha_peak:.2f} Hz</li>"
    qc_summary_html += f"<li>Band powers: {band_str}</li>"
    qc_summary_html += f"<li>Line-noise ratio: {line_noise_ratio:.2f}</li>"
    qc_summary_html += f"<li>HF/LF ratio: {hf_ratio:.2f}</li>"
    qc_summary_html += f"<li>Aperiodic slope: {slope:.2f}</li>"
    if metrics.get("condition_flags"):
        qc_summary_html += f"<li>Condition flags: {metrics['condition_flags']}</li>"
    if metrics.get("flag_reasons"):
        qc_summary_html += f"<li>Flag reasons: {metrics.get('flag_reasons')}</li>"
    if metrics.get("event_counts"):
        qc_summary_html += f"<li>Events: {metrics.get('event_counts')}</li>"
    qc_summary_html += "</ul>"
    report.add_html(qc_summary_html, title="QC Summary", section="Quality Control")
    
    # --- Signal Quality Figures ---
    if "amplitude_hist" in fig_paths:
        report.add_image(fig_paths["amplitude_hist"], title="Amplitude Distribution", section="Signal Quality")
    if "variance_topo" in fig_paths:
        report.add_image(fig_paths["variance_topo"], title="Channel Variance Topomap", section="Signal Quality")
        
    # --- Spectral / Topomaps ---
    if "spectral_topomaps_grid" in fig_paths:
        report.add_image(fig_paths["spectral_topomaps_grid"], title="Band Power Topomaps", section="Topographic Metrics")
    
    if "signal_quality_grid" in fig_paths:
        report.add_image(fig_paths["signal_quality_grid"], title="Signal Quality Topomaps", section="Topographic Metrics")
        
    for key, path in fig_paths.items():
        if key.endswith("_topo") and key not in ["variance_topo", "line_noise_topo"]:
             title = key.replace("_topo", "").replace("_", " ").title() + " Power"
             report.add_image(path, title=title, section="Topographic Metrics (Individual)")

    # --- PSD Figures ---
    if "psd_all" in fig_paths:
        report.add_image(fig_paths["psd_all"], title="PSD - All Channels", section="Power Spectral Density")
    if "psd_avg" in fig_paths:
        report.add_image(fig_paths["psd_avg"], title="PSD - Average", section="Power Spectral Density")
    if "psd_overlay" in fig_paths:
        report.add_image(fig_paths["psd_overlay"], title="PSD Overlay", section="Power Spectral Density")
    if "events" in fig_paths:
        report.add_image(fig_paths["events"], title="Annotation Counts", section="Events")
        
    # --- Raw Data (Last) ---
    try:
        report.add_raw(raw, title="Raw Data (with PSD)", psd=True, duration=30.0, start=0.0)
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


def save_segment_dataset_figures(
    segments_df: pd.DataFrame,
    fig_dir: Path,
    metric_specs: Sequence[Mapping[str, str]] = SEGMENT_REPORT_METRICS,
    topomap_aggregates: Mapping[str, Tuple[Sequence[str], np.ndarray]] | None = None,
) -> Dict[str, Path]:
    """Save dataset-level histograms for segment metrics."""
    paths: Dict[str, Path] = {}
    for spec in metric_specs:
        path = viz_qc.plot_segment_metric_distribution_by_type(
            segments_df,
            column=spec["column"],
            title=spec["title"],
            xlabel=spec["ylabel"],
            fig_dir=fig_dir,
        )
        if path:
            paths[spec["column"]] = path
    flagged_seg_pct, flagged_subj_pct, _subj_counts, _flagged_subj_counts = _compute_flagged_percentages_by_segment(segments_df)
    flagged_segments_path = viz_qc.plot_flagged_percentages(
        flagged_seg_pct,
        title="Flagged Segments by Type (%)",
        xlabel="Flagged segments (%)",
        fig_dir=fig_dir,
        filename="flagged_segments_pct.png",
    )
    if flagged_segments_path:
        paths["flagged_segments_pct"] = flagged_segments_path
    flagged_subjects_path = viz_qc.plot_flagged_percentages(
        flagged_subj_pct,
        title="Flagged Subjects by Type (%)",
        xlabel="Flagged subjects (%)",
        fig_dir=fig_dir,
        filename="flagged_subjects_pct.png",
    )
    if flagged_subjects_path:
        paths["flagged_subjects_pct"] = flagged_subjects_path
    flagged_subject_dist_path = viz_qc.plot_flagged_subject_distribution(segments_df, fig_dir)
    if flagged_subject_dist_path:
        paths["flagged_subjects_distribution"] = flagged_subject_dist_path

    if topomap_aggregates:
        for metric_key, (channels, values) in topomap_aggregates.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0 or len(channels) != arr.size:
                continue
            seg_type = None
            base_key = metric_key
            if "::" in metric_key:
                seg_type, base_key = metric_key.split("::", 1)
            if base_key.startswith("band_power_"):
                title = f"{base_key.replace('band_power_', '').title()} Band Power Topomap"
                cmap = "viridis"
            elif base_key in {"line_noise_ratio", "hf_lf_ratio", "aperiodic_slope"}:
                title = f"{base_key.replace('_', ' ').title()} Topomap"
                cmap = "RdBu_r"
            elif base_key == "variance":
                title = "Variance Topomap"
                cmap = "viridis"
            else:
                title = f"{base_key.replace('_', ' ').title()} Topomap"
                cmap = "viridis"
            if seg_type:
                title = f"{seg_type}: {title}"
            fig = viz_qc.plot_topomap_from_channel_values(channels, arr, title=title, cmap=cmap, unit=None)
            if fig is None:
                continue
            out_name = metric_key.replace("::", "_") + "_topomap.png"
            out_path = fig_dir / out_name
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths[f"{metric_key}_topomap"] = out_path
    return paths


import base64

def _make_gallery_html(
    images: Sequence[Tuple[str, Path]], 
    title: str = "Gallery",
    columns: int = 4
) -> str:
    """Create a responsive HTML gallery for a set of image paths."""
    if not images:
        return ""
        
    html = f'<h4>{title}</h4><div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: flex-start;">'
    
    # Calculate width rough percentage
    width_pct = int(100 / columns) - 2 # minus distinct gap
    
    for title_text, path in images:
        if not path.exists():
            continue
            
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            
        mime = "image/png"
        if str(path).endswith(".jpg") or str(path).endswith(".jpeg"):
            mime = "image/jpeg"
        elif str(path).endswith(".svg"):
            mime = "image/svg+xml"
            
        src = f"data:{mime};base64,{encoded}"
        
        card_style = (
            f"flex: 0 0 {width_pct}%; "
            "box-shadow: 0 2px 5px rgba(0,0,0,0.1); "
            "border-radius: 4px; "
            "overflow: hidden; "
            "margin-bottom: 20px; "
            "background: #fff; "
            "text-align: center;"
        )
        
        img_style = "width: 100%; height: auto; display: block;"
        
        html += f"""
        <div style="{card_style}">
            <div style="padding: 10px; font-weight: bold; background: #f8f9fa; border-bottom: 1px solid #eee;">{title_text}</div>
            <img src="{src}" style="{img_style}" alt="{title_text}"/>
        </div>
        """
        
    html += "</div>"
    return html

def create_segment_dataset_report(
    segments_df: pd.DataFrame,
    fig_paths: Mapping[str, Path],
    output_path: Path,
) -> None:
    """Dataset-level HTML report for segment QC."""
    if segments_df is None or segments_df.empty:
        return
    report = mne.Report(title="Segment QC Dataset Summary")
    subject_count = int(segments_df.get("subject_id", pd.Series(dtype=str)).nunique())
    flagged = segments_df.get("segment_flag_bad")
    flagged_count = int(pd.to_numeric(flagged, errors="coerce").fillna(0).astype(bool).sum()) if flagged is not None else 0
    durations = pd.to_numeric(segments_df.get("duration"), errors="coerce").dropna()
    total_duration = float(durations.sum()) if not durations.empty else float("nan")
    total_duration_readable = format_seconds_hms(total_duration)
    type_counts = (
        segments_df.get("segment_type", pd.Series(dtype=str)).fillna("Unknown").value_counts().to_dict()
        if not segments_df.empty
        else {}
    )
    summary_html = f"""
    <h3>Dataset Summary</h3>
    <ul>
        <li>Subjects: {subject_count}</li>
        <li>Total segments: {len(segments_df)}</li>
        <li>Total duration: {total_duration:.1f} s ({total_duration_readable})</li>
        <li>Flagged segments: {flagged_count}</li>
    </ul>
    """
    if type_counts:
        summary_html += "<p>Segment type counts:</p><ul>"
        for seg_type, count in type_counts.items():
            summary_html += f"<li>{seg_type}: {count}</li>"
        summary_html += "</ul>"
        
    flagged_seg_pct, flagged_subj_pct, subject_counts, flagged_subject_counts = _compute_flagged_percentages_by_segment(segments_df)
    if not flagged_seg_pct.empty:
        summary_html += "<p>Flagged segments (% of segments) by type:</p><ul>"
        for seg_type, pct in flagged_seg_pct.sort_values(ascending=False).items():
            summary_html += f"<li>{seg_type}: {pct:.1f}%</li>"
        summary_html += "</ul>"
    if not flagged_subj_pct.empty:
        summary_html += "<p>Flagged subjects (% of subjects with that segment type):</p><ul>"
        for seg_type, pct in flagged_subj_pct.sort_values(ascending=False).items():
            total = int(subject_counts.get(seg_type, 0))
            flagged_n = int(flagged_subject_counts.get(seg_type, 0))
            summary_html += f"<li>{seg_type}: {pct:.1f}% ({flagged_n}/{total} subjects)</li>"
        summary_html += "</ul>"
    report.add_html(summary_html, title="Summary", section="Overview")

    for spec in SEGMENT_REPORT_METRICS:
        path = fig_paths.get(spec["column"])
        if path and path.exists():
            report.add_image(path, title=f"{spec['title']} Distribution by Type", section="Metric Distributions")

    for title, key in [
        ("Flagged Segments by Type (%)", "flagged_segments_pct"),
        ("Flagged Subjects by Type (%)", "flagged_subjects_pct"),
        ("Flagged Segments per Subject Distribution", "flagged_subjects_distribution"),
    ]:
        fig_path = fig_paths.get(key)
        if fig_path and fig_path.exists():
            report.add_image(fig_path, title=title, section="Flagged Rates")

    # GRID GALLERY IMPLEMENTATION
    topo_items = sorted([item for item in fig_paths.items() if item[0].endswith("_topomap")])
    if topo_items:
        gallery_images = []
        for key, path in topo_items:
            title = key.replace("_topomap", "").replace("_", " ").title()
            gallery_images.append((title, path))
            
        gallery_html = _make_gallery_html(gallery_images, title="Topographic Metrics")
        report.add_html(gallery_html, title="Topomaps", section="Topographic Metrics")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


def save_meas_distribution_figures(meas_datetimes: pd.Series, fig_dir: Path) -> Dict[str, Path]:
    meas_datetimes = meas_datetimes.dropna()
    if meas_datetimes.empty:
        return {}

    paths: Dict[str, Path] = {}

    def _save_hist(values: np.ndarray, bins: np.ndarray, title: str, xlabel: str, filename: str,
                   xticks: np.ndarray | None = None, xlabels: List[str] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        if xticks is not None:
            ax.set_xticks(xticks)
            if xlabels is not None:
                ax.set_xticklabels(xlabels)
        plt.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths[filename.replace(".png", "")] = out_path

    hour_values = meas_datetimes.dt.hour + (meas_datetimes.dt.minute / 60.0)
    hour_bins = np.arange(0.0, 24.5, 0.5)
    _save_hist(
        hour_values.to_numpy(dtype=float),
        hour_bins,
        "Recording Start Hour",
        "Hour (30 min bins)",
        "meas_hour_distribution.png",
        xticks=np.arange(0, 25, 2),
    )

    day_values = meas_datetimes.dt.day
    day_bins = np.arange(0.5, 32.5, 1.0)
    _save_hist(
        day_values.to_numpy(dtype=float),
        day_bins,
        "Recording Day of Month",
        "Day of Month",
        "meas_day_distribution.png",
        xticks=np.arange(1, 32, 2),
    )

    dow_values = meas_datetimes.dt.dayofweek
    dow_bins = np.arange(-0.5, 7.5, 1.0)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _save_hist(
        dow_values.to_numpy(dtype=float),
        dow_bins,
        "Recording Day of Week",
        "Day of Week",
        "meas_dayofweek_distribution.png",
        xticks=np.arange(0, 7, 1),
        xlabels=dow_labels,
    )

    month_values = meas_datetimes.dt.month
    month_bins = np.arange(0.5, 12.5 + 1, 1.0)
    _save_hist(
        month_values.to_numpy(dtype=float),
        month_bins,
        "Recording Month",
        "Month",
        "meas_month_distribution.png",
        xticks=np.arange(1, 13, 1),
    )

    year_values = meas_datetimes.dt.year
    year_min = int(year_values.min())
    year_max = int(year_values.max())
    year_bins = np.arange(year_min - 0.5, year_max + 1.5, 1.0)
    _save_hist(
        year_values.to_numpy(dtype=float),
        year_bins,
        "Recording Year",
        "Year",
        "meas_year_distribution.png",
        xticks=np.arange(year_min, year_max + 1, 1),
    )

    return paths


def save_figures(
    df: pd.DataFrame,
    flags_counter: Mapping[str, int],
    fig_dir: Path,
    meas_datetimes: pd.Series | None = None,
    topomap_aggregates: Mapping[str, Tuple[Sequence[str], np.ndarray]] | None = None,
) -> Dict[str, Path]:
    from collections import Counter
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def _save_hist(column: str, title: str, filename: str):
        if column not in df:
            return
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths[column] = out_path

    _save_hist("duration_min", "Duration Distribution (min)", "dataset_duration_distribution.png")
    _save_hist("segment_amplitude_mean_uv", "Mean Amplitude Distribution (uV)", "dataset_amplitude_distribution.png")
    _save_hist("segment_alpha_peak_hz", "Alpha Peak Distribution (Hz)", "dataset_alpha_peak_distribution.png")
    _save_hist("segment_hf_lf_ratio", "HF/LF Ratio Distribution", "dataset_hf_ratio_distribution.png")
    _save_hist("segment_aperiodic_slope", "Aperiodic Slope Distribution", "dataset_slope_distribution.png")
    _save_hist("segment_line_noise_ratio", "Line Noise Ratio Distribution", "dataset_line_noise_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if flags_counter:
        if hasattr(flags_counter, "most_common"):
            labels, values = zip(*flags_counter.most_common())
        else:
            sorted_items = sorted(flags_counter.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*sorted_items)
        ax.bar(labels, values)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Flagged Subjects by Reason")
    plt.tight_layout()
    flag_path = fig_dir / "flagged_subjects_summary.png"
    fig.savefig(flag_path, dpi=150)
    plt.close(fig)
    paths["flag_reasons"] = flag_path

    event_specs = [
        ("Eyes Open", "eyes_open_event_count"),
        ("Eyes Closed", "eyes_closed_event_count"),
        ("Movement", "movement_event_count"),
        ("Artefact", "artefact_event_count"),
        ("Ignore System", "ignore_system_event_count"),
        ("HV", "hv_event_count"),
        ("PHOTO", "photo_event_count"),
        ("Yawning/Coughing", "yawning_coughing_event_count"),
        ("Jaw/Face Tension", "jaw_face_tension_event_count"),
        ("Sleepy", "sleepy_event_count"),
        ("Sleep", "sleep_event_count"),
        ("Collaboration", "collaboration_event_count"),
        ("Emotion/Behavior", "emotion_behavior_event_count"),
        ("Oral Activity", "oral_activity_event_count"),
        ("Eye Movement", "eye_movement_event_count"),
        ("Wakefulness", "wakefulness_event_count"),
        ("Respiration", "respiration_event_count"),
        ("Sensor Actions", "sensor_action_keyword_event_count"),
        ("Eye Movement Keywords", "eye_movement_keyword_event_count"),
        ("Clinical Comments", "clinical_comment_event_count"),
    ]
    events_distribution: Dict[str, np.ndarray] = {}
    for label, count_col in event_specs:
        if count_col not in df:
            continue
        series = pd.to_numeric(df[count_col], errors="coerce").dropna()
        if label not in {"Eyes Open", "Eyes Closed", "HV", "PHOTO"}:
            series = series[series > 0]
        if series.empty:
            continue
        events_distribution[label] = series.to_numpy(dtype=float)
    if events_distribution:
        fig = viz_qc.plot_events_distribution(events_distribution)
        if fig is not None:
            event_path = fig_dir / "event_count_distributions.png"
            fig.savefig(event_path, dpi=150)
            plt.close(fig)
            paths["event_stats"] = event_path

    if meas_datetimes is not None and not meas_datetimes.empty:
        meas_paths = save_meas_distribution_figures(meas_datetimes, fig_dir)
        paths.update(meas_paths)

    if topomap_aggregates:
        for metric_key, (channels, values) in topomap_aggregates.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0 or len(channels) != arr.size:
                continue
            seg_type = None
            base_key = metric_key
            if "::" in metric_key:
                seg_type, base_key = metric_key.split("::", 1)
            if base_key.startswith("band_power_"):
                title = f"{base_key.replace('band_power_', '').title()} Band Power Topomap"
                cmap = "viridis"
            elif base_key in {"line_noise_ratio", "hf_lf_ratio", "aperiodic_slope"}:
                title = f"{base_key.replace('_', ' ').title()} Topomap"
                cmap = "RdBu_r"
            elif base_key == "variance":
                title = "Variance Topomap"
                cmap = "viridis"
            else:
                title = f"{base_key.replace('_', ' ').title()} Topomap"
                cmap = "viridis"
            if seg_type:
                title = f"{seg_type}: {title}"
            fig = viz_qc.plot_topomap_from_channel_values(channels, arr, title=title, cmap=cmap, unit=None)
            if fig is None:
                continue
            out_name = metric_key.replace("::", "_") + "_topomap.png"
            out_path = fig_dir / out_name
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths[f"{metric_key}_topomap"] = out_path

    return paths


def create_summary_report(
    df: pd.DataFrame,
    fig_paths: Dict[str, Path],
    output_path: Path,
    total_files: int,
    flags_counter: Mapping[str, int],
    unknown_events: Dict[str, Dict[str, int]] | None = None,
) -> None:
    report = mne.Report(title="EEG QC Dataset Summary")
    valid_records = int((df["error"] == "").sum()) if "error" in df else len(df)
    flagged_count = int(df["flag_bad"].sum()) if "flag_bad" in df else 0
    summary_html = f"""
    <h3>Dataset Summary</h3>
    <ul>
        <li>Total files processed: {total_files}</li>
        <li>Valid records: {valid_records}</li>
        <li>Flagged bad: {flagged_count}</li>
    </ul>
    """
    if flags_counter:
        summary_html += "<p>Most common flag reasons:</p><ul>"
        # Handle if Counter object or dict
        items = []
        if hasattr(flags_counter, "most_common"):
            items = flags_counter.most_common()
        else:
            items = sorted(flags_counter.items(), key=lambda x: x[1], reverse=True)
            
        for reason, count in items:
            summary_html += f"<li>{reason}: {count}</li>"
        summary_html += "</ul>"

    report.add_html(summary_html, title="Summary", section="Overview")

    for title, path in [
        ("Duration Distribution", fig_paths.get("duration_min")),
        ("Mean Amplitude Distribution", fig_paths.get("segment_amplitude_mean_uv")),
        ("Alpha Peak Distribution", fig_paths.get("segment_alpha_peak_hz")),
        ("HF Ratio Distribution", fig_paths.get("segment_hf_lf_ratio")),
        ("Aperiodic Slope Distribution", fig_paths.get("segment_aperiodic_slope")),
        ("Line Noise Distribution", fig_paths.get("segment_line_noise_ratio")),
        ("Flag Reasons", fig_paths.get("flag_reasons")),
        ("Event Count Distributions", fig_paths.get("event_stats")),
        ("Recording Start Hour", fig_paths.get("meas_hour_distribution")),
        ("Recording Day of Month", fig_paths.get("meas_day_distribution")),
        ("Recording Day of Week", fig_paths.get("meas_dayofweek_distribution")),
        ("Recording Month", fig_paths.get("meas_month_distribution")),
        ("Recording Year", fig_paths.get("meas_year_distribution")),
    ]:
        if path and path.exists():
            report.add_image(path, title=title, section="Figures")

    # GRID GALLERY IMPLEMENTATION
    topo_items = sorted([item for item in fig_paths.items() if item[0].endswith("_topomap")])
    if topo_items:
        gallery_images = []
        for key, path in topo_items:
            title = key.replace("_topomap", "").replace("_", " ").title()
            gallery_images.append((title, path))
            
        gallery_html = _make_gallery_html(gallery_images, title="Topographic Metrics")
        report.add_html(gallery_html, title="Topomaps", section="Topographic Metrics")

    if unknown_events:
        unknown_html = "<p>Unrecognized annotation labels:</p><ul>"
        for label, stats in sorted(unknown_events.items(), key=lambda item: item[1]["occurrences"], reverse=True):
            unknown_html += (
                f"<li>{label}: {stats['occurrences']} occurrences; {stats['n_subjects']} subjects</li>"
            )
        unknown_html += "</ul>"
        report.add_html(unknown_html, title="Unrecognized Annotation Labels", section="Unrecognized Annotations")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)

def create_segment_subject_report(
    segments_df: pd.DataFrame,
    subject_id: str,
    output_path: Path,
    fig_paths: Mapping[str, Path | str] | None = None,
) -> None:
    """Create a HTML report for a single subject's segments."""
    if segments_df is None or segments_df.empty:
        return

    report = mne.Report(title=f"Segment QC Report - {subject_id}")
    
    # Summary of segments
    n_segments = len(segments_df)
    flagged = segments_df.get("segment_flag_bad")
    n_flagged = int(pd.to_numeric(flagged, errors="coerce").fillna(0).astype(bool).sum()) if flagged is not None else 0
    
    summary_html = f"""
    <h3>Segment Summary</h3>
    <ul>
        <li>Subject: {subject_id}</li>
        <li>Total segments extracted: {n_segments}</li>
        <li>Flagged segments: {n_flagged}</li>
    </ul>
    """
    
    # Table of segments
    cols_to_show = [col for col in [
        "segment_type", "t_start", "duration", "segment_flag_bad", 
        "segment_amplitude_mean_uv", "segment_pct_bad_channels", 
        "segment_hf_lf_ratio", "segment_line_noise_ratio"
    ] if col in segments_df.columns]
    
    table_html = segments_df[cols_to_show].to_html(classes="table table-striped", index=False, float_format="%.2f")
    
    summary_html += "<h4>Segment Details</h4>"
    summary_html += table_html
    
    report.add_html(summary_html, title="Segments", section="Overview")
    
    # Add Topomap Grids if available
    if fig_paths:
        topo_items = sorted([item for item in fig_paths.items() if "topomap" in item[0].lower()])
        if topo_items:
            for key, path in topo_items:
             # e.g. "Eyes Open_topomaps.png"
                 title = key.replace("_topomaps", "").replace("_", " ").title()
                 report.add_image(path, title=title, section="Segment Type Averages")
             
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)

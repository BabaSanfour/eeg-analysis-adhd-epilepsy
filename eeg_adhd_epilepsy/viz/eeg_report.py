"""Visualizations for the pre-base EEG report family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coco_pipe.viz import plot_bar, plot_histogram

import eeg_adhd_epilepsy.reports.eeg_report as report_eeg
from eeg_adhd_epilepsy.viz import utils

matplotlib.use("Agg")

plt.style.use("seaborn-v0_8-whitegrid")

FIGURE_FILENAMES = {
    "segment_duration": "segment_total_duration.png",
    "eye_state_breakdown": "segment_eye_state_breakdown.png",
    "photo_frequency": "photo_frequency_duration.png",
    "hv_blocks": "hv_block_eye_states.png",
    "post_hv_blocks": "post_hv_block_eye_states.png",
    "timeline": "segment_timeline.png",
    "runs_per_subject": "runs_per_subject.png",
    "recording_start_hour_distribution": "recording_start_hour_distribution.png",
    "run_duration_distribution": "run_duration_distribution.png",
    "availability_by_source_dataset": "availability_by_source_dataset.png",
    "availability_by_combined_diagnosis": "availability_by_combined_diagnosis.png",
    "duration_by_source_dataset": "duration_by_source_dataset.png",
    "duration_by_combined_diagnosis": "duration_by_combined_diagnosis.png",
    "dataset_event_distributions_conditions": "dataset_event_distributions_conditions.png",
    "dataset_event_distributions_clinical": "dataset_event_distributions_clinical.png",
}

CONDITION_EVENT_LABELS = ("Eyes Open", "Eyes Closed", "HV Start", "HV End", "Post-HV", "Photo")


def plot_total_duration_by_segment(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = df.groupby("segment_type", dropna=False)["duration"].sum().sort_values(ascending=True)
    if group.empty:
        return None
    fig, ax = plot_bar(
        group,
        sort=False,
        orientation="horizontal",
        color="#4C72B0",
        title="Total Duration by Segment Type",
        xlabel="Total Duration (s)",
        figsize=(7, max(3, len(group) * 0.35)),
    )
    ax.grid(True, axis="x", alpha=0.2)
    return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES["segment_duration"])


def plot_eye_state_breakdown(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    group = (
        df.groupby(["segment_type", "eye_state"], dropna=False)["duration"]
        .sum()
        .unstack(fill_value=0.0)
    )
    if group.empty:
        return None
    if "eo" in group.columns:
        group = group.sort_values(by="eo", ascending=False)
    labels = group.index.astype(str)
    eo_vals = group.get("eo", pd.Series(0.0, index=group.index)).to_numpy(dtype=float)
    ec_vals = group.get("ec", pd.Series(0.0, index=group.index)).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(3, len(group) * 0.35)))
    positions = np.arange(len(labels))
    ax.barh(positions, eo_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, ec_vals, left=eo_vals, label="Eyes Closed", color="#C44E52")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Eyes-Open vs Eyes-Closed Duration per Segment Type")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES["eye_state_breakdown"])


def plot_photo_frequency_durations(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    photo = df[df["block_family"].fillna("").astype(str).eq("photo")]
    if photo.empty or "freq_hz" not in photo:
        return None
    group = photo.dropna(subset=["freq_hz"]).groupby("freq_hz")["duration"].sum().sort_index()
    if group.empty:
        return None
    fig, ax = plot_bar(
        group,
        sort=False,
        color="#8172B2",
        title="PHOTO Block Duration by Frequency",
        xlabel="PHOTO Frequency (Hz)",
        ylabel="Total Duration (s)",
        figsize=(7, 4),
    )
    ax.grid(True, axis="y", alpha=0.2)
    return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES["photo_frequency"])


def plot_block_eye_states(df: pd.DataFrame, block_type: str, fig_dir: Path) -> Path | None:
    if block_type not in {"HV", "PostHV"}:
        raise ValueError("block_type must be 'HV' or 'PostHV'")
    family = "hv" if block_type == "HV" else "post_hv"
    block_df = df.loc[df["block_family"].fillna("").astype(str).eq(family)].copy()
    if block_df.empty:
        return None
    block_df["t_start"] = pd.to_numeric(block_df["t_start"], errors="coerce")
    block_df["t_stop"] = pd.to_numeric(block_df["t_stop"], errors="coerce")
    intervals = (
        block_df[["t_start", "t_stop"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["t_start", "t_stop"])
        .reset_index(drop=True)
    )
    if intervals.empty:
        return None
    block_ids = {
        (float(row.t_start), float(row.t_stop)): idx
        for idx, row in enumerate(intervals.itertuples(index=False), start=1)
    }
    block_df["block_id"] = [
        block_ids.get((float(start), float(stop)))
        for start, stop in zip(block_df["t_start"], block_df["t_stop"])
    ]
    block_df = block_df.dropna(subset=["block_id"]).copy()
    block_df["block_id"] = block_df["block_id"].astype(int)
    block_df["eye_state"] = block_df["eye_state"].fillna("unknown").astype(str).str.upper()
    grouped = block_df.groupby(["block_id", "eye_state"])["duration"].sum().unstack(fill_value=0.0)
    if grouped.empty:
        return None
    labels = [f"{block_type} #{idx}" for idx in grouped.index]
    positions = np.arange(len(labels))
    eo_vals = grouped.get("EO", pd.Series(0.0, index=grouped.index))
    ec_vals = grouped.get("EC", pd.Series(0.0, index=grouped.index))
    unknown_vals = grouped.drop(
        columns=[c for c in grouped.columns if c in {"EO", "EC"}], errors="ignore"
    ).sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, max(3, len(grouped) * 0.4)))
    ax.barh(positions, eo_vals, label="Eyes Open", color="#55A868")
    ax.barh(positions, ec_vals, left=eo_vals, label="Eyes Closed", color="#C44E52")
    if not (unknown_vals == 0).all():
        ax.barh(
            positions,
            unknown_vals,
            left=eo_vals + ec_vals,
            label="Unknown",
            color="#8172B2",
            alpha=0.7,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title(f"{block_type} Eyes-Open vs Eyes-Closed")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    key = "hv_blocks" if block_type == "HV" else "post_hv_blocks"
    return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES[key])


def plot_segment_timeline(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    if df.empty:
        return None
    ordered = df.sort_values("t_start")
    segment_types = list(dict.fromkeys(ordered["segment_type"]))
    if not segment_types:
        return None
    cmap = plt.get_cmap("tab20")
    type_to_y = {seg: idx for idx, seg in enumerate(segment_types)}
    run_ids = (
        ordered["run_id"].dropna().astype(str).drop_duplicates().tolist()
        if "run_id" in ordered.columns
        else []
    )
    if len(run_ids) <= 1:
        fig, ax = plt.subplots(figsize=(10, max(3, len(segment_types) * 0.6)))
        for _, row in ordered.iterrows():
            seg_type = row["segment_type"]
            y = type_to_y.get(seg_type)
            if y is None:
                continue
            start = float(row["t_start"])
            stop = float(row["t_stop"])
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            ax.plot(
                [start, stop], [y, y], linewidth=8, solid_capstyle="butt", color=cmap(y % cmap.N)
            )
        ax.set_yticks(list(type_to_y.values()))
        ax.set_yticklabels(segment_types)
        ax.set_xlabel("Time (s)")
        ax.set_title("Condition Timeline")
        ax.grid(True, axis="x", alpha=0.2)
        plt.tight_layout()
        return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES["timeline"])

    fig, axes = plt.subplots(
        len(run_ids),
        1,
        figsize=(10, max(3, len(segment_types) * 0.6) * len(run_ids)),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, run_id in zip(axes, run_ids):
        run_df = ordered.loc[ordered["run_id"].astype(str) == run_id]
        for _, row in run_df.iterrows():
            seg_type = row["segment_type"]
            y = type_to_y.get(seg_type)
            if y is None:
                continue
            start = float(row["t_start"])
            stop = float(row["t_stop"])
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            ax.plot(
                [start, stop], [y, y], linewidth=8, solid_capstyle="butt", color=cmap(y % cmap.N)
            )
        ax.set_title(f"Condition Timeline - Run {run_id}")
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_xlabel("Time within run (s)")
    axes[0].set_yticks(list(type_to_y.values()))
    axes[0].set_yticklabels(segment_types)
    plt.tight_layout()
    return utils.save_fig(fig, fig_dir / FIGURE_FILENAMES["timeline"])


def save_eeg_report_figures(df: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}
    for key, func in (
        ("segment_duration", plot_total_duration_by_segment),
        ("eye_state_breakdown", plot_eye_state_breakdown),
        ("photo_frequency", plot_photo_frequency_durations),
    ):
        path = func(df, fig_dir)
        if path:
            figure_paths[key] = path
    for block_type in ("HV", "PostHV"):
        path = plot_block_eye_states(df, block_type, fig_dir)
        if path:
            figure_paths["hv_blocks" if block_type == "HV" else "post_hv_blocks"] = path
    path = plot_segment_timeline(df, fig_dir)
    if path:
        figure_paths["timeline"] = path
    return figure_paths


def _plot_bar(
    values: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path
) -> Path | None:
    if values.empty:
        return None
    fig, ax = plot_bar(
        values,
        sort=False,
        orientation="horizontal",
        color="#4C72B0",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=(8, max(4, len(values) * 0.45)),
    )
    ax.grid(True, axis="x", alpha=0.2)
    return utils.save_fig(fig, out_path)


def plot_runs_per_subject(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if runs_df.empty or "subject_id" not in runs_df:
        return None
    counts = runs_df.groupby("subject_id")["run_id"].nunique()
    counts = counts.loc[counts.ge(report_eeg.MULTI_RUN_SUBJECT_THRESHOLD)].sort_values(
        ascending=True
    )
    if counts.empty:
        return None
    return _plot_bar(
        counts,
        f"Subjects With {report_eeg.MULTI_RUN_SUBJECT_THRESHOLD} Or More Runs",
        "Runs",
        "Subject",
        output_dir / FIGURE_FILENAMES["runs_per_subject"],
    )


def plot_recording_start_hour_distribution(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    starts = pd.to_datetime(runs_df.get("meas_datetime"), errors="coerce").dropna()
    if starts.empty:
        return None
    hours = starts.dt.hour.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plot_bar(
        hours,
        sort=False,
        color="#55A868",
        title="Recording Start Hour Distribution",
        xlabel="Hour of Day",
        ylabel="Runs",
        figsize=(10, 4),
    )
    ax.grid(True, axis="y", alpha=0.2)
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES["recording_start_hour_distribution"])


def plot_run_duration_distribution(runs_df: pd.DataFrame, output_dir: Path) -> Path | None:
    durations = pd.to_numeric(runs_df.get("raw_duration"), errors="coerce").dropna() / 60.0
    if durations.empty:
        return None
    capped = durations.clip(upper=50)
    bins = list(range(0, 55, 5))
    counts, _ = np.histogram(capped, bins=bins)
    labels = [f"{start}-{start + 5}" for start in bins[:-2]] + [">50"]
    counts[-1] = int((durations > 50).sum())
    series = pd.Series(counts, index=labels)
    fig, ax = plot_bar(
        series,
        sort=False,
        color="#4C72B0",
        title="Run Duration Distribution",
        xlabel="Duration (minutes)",
        ylabel="Runs",
        figsize=(9, 4.5),
    )
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES["run_duration_distribution"])


def _plot_condition_availability_by_group(
    runs_df: pd.DataFrame, group_col: str, output_dir: Path, out_key: str
) -> Path | None:
    if runs_df.empty or group_col not in runs_df.columns:
        return None
    grouped = runs_df.copy()
    grouped[group_col] = grouped[group_col].fillna("Unknown").replace("", "Unknown")
    summary = pd.DataFrame(
        {
            "Eyes Open": grouped.groupby(group_col)["total_eyes_open_duration"].apply(
                lambda s: s.gt(0).mean() * 100.0
            ),
            "Eyes Closed": grouped.groupby(group_col)["total_eyes_closed_duration"].apply(
                lambda s: s.gt(0).mean() * 100.0
            ),
            "HV": grouped.groupby(group_col)["hv_block_count"].apply(
                lambda s: s.gt(0).mean() * 100.0
            ),
            "PHOTO": grouped.groupby(group_col)["photo_block_count"].apply(
                lambda s: s.gt(0).mean() * 100.0
            ),
        }
    )
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.5)))
    summary.plot(kind="barh", ax=ax)
    ax.set_title(f"Condition Availability by {group_col.replace('_', ' ').title()}")
    ax.set_xlabel("Runs with Condition (%)")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES[out_key])


def _plot_duration_by_group(
    runs_df: pd.DataFrame, group_col: str, output_dir: Path, out_key: str
) -> Path | None:
    if runs_df.empty or group_col not in runs_df.columns:
        return None
    grouped = runs_df.copy()
    grouped[group_col] = grouped[group_col].fillna("Unknown").replace("", "Unknown")
    summary = grouped.groupby(group_col).agg(
        raw_duration_min=(
            "raw_duration",
            lambda s: pd.to_numeric(s, errors="coerce").mean() / 60.0,
        ),
        analysis_duration_min=(
            "total_duration",
            lambda s: pd.to_numeric(s, errors="coerce").mean() / 60.0,
        ),
    )
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.5)))
    summary.plot(kind="barh", ax=ax)
    ax.set_title(f"Mean Duration by {group_col.replace('_', ' ').title()}")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.legend(["Mean Raw Duration", "Mean Analysis Duration"], loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    return utils.save_fig(fig, output_dir / FIGURE_FILENAMES[out_key])


def _plot_event_distribution_matrix(
    event_counts_map: Mapping[str, np.ndarray], title: str, xlabel: str, out_path: Path
) -> Path | None:
    if not event_counts_map:
        return None
    labels = list(event_counts_map.keys())
    values = [np.asarray(event_counts_map[label], dtype=float) for label in labels]
    values = [arr[np.isfinite(arr)] for arr in values]
    labels = [label for label, arr in zip(labels, values) if arr.size > 0]
    values = [arr for arr in values if arr.size > 0]
    if not labels:
        return None
    cols = 2 if len(labels) > 1 else 1
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax_idx, (label, data) in enumerate(zip(labels, values)):
        plot_histogram(
            data,
            bins=min(20, max(3, int(np.sqrt(len(data))))),
            color="#4C72B0",
            title=label,
            xlabel=xlabel,
            ylabel="Runs",
            ax=axes[ax_idx],
        )
        axes[ax_idx].grid(True, axis="y", alpha=0.2)
    for extra_ax in axes[len(labels) :]:
        extra_ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return utils.save_fig(fig, out_path)


def save_dataset_event_distributions(
    event_counts_list: list[dict[str, int]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not event_counts_list:
        return {}
    all_keys = sorted(set().union(*(counts.keys() for counts in event_counts_list)))
    agg_data = {key: np.zeros(len(event_counts_list), dtype=float) for key in all_keys}
    for idx, counts in enumerate(event_counts_list):
        for key, value in counts.items():
            agg_data[key][idx] = float(value)

    condition_map = {key: agg_data[key] for key in CONDITION_EVENT_LABELS if key in agg_data}
    clinical_map = {
        key: values for key, values in agg_data.items() if key not in CONDITION_EVENT_LABELS
    }
    figure_paths: dict[str, Path] = {}
    cond_path = _plot_event_distribution_matrix(
        condition_map,
        "Condition Event Counts (EO/EC/HV/Photo)",
        "Count per Run",
        output_dir / FIGURE_FILENAMES["dataset_event_distributions_conditions"],
    )
    if cond_path:
        figure_paths["dataset_event_distributions_conditions"] = cond_path
    clin_path = _plot_event_distribution_matrix(
        clinical_map,
        "Clinical & Artifact Event Counts",
        "Count per Run",
        output_dir / FIGURE_FILENAMES["dataset_event_distributions_clinical"],
    )
    if clin_path:
        figure_paths["dataset_event_distributions_clinical"] = clin_path
    return figure_paths


def save_dataset_eeg_figures(
    runs_df: pd.DataFrame, event_counts_list: list[dict[str, int]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}
    for key, path in (
        ("runs_per_subject", plot_runs_per_subject(runs_df, output_dir)),
        (
            "recording_start_hour_distribution",
            plot_recording_start_hour_distribution(runs_df, output_dir),
        ),
        ("run_duration_distribution", plot_run_duration_distribution(runs_df, output_dir)),
        (
            "availability_by_source_dataset",
            _plot_condition_availability_by_group(
                runs_df, "source_dataset", output_dir, "availability_by_source_dataset"
            ),
        ),
        (
            "availability_by_combined_diagnosis",
            _plot_condition_availability_by_group(
                runs_df, "combined_diagnosis", output_dir, "availability_by_combined_diagnosis"
            ),
        ),
        (
            "duration_by_source_dataset",
            _plot_duration_by_group(
                runs_df, "source_dataset", output_dir, "duration_by_source_dataset"
            ),
        ),
        (
            "duration_by_combined_diagnosis",
            _plot_duration_by_group(
                runs_df, "combined_diagnosis", output_dir, "duration_by_combined_diagnosis"
            ),
        ),
    ):
        if path:
            figure_paths[key] = path
    figure_paths.update(save_dataset_event_distributions(event_counts_list, output_dir))
    return figure_paths

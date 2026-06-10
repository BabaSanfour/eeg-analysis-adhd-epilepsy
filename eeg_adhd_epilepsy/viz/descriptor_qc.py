"""Descriptor QC visualization helpers.

Thin wrappers around the generic :mod:`coco_pipe.viz` plotting primitives
(:func:`coco_pipe.viz.plot_bar`, :func:`coco_pipe.viz.plot_histogram`) that
turn descriptor-QC dataframes into the figure set used by the subject- and
dataset-level QC reports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from coco_pipe.viz import plot_bar, plot_histogram, save_figure


def _bar_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output_path: Path,
    color: str = "#4c78a8",
    top_n: int | None = None,
) -> Path | None:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return None
    series = pd.to_numeric(df[y], errors="coerce")
    series.index = df[x].astype(str)
    series = series.dropna()
    if series.empty:
        return None
    fig, _ax = plot_bar(
        series,
        top_n=top_n,
        color=color,
        title=title,
        ylabel=y.replace("_", " ").title(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(output_path))
    plt.close(fig)
    return output_path


def _hist_plot(
    series: pd.Series,
    *,
    title: str,
    output_path: Path,
    color: str = "#f58518",
) -> Path | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    try:
        fig, _ax = plot_histogram(clean, color=color, title=title)
    except ValueError:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(output_path))
    plt.close(fig)
    return output_path


def save_subject_descriptor_qc_figures(
    *,
    figures_dir: Path,
    family_summary_df: pd.DataFrame,
    failure_summary_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    epoch_feature_df: pd.DataFrame,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    path = _bar_plot(
        family_summary_df,
        x="family",
        y="missing_rate",
        title="Family Missingness",
        output_path=figures_dir / "family_missingness.png",
        color="#54a24b",
    )
    if path:
        figure_paths["family_missingness"] = path

    family_failures = failure_summary_df[failure_summary_df["group"] == "family"] if "group" in failure_summary_df.columns else pd.DataFrame()
    path = _bar_plot(
        family_failures.rename(columns={"value": "family", "count": "count"}),
        x="family",
        y="count",
        title="Failure Counts by Family",
        output_path=figures_dir / "failure_counts_by_family.png",
        color="#e45756",
    )
    if path:
        figure_paths["failure_counts_by_family"] = path

    path = _bar_plot(
        feature_missingness_df.rename(columns={"column": "feature"}),
        x="feature",
        y="missing_rate",
        title="Top Missing Features",
        output_path=figures_dir / "top_missing_features.png",
        top_n=20,
        color="#72b7b2",
    )
    if path:
        figure_paths["top_missing_features"] = path

    param_r2_cols = [column for column in epoch_feature_df.columns if "param_r_squared_" in column]
    if param_r2_cols:
        path = _hist_plot(
            epoch_feature_df[param_r2_cols].stack(),
            title="Parametric R-Squared",
            output_path=figures_dir / "param_r_squared_hist.png",
        )
        if path:
            figure_paths["param_r_squared_hist"] = path

    param_error_cols = [column for column in epoch_feature_df.columns if "param_fit_error_" in column]
    if param_error_cols:
        path = _hist_plot(
            epoch_feature_df[param_error_cols].stack(),
            title="Parametric Fit Error",
            output_path=figures_dir / "param_fit_error_hist.png",
        )
        if path:
            figure_paths["param_fit_error_hist"] = path

    return figure_paths


def save_dataset_descriptor_qc_figures(
    *,
    figures_dir: Path,
    shard_summary_df: pd.DataFrame,
    failure_family_df: pd.DataFrame,
    failure_channel_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    low_variance_df: pd.DataFrame,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    status_counts = (
        shard_summary_df["qc_status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
        if "qc_status" in shard_summary_df.columns
        else pd.DataFrame()
    )
    path = _bar_plot(
        status_counts,
        x="status",
        y="count",
        title="Shard QC Status Counts",
        output_path=figures_dir / "shard_status_counts.png",
        color="#4c78a8",
    )
    if path:
        figure_paths["shard_status_counts"] = path

    path = _bar_plot(
        failure_family_df.rename(columns={"value": "family"}),
        x="family",
        y="count",
        title="Failure Counts by Family",
        output_path=figures_dir / "failure_counts_by_family.png",
        color="#e45756",
    )
    if path:
        figure_paths["failure_counts_by_family"] = path

    path = _bar_plot(
        failure_channel_df.rename(columns={"value": "channel"}),
        x="channel",
        y="count",
        title="Failure Counts by Channel",
        output_path=figures_dir / "failure_counts_by_channel.png",
        top_n=20,
        color="#f58518",
    )
    if path:
        figure_paths["failure_counts_by_channel"] = path

    path = _bar_plot(
        feature_missingness_df.rename(columns={"column": "feature"}),
        x="feature",
        y="missing_rate",
        title="Top Missing Features",
        output_path=figures_dir / "top_missing_features.png",
        top_n=20,
        color="#72b7b2",
    )
    if path:
        figure_paths["top_missing_features"] = path

    low_variance_counts = (
        low_variance_df["family"].astype(str).value_counts().rename_axis("family").reset_index(name="count")
        if "family" in low_variance_df.columns and not low_variance_df.empty
        else pd.DataFrame()
    )
    path = _bar_plot(
        low_variance_counts,
        x="family",
        y="count",
        title="Low-Variance Features by Family",
        output_path=figures_dir / "low_variance_by_family.png",
        color="#54a24b",
    )
    if path:
        figure_paths["low_variance_by_family"] = path

    return figure_paths

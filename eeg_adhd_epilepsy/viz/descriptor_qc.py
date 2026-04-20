"""Descriptor QC visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from eeg_adhd_epilepsy.viz.utils import save_fig


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
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.nlargest(top_n, y)
    plot_df = plot_df.reset_index(drop=True)
    labels = plot_df[x].astype(str).tolist()
    positions = list(range(len(plot_df)))
    fig_width = 12 if len(plot_df) > 10 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(positions, plot_df[y].astype(float), color=color)
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return save_fig(fig, output_path)


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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=min(30, max(5, clean.shape[0])), color=color, edgecolor="white")
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, output_path)


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

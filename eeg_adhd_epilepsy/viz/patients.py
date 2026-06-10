"""Visualizations for clean cohort metadata reports."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from coco_pipe.viz import plot_bar, plot_distribution_groups, plot_heatmap
from eeg_adhd_epilepsy.viz import utils
import seaborn as sns

from eeg_adhd_epilepsy.utils.metadata_schema import (
    EPILEPSY_MED_COLS,
    NORMALIZED_PSYCHOSTIMULANT_CATEGORIES,
    SOURCE_DATASETS,
)

logger = logging.getLogger(__name__)


def _sorted_age_groups(series: pd.Series) -> list[str]:
    values = [value for value in series.dropna().astype(str).unique().tolist() if value]
    return sorted(values, key=lambda value: int(value.split("-")[0]))


def plot_eeg_date_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Plot EEG recordings over time, stratified by source dataset."""
    timeline = df.copy()
    timeline["eeg_date_dt"] = pd.to_datetime(timeline["eeg_date"], errors="coerce")
    timeline = timeline.dropna(subset=["eeg_date_dt"])
    if timeline.empty:
        return

    timeline["month"] = timeline["eeg_date_dt"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        timeline.groupby(["month", "source_dataset"], observed=True)
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=grouped,
        x="month",
        y="count",
        hue="source_dataset",
        hue_order=list(SOURCE_DATASETS),
        marker="o",
        ax=ax,
    )
    ax.set_title("EEG Recording Dates")
    ax.set_xlabel("Recording Month")
    ax.set_ylabel("Recordings")
    ax.legend(title="Source Dataset")
    sns.despine()
    utils.save_fig(fig, out_path, dpi=300)


def plot_source_dataset_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        df["source_dataset"]
        .value_counts()
        .reindex(SOURCE_DATASETS, fill_value=0)
        .rename_axis("source_dataset")
        .reset_index(name="count")
    )

    fig, ax = plot_bar(
        counts.set_index("source_dataset")["count"],
        sort=False,
        cmap="crest",
        title="Source Dataset Counts",
        ylabel="Rows",
        figsize=(7, 4),
    )
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_diagnosis_prevalence(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for column, label in [
        ("adhd", "ADHD"),
        ("autism", "Autism"),
        ("epilepsy", "Epilepsy"),
    ]:
        positive = int(df[column].fillna(0).sum())
        rows.append(
            {
                "diagnosis": label,
                "count": positive,
                "prevalence_pct": positive / len(df) * 100 if len(df) else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return

    fig, ax = plot_bar(
        summary.set_index("diagnosis")["prevalence_pct"],
        sort=True,
        ascending=False,
        cmap="viridis",
        title="Diagnosis Prevalence",
        ylabel="Prevalence (%)",
        figsize=(8, 4.5),
    )
    ax.set_ylim(0, 100)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_combined_diagnosis_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        df["combined_diagnosis"]
        .value_counts()
        .rename_axis("combined_diagnosis")
        .reset_index(name="count")
    )
    if counts.empty:
        return

    fig, ax = plot_bar(
        counts.set_index("combined_diagnosis")["count"],
        sort=True,
        ascending=False,
        cmap="rocket",
        title="Combined Diagnosis Counts",
        ylabel="Rows",
        figsize=(11, 5),
    )
    ax.tick_params(axis="x", rotation=35)
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_sex_age_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    age_order = _sorted_age_groups(df["age_group"])
    matrix = (
        df.assign(age_group=df["age_group"].astype(str))
        .pivot_table(
            index="sex",
            columns="age_group",
            values="study_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=["F", "M"], columns=age_order, fill_value=0)
    )
    if matrix.empty:
        return

    fig, ax = plot_heatmap(
        matrix.astype(int),
        cmap="Blues",
        annotate=True,
        annotation_format=".0f",
        colorbar=False,
        title="Sex by Age Group",
        xlabel="Age Group",
        ylabel="Sex",
        xtick_rotation=0,
        figsize=(8, 4.5),
    )
    utils.save_fig(fig, out_path, dpi=300)


def plot_age_by_diagnosis(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["age", "combined_diagnosis"]).copy()
    if plot_df.empty:
        return
    order = plot_df["combined_diagnosis"].value_counts().index.tolist()
    groups = [plot_df.loc[plot_df["combined_diagnosis"] == label, "age"] for label in order]

    fig, ax = plot_distribution_groups(
        groups,
        order,
        kind="box",
        title="Age by Combined Diagnosis",
        ylabel="Age (Years)",
        xtick_rotation=35,
        figsize=(11, 5.5),
    )
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_psychostimulant_category_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        df["psychostimulant_category"]
        .value_counts()
        .reindex(NORMALIZED_PSYCHOSTIMULANT_CATEGORIES, fill_value=0)
        .rename_axis("psychostimulant_category")
        .reset_index(name="count")
    )
    if counts["count"].sum() == 0:
        return

    fig, ax = plot_bar(
        counts.set_index("psychostimulant_category")["count"],
        sort=False,
        cmap="mako",
        title="Psychostimulant Categories",
        ylabel="Rows",
        figsize=(10, 5),
    )
    ax.tick_params(axis="x", rotation=35)
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_asm_exposure_counts(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for column in EPILEPSY_MED_COLS:
        count = int(df[column].fillna(0).sum())
        if count > 0:
            rows.append({"asm_name": column, "count": count})
    other_count = int(df["other_asm"].fillna(0).sum())
    if other_count > 0:
        rows.append({"asm_name": "Other ASM", "count": other_count})

    counts = pd.DataFrame(rows).sort_values("count", ascending=False)
    if counts.empty:
        return

    fig, ax = plot_bar(
        counts.set_index("asm_name")["count"],
        sort=True,
        ascending=False,
        cmap="flare",
        title="ASM Exposure Counts",
        ylabel="Rows",
        figsize=(10, 5),
    )
    ax.tick_params(axis="x", rotation=35)
    sns.despine(ax=ax)
    utils.save_fig(fig, out_path, dpi=300)


def plot_medication_overlap(df: pd.DataFrame, out_path: Path) -> None:
    matrix = pd.crosstab(df["asm"], df["psychostimulant"])
    matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    fig, ax = plot_heatmap(
        matrix.astype(int),
        x_labels=["0", "1"],
        y_labels=["0", "1"],
        cmap="PuBu",
        annotate=True,
        annotation_format=".0f",
        colorbar=False,
        title="Psychostimulant vs ASM Exposure",
        xlabel="Psychostimulant",
        ylabel="ASM",
        xtick_rotation=0,
        figsize=(5.5, 4.5),
    )
    utils.save_fig(fig, out_path, dpi=300)


def plot_drug_resistance_summary(df: pd.DataFrame, out_path: Path) -> None:
    overall = (
        df["asm_resistant"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename_axis("asm_resistant")
        .reset_index(name="count")
    )
    by_source = (
        df.groupby(["source_dataset", "asm_resistant"], observed=True)
        .size()
        .reset_index(name="count")
        .pivot(index="source_dataset", columns="asm_resistant", values="count")
        .reindex(index=SOURCE_DATASETS, columns=[0, 1], fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(data=overall, x="asm_resistant", y="count", ax=axes[0], palette="coolwarm")
    axes[0].set_title("Drug Resistance Overall")
    axes[0].set_xlabel("ASM Resistant")
    axes[0].set_ylabel("Rows")
    axes[0].set_xticklabels(["0", "1"])

    by_source.plot(kind="bar", stacked=True, ax=axes[1], color=["#8ecae6", "#d00000"])
    axes[1].set_title("Drug Resistance by Source Dataset")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Rows")
    axes[1].legend(title="ASM Resistant")
    sns.despine()
    utils.save_fig(fig, out_path, dpi=300)

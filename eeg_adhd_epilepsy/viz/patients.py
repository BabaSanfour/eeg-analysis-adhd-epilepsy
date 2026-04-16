"""Visualizations for clean cohort metadata reports."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=counts, x="source_dataset", y="count", ax=ax, palette="crest")
    ax.set_title("Source Dataset Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
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
    sns.despine()
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=summary.sort_values("prevalence_pct", ascending=False),
        x="diagnosis",
        y="prevalence_pct",
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Diagnosis Prevalence")
    ax.set_xlabel("")
    ax.set_ylabel("Prevalence (%)")
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
    sns.despine()
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

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=counts,
        x="combined_diagnosis",
        y="count",
        ax=ax,
        palette="rocket",
    )
    ax.set_title("Combined Diagnosis Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=35)
    sns.despine()
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Sex by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Sex")
    utils.save_fig(fig, out_path, dpi=300)


def plot_age_by_diagnosis(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["age", "combined_diagnosis"]).copy()
    if plot_df.empty:
        return
    order = plot_df["combined_diagnosis"].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=plot_df,
        x="combined_diagnosis",
        y="age",
        order=order,
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Age by Combined Diagnosis")
    ax.set_xlabel("")
    ax.set_ylabel("Age (Years)")
    ax.tick_params(axis="x", rotation=35)
    sns.despine()
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

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=counts,
        x="psychostimulant_category",
        y="count",
        ax=ax,
        palette="mako",
    )
    ax.set_title("Psychostimulant Categories")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=35)
    sns.despine()
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

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x="asm_name", y="count", ax=ax, palette="flare")
    ax.set_title("ASM Exposure Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=35)
    sns.despine()
    utils.save_fig(fig, out_path, dpi=300)


def plot_medication_overlap(df: pd.DataFrame, out_path: Path) -> None:
    matrix = pd.crosstab(df["asm"], df["psychostimulant"])
    matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="PuBu", cbar=False, ax=ax)
    ax.set_title("Psychostimulant vs ASM Exposure")
    ax.set_xlabel("Psychostimulant")
    ax.set_ylabel("ASM")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)
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

"""Shared helpers for eeg_adhd_epilepsy report generation.

Consolidates small building blocks (image/table embedding, value formatting,
metric-table construction) that were previously duplicated, with slight
drift, across `raw_qc.py`, `eeg_report.py`, `preproc_qc.py`, and
`cohort_report.py`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.report.core import ImageElement, Section, TableElement

from eeg_adhd_epilepsy.utils.formatting import format_duration_hms


def clean_scalar(value: object) -> object:
    """Return ``None`` for NaN/NaT scalars, otherwise the value unchanged."""
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def format_value(value: object, digits: int = 2, suffix: str = "") -> str:
    """Format a numeric value with fixed precision, or "" if not finite."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}{suffix}"


def add_optional_table(section: Section, data: pd.DataFrame | None, title: str) -> None:
    """Add `data` as a TableElement to `section` if it is a non-empty DataFrame."""
    if data is not None and not data.empty:
        section.add_element(TableElement(data, title=title))


def add_images(
    section: Section,
    figures: Mapping[str, Path],
    ordered_keys: Sequence[str],
    *,
    caption_from_key: bool = True,
) -> None:
    """Add images from a `{key: path}` mapping, in `ordered_keys` order.

    Skips missing/non-existent paths and de-duplicates repeated paths (the
    same figure referenced under two keys is only embedded once).
    """
    seen_paths: set[Path] = set()
    for key in ordered_keys:
        path = figures.get(key)
        if path and path.exists() and path not in seen_paths:
            seen_paths.add(path)
            caption = key.replace("_", " ").title() if caption_from_key else None
            section.add_element(ImageElement(str(path), caption=caption))


def add_image_list(section: Section, figures: Sequence[tuple[str, Path]]) -> None:
    """Add images from a sequence of `(caption, path)` pairs."""
    for title, path in figures:
        if path.exists():
            section.add_element(ImageElement(str(path), caption=title))


def build_subject_overview_table(record: Mapping[str, object]) -> pd.DataFrame:
    """Common subject/session/source-dataset overview row used by raw QC and EEG reports."""
    return pd.DataFrame(
        [
            {
                "Subject": record.get("subject_id", ""),
                "Session": record.get("session_id", ""),
                "Runs": int(record.get("n_runs", 0) or 0),
                "Source Dataset": record.get("source_dataset", ""),
                "Total Duration": format_duration_hms(record.get("raw_duration", 0.0)),
                "Age Group": record.get("age_group", ""),
                "Sex": record.get("sex", ""),
                "Combined Diagnosis": record.get("combined_diagnosis", ""),
            }
        ]
    )


def build_record_metric_table(
    record: Mapping[str, object],
    specs: Sequence[tuple[str, str, str]],
    *,
    value_col: str = "Value",
    skip_empty: bool = False,
) -> pd.DataFrame:
    """Build a `Metric` / `value_col` table from a single record.

    `specs` is a sequence of `(label, key, suffix)` tuples; `record[key]` is
    formatted via `format_value(..., suffix=suffix)`. If `skip_empty` is
    True, rows whose formatted value is empty are omitted.
    """
    rows = []
    for label, key, suffix in specs:
        value = format_value(record.get(key), suffix=suffix)
        if skip_empty and not value:
            continue
        rows.append({"Metric": label, value_col: value})
    return pd.DataFrame(rows)


def build_dataset_mean_metric_table(
    runs_df: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    *,
    value_col: str = "Value",
) -> pd.DataFrame:
    """Build a `Metric` / `value_col` table of column means from `runs_df`.

    `specs` is a sequence of `(label, column, suffix)` tuples; the mean of
    `runs_df[column]` (coerced to numeric) is formatted via
    `format_value(..., suffix=suffix)`.
    """
    rows = []
    for label, column, suffix in specs:
        series = pd.to_numeric(runs_df.get(column), errors="coerce")
        rows.append({"Metric": label, value_col: format_value(series.mean(), suffix=suffix)})
    return pd.DataFrame(rows)


def build_flag_reason_table(
    runs_df: pd.DataFrame,
    *,
    reasons_column: str,
    count_label: str = "Runs",
) -> pd.DataFrame:
    """Count `;`-separated flag reasons in `runs_df[reasons_column]`.

    Returns a `Reason` / `count_label` table sorted by count descending.
    """
    counts: dict[str, int] = {}
    for reasons in runs_df.get(reasons_column, pd.Series(dtype=str)).fillna(""):
        for reason in str(reasons).split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    rows = [
        {"Reason": reason, count_label: count}
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return pd.DataFrame(rows)


def overview_containers(
    args: Any,
    containers_by_scope: dict[tuple[str, str], Any] | None,
    pooled_condition: str,
) -> dict[str, Any]:
    if not containers_by_scope:
        return {}

    containers = {}
    if args.run_pooled:
        pooled = containers_by_scope.get(("pooled", pooled_condition))
        if pooled is not None:
            containers[f"Pooled ({pooled_condition})"] = pooled
    for condition in args.conditions:
        cond = containers_by_scope.get(("condition", condition))
        if cond is not None:
            containers[f"Condition: {condition}"] = cond
    return containers


def add_overview_cohort_summary(
    overview_sec: Section,
    args: Any,
    eval_specs: Sequence[dict[str, Any]],
    containers_by_scope: dict[tuple[str, str], Any] | None,
    pooled_condition: str,
) -> None:
    primary_spec = eval_specs[0] if eval_specs else None
    if primary_spec is None:
        raise ValueError(
            "No evaluation specifications (eval_specs) provided. Cannot generate cohort summary."
        )
    containers = overview_containers(args, containers_by_scope, pooled_condition)
    if not containers:
        raise ValueError(
            "No valid datasets (containers) were found. Cannot generate cohort summary."
        )

    for condition_name, container in containers.items():
        frame = container.observation_frame()
        if args.subject_col in frame.columns:
            frame = frame.drop_duplicates(subset=[args.subject_col], keep="first").reset_index(
                drop=True
            )

        if primary_spec["target_col"] not in frame.columns:
            raise ValueError(
                f"Dataset for condition '{condition_name}' is missing the primary target column "
                f"'{primary_spec['target_col']}' in its metadata. Cannot generate cohort summary."
            )

        labels = frame[primary_spec["target_col"]].astype(str)
        label_map = primary_spec.get("label_map") or {}
        if label_map:
            labels = labels.map(lambda value: label_map.get(value, value))
        frame = frame.assign(_primary_class=labels.astype(str))
        if frame["_primary_class"].nunique(dropna=True) <= 1:
            raise ValueError(
                f"Dataset for condition '{condition_name}' contains 1 or fewer unique classes "
                f"for target '{primary_spec['target_col']}'. "
                "Cannot generate a comparative cohort summary."
            )

        n_subj = (
            frame[args.subject_col].nunique() if args.subject_col in frame.columns else len(frame)
        )
        overview_sec.add_markdown(
            f"### {condition_name}\n"
            f"Primary cohort summary for **{primary_spec['name']}** "
            f"using **{n_subj}** unique subjects."
        )

        summary_rows = []
        for class_value, class_df in frame.groupby("_primary_class", dropna=False):
            row = {
                "class": class_value,
                "n_subjects": int(len(class_df)),
                "pct_subjects": round(100.0 * len(class_df) / len(frame), 1),
            }
            if "age" in class_df.columns:
                age = pd.to_numeric(class_df["age"], errors="coerce")
                if age.notna().any():
                    row["mean_age"] = round(float(age.mean()), 2)
                    row["sd_age"] = round(float(age.std(ddof=0)), 2)
            summary_rows.append(row)
        overview_sec.add_element(
            TableElement(
                pd.DataFrame(summary_rows), title=f"Primary class counts ({condition_name})"
            )
        )

        if "sex" in frame.columns:
            sex_table = (
                frame.assign(sex=frame["sex"].astype(str))
                .groupby(["_primary_class", "sex"], dropna=False)
                .size()
                .unstack(fill_value=0)
                .reset_index()
                .rename(columns={"_primary_class": "class"})
            )
            overview_sec.add_element(
                TableElement(sex_table, title=f"Sex by class ({condition_name})")
            )

        if "age_group" in frame.columns:
            age_group_table = (
                frame.assign(age_group=frame["age_group"].astype(str))
                .groupby(["_primary_class", "age_group"], dropna=False)
                .size()
                .unstack(fill_value=0)
                .reset_index()
                .rename(columns={"_primary_class": "class"})
            )
            overview_sec.add_element(
                TableElement(age_group_table, title=f"Age group by class ({condition_name})")
            )

        clinical_columns = [
            column
            for column in ["autism", "epilepsy", "asm", "asm_resistant", "psychostimulant"]
            if column in frame.columns
        ]
        if clinical_columns:
            clinical_rows = []
            for class_value, class_df in frame.groupby("_primary_class", dropna=False):
                row = {"class": class_value}
                for column in clinical_columns:
                    values = class_df[column].astype(str).str.strip().str.lower()
                    present = values.isin({"1", "true", "yes", "y", "present"})
                    total = int(present.notna().sum())
                    if total == 0:
                        row[column] = ""
                    else:
                        count = int(present.sum())
                        row[column] = f"{count} ({(100.0 * count / total):.1f}%)"
                clinical_rows.append(row)
            overview_sec.add_element(
                TableElement(
                    pd.DataFrame(clinical_rows),
                    title=f"Clinical composition by class ({condition_name})",
                )
            )

        if "psychostimulant_category" in frame.columns:
            medication_rows = []
            for class_value, class_df in frame.groupby("_primary_class", dropna=False):
                counts = (
                    class_df["psychostimulant_category"].fillna("None").astype(str).value_counts()
                )
                medication_rows.append(
                    {
                        "class": class_value,
                        "psychostimulant_category_counts": "; ".join(
                            f"{name}={count}" for name, count in counts.items()
                        ),
                    }
                )
            overview_sec.add_element(
                TableElement(
                    pd.DataFrame(medication_rows),
                    title=f"Medication category by class ({condition_name})",
                )
            )


def family_label(args: Any) -> str:
    """Human-readable descriptor-family label, or empty string for raw inputs."""
    if args.input_mode == "descriptors" and args.descriptor_families:
        return ", ".join(args.descriptor_families)
    if args.input_mode == "descriptors":
        return "all descriptor families"
    return ""


def get_feature_names(container) -> list[str] | None:
    """Return the feature-axis names from a container, or None if unavailable."""
    if container is None:
        return None
    try:
        feat = (container.coords or {}).get("feature")
        if feat is not None:
            names = [str(f) for f in np.asarray(feat)]
            return names if names else None
    except Exception:
        pass
    return None

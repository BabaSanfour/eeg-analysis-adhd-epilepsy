"""Shared helpers for eeg_adhd_epilepsy report generation.

Consolidates small building blocks (image/table embedding, value formatting,
metric-table construction) that were previously duplicated, with slight
drift, across `raw_qc.py`, `eeg_report.py`, `preproc_qc.py`, and
`cohort_report.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

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

"""Shared formatting helpers for reports and summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def reorder_columns_front(df: pd.DataFrame, front: Sequence[str]) -> pd.DataFrame:
    """Return *df* with *front* columns first, remaining columns in order.

    Front columns that are absent from *df* are skipped, so callers can pass a
    canonical front-ordering without first checking which columns exist.
    """
    front_present = [column for column in front if column in df.columns]
    rest = [column for column in df.columns if column not in front_present]
    return df[front_present + rest]


def format_duration_hms(seconds: float | None) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "0s"
    if not math.isfinite(value):
        return "0s"
    hours, value = divmod(value, 3600)
    minutes, value = divmod(value, 60)
    sec_component = f"{f'{value:.2f}'.rstrip('0').rstrip('.') or '0'}s"
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {sec_component}"
    if minutes >= 1:
        return f"{int(minutes)}m {sec_component}"
    return sec_component


def format_clock_time(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%H:%M:%S")

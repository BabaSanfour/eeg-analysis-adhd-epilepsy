"""
features.py

Scans a feature CSV (e.g., data/csv/aggregate@raw.csv), focusing on
the subject column and feature columns named like:
  feature-<FEATURENAME>.spaces-<SENSOR>

It reports, for each base FEATURENAME, which subjects have NaN/Inf values
across any of its sensor columns and how many sensors are affected.

Output format per feature:
  Feature <FEATURENAME>: bad values for rows: [<subject1>, <subject2>, ...] across <N> sensors
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_adhd_epilepsy_psychostimulant.utils.config import csv_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _is_feature_col(col: str) -> bool:
    return col.startswith("feature-") and ".spaces-" in col


def _feature_base(col: str) -> str:
    # Strip the "feature-" prefix and the ".spaces-<SENSOR>" suffix
    # Keep everything between them as the base feature name
    assert col.startswith("feature-") and ".spaces-" in col
    base = col[len("feature-") : col.index(".spaces-")]
    return base


def find_bad_values_by_feature(df: pd.DataFrame, subject_col: str = "subject") -> None:
    if subject_col not in df.columns:
        # Try a fallback
        for fallback in ("Subject", "ID", "id", "participant_id"):
            if fallback in df.columns:
                subject_col = fallback
                break
        else:
            raise KeyError("No subject-like column found.")

    # Keep only subject and feature columns of interest
    feature_cols = [c for c in df.columns if _is_feature_col(c)]
    cols = [subject_col] + feature_cols
    data = df.loc[:, cols].copy()

    # Group feature columns by base feature name
    groups: Dict[str, List[str]] = {}
    for c in feature_cols:
        base = _feature_base(c)
        groups.setdefault(base, []).append(c)

    for base, cols in sorted(groups.items()):
        sub = data[cols].apply(pd.to_numeric, errors="coerce")
        arr = sub.to_numpy()
        isinf = pd.DataFrame(np.isinf(arr), index=sub.index, columns=sub.columns)
        zeros = pd.DataFrame(arr == 0, index=sub.index, columns=sub.columns)
        bad = sub.isna() | isinf | zeros
        rows_bad = bad.any(axis=1)
        if not rows_bad.any():
            continue
        subjects_with_bad = data.loc[rows_bad, subject_col].tolist()
        n_sensors_bad = int(bad.any(axis=0).sum())
        print(
            f"Feature {base}: bad values for {len(subjects_with_bad)} rows: {subjects_with_bad} across {n_sensors_bad} sensors"
        )


def rank_subjects_by_bad_values(
    df: pd.DataFrame, subject_col: str = "subject", top: Optional[int] = None
) -> None:
    """Rank subjects by number of base features with any bad values.

    Counts one per base feature (feature-<FEATURE>.spaces-<SENSOR>) if any sensor/row
    for that subject has NaN or ±Inf for that base. Multiple sensors or rows for the
    same base feature count only once per subject.
    """
    if subject_col not in df.columns:
        for fallback in ("Subject", "ID", "id", "participant_id"):
            if fallback in df.columns:
                subject_col = fallback
                break
        else:
            raise KeyError("No subject-like column found.")

    feature_cols = [c for c in df.columns if _is_feature_col(c)]
    if not feature_cols:
        print("No feature columns found matching 'feature-*.spaces-*'.")
        return

    data = df[[subject_col] + feature_cols].copy()

    # Group columns by base feature name
    groups: Dict[str, List[str]] = {}
    for c in feature_cols:
        groups.setdefault(_feature_base(c), []).append(c)

    # Accumulate per-subject counts: one per base feature with any bad value
    counts: Dict[object, int] = {}
    for base, cols in groups.items():
        sub = data[cols].apply(pd.to_numeric, errors="coerce")
        arr = sub.to_numpy()
        bad = np.isnan(arr) | np.isinf(arr) | (arr == 0)
        # Any bad across sensors for each row
        row_has_bad = bad.any(axis=1)
        # Unique subjects that have this base feature bad in any row
        bad_subjects = pd.unique(data.loc[row_has_bad, subject_col])
        for subj in bad_subjects:
            counts[subj] = counts.get(subj, 0) + 1

    # Build Series, drop zeroes, sort desc
    if not counts:
        print("No subjects with bad feature values found.")
        return
    per_subject = pd.Series(counts, name="bad_features").sort_values(ascending=False)
    per_subject = per_subject[per_subject > 0]
    if per_subject.empty:
        print("No subjects with bad feature values found.")
        return

    print("Subjects ranked by total bad features:")
    items = per_subject.items()
    if top is not None:
        items = list(items)[:top]
    for subj, cnt in items:
        print(f"- {subj}: {int(cnt)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore aggregate feature CSV for NaN/Inf values by feature base."
    )
    default_csv = Path(csv_dir) / "aggregate@raw.csv"
    parser.add_argument(
        "--csv_file",
        type=str,
        default=str(default_csv),
        help="Path to aggregate CSV (defaults to csv_dir/aggregate@raw.csv)",
    )
    parser.add_argument(
        "--subject_col",
        type=str,
        default="subject",
        help="Name of the subject column (default: subject)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit the number of ranked subjects to print (default: all)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    find_bad_values_by_feature(df, subject_col=args.subject_col)
    rank_subjects_by_bad_values(df, subject_col=args.subject_col, top=args.top)


if __name__ == "__main__":
    main()

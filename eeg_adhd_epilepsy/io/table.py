from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from coco_pipe.io.structures import DataContainer

logging.basicConfig(level=logging.INFO)


def parse_descriptor_feature_column(column: str) -> dict[str, str]:
    match = re.compile(r"(?P<body>.+)_(?P<scope>chgrp|ch)-(?P<sensor>.+)$").match(str(column))
    if match is None:
        raise ValueError(
            f"Could not parse descriptor column '{column}'. "
            "Expected names like 'mean_complexity_sample_entropy_chgrp-front_left' "
            "or 'band_abs_alpha_ch-Fz'."
        )
    body = match.group("body")
    family = None
    prefix = ""
    feature = ""
    for family_name in ("band", "complexity", "param"):
        token = f"_{family_name}_"
        if body.startswith(f"{family_name}_"):
            family = family_name
            feature = body[len(f"{family_name}_"):]
            break
        if token in body:
            prefix, feature = body.split(token, 1)
            family = family_name
            break
    if family is None:
        raise ValueError(
            f"Could not parse descriptor column '{column}'. "
            "Expected names like 'mean_complexity_sample_entropy_chgrp-front_left' "
            "or 'band_abs_alpha_ch-Fz'."
        )
    return {
        "column": str(column),
        "family": family,
        "feature": f"{prefix}_{feature}" if prefix else feature,
        "scope": "sensor_group" if match.group("scope") == "chgrp" else "sensor",
        "sensor": match.group("sensor"),
    }

def load(fpath: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV or parquet table with basic exploration logs.

    - Auto-detects delimiter when ``sep`` is None for CSVs.
    - Drops unnamed/all-NaN columns created by trailing separators.
    - Logs shape, head, columns, missing values, duplicates, and nunique.
    """
    path = Path(fpath)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        # Use pandas' engine to auto-detect when sep is None
        read_kwargs = dict(encoding="utf-8")
        if sep is None:
            read_kwargs.update(dict(sep=None, engine='python'))
        else:
            read_kwargs.update(dict(sep=sep, low_memory=False))
        df = pd.read_csv(fpath, **read_kwargs)
    else:
        raise ValueError(f"Unsupported table format: {fpath}")

    # Drop unnamed/all-NaN columns (often from trailing separators)
    before_cols = list(df.columns)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]  # drop unnamed
    df = df.dropna(axis=1, how="all")  # drop entirely empty columns
    dropped = [c for c in before_cols if c not in df.columns]
    if dropped:
        logging.info(f"Dropped {len(dropped)} empty/unnamed columns: {dropped}")

    logging.info(f"Loaded file: {fpath}")
    logging.info(f"{df.shape[0]} rows and {df.shape[1]} columns")
    logging.info(f"First 5 rows:\n{df.head()}")
    logging.info(f"Columns:\n{df.columns.tolist()}")
    missing_counts = df.isnull().sum()
    if int(missing_counts.sum()) > 0:
        logging.info(f"Missing values per column:\n{missing_counts}")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        logging.info(f"Duplicate rows: {duplicate_count}")
    logging.info(f"Unique values per column:\n{df.nunique()}")
    return df


def _normalize_subject_value(value: object) -> str:
    text = str(value).strip().replace("sub-", "")
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return f"{int(numeric):04d}"
    return text


def load_tabular_data(
    table_path: Path,
    feature_columns_path: Path,
    condition: str | None = None,
    target_col: str | None = None,
    subjects: Optional[Sequence[str]] = None,
    subject_col: str = "study_id",
    analysis_mode: str = "flat",
    descriptor_families: Optional[Sequence[str]] = None,
    descriptor_max_abs_value: float | None = None,
) -> DataContainer:
    df = load(str(table_path), sep=None)
    if condition is not None:
        df = df[df["condition"].astype(str) == str(condition)].copy()
    if subjects:
        if subject_col not in df.columns:
            raise ValueError(f"Subject filter column '{subject_col}' is not available in {table_path}.")
        wanted_subjects = {_normalize_subject_value(subject) for subject in subjects}
        subject_values = df[subject_col].map(_normalize_subject_value)
        df = df[subject_values.isin(wanted_subjects)].copy()
    if df.empty:
        raise RuntimeError(f"No feature-table rows survived filtering for condition='{condition}'.")

    raw_feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    parsed_feature_columns = [parse_descriptor_feature_column(column) for column in raw_feature_columns]
    if descriptor_families:
        allowed_families = {str(value).strip() for value in descriptor_families}
        parsed_feature_columns = [
            item for item in parsed_feature_columns if item["family"] in allowed_families
        ]
        if not parsed_feature_columns:
            raise RuntimeError(
                f"No descriptor features matched descriptor_families={list(descriptor_families)}."
            )
    feature_columns = [item["column"] for item in parsed_feature_columns]
    feature_df = df.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    valid_mask = ~feature_df.isna().any(axis=1)
    if not valid_mask.all():
        dropped_df = df.loc[~valid_mask].copy()
        dropped_feature_df = feature_df.loc[~valid_mask]
        summary_cols = [
            column
            for column in ("obs_id", "study_id")
            if column in dropped_df.columns
        ]
        dropped_summary = dropped_df.loc[:, summary_cols].copy() if summary_cols else pd.DataFrame(index=dropped_df.index)
        dropped_summary["missing_feature_count"] = dropped_feature_df.isna().sum(axis=1).to_numpy()
        dropped_summary["missing_features"] = [
            ", ".join(dropped_feature_df.columns[row.isna()].tolist())
            for _, row in dropped_feature_df.iterrows()
        ]
        logging.warning(
            "Dropping %d row(s) with NaN/Inf features from %s for condition=%r:\n%s",
            int((~valid_mask).sum()),
            table_path,
            condition,
            dropped_summary.to_string(index=False),
        )
        df = df.loc[valid_mask].copy()
        feature_df = feature_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError(
            f"No feature-table rows survived missing-feature filtering for condition='{condition}'."
        )
    dropped_extreme_rows = 0
    if descriptor_max_abs_value is not None:
        max_abs = float(descriptor_max_abs_value)
        if max_abs <= 0:
            raise ValueError("descriptor_max_abs_value must be positive when provided.")
        extreme_mask = feature_df.abs().gt(max_abs).any(axis=1)
        if extreme_mask.any():
            dropped_extreme_rows = int(extreme_mask.sum())
            dropped_df = df.loc[extreme_mask].copy()
            dropped_feature_df = feature_df.loc[extreme_mask]
            summary_cols = [
                column
                for column in ("obs_id", "study_id")
                if column in dropped_df.columns
            ]
            dropped_summary = (
                dropped_df.loc[:, summary_cols].copy()
                if summary_cols
                else pd.DataFrame(index=dropped_df.index)
            )
            abs_feature_df = dropped_feature_df.abs()
            dropped_summary["extreme_feature_count"] = abs_feature_df.gt(max_abs).sum(axis=1).to_numpy()
            dropped_summary["max_abs_feature_value"] = abs_feature_df.max(axis=1).to_numpy()
            dropped_summary["extreme_features"] = [
                ", ".join(abs_feature_df.columns[row.gt(max_abs)].tolist())
                for _, row in abs_feature_df.iterrows()
            ]
            logging.warning(
                "Dropping %d row(s) with descriptor feature abs(value) > %g from %s for condition=%r:\n%s",
                dropped_extreme_rows,
                max_abs,
                table_path,
                condition,
                dropped_summary.to_string(index=False),
            )
            df = df.loc[~extreme_mask].copy()
            feature_df = feature_df.loc[~extreme_mask].copy()
        if df.empty:
            raise RuntimeError(
                "No feature-table rows survived finite-extreme filtering "
                f"for condition='{condition}' with descriptor_max_abs_value={max_abs}."
            )
    metadata_df = df.drop(columns=feature_columns)

    y = df[target_col].astype(str).to_numpy() if target_col else None
    if "obs_id" in df.columns:
        ids = df["obs_id"].astype(str).to_numpy()
    elif "recording_id" in df.columns:
        ids = df["recording_id"].astype(str).to_numpy()
    elif "subject" in df.columns:
        ids = df["subject"].astype(str).to_numpy()
    elif "study_id" in df.columns:
        ids = df["study_id"].astype(str).to_numpy()
    else:
        raise ValueError(
            f"Could not infer ids for {table_path}. Expected one of 'obs_id', 'subject', or 'study_id'."
        )
    coords = {}
    coords.update(
        {column: metadata_df[column].to_numpy() for column in metadata_df.columns}
    )

    if analysis_mode == "flat":
        coords["feature"] = np.asarray(feature_columns, dtype=object)
        return DataContainer(
            X=feature_df.to_numpy(dtype=float),
            dims=("obs", "feature"),
            coords=coords,
            y=y,
            ids=ids,
            meta={
                "source": str(table_path),
                "descriptor_max_abs_value": descriptor_max_abs_value,
                "dropped_extreme_rows": dropped_extreme_rows,
            },
        )

    sensors = list(dict.fromkeys(item["sensor"] for item in parsed_feature_columns))
    features = list(dict.fromkeys(item["feature"] for item in parsed_feature_columns))
    sensor_to_index = {sensor: idx for idx, sensor in enumerate(sensors)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    feature_family = {
        item["feature"]: item["family"] for item in parsed_feature_columns
    }

    X_tensor = np.full(
        (len(feature_df), len(sensors), len(features)),
        np.nan,
        dtype=float,
    )
    for item in parsed_feature_columns:
        X_tensor[:, sensor_to_index[item["sensor"]], feature_to_index[item["feature"]]] = feature_df[
            item["column"]
        ].to_numpy(dtype=float)
    coords["sensor"] = np.asarray(sensors, dtype=object)
    coords["feature"] = np.asarray(features, dtype=object)
    coords["feature_family"] = np.asarray(
        [feature_family[feature_key] for feature_key in features],
        dtype=object,
    )
    return DataContainer(
        X=X_tensor,
        dims=("obs", "sensor", "feature"),
        coords=coords,
        y=y,
        ids=ids,
        meta={
            "source": str(table_path),
            "descriptor_families": list(dict.fromkeys(coords["feature_family"].tolist())),
            "descriptor_max_abs_value": descriptor_max_abs_value,
            "dropped_extreme_rows": dropped_extreme_rows,
        },
    )


def save(
    df: pd.DataFrame,
    base_path: Path,
    feature_columns: list[str] | None = None,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    if feature_columns is not None:
        (base_path.parent / f"{base_path.name}_feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2),
            encoding="utf-8",
        )

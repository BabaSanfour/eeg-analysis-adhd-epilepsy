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


def load_tabular_data(
    table_path: Path,
    feature_columns_path: Path,
    condition: str | None = None,
    target_col: str | None = None,
    analysis_mode: str = "flat",
    descriptor_families: Optional[Sequence[str]] = None,
) -> DataContainer:
    df = load(str(table_path), sep=None)
    if condition is not None:
        df = df[df["condition"].astype(str) == str(condition)].copy()
    if df.empty:
        raise RuntimeError(f"No feature-table rows survived filtering for condition='{condition}'.")

    raw_feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    parsed_feature_columns: list[tuple[str, str, str, str]] = []
    for column in raw_feature_columns:
        match = re.match(
            r"(?P<prefix>.+?)_(?P<family>band|complexity|param)_(?P<feature>.+?)_(?:chgrp|ch)-(?P<sensor>.+)$",
            str(column),
        )
        if match is None:
            raise ValueError(
                f"Could not parse descriptor column '{column}'. "
                "Expected names like 'mean_complexity_sample_entropy_chgrp-front_left' "
                "or 'band_abs_alpha_ch-Fz'."
            )
        parsed_feature_columns.append(
            (
                column,
                match.group("family"),
                f"{match.group('prefix')}_{match.group('feature')}",
                match.group("sensor"),
            )
        )
    if descriptor_families:
        allowed_families = {str(value).strip() for value in descriptor_families}
        parsed_feature_columns = [
            item for item in parsed_feature_columns if item[1] in allowed_families
        ]
        if not parsed_feature_columns:
            raise RuntimeError(
                f"No descriptor features matched descriptor_families={list(descriptor_families)}."
            )
    feature_columns = [column for column, _, _, _ in parsed_feature_columns]
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
    metadata_df = df.drop(columns=feature_columns)

    y = df[target_col].astype(str).to_numpy() if target_col else None
    if "obs_id" in df.columns:
        ids = df["obs_id"].astype(str).to_numpy()
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
            meta={"source": str(table_path)},
        )

    sensors = list(dict.fromkeys(sensor for _, _, _, sensor in parsed_feature_columns))
    features = list(dict.fromkeys(feature_key for _, _, feature_key, _ in parsed_feature_columns))
    sensor_to_index = {sensor: idx for idx, sensor in enumerate(sensors)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    feature_family = {
        feature_key: family for _, family, feature_key, _ in parsed_feature_columns
    }

    X_tensor = np.full(
        (len(feature_df), len(sensors), len(features)),
        np.nan,
        dtype=float,
    )
    for column, _, feature_key, sensor in parsed_feature_columns:
        X_tensor[:, sensor_to_index[sensor], feature_to_index[feature_key]] = feature_df[
            column
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

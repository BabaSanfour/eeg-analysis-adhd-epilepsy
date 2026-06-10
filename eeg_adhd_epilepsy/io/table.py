from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from coco_pipe.io.descriptors import save_descriptor_table
from coco_pipe.io.utils import read_table

logger = logging.getLogger(__name__)


def load(fpath: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV or parquet table with diagnostic logging.

    Delegates file reading and column cleanup to :func:`coco_pipe.io.utils.read_table`,
    then logs shape, head, columns, missing values, duplicates, and nunique.
    """
    df = read_table(fpath, sep=sep)

    logger.info(f"Loaded file: {fpath}")
    logger.info(f"{df.shape[0]} rows and {df.shape[1]} columns")
    logger.info(f"First 5 rows:\n{df.head()}")
    logger.info(f"Columns:\n{df.columns.tolist()}")
    missing_counts = df.isnull().sum()
    if int(missing_counts.sum()) > 0:
        logger.info(f"Missing values per column:\n{missing_counts}")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        logger.info(f"Duplicate rows: {duplicate_count}")
    logger.info(f"Unique values per column:\n{df.nunique()}")
    return df


def save(
    df: pd.DataFrame,
    base_path: Path,
    feature_columns: list[str] | None = None,
) -> None:
    """Write *df* as parquet + csv, with an optional feature-columns sidecar.

    Thin wrapper around :func:`coco_pipe.io.descriptors.save_descriptor_table`.
    """
    save_descriptor_table(df, base_path, feature_columns=feature_columns)

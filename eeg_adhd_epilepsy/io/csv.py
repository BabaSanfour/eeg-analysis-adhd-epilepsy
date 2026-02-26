
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)

def load(fpath: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV with basic exploration logs.

    - Auto-detects delimiter when ``sep`` is None (handles comma/semicolon files).
    - Drops unnamed/all-NaN columns created by trailing separators.
    - Logs shape, head, columns, missing values, duplicates, and nunique.
    """
    # Use pandas' engine to auto-detect when sep is None
    read_kwargs = dict(encoding="utf-8")
    if sep is None:
        read_kwargs.update(dict(sep=None, engine='python'))
    else:
        read_kwargs.update(dict(sep=sep, low_memory=False))

    df = pd.read_csv(fpath, **read_kwargs)

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
    logging.info(f"Missing values per column:\n{df.isnull().sum()}")
    logging.info(f"Duplicate rows: {df.duplicated().sum()}")
    logging.info(f"Unique values per column:\n{df.nunique()}")
    return df

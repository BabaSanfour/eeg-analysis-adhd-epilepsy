"""Data filtering utilities for analysis datasets.

This module centralizes helper functions previously spread across
``analysis_design.py`` and exposes a high level ``build_analysis_dataset``
function that merges demographic and feature tables then applies the
requested filters.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional

import pandas as pd

# Minimal mapping to compute psychostimulant categories without relying on
# heavy configuration modules. Only keys used in tests are included.
MAPPING_PSYCHOSTIMULANT = {
    "no psychostimulants": 0,
    "Lisdexamfetamine": 1,
    "Lisdexamfetamine (d/c)": 1,
    "Methylphenidate (d/c)": 2,
    "Methylphenidate": 2,
    "Methylphenidate, Methylphenidate": 2,
    "Lisdexamfetamine, Methylphenidate": 3,
    "Dextroamphetamine (Dexedrine)": 4,
    "Amphetamine/dextroamphetamine salt (Adderall)": 5,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    file_path:
        Location of the CSV file on disk.
    """
    df = pd.read_csv(file_path)
    logging.info("Loaded %s with %d rows and %d columns", file_path, df.shape[0], df.shape[1])
    return df


def merge_data(demo_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Merge demographic and feature DataFrames on ``Study ID``."""
    if "Study ID" not in demo_df.columns or "Study ID" not in features_df.columns:
        raise ValueError("Both dataframes must contain a 'Study ID' column")
    merged_df = pd.merge(demo_df, features_df, on="Study ID", how="inner")
    logging.info("Merged data has %d rows and %d columns", merged_df.shape[0], merged_df.shape[1])
    return merged_df


def compute_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an ``age_groups`` column exists based on ``Age``."""
    if "age_groups" not in df.columns:
        df["age_groups"] = pd.cut(df["Age"], bins=[0, 12, 19], labels=[1, 2], right=False)
        logging.info("Computed 'age_groups' from 'Age' column")
    return df


def filter_by_sex(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    """Filter a DataFrame by ``Sex``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Sex": ["F", "M", "F"]})
    >>> filter_by_sex(df, "F").shape[0]
    2
    """
    if sex.lower() in {"f", "m"}:
        filtered = df[df["Sex"] == sex.upper()]
        logging.info("Filtered by Sex = %s: %d rows remain", sex.upper(), filtered.shape[0])
        return filtered
    logging.info("Sex filter set to 'combined'; no filtering applied")
    return df


def filter_by_age_group(df: pd.DataFrame, age_group: str) -> pd.DataFrame:
    """Filter a DataFrame by ``age_groups``."""
    df = compute_age_groups(df)
    if age_group.lower() != "combined":
        age_group_val = int(age_group)
        filtered = df[df["age_groups"] == age_group_val]
        logging.info(
            "Filtered by age_group = %s: %d rows remain", age_group_val, filtered.shape[0]
        )
        return filtered
    logging.info("Age group filter set to 'combined'; no filtering applied")
    return df


def filter_diagnosis(
    df: pd.DataFrame,
    diag_col: str,
    condition: str,
    include_potential: bool = True,
) -> pd.DataFrame:
    """Filter ``df`` according to diagnosis column ``diag_col``.

    The ``condition`` parameter accepts ``with``, ``without`` or ``combined``.
    When ``include_potential`` is ``True`` potential diagnoses (``"0 (potentiel)"``)
    are considered positive cases for the ``with`` condition.
    """
    if condition.lower() == "combined":
        logging.info("No filtering applied for %s (combined)", diag_col)
        return df

    if condition.lower() == "with":
        valid = ["1", "0 (potentiel)"] if include_potential else ["1"]
    elif condition.lower() == "without":
        valid = ["0"] if include_potential else ["0", "0 (potentiel)"]
    else:
        logging.warning("Unrecognized diagnosis filter for %s: %s", diag_col, condition)
        return df

    filtered = df[df[diag_col].astype(str).isin(valid)]
    logging.info(
        "Filtered %s with condition '%s': %d rows remain", diag_col, condition, filtered.shape[0]
    )
    return filtered


def create_target_column(df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    """Create a ``target`` column for the chosen analysis type."""
    if analysis_type.lower() == "medications":
        if "psychostimulant_category" not in df.columns:
            if "psychostimulant_description" in df.columns:
                df["psychostimulant_category"] = df["psychostimulant_description"].map(
                    MAPPING_PSYCHOSTIMULANT
                )
                logging.info(
                    "Computed 'psychostimulant_category' from 'psychostimulant_description'"
                )
            else:
                raise ValueError(
                    "For 'medications' analysis, a psychostimulant column is required"
                )
        df = df[df["psychostimulant_category"].isin([1, 2])].copy()
        df["target"] = df["psychostimulant_category"].apply(
            lambda x: "Med1" if x == 1 else "Med2"
        )
        logging.info("Created target column for 'medications' analysis")
    elif analysis_type.lower() == "general":
        if "Psychostimulant (y/n)" not in df.columns:
            raise ValueError("'Psychostimulant (y/n)' column is required for general analysis")
        df["target"] = df["Psychostimulant (y/n)"].apply(
            lambda x: 0 if pd.isna(x) or x == 0 else 1
        )
        logging.info(
            "Created target column for 'general' analysis (0: Control, 1: Psychostimulant)"
        )
    else:
        raise ValueError("analysis_type must be either 'medications' or 'general'")
    return df


def build_analysis_dataset(
    demographic_csv: str,
    features_csv: str,
    analysis_type: str,
    filters: Dict[str, str],
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build a filtered dataset ready for analysis.

    Parameters
    ----------
    demographic_csv, features_csv:
        Paths to the demographic and feature CSV files to merge.
    analysis_type:
        Either ``"medications"`` or ``"general"`` determining the target column.
    filters:
        Mapping specifying filters for ``sex``, ``age_groups``, ``ADHD``, ``TSA`` and
        ``Epilepsy``.
    output_csv:
        When provided, the resulting dataset is written to this path.
    """
    demo_df = load_csv(demographic_csv)
    feat_df = load_csv(features_csv)
    merged = merge_data(demo_df, feat_df)

    filtered = filter_by_sex(merged, filters.get("sex", "combined"))
    filtered = filter_by_age_group(filtered, filters.get("age_groups", "combined"))
    filtered = filter_diagnosis(filtered, "ADHD", filters.get("ADHD", "combined"))
    filtered = filter_diagnosis(filtered, "TSA", filters.get("TSA", "combined"))
    filtered = filter_diagnosis(filtered, "Epilepsy", filters.get("Epilepsy", "combined"))

    final_df = create_target_column(filtered, analysis_type)
    if output_csv:
        final_df.to_csv(output_csv, index=False)
        logging.info("Final results saved to %s", output_csv)
    return final_df


def main() -> None:
    """CLI entry point for dataset construction."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge demographic and feature CSVs, apply filters and create a target column"
        )
    )
    parser.add_argument("--demographic-csv", required=True, help="Path to demographic CSV")
    parser.add_argument("--features-csv", required=True, help="Path to features CSV")
    parser.add_argument(
        "--analysis-type",
        choices=["medications", "general"],
        required=True,
        help="Type of analysis to prepare",
    )
    parser.add_argument("--sex", choices=["F", "M", "combined"], default="combined")
    parser.add_argument(
        "--age-groups", choices=["1", "2", "combined"], default="combined"
    )
    parser.add_argument(
        "--ADHD", choices=["with", "without", "combined"], default="combined"
    )
    parser.add_argument(
        "--TSA", choices=["with", "without", "combined"], default="combined"
    )
    parser.add_argument(
        "--Epilepsy", choices=["with", "without", "combined"], default="combined"
    )
    parser.add_argument("--output-csv", help="Where to write the filtered dataset")

    args = parser.parse_args()
    filters = {
        "sex": args.sex,
        "age_groups": args.age_groups,
        "ADHD": args.ADHD,
        "TSA": args.TSA,
        "Epilepsy": args.Epilepsy,
    }

    build_analysis_dataset(
        args.demographic_csv,
        args.features_csv,
        args.analysis_type,
        filters,
        args.output_csv,
    )


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()

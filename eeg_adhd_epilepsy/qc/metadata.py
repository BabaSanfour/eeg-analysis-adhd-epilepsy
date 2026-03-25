"""
Build canonical patient metadata tables from the ADHD and drug-resistant CSVs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_adhd_epilepsy.io.csv import load as load_csv
from eeg_adhd_epilepsy.utils.metadata_schema import (
    EPILEPSY_MED_COLS,
    PATIENTS_METADATA_AUDIT_COLUMNS,
    PATIENTS_METADATA_COLUMNS,
    PSYCHOSTIMULANT_RAW_PAIR_TO_CATEGORY,
)

DEFAULT_CSV_DIR = Path(
    "/Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/"
    "EEG_psychostimulants_2025-02/csv"
)
DEFAULT_ADHD_CSV = DEFAULT_CSV_DIR / "EEG_Psychostimulants_PatientList_08-2025.csv"
DEFAULT_DRUG_RESISTANT_CSV = DEFAULT_CSV_DIR / "IRSC_data_03-22-2026.csv"

_AUDIT_OUTPUT_COLUMNS = [*PATIENTS_METADATA_AUDIT_COLUMNS, "drop_reason"]
_RAW_MERGED_COLUMNS = [
    "source_dataset",
    "study_id",
    "patient_id",
    "eeg_date",
    "first_eeg",
    "age",
    "sex",
    "adhd",
    "autism",
    "epilepsy",
    "psychostimulant_description_input",
    "psychostimulant_category_input",
    *EPILEPSY_MED_COLS,
    "other_asm",
    "asm_resistant",
]

POTENTIAL_PATTERN = re.compile(r"^\s*0\s*\(potentiel\)\s*$", flags=re.IGNORECASE)


def _normalize_binary_flag_series(series: pd.Series, allow_missing: bool) -> pd.Series:
    """Normalize a true binary field to 0/1, optionally preserving missing values.

    This helper is only for binary columns such as diagnoses, ASM flags, or
    `first_eeg`. It must not be used for psychostimulant category codes or any
    other multi-value field.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if allow_missing:
        out = pd.Series(pd.array([pd.NA] * len(series), dtype="Int64"), index=series.index)
        mask = numeric.notna()
        out.loc[mask] = (numeric.loc[mask] == 1).astype("Int64")
        return out
    return numeric.fillna(0).eq(1).astype(int)


def _parse_age_years(value: object) -> float | None:
    if pd.isna(value) or isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    match = re.compile(
        r"(?:(?P<years>\d+)y)?(?:(?P<months>\d+)m)?(?:(?P<days>\d+)d)?$"
    ).match(text)
    years = int(match.group("years") or 0)
    months = int(match.group("months") or 0)
    days = int(match.group("days") or 0)
    total_years = years + (months / 12.0) + (days / 365.25)
    return round(total_years, 2)


def _normalize_sex(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def _build_psychostimulant_fields(
    description_series: pd.Series,
    category_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    categories = []
    for description, raw_category in zip(description_series, category_series):
        label = "" if pd.isna(description) else str(description).strip()
        category = "" if pd.isna(raw_category) else str(raw_category).strip()
        key = (label, category)

        if key not in PSYCHOSTIMULANT_RAW_PAIR_TO_CATEGORY:
            raise ValueError(
                "Unknown psychostimulant mapping "
                f"for description={label!r}, category={category!r}"
            )

        categories.append(PSYCHOSTIMULANT_RAW_PAIR_TO_CATEGORY[key])

    category = pd.Series(categories, index=description_series.index, dtype=object)
    psychostimulant = category.ne("No Psychostimulant").astype(int)
    return psychostimulant, category


def _compute_medication_mismatch_mask(df: pd.DataFrame) -> pd.Series:
    """Return a mask for rows with ADHD/ASM medication mismatches."""
    mask = pd.Series(False, index=df.index)
    mask |= df["adhd"].fillna(0).eq(0) & df["psychostimulant"].fillna(0).eq(1)
    mask |= df["epilepsy"].fillna(0).eq(0) & df["asm"].fillna(0).eq(1)
    return mask


def _add_metadata_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived columns used by downstream metadata reports."""
    out = df.copy()

    out["age_group"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce"),
        bins=[0, 5, 9, 13, 19],
        labels=["0-4", "5-8", "9-12", "13-18"],
        right=False,
    )

    def is_one(value: object) -> bool:
        return pd.notna(value) and int(value) == 1

    def combine_diagnosis(row: pd.Series) -> str:
        labels = []
        if is_one(row.get("adhd")):
            labels.append("ADHD")
        if is_one(row.get("autism")):
            labels.append("ASD")
        if is_one(row.get("epilepsy")):
            labels.append("Epilepsy")
        return "+".join(labels) if labels else "Control"

    def asm_types(row: pd.Series) -> str:
        labels = [column for column in EPILEPSY_MED_COLS if is_one(row.get(column))]
        return "+".join(labels) if labels else "No_ASM"

    def meds_summary(row: pd.Series) -> str:
        has_asm = is_one(row.get("asm"))
        has_psychostim = is_one(row.get("psychostimulant"))
        if has_asm and has_psychostim:
            return "ASM+Psychostim"
        if has_asm:
            return "ASM"
        if has_psychostim:
            return "Psychostim"
        return "No Med"

    out["combined_diagnosis"] = out.apply(combine_diagnosis, axis=1)
    out["asm_types"] = out.apply(asm_types, axis=1)
    out["meds_summary"] = out.apply(meds_summary, axis=1)
    return out


def _rename_adhd_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Study ID": "study_id",
            "Pt ID": "patient_id",
            "EEG_date": "eeg_date",
            "Age": "age",
            "Sex": "sex",
            "TDAH": "adhd",
            "TSA": "autism",
            "Epilepsy": "epilepsy",
            "psychostimulant_description": "psychostimulant_description_input",
            "psychostimulant_category": "psychostimulant_category_input",
            "Resistant": "asm_resistant",
        }
    ).copy()
    out["source_dataset"] = "adhd"
    out["first_eeg"] = 0
    out["other_asm"] = 0
    return out[_RAW_MERGED_COLUMNS].copy()


def _rename_drug_resistant_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Study ID": "study_id",
            "Patient ID": "patient_id",
            "EEG date": "eeg_date",
            "Age": "age",
            "Sex": "sex",
            "TDAH": "adhd",
            "TSA": "autism",
            "Epilepsy": "epilepsy",
            "Psychostimulant - description": "psychostimulant_description_input",
            "Psychostimulant - category": "psychostimulant_category_input",
            "First EEG": "first_eeg",
            "Other ASM": "other_asm",
            "Resistant": "asm_resistant",
        }
    ).copy()
    out["source_dataset"] = "drug_resistant"
    return out[_RAW_MERGED_COLUMNS].copy()


def _normalize_merged_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype("Int64")
    out["age"] = out["age"].apply(_parse_age_years)
    out["sex"] = out["sex"].apply(_normalize_sex)

    for column in ["adhd", "autism", "epilepsy"]:
        out[column] = _normalize_binary_flag_series(out[column], allow_missing=True)

    for column in ["first_eeg", *EPILEPSY_MED_COLS, "other_asm", "asm_resistant"]:
        out[column] = _normalize_binary_flag_series(out[column], allow_missing=False)

    psychostimulant, category = _build_psychostimulant_fields(
        out["psychostimulant_description_input"],
        out["psychostimulant_category_input"],
    )
    out["psychostimulant"] = psychostimulant
    out["psychostimulant_category"] = category
    out["asm"] = (
        out[EPILEPSY_MED_COLS].sum(axis=1).gt(0) | out["other_asm"].eq(1)
    ).astype(int)

    return _add_metadata_derived_columns(out)


def _append_removed(
    removed_frames: list[pd.DataFrame],
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
) -> pd.DataFrame:
    if not mask.any():
        return df

    removed = df.loc[mask, PATIENTS_METADATA_AUDIT_COLUMNS].copy()
    removed["drop_reason"] = reason
    removed_frames.append(removed[_AUDIT_OUTPUT_COLUMNS])
    return df.loc[~mask].copy()


def _drop_same_source_duplicates(
    df: pd.DataFrame,
    removed_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_has_real_eeg"] = ~(
        sortable["eeg_date"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({"NO EEG", "SEEG"})
    )
    sortable["_eeg_sort"] = pd.to_datetime(sortable["eeg_date"], errors="coerce")

    nonmissing = sortable["patient_id"].notna()
    ordered = sortable.loc[nonmissing].sort_values(
        by=["source_dataset", "patient_id", "_has_real_eeg", "_eeg_sort", "study_id"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )
    duplicate_mask = ordered.duplicated(subset=["source_dataset", "patient_id"], keep="first")
    duplicate_indices = ordered.loc[duplicate_mask].index

    keep = sortable.drop(columns=["_has_real_eeg", "_eeg_sort"])
    if len(duplicate_indices) == 0:
        return keep

    removed = keep.loc[duplicate_indices, PATIENTS_METADATA_AUDIT_COLUMNS].copy()
    removed["drop_reason"] = "duplicate_same_source_patient_id"
    removed_frames.append(removed[_AUDIT_OUTPUT_COLUMNS])
    return keep.drop(index=duplicate_indices).copy()


def _serialize_removed_rows(removed_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if removed_rows.empty:
        return []
    serializable = removed_rows.astype(object).where(pd.notna(removed_rows), None)
    records = serializable.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if hasattr(value, "item"):
                record[key] = value.item()
    return records


def build_patients_metadata(
    adhd_csv: Path = DEFAULT_ADHD_CSV,
    drug_resistant_csv: Path = DEFAULT_DRUG_RESISTANT_CSV,
    output_dir: Path = DEFAULT_CSV_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_adhd = load_csv(str(adhd_csv), sep=None)
    raw_drug_resistant = load_csv(str(drug_resistant_csv), sep=None)

    merged_raw = pd.concat(
        [_rename_adhd_source(raw_adhd), _rename_drug_resistant_source(raw_drug_resistant)],
        ignore_index=True,
    )
    potential_mask = merged_raw[["adhd", "autism", "epilepsy"]].apply(
        lambda column: column.astype(str).str.match(POTENTIAL_PATTERN)
    ).any(axis=1)
    merged_normalized = _normalize_merged_metadata(merged_raw)

    raw_output = output_dir / "patients_metadata.csv"
    clean_output = output_dir / "patients_metadata_clean.csv"
    removed_output = output_dir / "patients_metadata_removed.json"

    (
        merged_normalized[PATIENTS_METADATA_COLUMNS]
        .sort_values("study_id")
        .reset_index(drop=True)
        .to_csv(raw_output, index=False)
    )

    working = merged_normalized.copy()
    removed_frames: list[pd.DataFrame] = []

    working = _drop_same_source_duplicates(working, removed_frames)
    working = _append_removed(
        removed_frames,
        working,
        potential_mask.reindex(working.index, fill_value=False),
        "non_confirmed_diagnosis",
    )
    working = _append_removed(
        removed_frames,
        working,
        working["eeg_date"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({"NO EEG", "SEEG"}),
        "no_eeg_files",
    )
    working = _append_removed(
        removed_frames,
        working,
        working["adhd"].isna() | working["autism"].isna(),
        "missing_diagnosis",
    )
    working = _append_removed(
        removed_frames,
        working,
        _compute_medication_mismatch_mask(working),
        "medication_mismatch",
    )

    clean_df = working.copy()
    (
        clean_df[PATIENTS_METADATA_COLUMNS]
        .sort_values("study_id")
        .reset_index(drop=True)
        .to_csv(clean_output, index=False)
    )

    if removed_frames:
        removed_rows = pd.concat(removed_frames, ignore_index=True)
        removed_rows = removed_rows.sort_values(
            by=["study_id", "source_dataset", "patient_id"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        removed_rows = pd.DataFrame(columns=_AUDIT_OUTPUT_COLUMNS)

    summary = {
        "raw_rows": int(len(merged_normalized)),
        "clean_rows": int(len(clean_df)),
        "removed_rows": int(len(removed_rows)),
        "drop_reason_counts": {
            key: int(value)
            for key, value in Counter(removed_rows["drop_reason"]).items()
        },
        "source_dataset_counts": {
            key: int(value)
            for key, value in Counter(removed_rows["source_dataset"]).items()
        },
    }
    removed_output.write_text(
        json.dumps(
            {
                "summary": summary,
                "removed_rows": _serialize_removed_rows(removed_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "raw_csv": raw_output,
        "clean_csv": clean_output,
        "removed_json": removed_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical patient metadata tables.")
    parser.add_argument("--adhd_csv", type=Path, default=DEFAULT_ADHD_CSV)
    parser.add_argument("--drug_resistant_csv", type=Path, default=DEFAULT_DRUG_RESISTANT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_CSV_DIR)
    args = parser.parse_args()
    build_patients_metadata(
        adhd_csv=args.adhd_csv,
        drug_resistant_csv=args.drug_resistant_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

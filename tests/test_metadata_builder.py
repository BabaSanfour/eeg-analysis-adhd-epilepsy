from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eeg_adhd_epilepsy.utils.metadata_schema import (
    EPILEPSY_MED_COLS,
    PATIENTS_METADATA_AUDIT_COLUMNS,
    PATIENTS_METADATA_COLUMNS,
)
from eeg_adhd_epilepsy.qc.metadata import (
    _normalize_binary_flag_series,
    _normalize_merged_metadata,
    _rename_adhd_source,
    _rename_drug_resistant_source,
    build_patients_metadata,
)


def _old_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Study ID": 1,
        "Pt ID": 1001,
        "psychostimulant_description": None,
        "psychostimulant_category": 0,
        "Age": 10,
        "Sex": "F",
        "TDAH": 0,
        "TSA": 0,
        "Epilepsy": 0,
        "Psychostimulant (y/n)": 0,
        "Resistant": 0,
        "EEG_date": "1/1/2024",
    }
    for column in EPILEPSY_MED_COLS:
        row[column] = 0
    row.update(overrides)
    return row


def _new_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "EEG date": "4/16/2019",
        "Study ID": 1014,
        "Patient ID": 2001,
        "Psychostimulant - description": None,
        "Psychostimulant - category": None,
        "Age": "07y00m",
        "Sex": "M",
        "TDAH": 0,
        "TSA": 0,
        "Epilepsy": 1,
        "Other ASM": 0,
        "Other AEDs - description": None,
        "Resistant": 1,
        "First EEG": 1,
    }
    for column in EPILEPSY_MED_COLS:
        row[column] = 0
    row.update(overrides)
    return row


def test_source_specific_rename_defaults_and_shared_normalize_pass() -> None:
    old_df = pd.DataFrame(
        [
            _old_row(
                **{
                    "Study ID": "1",
                    "Pt ID": "1001",
                    "Age": 10,
                    "Sex": "f",
                    "TDAH": "0 (potentiel)",
                    "psychostimulant_description": "Dextroamphetamine (Dexedrine)",
                    "psychostimulant_category": "3.1",
                }
            )
        ]
    )
    new_df = pd.DataFrame(
        [
            _new_row(
                **{
                    "Study ID": "1014",
                    "Patient ID": "2001",
                    "Age": "04m16d",
                    "Sex": "m",
                    "Psychostimulant - description": "Dextroamphetamine",
                    "Psychostimulant - category": "3.1",
                    "Other ASM": "1",
                }
            )
        ]
    )

    renamed_old = _rename_adhd_source(old_df)
    renamed_new = _rename_drug_resistant_source(new_df)

    assert renamed_old.loc[0, "source_dataset"] == "adhd"
    assert renamed_old.loc[0, "first_eeg"] == 0
    assert renamed_old.loc[0, "other_asm"] == 0
    assert renamed_old.loc[0, "adhd"] == "0 (potentiel)"
    assert renamed_old.loc[0, "sex"] == "f"
    assert renamed_old.loc[0, "age"] == 10

    assert renamed_new.loc[0, "source_dataset"] == "drug_resistant"
    assert renamed_new.loc[0, "first_eeg"] == 1
    assert renamed_new.loc[0, "other_asm"] == "1"
    assert renamed_new.loc[0, "sex"] == "m"
    assert renamed_new.loc[0, "age"] == "04m16d"

    normalized = _normalize_merged_metadata(
        pd.concat([renamed_old, renamed_new], ignore_index=True)
    )

    assert normalized["study_id"].tolist() == [1, 1014]
    assert normalized["patient_id"].tolist() == [1001, 2001]
    assert normalized["patient_group_id"].tolist() == [0, 1]
    assert normalized["sex"].tolist() == ["F", "M"]
    assert normalized["eeg_date"].tolist() == ["1/1/2024", "4/16/2019"]
    assert normalized["age"].tolist() == [10.0, 0.38]
    assert normalized["psychostimulant_category"].tolist() == [
        "Dextroamphetamine",
        "Dextroamphetamine",
    ]
    assert normalized["psychostimulant"].tolist() == [1, 1]
    assert normalized["asm"].tolist() == [0, 1]


def test_normalize_binary_flag_series_only_keeps_binary_ones_and_missing() -> None:
    series = pd.Series([1, 0, None, "0 (potentiel)"])
    normalized = _normalize_binary_flag_series(series, allow_missing=True)

    assert normalized.iloc[0] == 1
    assert normalized.iloc[1] == 0
    assert pd.isna(normalized.iloc[2])
    assert pd.isna(normalized.iloc[3])


def test_psychostimulant_normalization_raises_on_unknown_pair() -> None:
    bad_df = pd.concat(
        [
            _rename_adhd_source(
                pd.DataFrame(
                    [
                        _old_row(
                            **{
                                "psychostimulant_description": "Unknown med",
                                "psychostimulant_category": "99",
                            }
                        )
                    ]
                )
            )
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        _normalize_merged_metadata(bad_df)


def test_build_patients_metadata_writes_raw_clean_and_removed_outputs(tmp_path: Path) -> None:
    old_rows = [
        _old_row(
            **{
                "Study ID": 1,
                "Pt ID": 1001,
                "psychostimulant_description": "Methylphenidate",
                "psychostimulant_category": 2,
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 0,
                "Psychostimulant (y/n)": 1,
                "EEG_date": "1/5/2024",
            }
        ),
        _old_row(
            **{
                "Study ID": 2,
                "Pt ID": 1001,
                "psychostimulant_description": "Methylphenidate",
                "psychostimulant_category": 2,
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 0,
                "Psychostimulant (y/n)": 1,
                "EEG_date": "NO EEG",
            }
        ),
        _old_row(
            **{
                "Study ID": 8,
                "Pt ID": 1001,
                "psychostimulant_description": "Methylphenidate",
                "psychostimulant_category": 2,
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 0,
                "Psychostimulant (y/n)": 1,
                "EEG_date": "2/5/2024",
            }
        ),
        _old_row(
            **{
                "Study ID": 3,
                "Pt ID": 1003,
                "TDAH": "0 (potentiel)",
                "TSA": 0,
                "Epilepsy": 0,
            }
        ),
        _old_row(
            **{
                "Study ID": 4,
                "Pt ID": 1004,
                "EEG_date": "NO EEG",
            }
        ),
        _old_row(
            **{
                "Study ID": 5,
                "Pt ID": 1005,
                "TDAH": 1,
                "TSA": None,
            }
        ),
        _old_row(
            **{
                "Study ID": 6,
                "Pt ID": 1006,
                "psychostimulant_description": "Lisdexamfetamine",
                "psychostimulant_category": 1,
                "TDAH": 0,
                "TSA": 0,
                "Epilepsy": 0,
                "Psychostimulant (y/n)": 1,
            }
        ),
        _old_row(
            **{
                "Study ID": 7,
                "Pt ID": 1007,
                "TDAH": 0,
                "TSA": 0,
                "Epilepsy": 0,
                "LEV": 1,
            }
        ),
    ]
    new_rows = [
        _new_row(**{"Study ID": 1014, "Patient ID": 2001, "LEV": 1}),
        _new_row(
            **{
                "EEG date": "5/11/2021",
                "Study ID": 1015,
                "Patient ID": 2002,
                "Psychostimulant - description": "Methylphenidate - Concerta",
                "Psychostimulant - category": 2.0,
                "Age": "04m16d",
                "Sex": "F",
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 1,
                "First EEG": 0,
            }
        ),
        _new_row(
            **{
                "EEG date": "6/1/2020",
                "Study ID": 1016,
                "Patient ID": 2003,
                "Age": "09d",
                "Sex": "M",
                "TDAH": None,
                "TSA": None,
                "Epilepsy": 1,
            }
        ),
        _new_row(
            **{
                "EEG date": "6/3/2020",
                "Study ID": 1017,
                "Patient ID": 2004,
                "Psychostimulant - description": "Dextroamphetamine",
                "Psychostimulant - category": "3.1",
                "Age": "08y00m",
                "Sex": "F",
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 1,
                "First EEG": 1,
            }
        ),
        _new_row(
            **{
                "EEG date": "7/3/2020",
                "Study ID": 1018,
                "Patient ID": 1001,
                "Psychostimulant - description": "Lisdexamfetamine - Vyvanse",
                "Psychostimulant - category": 1.0,
                "Age": "09y00m",
                "Sex": "F",
                "TDAH": 1,
                "TSA": 0,
                "Epilepsy": 1,
                "First EEG": 0,
            }
        ),
    ]

    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    pd.DataFrame(old_rows).to_csv(old_path, index=False)
    pd.DataFrame(new_rows).to_csv(new_path, index=False)

    outputs = build_patients_metadata(
        adhd_csv=old_path,
        drug_resistant_csv=new_path,
        output_dir=tmp_path,
    )

    raw_df = pd.read_csv(outputs["raw_csv"])
    clean_df = pd.read_csv(outputs["clean_csv"])
    removed = json.loads(outputs["removed_json"].read_text(encoding="utf-8"))

    assert list(raw_df.columns) == PATIENTS_METADATA_COLUMNS
    assert raw_df["study_id"].tolist() == sorted(raw_df["study_id"].tolist())
    assert "_potential_row" not in raw_df.columns
    assert "psychostimulant_description_input" not in raw_df.columns
    assert "psychostimulant_category_input" not in raw_df.columns

    assert clean_df["study_id"].tolist() == [1, 8, 1014, 1015, 1017, 1018]
    assert clean_df["patient_id"].tolist() == [1001, 1001, 2001, 2002, 2004, 1001]
    assert clean_df.loc[
        clean_df["patient_id"].eq(1001), "patient_group_id"
    ].nunique() == 1
    assert (
        clean_df.loc[clean_df["patient_id"].eq(1001), "patient_group_id"]
        != clean_df.loc[clean_df["patient_id"].eq(1001), "patient_id"]
    ).all()
    assert (
        clean_df.loc[clean_df["patient_id"].eq(1001), "patient_group_id"]
        != clean_df.loc[clean_df["patient_id"].eq(1001), "study_id"]
    ).all()
    assert clean_df["eeg_date"].tolist() == [
        "1/5/2024",
        "2/5/2024",
        "4/16/2019",
        "5/11/2021",
        "6/3/2020",
        "7/3/2020",
    ]
    assert clean_df["psychostimulant_category"].tolist() == [
        "Methylphenidate",
        "Methylphenidate",
        "No Psychostimulant",
        "Methylphenidate",
        "Dextroamphetamine",
        "Lisdexamfetamine",
    ]
    assert clean_df["psychostimulant"].tolist() == [1, 1, 0, 1, 1, 1]
    assert clean_df["asm"].tolist() == [0, 0, 1, 0, 0, 0]
    assert clean_df["asm_types"].tolist() == [
        "No_ASM",
        "No_ASM",
        "LEV",
        "No_ASM",
        "No_ASM",
        "No_ASM",
    ]
    assert clean_df["meds_summary"].tolist() == [
        "Psychostim",
        "Psychostim",
        "ASM",
        "Psychostim",
        "Psychostim",
        "Psychostim",
    ]
    assert clean_df["combined_diagnosis"].tolist() == [
        "ADHD",
        "ADHD",
        "Epilepsy",
        "ADHD+Epilepsy",
        "ADHD+Epilepsy",
        "ADHD+Epilepsy",
    ]
    assert clean_df["age_group"].fillna("nan").tolist() == [
        "9-12",
        "9-12",
        "5-8",
        "0-4",
        "5-8",
        "9-12",
    ]
    assert clean_df.loc[clean_df["study_id"] == 1014, "age"].iat[0] == 7.0
    assert clean_df.loc[clean_df["study_id"] == 1015, "age"].iat[0] == 0.38
    assert clean_df.loc[clean_df["study_id"] == 1017, "age"].iat[0] == 8.0
    assert "NO EEG" not in clean_df["eeg_date"].tolist()
    assert not clean_df[["adhd", "autism"]].isna().any().any()

    assert removed["summary"]["drop_reason_counts"] == {
        "non_confirmed_diagnosis": 1,
        "no_eeg_files": 2,
        "missing_diagnosis": 2,
        "medication_mismatch": 2,
    }
    assert removed["summary"]["source_dataset_counts"] == {
        "adhd": 6,
        "drug_resistant": 1,
    }
    assert list(removed["removed_rows"][0].keys()) == [
        *PATIENTS_METADATA_AUDIT_COLUMNS,
        "drop_reason",
    ]
    removed_reasons = {row["study_id"]: row["drop_reason"] for row in removed["removed_rows"]}
    assert removed_reasons[2] == "no_eeg_files"
    assert removed_reasons[3] == "non_confirmed_diagnosis"
    assert removed_reasons[4] == "no_eeg_files"
    assert removed_reasons[5] == "missing_diagnosis"
    assert removed_reasons[6] == "medication_mismatch"
    assert removed_reasons[7] == "medication_mismatch"
    assert removed_reasons[1016] == "missing_diagnosis"

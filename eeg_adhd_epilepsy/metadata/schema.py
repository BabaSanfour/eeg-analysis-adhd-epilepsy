"""
Shared schema constants for canonical patient metadata tables.
"""

from __future__ import annotations

EPILEPSY_MED_COLS = [
    "LEV",
    "LTG",
    "LCS",
    "CLB",
    "CBZ",
    "VPA",
    "ETH",
    "TPM",
    "RUF",
    "BRV",
    "STP",
    "OXZ",
    "CBM",
]

SOURCE_DATASETS = ("adhd", "drug_resistant")

NORMALIZED_PSYCHOSTIMULANT_CATEGORIES = (
    "No Psychostimulant",
    "Lisdexamfetamine",
    "Methylphenidate",
    "Lisdexamfetamine + Methylphenidate",
    "Dextroamphetamine",
)

PSYCHOSTIMULANT_RAW_PAIR_TO_CATEGORY = {
    ("", ""): "No Psychostimulant",
    ("", "0"): "No Psychostimulant",
    ("", "0.0"): "No Psychostimulant",
    ("Amphetamine/dextroamphetamine salt (Adderall)", "3,2"): "Dextroamphetamine",
    ("Amphetamine/dextroamphetamine salt (Adderall)", "3.2"): "Dextroamphetamine",
    ("Dextroamphetamine (Dexedrine)", "3,1"): "Dextroamphetamine",
    ("Dextroamphetamine (Dexedrine)", "3.1"): "Dextroamphetamine",
    ("Dextroamphetamine", "3.1"): "Dextroamphetamine",
    ("Lisdexamfetamine", "1"): "Lisdexamfetamine",
    ("Lisdexamfetamine", "1.0"): "Lisdexamfetamine",
    ("Lisdexamfetamine - Vyvanse", "1.0"): "Lisdexamfetamine",
    ("Lisdexamfetamine (d/c)", "1"): "Lisdexamfetamine",
    ("Lisdexamfetamine (d/c)", "1.0"): "Lisdexamfetamine",
    ("Methylphenidate", "2"): "Methylphenidate",
    ("Methylphenidate", "2.0"): "Methylphenidate",
    ("Methylphenidate (d/c 2019)", "2"): "Methylphenidate",
    ("Methylphenidate (d/c 2019)", "2.0"): "Methylphenidate",
    ("Methylphenidate (d/c)", "2"): "Methylphenidate",
    ("Methylphenidate (d/c)", "2.0"): "Methylphenidate",
    ("Methylphenidate, Methylphenidate", "2"): "Methylphenidate",
    ("Methylphenidate, Methylphenidate", "2.0"): "Methylphenidate",
    ("Methylphenidate - Biphentin", "2.0"): "Methylphenidate",
    ("Methylphenidate - Biphentin, Methylphenidate - Ritalin", "2.0"): "Methylphenidate",
    ("Methylphenidate - Concerta", "2.0"): "Methylphenidate",
    ("Methylphenidate - Concerta, Methylphenidate - Ritalin", "2.0"): "Methylphenidate",
    ("Methylphenidate - Ritalin", "2.0"): "Methylphenidate",
    ("Methylphenidate - Ritalin, Methylphenidate - Biphentin", "2.0"): "Methylphenidate",
    ("Lisdexamfetamine, Methylphenidate", "mixed"): "Lisdexamfetamine + Methylphenidate",
}

PATIENTS_METADATA_COLUMNS = [
    "source_dataset",
    "study_id",
    "patient_id",
    "patient_group_id",
    "eeg_date",
    "first_eeg",
    "age",
    "age_group",
    "sex",
    "adhd",
    "autism",
    "epilepsy",
    "combined_diagnosis",
    "psychostimulant",
    "psychostimulant_category",
    "asm",
    *EPILEPSY_MED_COLS,
    "other_asm",
    "asm_types",
    "meds_summary",
    "asm_resistant",
]

# Subset of fields stored in `patients_metadata_removed.json` for drop auditing.
PATIENTS_METADATA_AUDIT_COLUMNS = [
    "source_dataset",
    "study_id",
    "patient_id",
    "patient_group_id",
    "eeg_date",
    "adhd",
    "autism",
    "epilepsy",
    "psychostimulant",
    "asm",
]

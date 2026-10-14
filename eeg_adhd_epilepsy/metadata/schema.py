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

# Columns of the BIDS participants.tsv, after participant_id: the metadata without the
# hospital patient ID and the EEG date. study_id is the number in participant_id.
PARTICIPANTS_COLUMNS = [
    column
    for column in PATIENTS_METADATA_COLUMNS
    if column not in ("study_id", "patient_id", "eeg_date")
]

# Written as integers: pandas reads a column with a missing value as float.
PARTICIPANTS_INTEGER_COLUMNS = [
    "patient_group_id",
    "first_eeg",
    "adhd",
    "autism",
    "epilepsy",
    "psychostimulant",
    "asm",
    *EPILEPSY_MED_COLS,
    "other_asm",
    "asm_resistant",
]

_FLAG_LEVELS = {"0": "no", "1": "yes"}

# participants.json entries for the columns mne-bids does not describe.
PARTICIPANTS_SIDECAR = {
    "source_dataset": {
        "Description": "Patient list the participant comes from",
        "Levels": {
            "adhd": "ADHD patient list",
            "drug_resistant": "drug-resistant epilepsy patient list",
        },
    },
    "patient_group_id": {
        "Description": (
            "Surrogate patient identifier shared by the participants that are EEGs of the same"
            " patient; group cross-validation folds by it"
        ),
    },
    "first_eeg": {
        "Description": (
            "First EEG of the patient, from the drug-resistant list; 0 for every participant"
            " of the ADHD list, which does not record it"
        ),
        "Levels": _FLAG_LEVELS,
    },
    "age_group": {"Description": "Age group, in years, from age"},
    "adhd": {"Description": "ADHD diagnosis", "Levels": _FLAG_LEVELS},
    "autism": {"Description": "Autism spectrum disorder diagnosis", "Levels": _FLAG_LEVELS},
    "epilepsy": {"Description": "Epilepsy diagnosis", "Levels": _FLAG_LEVELS},
    "combined_diagnosis": {
        "Description": "ADHD, ASD and Epilepsy diagnoses joined with '+', or Control for none",
    },
    "psychostimulant": {"Description": "Takes a psychostimulant", "Levels": _FLAG_LEVELS},
    "psychostimulant_category": {
        "Description": "Psychostimulant taken",
        "Levels": {category: category for category in NORMALIZED_PSYCHOSTIMULANT_CATEGORIES},
    },
    "asm": {"Description": "Takes an anti-seizure medication", "Levels": _FLAG_LEVELS},
    **{
        column: {
            "Description": f"Takes the anti-seizure medication coded {column} in the patient list",
            "Levels": _FLAG_LEVELS,
        }
        for column in EPILEPSY_MED_COLS
    },
    "other_asm": {
        "Description": "Takes an anti-seizure medication not in the coded columns",
        "Levels": _FLAG_LEVELS,
    },
    "asm_types": {
        "Description": "Codes of the anti-seizure medications taken, joined with '+', or No_ASM",
    },
    "meds_summary": {
        "Description": "Medication classes taken",
        "Levels": {
            "ASM+Psychostim": "anti-seizure medication and psychostimulant",
            "ASM": "anti-seizure medication only",
            "Psychostim": "psychostimulant only",
            "No Med": "neither",
        },
    },
    "asm_resistant": {
        "Description": "Drug-resistant epilepsy, as marked in the patient list",
        "Levels": _FLAG_LEVELS,
    },
}

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

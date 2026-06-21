"""
Raw EEG recording discovery and `.pnt` parsing utilities.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_pnt_metadata(pnt_path: Path) -> dict[str, str | datetime | None]:
    """
    Parse a brainvision-style .pnt file (text) to extract:
    - 'original_id': The ID string found in the file
    - 'meas_date': A timezone-aware (UTC) datetime of recording start
    """
    if not pnt_path.exists():
        return {"original_id": None, "meas_date": None}

    try:
        text = pnt_path.read_bytes().decode("ISO-8859-1", errors="ignore").replace("\x00", "")
    except Exception as e:
        LOGGER.warning("Failed to read %s: %s", pnt_path, e)
        return {"original_id": None, "meas_date": None}

    # --- parse numeric ID ---
    original_id = None
    match = re.search(r"ID(\d+(?:\.\d+)?)", text)
    if match:
        original_id = match.group(1).split(".")[0]  # strip potential suffix

    # --- parse Date + Start Time ---
    meas_dt = None
    date_match = re.search(r"Date(\d{4})/(\d{2})/(\d{2})", text)
    start_match = re.search(r"Start Time(\d{2})(\d{2})(\d{2})", text)
    if date_match and start_match:
        try:
            year, month, day = map(int, date_match.groups())
            sh, sm, ss = map(int, start_match.groups())
            # Recording timestamps need to be UTC-aware for BIDS export
            meas_dt = datetime(year, month, day, sh, sm, ss, tzinfo=timezone.utc)
        except ValueError:
            pass

    return {"original_id": original_id, "meas_date": meas_dt}


def _build_lookup(
    metadata_df: pd.DataFrame,
) -> dict[str, tuple[dict[int, tuple[int, int | None]], dict[int, tuple[int, int | None]]]]:
    lookups: dict[
        str, tuple[dict[int, tuple[int, int | None]], dict[int, tuple[int, int | None]]]
    ] = {}
    for source_dataset, group in metadata_df.groupby("source_dataset", dropna=False):
        study_lookup: dict[int, tuple[int, int | None]] = {}
        patient_lookup: dict[int, tuple[int, int | None]] = {}
        for row in group.itertuples(index=False):
            study_id = pd.to_numeric(getattr(row, "study_id", None), errors="coerce")
            patient_id = pd.to_numeric(getattr(row, "patient_id", None), errors="coerce")
            study_id = None if pd.isna(study_id) else int(study_id)
            patient_id = None if pd.isna(patient_id) else int(patient_id)
            if study_id is not None:
                study_lookup[study_id] = (study_id, patient_id)
            if patient_id is not None:
                patient_lookup[patient_id] = (study_id, patient_id)
        lookups[str(source_dataset)] = (study_lookup, patient_lookup)
    return lookups


def discover_raw_records(raw_root: Path, metadata_df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Discover raw recordings under `raw_root` and resolve them to canonical metadata.

    Resolution order:
    - cohort 1: `.pnt original_id -> study_id`, then `.pnt original_id -> patient_id`
    - cohort 2: `.pnt original_id -> study_id`, then `.pnt original_id -> patient_id`,
      then subject folder name
    """
    raw_root = Path(raw_root)
    lookups = _build_lookup(metadata_df)
    records: list[dict[str, object]] = []

    for pnt_path in sorted(raw_root.rglob("*.pnt")):
        meta = parse_pnt_metadata(pnt_path)
        folder_id = None
        if raw_root.name == "cohort2":
            try:
                relative = pnt_path.relative_to(raw_root)
                if relative.parts and relative.parts[0].isdigit():
                    folder_id = int(relative.parts[0])
            except ValueError:
                pass
        if folder_id is None and "cohort2" in pnt_path.parts:
            idx = pnt_path.parts.index("cohort2")
            if idx + 1 < len(pnt_path.parts) and pnt_path.parts[idx + 1].isdigit():
                folder_id = int(pnt_path.parts[idx + 1])
        if folder_id is None and len(pnt_path.parents) >= 3:
            candidate = pnt_path.parents[2].name
            if candidate.isdigit():
                folder_id = int(candidate)

        source_dataset = "drug_resistant" if folder_id is not None else "adhd"
        study_lookup, patient_lookup = lookups.get(source_dataset, ({}, {}))

        eeg_path = pnt_path.with_suffix(".EEG")
        meas_datetime = meta["meas_date"]
        record_date = None if meas_datetime is None else meas_datetime.date()

        study_id = None
        patient_id = None
        resolved_by = None
        status = "ready"

        raw_id = pd.to_numeric(meta["original_id"], errors="coerce")
        raw_id = None if pd.isna(raw_id) else int(raw_id)
        if raw_id is not None and raw_id in study_lookup:
            study_id, patient_id = study_lookup[raw_id]
            resolved_by = "study_id"
        elif raw_id is not None and raw_id in patient_lookup:
            study_id, patient_id = patient_lookup[raw_id]
            resolved_by = "patient_id"
        elif source_dataset == "drug_resistant":
            if folder_id is not None and folder_id in study_lookup:
                study_id, patient_id = study_lookup[folder_id]
                resolved_by = "folder_name"

        if study_id is None:
            status = "unresolved_subject"
        elif not eeg_path.exists():
            status = "missing_eeg"

        records.append(
            {
                "source_dataset": source_dataset,
                "study_id": study_id,
                "patient_id": patient_id,
                "resolved_by": resolved_by,
                "record_stem": pnt_path.stem,
                "pnt_path": str(pnt_path),
                "eeg_path": None if not eeg_path.exists() else str(eeg_path),
                "meas_datetime": (
                    None
                    if meas_datetime is None
                    else meas_datetime.isoformat().replace("+00:00", "Z")
                ),
                "record_date": None if record_date is None else record_date.isoformat(),
                "status": status,
            }
        )

    return records

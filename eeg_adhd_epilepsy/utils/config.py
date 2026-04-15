import os
import numpy as np


# ---------------------------------------------------------------------------
# Environment-based configuration
# ---------------------------------------------------------------------------

def _get_env_path(var_name: str, default: str | None = None) -> str:
    """Return the path from an environment variable or a default value.

    Args:
        var_name: Name of the environment variable to read.
        default: Default path to use if the environment variable is unset.

    Returns:
        The resolved path as a string.

    Raises:
        EnvironmentError: If the variable is not set and no default is provided.
    """

    value = os.environ.get(var_name, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is not set."
        )
    return value


# Base directory for EEG data (default to repo-local ./data if unset)
_default_data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
data_dir = _get_env_path("EEG_DATA_DIR", _default_data_dir)

# Subdirectories with sensible defaults that can also be overridden
embeddings_dir = _get_env_path("EEG_EMBEDDINGS_DIR", os.path.join(data_dir, "embeddings"))
results_dir = _get_env_path("EEG_RESULTS_DIR", os.path.join(data_dir, "results"))
csv_dir = _get_env_path("EEG_CSV_DIR", os.path.join(data_dir, "csv"))
bids_dir = _get_env_path("EEG_BIDS_DIR", os.path.join(data_dir, "BIDS"))
derivatives_dir = _get_env_path("EEG_DERIVATIVES_DIR", os.path.join(data_dir, "derivatives"))

source_dirs = {"control": "Controls", "patients": "patients"}

import re
import yaml
import logging
from typing import Dict, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Load annotations from YAML config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "annotations.yaml"

def _load_annotations() -> Dict[str, Tuple[str, ...]]:
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                # Return nested "annotations" key if present, else root
                return data.get("annotations", data)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load annotations from %s: %s", _CONFIG_PATH, e)
    return {}

_annot_config = _load_annotations()

EYES_OPEN_LABELS = tuple(_annot_config.get("eyes_open", ()))
EYES_CLOSED_LABELS = tuple(_annot_config.get("eyes_closed", ()))
HV_LABELS = tuple(_annot_config.get("hv", ("hv",)))
POST_HV_LABELS = tuple(_annot_config.get("post_hv", ("post hv",)))
PHOTO_LABELS = tuple(_annot_config.get("photo", ("photo",)))
MOVEMENT_LABELS = tuple(_annot_config.get("movement", ()))
ARTEFACT_LABELS = tuple(_annot_config.get("artefact", ()))
YAWN_COUGH_LABELS = tuple(_annot_config.get("yawn_cough", ()))
SLEEPY_LABELS = tuple(_annot_config.get("sleepy", ()))
SLEEP_LABELS = tuple(_annot_config.get("sleep", ()))
JAW_FACE_TENSION_LABELS = tuple(_annot_config.get("jaw_face_tension", ()))
EMOTION_BEHAVIOR_LABELS = tuple(_annot_config.get("emotion_behavior", ()))
ORAL_ACTIVITY_LABELS = tuple(_annot_config.get("oral_activity", ()))
EYE_MOVEMENT_LABELS = tuple(_annot_config.get("eye_movement", ()))
WAKEFULNESS_LABELS = tuple(_annot_config.get("wakefulness", ()))
RESPIRATION_LABELS = tuple(_annot_config.get("respiration", ()))
HYPER_STATE_LABELS = tuple(_annot_config.get("hyper_state", ()))
CLINICAL_COMMENT_LABELS = {
    "Clinical - Spikes": tuple(_annot_config.get("clinical_spikes", ())),
    "Clinical - Slowing": tuple(_annot_config.get("clinical_slowing", ())),
    "Clinical - Seizure": tuple(_annot_config.get("clinical_seizure", ())),
    "Clinical - Background": tuple(_annot_config.get("clinical_background", ())),
}
DEMOGRAPHIC_LABELS = tuple(_annot_config.get("demographics", ()))
IGNORE_SYSTEM_LABELS = tuple(
    _annot_config.get("ignore_system", _annot_config.get("pat_montage", ("pat montage",)))
)
COLLABORATION_LABELS = tuple(_annot_config.get("collaboration", ()))
EFFORT_LABELS = tuple(_annot_config.get("effort", ()))
HV_IGNORE = tuple(_annot_config.get("HV_ignore", ()))
IGNORE_MISC_LABELS = tuple(_annot_config.get("ignore_misc", ()))
IGNORE_PATTERNS = tuple(
    _annot_config.get(
        "ignore_patterns",
        ("x", "xx", "xxx", "xxxx", "ml", "x *", "recording gap*", "*storage ring buffer overflow*"),
    )
)
SENSOR_ARTEFACT_KEYWORDS = tuple(
    _annot_config.get("sensor_artefact_keywords", _annot_config.get("sensor_missing_keywords", ("manquantes",)))
)

IGNORED_LABELS = (
    DEMOGRAPHIC_LABELS
    + IGNORE_SYSTEM_LABELS
    + COLLABORATION_LABELS
    + EFFORT_LABELS
    + HYPER_STATE_LABELS
    + HV_IGNORE
    + IGNORE_MISC_LABELS
)
_recording_start_labels = tuple(_annot_config.get("recording_start", ("a1+a2 off",)))
RECORDING_START_LABEL = (
    str(_recording_start_labels[0]).lower().strip() if _recording_start_labels else "a1+a2 off"
)
SENSOR_ACTION_KEYWORDS = tuple(_annot_config.get("sensor_action_keywords", ()))

# ---------------------------------------------------------------------------
# Frequency bands for PSD summaries
# ---------------------------------------------------------------------------
BAND_LIMITS: Dict[str, Tuple[int, int]] = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}

# 10-20 montage channel references
# (Aliasing existing internal sensors_to_keep mostly, but ensuring explicit list)
BASIC_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

ANNOTATION_INTEREST_MAP = {
    "Eyes Open": EYES_OPEN_LABELS,
    "Eyes Closed": EYES_CLOSED_LABELS,
    "Movement": MOVEMENT_LABELS,
    "Artefact": ARTEFACT_LABELS,
    "Effort": EFFORT_LABELS,
    "HV": HV_LABELS,
    "Post HV": POST_HV_LABELS,
    "PHOTO": PHOTO_LABELS,
    "Yawning/Coughing": YAWN_COUGH_LABELS,
    "Sleepy": SLEEPY_LABELS,
    "Sleep": SLEEP_LABELS,
    "Jaw/Face Tension": JAW_FACE_TENSION_LABELS,
    "Emotion/Behavior": EMOTION_BEHAVIOR_LABELS,
    "Oral Activity": ORAL_ACTIVITY_LABELS,
    "Eye Movement": EYE_MOVEMENT_LABELS,
    "Wakefulness": WAKEFULNESS_LABELS,
    "Respiration": RESPIRATION_LABELS,
}


SEGMENT_COLUMNS = [
    "segment_type",
    "block_family",
    "eye_state",
    "t_start",
    "t_stop",
    "duration",
    "freq_hz",
]

CANONICAL_SEGMENT_TYPES = (
    "RAW_baseline",
    "EO_baseline",
    "EC_baseline",
    "HV_EO",
    "HV_EC",
    "HV_UNKNOWN",
    "PostHV_EO",
    "PostHV_EC",
    "PostHV_UNKNOWN",
    "PHOTO_EO",
    "PHOTO_EC",
    "PHOTO_UNKNOWN",
)

DEFAULT_ANALYSIS_CONDITIONS = (
    "EO_baseline",
    "EC_baseline",
    "HV_EO",
    "HV_EC",
    "PostHV_EO",
    "PostHV_EC",
    "PHOTO_EO",
    "PHOTO_EC",
)

KNOWN_EVENT_LABELS = set(ANNOTATION_INTEREST_MAP.keys()) | set(CLINICAL_COMMENT_LABELS.keys())

AGE_YEARS_PATTERN = re.compile(r"\b\d{1,2}\s*ans\b")
DIGIT_PATTERN = re.compile(r"\d+")
PHOTO_FREQ_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*hz", flags=re.IGNORECASE)

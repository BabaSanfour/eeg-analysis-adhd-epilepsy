"""Helper utilities for EEG QC annotation normalization."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Dict

import mne
import numpy as np

from .qc_config import (
    AGE_YEARS_PATTERN,
    ANNOTATION_INTEREST_MAP,
    CLINICAL_COMMENT_LABELS,
    DIGIT_PATTERN,
    EYE_MOVEMENT_LABELS,
    IGNORED_DEMOGRAPHIC_LABELS,
    REFERENCE_EVENT_KEYWORDS,
    SENSOR_ACTION_KEYWORDS,
    SENSOR_CHANNEL_TOKEN_SET,
    SENSOR_ELECTRODE_VERBS,
)


def strip_accents(text: str) -> str:
    """Remove accents from text for easier fuzzy matching."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def clean_annotation_text(desc: str) -> str:
    return strip_accents(desc).lower().strip()


def is_demographic_label(clean_text: str) -> bool:
    if not clean_text:
        return False
    if not any(token in clean_text for token in IGNORED_DEMOGRAPHIC_LABELS):
        return False
    if AGE_YEARS_PATTERN.search(clean_text):
        return True
    if ("ans" in clean_text or "age" in clean_text) and DIGIT_PATTERN.search(clean_text):
        return True
    return False


def matches_sensor_channel_event(clean_text: str) -> bool:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", clean_text) if tok]
    has_channel = any(tok in SENSOR_CHANNEL_TOKEN_SET for tok in tokens)
    if not has_channel:
        return False
    return any(keyword in clean_text for keyword in SENSOR_ACTION_KEYWORDS)


def match_annotation_category(clean_text: str) -> str | None:
    for canonical, patterns in ANNOTATION_INTEREST_MAP.items():
        if canonical == "Sensor/Electrode":
            if any(pattern in clean_text for pattern in patterns) or matches_sensor_channel_event(clean_text):
                return canonical
            continue
        if any(pattern in clean_text for pattern in patterns):
            return canonical
    return None


def match_clinical_category(clean_text: str) -> str | None:
    for canonical, patterns in CLINICAL_COMMENT_LABELS.items():
        if any(pattern in clean_text for pattern in patterns):
            return canonical
    return None


def categorize_annotation_label(desc: str, clean_text: str | None = None) -> str | None:
    clean = clean_text if clean_text is not None else clean_annotation_text(desc)
    if not clean or is_demographic_label(clean):
        return None
    return match_annotation_category(clean)


def summarize_annotations(annotations: mne.Annotations) -> Dict[str, int]:
    counts: Counter = Counter()
    if annotations is None or len(annotations) == 0:
        return {}
    for desc in annotations.description:
        if desc is None:
            continue
        clean = clean_annotation_text(desc)
        if not clean:
            continue
        canonical = categorize_annotation_label(desc, clean_text=clean)
        if canonical:
            counts[canonical] += 1
            continue
        if is_demographic_label(clean):
            continue
        clinical = match_clinical_category(clean)
        if clinical:
            counts[clinical] += 1
            continue
        original = desc.strip()
        if not original:
            continue
        counts[original] += 1
    return dict(counts)


def compute_special_event_counts(annotations: mne.Annotations | None) -> Dict[str, int]:
    counts = {
        "sensor_action_keyword_events": 0,
        "sensor_electrode_verb_events": 0,
        "eye_movement_keyword_events": 0,
        "clinical_comment_events": 0,
    }
    if annotations is None or len(annotations) == 0:
        return counts
    for desc in annotations.description:
        if not desc:
            continue
        clean = clean_annotation_text(desc)
        if not clean:
            continue
        if any(keyword in clean for keyword in SENSOR_ACTION_KEYWORDS):
            counts["sensor_action_keyword_events"] += 1
        if any(verb in clean for verb in SENSOR_ELECTRODE_VERBS):
            counts["sensor_electrode_verb_events"] += 1
        if any(label in clean for label in EYE_MOVEMENT_LABELS):
            counts["eye_movement_keyword_events"] += 1
        for patterns in CLINICAL_COMMENT_LABELS.values():
            if any(pattern in clean for pattern in patterns):
                counts["clinical_comment_events"] += 1
                break
    return counts


def crop_raw_after_reference_event(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations | None,
    logger: logging.Logger | None = None,
) -> float:
    """Crop raw data if the reference event occurs near the beginning."""
    if annotations is None or len(annotations) == 0:
        return 0.0
    ref_onset = None
    for onset, desc in zip(annotations.onset, annotations.description):
        if not desc:
            continue
        normalized = clean_annotation_text(desc)
        if any(keyword in normalized for keyword in REFERENCE_EVENT_KEYWORDS):
            ref_onset = float(onset)
            break
    if ref_onset is None or not np.isfinite(ref_onset):
        return 0.0
    if ref_onset >= 60.0:
        return 0.0
    total_duration = raw.times[-1]
    if ref_onset >= total_duration:
        if logger:
            logger.warning(
                "Reference event onset %.2fs occurs after raw duration %.2fs", ref_onset, total_duration
            )
        return 0.0
    try:
        raw.crop(tmin=ref_onset, verbose="ERROR")
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Failed to crop after reference event: %s", exc)
        return 0.0
    return ref_onset


__all__ = [
    "categorize_annotation_label",
    "clean_annotation_text",
    "compute_special_event_counts",
    "crop_raw_after_reference_event",
    "match_annotation_category",
    "match_clinical_category",
    "matches_sensor_channel_event",
    "summarize_annotations",
]

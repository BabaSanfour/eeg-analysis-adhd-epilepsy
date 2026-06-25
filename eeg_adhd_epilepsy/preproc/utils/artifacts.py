"""Artifact annotation helpers shared across pipeline stages."""

from __future__ import annotations

import mne
import numpy as np


def _compute_artifact_overlap(
    raw: mne.io.BaseRaw,
    new_annots: list[tuple[float, float, str, tuple[str, ...]]],
) -> float:
    """Calculate percentage of existing manual BAD segments re-detected by AutoReject.

    Assumes any pre-existing ``BAD_*`` annotation is a manual reference and
    that all BAD annotations have already been inflated to positive durations.

    Returns:
        Overlap percentage (0–100).
    """
    existing_bads = [
        (float(a["onset"]), float(a["onset"]) + float(a.get("duration", 0)))
        for a in raw.annotations
        if str(a["description"]).startswith("BAD_")
    ]
    if not existing_bads or not new_annots:
        return 0.0

    detected_intervals = [(onset, onset + dur) for onset, dur, _, _ in new_annots]

    max_time = raw.times[-1]
    res = 0.1  # 100 ms resolution
    n_points = int(max_time / res) + 1
    if n_points <= 0:
        return 0.0

    mask_existing = np.zeros(n_points, dtype=bool)
    mask_detected = np.zeros(n_points, dtype=bool)

    for s, e in existing_bads:
        i_s = max(0, min(int(s / res), n_points))
        i_e = max(0, min(int(e / res) + 1, n_points))
        if i_e > i_s:
            mask_existing[i_s:i_e] = True

    for s, e in detected_intervals:
        i_s = max(0, min(int(s / res), n_points))
        i_e = max(0, min(int(e / res) + 1, n_points))
        if i_e > i_s:
            mask_detected[i_s:i_e] = True

    n_existing = int(mask_existing.sum())
    if n_existing == 0:
        return 0.0
    return float((mask_existing & mask_detected).sum()) / n_existing * 100.0


def inflate_bad_annotations(
    raw: mne.io.BaseRaw,
    default_duration: float = 3.0,
    major_duration: float = 5.0,
) -> mne.io.BaseRaw:
    """Assign durations to point-like BAD annotations based on label severity.

    Groups are defined by slug matching on the annotation description:

    * **Major** (``major_duration``, default 5 s): yawn, cough, eye movement,
      oral activity, jaw tension, sleep/wakefulness events.
    * **Common** (``default_duration``, default 3 s): all other ``BAD_*``
      annotations.

    Also normalises any ``bad_*`` annotation to the ``BAD_`` prefix required
    by MNE for rejection.
    """
    major_slugs = [
        "yawn",
        "cough",
        "yawning_coughing",
        "emotion_behavior",
        "oral_activity",
        "sensor_artefact",
        "sensor_action",
        "eye_movement",
        "blink",
        "jaw_face_tension",
        "sleep",
        "sleepy",
        "wakefulness",
    ]

    new_onsets, new_durations, new_descs = [], [], []

    for annot in raw.annotations:
        desc = str(annot["description"])
        onset = float(annot["onset"])
        duration = float(annot["duration"])
        desc_lower = desc.lower()

        if not desc_lower.startswith("bad"):
            new_onsets.append(onset)
            new_durations.append(duration)
            new_descs.append(desc)
            continue

        # Assign duration by label family
        target_duration = (
            major_duration if any(slug in desc_lower for slug in major_slugs) else default_duration
        )

        # Normalise to BAD_ prefix
        if not desc.startswith("BAD_"):
            clean = desc_lower.replace("bad_", "").replace("bad", "").strip("_").strip()
            desc = f"BAD_{clean}" if clean else "BAD_manual"

        new_onsets.append(onset)
        new_durations.append(target_duration)
        new_descs.append(desc)

    raw.set_annotations(
        mne.Annotations(
            onset=new_onsets,
            duration=new_durations,
            description=new_descs,
            orig_time=raw.annotations.orig_time,
        )
    )
    return raw

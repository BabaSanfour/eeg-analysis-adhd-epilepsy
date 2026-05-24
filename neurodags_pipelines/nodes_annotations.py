"""Annotation manipulation nodes: inflate BAD_ annotations."""

from __future__ import annotations

import os

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node
def inflate_bad_annotations(
    mne_object,
    default_duration: float = 3.0,
    major_duration: float = 5.0,
) -> NodeResult:
    """Expand point-like manual BAD_ annotations to fixed durations by label type.

    Rare/disruptive labels (yawn, cough, blink, etc.) → major_duration (5 s).
    All other BAD_ labels → default_duration (3 s), or keep existing if longer.
    Non-BAD_ annotations are kept unchanged.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    major_slugs = [
        "yawn", "cough", "yawning_coughing",
        "emotion_behavior", "oral_activity",
        "sensor_artefact", "sensor_action",
        "eye_movement", "blink",
        "jaw_face_tension",
        "sleep", "sleepy", "wakefulness",
    ]

    new_onsets, new_durations, new_descs = [], [], []
    for annot in raw.annotations:
        desc = str(annot["description"])
        onset = float(annot["onset"])
        duration = float(annot["duration"])
        if not desc.lower().startswith("bad"):
            new_onsets.append(onset)
            new_durations.append(duration)
            new_descs.append(desc)
            continue
        desc_lower = desc.lower()
        if any(slug in desc_lower for slug in major_slugs):
            new_durations.append(major_duration)
        else:
            new_durations.append(max(duration, default_duration))
        new_onsets.append(onset)
        new_descs.append(desc)

    raw.set_annotations(_mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descs,
        orig_time=raw.annotations.orig_time,
    ))

    return NodeResult(artifacts={
        ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
    })

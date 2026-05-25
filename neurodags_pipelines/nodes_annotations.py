"""Annotation manipulation nodes: inflate BAD_ annotations, inject block segments."""

from __future__ import annotations

import os
from pathlib import Path

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node
def inject_block_annotations(mne_object) -> NodeResult:
    """Inject BLOCK_* annotations from the *_segments.csv sidecar next to the source .vhdr.

    Reads segment_type / t_start / t_stop columns and adds
    BLOCK_{segment_type} annotations to the raw.  Skips silently when no CSV
    is found (e.g. synthetic data without a sidecar).
    """
    import mne as _mne
    import pandas as pd
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    csv_path = None
    if raw.filenames and raw.filenames[0]:
        raw_path = Path(raw.filenames[0])
        stem = raw_path.stem
        for suffix in ("_eeg", "_meg", "_ieeg"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidate = raw_path.parent / f"{stem}_segments.csv"
        if candidate.exists():
            csv_path = candidate

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        mask = (
            df["segment_type"].notna()
            & df["t_start"].notna()
            & df["t_stop"].notna()
            & (df["t_stop"] > df["t_start"])
        )
        blocks = df.loc[mask]
        if not blocks.empty:
            block_annots = _mne.Annotations(
                onset=blocks["t_start"].tolist(),
                duration=(blocks["t_stop"] - blocks["t_start"]).tolist(),
                description=("BLOCK_" + blocks["segment_type"].astype(str)).tolist(),
                orig_time=raw.annotations.orig_time,
            )
            raw.set_annotations(raw.annotations + block_annots)

    return NodeResult(artifacts={
        ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
    })


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

"""Preprocessing nodes: filter/resample Raw and extract condition epochs."""

from __future__ import annotations

import os
from typing import Any

from neurodags.definitions import Artifact, NodeResult, SkipDerivative
from neurodags.nodes import register_node


@register_node
def preprocess_raw(
    mne_object,
    filter_args: dict[str, Any] | None = None,
    notch_filter: dict[str, Any] | None = None,
    resample: float | None = None,
    resample_first: bool = False,
) -> NodeResult:
    """Filter and resample a Raw recording without epoching.

    resample_first=True applies resample before bandpass (anti-alias then filter),
    matching base.py behaviour.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    if notch_filter is not None:
        raw.notch_filter(**notch_filter, verbose=False)
    if resample_first and resample is not None:
        raw.resample(float(resample), verbose=False)
    if filter_args is not None:
        raw.filter(**filter_args, verbose=False)
    if not resample_first and resample is not None:
        raw.resample(float(resample), verbose=False)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def extract_condition_epochs(
    mne_object,
    condition_name: str,
    annotation_prefix: str = "BLOCK_",
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
    reject_by_annotation: str | None = None,
) -> NodeResult:
    """Extract fixed-length epochs from BLOCK_<condition_name> annotation windows.

    Parameters
    ----------
    mne_object
        Preprocessed MNE Raw (e.g. from preprocess_raw).
    condition_name
        Condition label (appended to *annotation_prefix*).
    annotation_prefix
        Prefix used in the Raw annotations (default ``"BLOCK_"``).
    epoch_duration
        Length of each fixed epoch in seconds.
    epoch_overlap
        Overlap between consecutive epochs in seconds.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = _mne.io.read_raw_fif(str(mne_object), preload=True, verbose="ERROR")

    target_desc = f"{annotation_prefix}{condition_name}"

    def _condition_matches(desc: str) -> bool:
        # Exact match: BLOCK_EO matches BLOCK_EO
        if desc == target_desc:
            return True
        # Token match for simple (no-underscore) condition names:
        # "EO" matches BLOCK_EO_baseline, BLOCK_HV_EO, BLOCK_PHOTO_EO, etc.
        if "_" not in condition_name and desc.startswith(annotation_prefix):
            suffix = desc[len(annotation_prefix):]
            return condition_name in suffix.split("_")
        return False

    windows: list[tuple[float, float]] = []
    for annot in mne_object.annotations:
        desc = str(annot["description"])
        if desc.startswith("Comment/"):
            desc = desc[len("Comment/"):]
        if _condition_matches(desc):
            onset = float(annot["onset"])
            windows.append((onset, onset + float(annot["duration"])))

    if not windows:
        normalized = sorted({
            str(a["description"]).removeprefix("Comment/")
            for a in mne_object.annotations
        })
        raise SkipDerivative(
            f"Condition '{condition_name}' not present in this recording "
            f"(no annotations matching '{target_desc}'). "
            f"Present descriptions: {normalized}"
        )

    epoch_chunks: list[_mne.BaseEpochs] = []
    for onset, offset in windows:
        crop = mne_object.copy().crop(onset, min(offset, mne_object.times[-1] + mne_object.first_time))
        if crop.n_times < int(epoch_duration * crop.info["sfreq"]):
            continue
        if reject_by_annotation:
            # Rename BAD_ annotations that should not cause epoch rejection here:
            #   - per-channel AR spans (have ch_names): they mark channels, not epochs
            #   - BAD_epoch_{other_condition}: epoch markers from a different condition
            # Manual BAD_ annotations (no ch_names, not BAD_epoch_) are kept as BAD_.
            new_onsets, new_durations, new_descs, new_ch_names = [], [], [], []
            for a in crop.annotations:
                desc = str(a["description"]).removeprefix("Comment/")
                ch_names_val = a.get("ch_names") or ()
                mask_out = False
                if desc.startswith("BAD_"):
                    if ch_names_val:
                        mask_out = True
                    elif desc.startswith("BAD_epoch_"):
                        cond_part = desc[len("BAD_epoch_"):]
                        if "_" not in condition_name:
                            mask_out = condition_name not in cond_part.split("_")
                        else:
                            mask_out = cond_part != condition_name
                if mask_out:
                    desc = "SKIP_" + desc[4:]
                new_onsets.append(float(a["onset"]))
                new_durations.append(float(a["duration"]))
                new_descs.append(desc)
                new_ch_names.append(ch_names_val)
            crop.set_annotations(_mne.Annotations(
                onset=new_onsets, duration=new_durations, description=new_descs,
                ch_names=new_ch_names, orig_time=crop.annotations.orig_time,
            ))
        eps = _mne.make_fixed_length_epochs(
            crop,
            duration=epoch_duration,
            overlap=epoch_overlap,
            preload=True,
            verbose="ERROR",
            reject_by_annotation=reject_by_annotation or False,
        )
        if len(eps) > 0:
            epoch_chunks.append(eps)

    if not epoch_chunks:
        raise ValueError(
            f"Condition '{condition_name}' found in annotations but all windows "
            f"were too short for {epoch_duration}s epochs."
        )

    epochs = (
        epoch_chunks[0]
        if len(epoch_chunks) == 1
        else _mne.concatenate_epochs(epoch_chunks, verbose="ERROR")
    )

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=epochs,
                writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def epoch_by_condition(
    mne_object,
    annotation_prefix: str = "BLOCK_",
    segment_duration: float = 2.0,
    overlap: float = 0.0,
    ignore_annotations: bool = True,
) -> NodeResult:
    """Create multi-condition fixed-length Epochs from BLOCK_* annotated Raw.

    Port of make_epochs_from_preproc_raw: one event_id per BLOCK_* segment type,
    all conditions in a single Epochs object. Matches original _desc-base_epo.fif.

    ignore_annotations=True (default) matches original: BAD_ annotations are noted
    but do not cause epoch rejection. Set False to omit bad spans.
    """
    import mne as _mne
    import numpy as np
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object
    if not raw.preload:
        raw = raw.load_data()

    blocks: list[tuple[str, float, float]] = []
    for annot in raw.annotations:
        desc = str(annot["description"]).removeprefix("Comment/")
        if desc.startswith(annotation_prefix):
            onset = float(annot["onset"])
            duration = float(annot["duration"])
            if duration >= segment_duration:
                condition = desc[len(annotation_prefix):]
                blocks.append((condition, onset, onset + duration))

    if not blocks:
        raise ValueError(
            f"No '{annotation_prefix}*' annotations found or all too short for "
            f"{segment_duration}s epochs."
        )

    events_by_condition: dict[str, np.ndarray] = {}
    for condition, onset, stop in blocks:
        evs = _mne.make_fixed_length_events(
            raw,
            id=1,
            start=onset,
            stop=stop,
            duration=segment_duration,
            overlap=overlap,
            first_samp=True,
        )
        if len(evs) == 0:
            continue
        if condition in events_by_condition:
            events_by_condition[condition] = np.concatenate(
                [events_by_condition[condition], evs]
            )
        else:
            events_by_condition[condition] = evs

    if not events_by_condition:
        raise ValueError("No epochs could be constructed from block annotations.")

    event_id = {
        name: idx
        for idx, name in enumerate(sorted(events_by_condition), start=1)
    }
    remapped = []
    for name, evs in events_by_condition.items():
        ec = evs.copy()
        ec[:, 2] = event_id[name]
        remapped.append(ec)
    events = np.concatenate(remapped)
    events = events[events[:, 0].argsort()]

    epochs = _mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=segment_duration,
        baseline=None,
        reject=None,
        verbose="ERROR",
        preload=True,
        proj=False,
        reject_by_annotation=not ignore_annotations,
        event_repeated="drop",
    )

    artifacts = {}
    for name in sorted(event_id.keys()):
        cond_epochs = epochs[name]
        artifacts[f".{name}.fif"] = Artifact(
            item=cond_epochs,
            writer=lambda path, e=cond_epochs: e.save(path, overwrite=True, verbose="ERROR"),
        )
    return NodeResult(artifacts=artifacts)

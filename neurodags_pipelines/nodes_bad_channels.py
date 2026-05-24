"""Bad channel detection and referencing nodes: RANSAC, CAR."""

from __future__ import annotations

import os

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node
def ransac_bad_channels(
    mne_object,
    block_label: str | None = None,
    annotation_prefix: str = "BLOCK_",
) -> NodeResult:
    """Detect and mark bad channels using RANSAC (pyprep).

    Marks detected channels in raw.info['bads'] without removing them.

    Parameters
    ----------
    block_label
        If set, RANSAC runs only on segments matching
        ``{annotation_prefix}{block_label}`` annotations.  None = full recording.
    annotation_prefix
        Prefix for block annotations (default ``"BLOCK_"``).
    """
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except ImportError as exc:
        raise ImportError("pyprep required for ransac_bad_channels") from exc

    import mne as _mne

    raw = mne_object.copy().load_data()
    eeg_picks = _mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return NodeResult(
            artifacts={".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))}
        )

    raw_for_ransac = raw
    if block_label is not None:
        target = f"{annotation_prefix}{block_label}"
        crops: list = []
        for annot in raw.annotations:
            desc = str(annot["description"])
            if desc.startswith("Comment/"):
                desc = desc[len("Comment/"):]
            if desc == target:
                onset = float(annot["onset"])
                offset = onset + float(annot["duration"])
                crop = raw.copy().crop(onset, min(offset, raw.times[-1] + raw.first_time))
                if crop.n_times > 0:
                    crops.append(crop)
        if crops:
            raw_for_ransac = crops[0] if len(crops) == 1 else _mne.concatenate_raws(crops, verbose="ERROR")

    try:
        nc = NoisyChannels(raw_for_ransac, random_state=42)
        nc.find_bad_by_ransac()
        bads = nc.get_bads(verbose=False) or []
        bads = sorted(ch for ch in bads if ch in raw.ch_names)
        raw.info["bads"] = sorted(set(raw.info.get("bads") or []) | set(bads))
    except (ValueError, OSError):
        pass  # RANSAC failed silently — common on short/low-channel data

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def apply_car(mne_object) -> NodeResult:
    """Apply Common Average Reference."""
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

"""Denoising nodes: ZapLine power-line noise removal."""

from __future__ import annotations

import os

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node
def zapline_denoise(
    mne_object,
    line_freq: float = 60.0,
    adaptive: bool = False,
) -> NodeResult:
    """Remove power-line noise using ZapLine (mne-denoise)."""
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from mne_denoise.zapline import ZapLine
    except ImportError as exc:
        raise ImportError("mne-denoise required for zapline_denoise") from exc

    raw = mne_object.copy().load_data()
    zapline = ZapLine(sfreq=raw.info["sfreq"], line_freq=line_freq, adaptive=adaptive)
    raw = zapline.fit_transform(raw)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

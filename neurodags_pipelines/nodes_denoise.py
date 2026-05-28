"""Denoising nodes: ZapLine power-line noise removal and residual transient denoising."""

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


@register_node
def residual_denoise(
    mne_object,
    transient_method: str = "wiener",
    wiener_n_components: int = 10,
    wiener_window_duration: float = 0.2,
    wiener_noise_percentile: float = 85.0,
    wiener_max_iter: int = 5,
    autoreject_max_chunk_minutes: float = 30.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> NodeResult:
    """Residual denoising: Wiener/ASR transient removal + final AutoReject refinement.

    Port of eeg_adhd_epilepsy.preproc.denoise.run_residual_denoising.
    """
    from neurodags.loaders import load_meeg
    from eeg_adhd_epilepsy.preproc.denoise import ArtifactDenoisingConfig, run_residual_denoising

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    config = ArtifactDenoisingConfig(
        transient_method=transient_method if transient_method != "none" else None,
        wiener_n_components=wiener_n_components,
        wiener_window_duration=wiener_window_duration,
        wiener_noise_percentile=wiener_noise_percentile,
        wiener_max_iter=wiener_max_iter,
        autoreject_max_chunk_minutes=autoreject_max_chunk_minutes,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    denoised_raw, _provenance = run_residual_denoising(raw, config)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=denoised_raw,
                writer=lambda path, r=denoised_raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

"""ICA artifact correction node."""

from __future__ import annotations

import os

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node
def ica_artifact_correction(
    mne_object,
    n_components: int = 20,
    remove_eog: bool = True,
    remove_ecg: bool = True,
    random_state: int = 42,
) -> NodeResult:
    """Remove physiological artifacts using ICA.

    Fits ICA on bandpass-filtered copy (1-100 Hz), then applies to original.
    EOG uses frontal channels (Fp1/Fp2 or first 2 channels) as proxy.
    ECG uses cardiac-channel heuristic or skips silently if no ECG found.
    """
    import mne as _mne
    from mne.preprocessing import ICA
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    if isinstance(raw, _mne.BaseEpochs):
        raw_for_ica = raw.copy()
        filt = raw_for_ica.filter(1.0, 100.0, verbose=False)
        ica = ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(filt)
        if remove_eog:
            try:
                eog_inds, _ = ica.find_bads_eog(raw_for_ica)
                ica.exclude.extend(eog_inds)
            except (RuntimeError, ValueError):
                pass
        cleaned = ica.apply(raw.copy(), verbose=False)
    else:
        filt = raw.copy().filter(1.0, 100.0, verbose=False)
        ica = ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(filt)

        if remove_eog:
            try:
                eog_inds, _ = ica.find_bads_eog(raw)
                ica.exclude.extend(eog_inds)
            except (RuntimeError, ValueError):
                pass
        if remove_ecg:
            try:
                ecg_inds, _ = ica.find_bads_ecg(raw)
                ica.exclude.extend(ecg_inds)
            except (RuntimeError, ValueError):
                pass

        cleaned = ica.apply(raw.copy(), verbose=False)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=cleaned,
                writer=lambda path, r=cleaned: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

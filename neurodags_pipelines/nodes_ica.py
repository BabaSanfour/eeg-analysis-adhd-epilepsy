"""ICA artifact correction nodes."""

from __future__ import annotations

import os
from pathlib import Path

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

    picks = _mne.pick_types(raw.info, eeg=True, exclude="bads")
    n_components = min(n_components, len(picks))

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


@register_node
def source_correction(
    mne_object,
    eog_method: str = "dss",
    ecg_method: str = "dss",
    emg_method: str = "mwf",
    ica_n_components: int = 20,
    random_state: int = 42,
    dss_n_components: int = 10,
    dss_n_remove_eog: int = 1,
    dss_n_remove_ecg: int = 1,
    dss_n_remove_emg: int = 2,
    mwf_n_components: int = 30,
) -> NodeResult:
    """DSS+MWF artifact correction (EOG/ECG via DSS, EMG via MWF).

    Port of eeg_adhd_epilepsy.preproc.correct.run_source_correction.
    Replaces basic find_bads_eog/ecg with profile-based DSS and adds MWF for EMG.

    Auto-tuning: if @CleanedPrepRaw_prov.json exists alongside the input fif,
    its integrity_stats are passed as artifact_profile to enable adaptive
    component-count boosting when autoreject_bad_fraction > 0.15.
    """
    import json
    from neurodags.loaders import load_meeg
    from eeg_adhd_epilepsy.preproc.correct import ArtifactCorrectionConfig, run_source_correction

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    subject_id = "unknown"
    output_dir = None
    artifact_profile: dict = {}
    if raw.filenames and raw.filenames[0]:
        rec_path = Path(raw.filenames[0])
        subject_id = rec_path.stem
        try:
            from eeg_adhd_epilepsy.io import bids as bids_io
            ids = bids_io.build_bids_report_ids(rec_path)
            subject_id = str(ids["run_prefix"])
        except Exception:
            pass
        output_dir = rec_path.parent
        # Load CleanedPrepRaw provenance for auto-tuning (same dir, _prov.json suffix).
        prov_path = rec_path.parent / (rec_path.stem + "_prov.json")
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
                artifact_profile = prov.get("integrity_stats", {})
            except Exception:
                pass

    config = ArtifactCorrectionConfig(
        eog_method=eog_method if eog_method != "none" else None,
        ecg_method=ecg_method if ecg_method != "none" else None,
        emg_method=emg_method if emg_method != "none" else None,
        ica_n_components=ica_n_components,
        random_state=random_state,
        dss_n_components=dss_n_components,
        dss_n_remove_eog=dss_n_remove_eog,
        dss_n_remove_ecg=dss_n_remove_ecg,
        dss_n_remove_emg=dss_n_remove_emg,
        mwf_n_components=mwf_n_components,
    )

    corrected_raw, _provenance = run_source_correction(
        raw,
        config,
        output_dir=output_dir,
        subject_id=subject_id,
        artifact_profile=artifact_profile,
    )

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=corrected_raw,
                writer=lambda path, r=corrected_raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

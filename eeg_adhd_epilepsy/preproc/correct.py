"""Source Correction Module (Stage 1).

Implements removal of specific physiological artifacts (EOG, ECG, EMG) using
DSS (Denoising Source Separation), ICA (Independent Component Analysis), and
other targeted methods.

This module is designed to work on MNE Raw objects, optionally leveraging
annotations from `base.py` to exclude bad segments during model fitting.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import scipy.linalg
import sys
import argparse
import json
from pathlib import Path

from .utils import benchmark_step, NumpyEncoder
from .base import _collect_block_windows
from eeg_adhd_epilepsy.reports.correct import create_correction_report
from eeg_adhd_epilepsy.features.spectral import compute_spectral_metrics
from eeg_adhd_epilepsy.utils.logs import setup_logging
from eeg_adhd_epilepsy.io import bids

LOGGER = logging.getLogger(__name__)

from mne_denoise.dss import DSS, IterativeDSS, AverageBias, BandpassBias
from mne_denoise.viz import (
    plot_score_curve,
    plot_component_summary,
    plot_component_time_series,
    plot_evoked_comparison,
    plot_overlay_comparison,
    plot_psd_comparison,
    plot_spatial_patterns,
    plot_spectral_psd_comparison,
    plot_time_course_comparison,
)

from mne_denoise.dss.denoisers import QuasiPeriodicDenoiser

from mne_icalabel import label_components
import pywt


@dataclass
class ArtifactCorrectionConfig:
    """Configuration for Stage 1 Artifact Correction."""
    
    # EOG Removal (Part 5 vs Part 2)
    eog_method: Optional[str] = "dss"  # 'dss', 'ica', 'blind', None
    
    # ECG Removal (Part 5 vs Part 2)
    ecg_method: Optional[str] = "dss"  # 'dss', 'ica', 'quasiperiodic', None
    
    # EMG Removal (Part 2.1 options + Part 5)
    emg_method: Optional[str] = "mwf"  # 'mwf', 'wica', 'ica', 'dss', None
    
    # Shared ICA Parameters (when ICA is used)
    ica_n_components: int = 20
    exclude_probability: float = 0.8
    
    # DSS Parameters (when DSS is used)
    dss_n_components: int = 10
    dss_emg_n_remove: int = 2  # For DSS-EMG: how many components to remove
    
    # MWF Parameters (Part 2.1, RELAX)
    mwf_n_components: int = 30
    
    # wICA Parameters (Part 2.1, RELAX-Jr)
    wavelet_type: str = 'db4'
    wavelet_level: int = 5
    
    # random state for reproducibility
    random_state: int = 42


def _save_eeg_snapshot(
    raw: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    label: str,
    start: float = 5.0,
    duration: float = 30.0,
    n_channels: int = 20
) -> str:
    """Save a butterfly plot snapshot of EEG channels.
    
    Captures `duration` seconds of raw EEG data as a static butterfly plot.
    Starts at `start` seconds to skip initial transients.
    """
    import matplotlib.pyplot as plt
    
    raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    # Ensure start is within bounds
    max_start = max(0, raw_eeg.times[-1] - duration)
    start = min(start, max_start)
    ch_names = raw_eeg.ch_names[:min(n_channels, len(raw_eeg.ch_names))]
    stop = min(start + duration, raw_eeg.times[-1])
    sfreq = raw_eeg.info['sfreq']
    data, times = raw_eeg[ch_names, int(start * sfreq):int(stop * sfreq)]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    data_uv = data * 1e6
    for i, ch in enumerate(ch_names):
        ax.plot(times[:data.shape[1]], data_uv[i], linewidth=0.4, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'{subject_id} — {label.replace("_", " ").title()}')
    ax.set_xlim(times[0], times[min(data.shape[1]-1, len(times)-1)])
    ax.grid(True, alpha=0.3)
    
    path = fig_dir / f'{subject_id}_{label}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def _save_artifact_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    fig_dir: Path,
    subject_id: str,
    artifact_type: str,
    window: float = 3.0,
    n_channels: int = 8
) -> str:
    """Save a 3-panel before/after/removed comparison at the largest artifact peak.
    
    Finds the time of maximum artifact removal (largest absolute difference),
    then plots a window around that peak showing the Original, Cleaned,
    and what was Removed.
    
    For EOG: uses frontal channels to detect blinks.
    For ECG/EMG: uses the channel with the largest removed power.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    eeg_before = raw_before.copy().pick_types(eeg=True, exclude='bads')
    eeg_after = raw_after.copy().pick_types(eeg=True, exclude='bads')
    sfreq = eeg_before.info['sfreq']
    
    # Compute the removed signal
    data_before = eeg_before.get_data()
    data_after = eeg_after.get_data()
    
    # Ensure same shape
    n_samples = min(data_before.shape[1], data_after.shape[1])
    data_before = data_before[:, :n_samples]
    data_after = data_after[:, :n_samples]
    removed = data_before - data_after
    
    # Choose channels to display and find peaks
    ch_names = eeg_before.ch_names
    if artifact_type == 'eog':
        # Prefer frontal channels for EOG
        frontal = [i for i, ch in enumerate(ch_names)
                   if any(f in ch.upper() for f in ['FP1', 'FP2', 'F3', 'F4', 'FZ', 'AF'])]
        if not frontal:
            frontal = list(range(min(4, len(ch_names))))
        # Find blink peaks using envelope of frontal removed signal
        envelope = np.abs(removed[frontal]).mean(axis=0)
        display_idx = frontal[:n_channels]
    else:
        # For ECG/EMG — find channel with max removed power
        ch_power = np.sum(removed ** 2, axis=1)
        top_ch = np.argsort(ch_power)[::-1][:n_channels]
        envelope = np.abs(removed[top_ch[0]])
        display_idx = list(top_ch)
    
    # Find peaks in the envelope
    min_dist = int(0.5 * sfreq)  # at least 0.5s apart
    peaks, properties = find_peaks(envelope, distance=min_dist, height=np.percentile(envelope, 95))
    
    if len(peaks) == 0:
        # Fallback: use the point of maximum absolute difference
        peak_idx = int(np.argmax(envelope))
    else:
        # Use the tallest peak
        peak_idx = peaks[np.argmax(properties['peak_heights'])]
    
    # Convert to time
    peak_time = peak_idx / sfreq
    half_win = window / 2
    t_start = max(0, peak_time - half_win)
    t_end = min(n_samples / sfreq, peak_time + half_win)
    s_start = int(t_start * sfreq)
    s_end = int(t_end * sfreq)
    times = np.arange(s_start, s_end) / sfreq
    
    display_names = [ch_names[i] for i in display_idx]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    artifact_label = artifact_type.upper()
    colors = plt.cm.tab10(np.linspace(0, 1, len(display_idx)))
    
    for panel_idx, (ax, title_sfx, signal) in enumerate([
        (axes[0], 'Before (Original)', data_before),
        (axes[1], 'After (Cleaned)', data_after),
        (axes[2], f'Removed ({artifact_label} Artifact)', removed),
    ]):
        for j, ch_idx in enumerate(display_idx):
            sig_uv = signal[ch_idx, s_start:s_end] * 1e6
            ax.plot(times[:len(sig_uv)], sig_uv, linewidth=0.6, alpha=0.8,
                    color=colors[j], label=display_names[j] if panel_idx == 0 else None)
        ax.set_ylabel('µV')
        ax.set_title(f'{title_sfx}', fontsize=11)
        ax.grid(True, alpha=0.3)
        # Mark the peak time
        ax.axvline(peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[2].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right', fontsize=7, ncol=min(4, len(display_idx)))
    
    fig.suptitle(f'{subject_id} — {artifact_label} Artifact Removal (peak at {peak_time:.2f}s)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = fig_dir / f'{subject_id}_{artifact_type}_artifact_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def run_source_correction(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    condition_name: Optional[str] = None,
    fit_segments: Optional[List[Tuple[float, float]]] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Orchestrate Stage 1 artifact correction.
    
    Args:
        raw: The raw data object. Corrected version is returned.
        config: Configuration for artifact correction.
        condition_name: If provided, only process blocks matching this condition.
                        The returned raw will contain ONLY the processed data for this condition.
        fit_segments: Optional list of (onset, duration) tuples defining segments to use 
                      for model fitting (ICA/DSS). If None, fits on the data being corrected.
                      Useful for training on stable 'Rest' blocks and applying to task blocks.
        output_dir: Directory to save plots.
        subject_id: Subject identifier for plot filenames.
    
    Returns:
        (corrected_raw, provenance_dict)
    """
    provenance = {
        "steps_completed": [],
        "correction_stats": {},
        "timings": {},
        "condition_name": condition_name,
        "fit_segments_used": fit_segments is not None
    }
    
    # 1. Determine Target Data (Data to be Corrected)
    if condition_name:
        LOGGER.info(f" Selecting data for condition: {condition_name}")
        all_blocks = _collect_block_windows(raw)
        cond_blocks = [b for b in all_blocks if b.name == condition_name]
        
        if not cond_blocks:
            LOGGER.warning(f"No blocks found for condition '{condition_name}'. Returning original raw.")
            return raw, provenance
        
        crops = []
        for b in cond_blocks:
             if b.stop > b.onset:
                crops.append(raw.copy().crop(b.onset, b.stop, include_tmax=False))
        
        if not crops:
             LOGGER.warning(f"No valid data found for condition '{condition_name}'.")
             return raw, provenance
             
        corrected_raw = mne.concatenate_raws(crops) # This is a copy
        LOGGER.info(f"Created concatenated raw for '{condition_name}': {corrected_raw.times[-1]:.2f}s")
    else:
        # Work on full raw (copy for safety)
        corrected_raw = raw.copy()

    # 2. Determine Training Data (Data to Fit Models)
    raw_fit = None
    if fit_segments:
        # Create concatenated raw from fit_segments
        fit_crops = []
        for onset, duration in fit_segments:
            stop = onset + duration
            # Ensure valid bounds
            if stop > raw.times[-1]: 
                stop = raw.times[-1]
                duration = stop - onset
            if duration <= 0: continue
            
            fit_crops.append(raw.copy().crop(onset, stop, include_tmax=False))
            
        if fit_crops:
            raw_fit = mne.concatenate_raws(fit_crops)
            LOGGER.info(f"Created training raw from {len(fit_segments)} segments ({raw_fit.times[-1]:.2f}s)")
        else:
            LOGGER.warning("fit_segments provided but no valid data extracted. Fallback to target data.")
            raw_fit = None # Will fallback to corrected_raw inside functions
    
    # If raw_fit is still None, functions default to using corrected_raw for training.
    
    # Extract bad segments to exclude from fitting (if fitting on corrected_raw)
    # If fitting on raw_fit, ideally it should also respect bad segments within it.
    # The helper extracts from annotations.
    # MNE's reject_by_annotation handles this if annotations are present.
    # When we crop/concatenate, annotations are preserved.
    bad_segments = _extract_bad_segments(corrected_raw)
    provenance["n_bad_segments"] = len(bad_segments)
    
    # EEG Snapshots — before/after each correction step
    eeg_snapshots = {}
    artifact_comparisons = {}
    snap_dir = None
    if output_dir:
        snap_dir = output_dir / 'figures' / 'eeg_snapshots'
        snap_dir.mkdir(parents=True, exist_ok=True)
        try:
            eeg_snapshots['before_correction'] = _save_eeg_snapshot(
                corrected_raw, snap_dir, subject_id, 'before_correction')
        except Exception as e:
            LOGGER.warning(f"EEG snapshot failed (before_correction): {e}")
    
    # Step 1.1: EOG (Eye) Removal
    if config.eog_method:
        raw_before_eog = corrected_raw.copy()
        with benchmark_step("eog_removal", provenance):
            s = {}
            if config.eog_method == "dss":
                corrected_raw, s = _remove_eog_dss(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
                if s.get('skipped'):
                    LOGGER.info("DSS EOG skipped. Falling back to Blind DSS.")
                    corrected_raw, s = _remove_eog_blind(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.eog_method == "ica":
                corrected_raw, s = _remove_eog_ica(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.eog_method == "blind":
                corrected_raw, s = _remove_eog_blind(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            provenance["correction_stats"]["eog"] = s
        provenance["steps_completed"].append("eog_removal")
        if snap_dir:
            try:
                eeg_snapshots['after_eog'] = _save_eeg_snapshot(
                    corrected_raw, snap_dir, subject_id, 'after_eog')
            except Exception as e:
                LOGGER.warning(f"EEG snapshot failed (after_eog): {e}")
            try:
                artifact_comparisons['eog'] = _save_artifact_comparison(
                    raw_before_eog, corrected_raw, snap_dir, subject_id, 'eog')
            except Exception as e:
                LOGGER.warning(f"Artifact comparison failed (eog): {e}")
    
    # Step 1.2: ECG (Heart) Removal
    if config.ecg_method:
        raw_before_ecg = corrected_raw.copy()
        with benchmark_step("ecg_removal", provenance):
            s = {}
            if config.ecg_method == "dss":
                corrected_raw, s = _remove_ecg_dss(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
                if s.get('skipped'):
                    LOGGER.info("DSS ECG skipped. Falling back to QuasiPeriodic Denoiser.")
                    corrected_raw, s = _remove_ecg_quasiperiodic(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.ecg_method == "ica":
                corrected_raw, s = _remove_ecg_ica(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.ecg_method == "quasiperiodic":
                corrected_raw, s = _remove_ecg_quasiperiodic(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            provenance["correction_stats"]["ecg"] = s
        provenance["steps_completed"].append("ecg_removal")
        if snap_dir:
            try:
                eeg_snapshots['after_ecg'] = _save_eeg_snapshot(
                    corrected_raw, snap_dir, subject_id, 'after_ecg')
            except Exception as e:
                LOGGER.warning(f"EEG snapshot failed (after_ecg): {e}")
            try:
                artifact_comparisons['ecg'] = _save_artifact_comparison(
                    raw_before_ecg, corrected_raw, snap_dir, subject_id, 'ecg')
            except Exception as e:
                LOGGER.warning(f"Artifact comparison failed (ecg): {e}")

    # Step 1.3: EMG (Muscle) Removal
    if config.emg_method:
        raw_before_emg = corrected_raw.copy()
        with benchmark_step("emg_removal", provenance):
            s = {}
            if config.emg_method == "mwf":
                cleaned, s = _remove_emg_mwf(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
                if cleaned is not None:
                     corrected_raw = cleaned
            elif config.emg_method == "wica":
                corrected_raw, s = _remove_emg_wica(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.emg_method == "ica":
                corrected_raw, s = _remove_emg_ica(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            elif config.emg_method == "dss":
                corrected_raw, s = _remove_emg_dss(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            provenance["correction_stats"]["emg"] = s
        provenance["steps_completed"].append("emg_removal")
        if snap_dir:
            try:
                eeg_snapshots['after_emg'] = _save_eeg_snapshot(
                    corrected_raw, snap_dir, subject_id, 'after_emg')
            except Exception as e:
                LOGGER.warning(f"EEG snapshot failed (after_emg): {e}")
            try:
                artifact_comparisons['emg'] = _save_artifact_comparison(
                    raw_before_emg, corrected_raw, snap_dir, subject_id, 'emg')
            except Exception as e:
                LOGGER.warning(f"Artifact comparison failed (emg): {e}")
        
    provenance['eeg_snapshots'] = eeg_snapshots
    provenance['artifact_comparisons'] = artifact_comparisons
        
    return corrected_raw, provenance


def _extract_bad_segments(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """Helper: Extract bad segment timestamps from base.py annotations."""
    return [
        (a['onset'], a['duration']) 
        for a in raw.annotations 
        if a['description'].startswith('BAD_')
    ]


# ------------------------------------------------------------------------------
# EOG (Eye) Removal Functions
# ------------------------------------------------------------------------------

def _remove_eog_dss(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove eye blinks using DSS with blink event bias."""
        
    from mne_denoise.dss import DSS, AverageBias
    from mne.preprocessing import create_eog_epochs

    train_raw = raw_fit if raw_fit is not None else raw
    
    # 1. Create EOG epochs on training data
    # 1. Create EOG epochs on training data
    try:
        eog_epochs = create_eog_epochs(
            train_raw, 
            baseline=(-0.5, -0.2), 
            tmin=-0.5, 
            tmax=0.5,
            reject_by_annotation=True,
            verbose="ERROR"
        )
    except RuntimeError:
        # Fallback: Use Frontal channels if available
        eog_ch = None
        if 'Fp1' in train_raw.ch_names:
            eog_ch = 'Fp1'
        elif 'Fp2' in train_raw.ch_names:
            eog_ch = 'Fp2'
        elif 'Fpz' in train_raw.ch_names:
            eog_ch = 'Fpz'
            
        if eog_ch:
            LOGGER.info(f"No EOG channels found. Using {eog_ch} for blink detection.")
            try:
                eog_epochs = create_eog_epochs(
                    train_raw,
                    ch_name=eog_ch,
                    baseline=(-0.5, -0.2), 
                    tmin=-0.5, 
                    tmax=0.5,
                    reject_by_annotation=True,
                    verbose="ERROR"
                )
            except Exception as e:
                LOGGER.warning(f"Blink detection on {eog_ch} failed: {e}")
                return raw, {'skipped': True, 'reason': 'blink_detection_failed'}
        else:
            LOGGER.warning("No EOG or Frontal channels (Fp1/Fp2/Fpz) found, skipping EOG-DSS.")
            return raw, {'skipped': True, 'reason': 'no_eog_or_frontal'}
    
    if len(eog_epochs) < 5:
        LOGGER.warning("Too few blinks detected (<5), skipping EOG-DSS")
        return raw, {'n_blinks': len(eog_epochs), 'skipped': True}
    
    eog_epochs.pick_types(eeg=True, eog=False, exclude='bads')
    
    # 2. Fit DSS with Trial Average Bias
    dss = DSS(n_components=config.dss_n_components, bias=AverageBias(axis='epochs'))
    dss.fit(eog_epochs.get_data())
    
    # PLOTTING
    plot_paths = {}
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_eog'
            fig_dir.mkdir(parents=True, exist_ok=True)
            eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude='bads'))

            # 1. Score curve
            fig_score = plot_score_curve(dss, show=False)
            score_path = fig_dir / f'{subject_id}_eog_score.png'
            fig_score.savefig(score_path, dpi=150, bbox_inches='tight')
            plt.close(fig_score)
            plot_paths['score_curve'] = str(score_path)

            # 2. Component Summary (Top 3)
            fig_comp = plot_component_summary(dss, eog_epochs.get_data(), info=eeg_info, n_components=3, show=False)
            comp_path = fig_dir / f'{subject_id}_eog_comps.png'
            fig_comp.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close(fig_comp)
            plot_paths['component_summary'] = str(comp_path)

            # 3. Spatial Patterns (topomaps)
            fig_topo = plot_spatial_patterns(dss, info=eeg_info, n_components=3, show=False)
            topo_path = fig_dir / f'{subject_id}_eog_topo.png'
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            plt.close(fig_topo)
            plot_paths['spatial_patterns'] = str(topo_path)

            # 4. Component Time Series (stacked traces)
            fig_ts = plot_component_time_series(dss, eog_epochs.get_data(), n_components=5, show=False)
            ts_path = fig_dir / f'{subject_id}_eog_timeseries.png'
            fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ts)
            plot_paths['component_time_series'] = str(ts_path)

            # 5. Evoked Comparison (GFP before/after)
            sources_viz = dss.transform(eog_epochs.get_data())
            n_remove = config.dss_eog_n_remove or 1
            if sources_viz.shape[1] >= n_remove:
                sources_viz[:, :n_remove, :] = 0
            clean_data = dss.inverse_transform(sources_viz)
            clean_epochs = eog_epochs.copy()
            clean_epochs._data = clean_data

            fig_evoked = plot_evoked_comparison(eog_epochs, clean_epochs, show=False)
            evoked_path = fig_dir / f'{subject_id}_eog_evoked.png'
            fig_evoked.savefig(evoked_path, dpi=150, bbox_inches='tight')
            plt.close(fig_evoked)
            plot_paths['evoked_comparison'] = str(evoked_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in EOG-DSS: {e}")

    # 3. Transform Target
    target_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    # Use config.dss_psd_threshold if implemented, else hardcode
    
    # dss.transform expects (n_epochs, n_channels, n_times) if fitted on epochs,
    # or (n_channels, n_times) if fitted on continuous data.
    # Here, dss was fitted on eog_epochs.get_data() which is (n_epochs, n_channels, n_times).
    # To transform continuous raw data, we need to pass it as (1, n_channels, n_times)
    # or ensure the DSS object can handle 2D continuous data directly.
    # MNE-denoise DSS, when fitted on epochs, can transform 2D continuous data.
    # It applies the spatial filters (n_comp, n_chan) to (n_chan, n_time) -> (n_comp, n_time).
    
    sources = dss.transform(target_eeg.get_data()) # (n_comp, n_samples)
    
    n_remove = config.dss_eog_n_remove or 1
    if sources.shape[0] >= n_remove:
         # Zero out blink components (first ones)
         sources[:n_remove, :] = 0
         
    cleaned_data = dss.inverse_transform(sources)
    
    try:
        raw_out = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        raw_out._data[picks, :] = cleaned_data
    except Exception as e:
        return raw, {'error': str(e)}
    
    # Post-cleaning comparison plots (need raw_out)
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_eog'
            fig_dir.mkdir(parents=True, exist_ok=True)

            # 6. PSD Comparison (before/after)
            fig_psd = plot_psd_comparison(raw, raw_out, fmax=50, show=False)
            psd_path = fig_dir / f'{subject_id}_eog_psd.png'
            fig_psd.savefig(psd_path, dpi=150, bbox_inches='tight')
            plt.close(fig_psd)
            plot_paths['psd_comparison'] = str(psd_path)

            # 7. Overlay Comparison (first 5 seconds)
            fig_ov = plot_overlay_comparison(raw, raw_out, start=0.0, stop=5.0, title='EOG DSS: Signal Overlay (0-5s)', show=False)
            ov_path = fig_dir / f'{subject_id}_eog_overlay.png'
            fig_ov.savefig(ov_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ov)
            plot_paths['overlay_comparison'] = str(ov_path)

            # 8. Time Course Comparison (first 5 channels)
            fig_tc = plot_time_course_comparison(raw, raw_out, start=0, stop=int(5 * raw.info['sfreq']), show=False)
            tc_path = fig_dir / f'{subject_id}_eog_timecourse.png'
            fig_tc.savefig(tc_path, dpi=150, bbox_inches='tight')
            plt.close(fig_tc)
            plot_paths['time_course_comparison'] = str(tc_path)
        except Exception as e:
            LOGGER.warning(f"Post-cleaning plots failed in EOG-DSS: {e}")

    return raw_out, {
        'method': 'dss',
        'n_components_removed': n_remove,
        'bias_type': 'blink_average',
        'n_blinks': len(eog_epochs),
        'plot_paths': plot_paths
    }


def _remove_eog_ica(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig,
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove eye artifacts using ICA + ICLabel."""
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    train_raw = raw_fit if raw_fit is not None else raw
    
    # 1. Fit ICA
    ica = ICA(
        n_components=config.ica_n_components, 
        method='fastica',
        max_iter='auto',
        random_state=config.random_state,
        verbose="ERROR"
    )
    # MNE's ICA.fit respects annotations if reject_by_annotation=True is passed.
    ica.fit(train_raw, reject_by_annotation=True)
    
    # 2. Classify
    labels = label_components(train_raw, ica, method='iclabel')
    
    # 3. Exclude 'eye' components
    exclude_idx = []
    probas = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'eye' and prob > config.exclude_probability:
            exclude_idx.append(i)
            probas.append(prob)
            
    ica.exclude = exclude_idx
    
    # 4. Apply to target
    raw_clean = ica.apply(raw.copy())
    
    # No plotting specified for ICA in the instruction, so plot_paths will be empty.
    plot_paths = {}

    return raw_clean, {
        'method': 'ica',
        'n_components_removed': len(exclude_idx),
        'probabilities': probas,
        'total_components': config.ica_n_components,
        'plot_paths': plot_paths
    }


def _remove_eog_blind(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove eye artifacts using Blind DSS (Kurtosis/Tanh)."""
    from mne_denoise.dss import IterativeDSS
    from mne_denoise.dss.denoisers import KurtosisDenoiser
    
    train_raw = raw_fit if raw_fit is not None else raw
    train_eeg = train_raw.copy().pick_types(eeg=True, exclude='bads')
    
    # Fit Kurtosis-DSS (FastICA-like)
    # Use 2D data to avoid mne-denoise 3D bugs
    train_data = train_eeg.get_data()
    dss = IterativeDSS(
        denoiser=KurtosisDenoiser(nonlinearity="cube"),
        method="deflation",
        n_components=config.dss_n_components,
        beta=-3.0
    )
    dss.fit(train_data)
    
    # PLOTTING
    plot_paths = {}
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_blind_eog'
            fig_dir.mkdir(parents=True, exist_ok=True)
            eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude='bads'))

            # 1. Score curve
            fig_score = plot_score_curve(dss, show=False)
            if fig_score is not None:
                score_path = fig_dir / f'{subject_id}_blind_score.png'
                fig_score.savefig(score_path, dpi=150, bbox_inches='tight')
                plt.close(fig_score)
                plot_paths['score_curve'] = str(score_path)

            # 2. Spatial Patterns (topomaps)
            fig_topo = plot_spatial_patterns(dss, info=eeg_info, n_components=3, show=False)
            topo_path = fig_dir / f'{subject_id}_blind_topo.png'
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            plt.close(fig_topo)
            plot_paths['spatial_patterns'] = str(topo_path)

            # 3. Component Time Series (first 60s for speed)
            n_plot = min(train_data.shape[1], int(raw.info['sfreq'] * 60))
            fig_ts = plot_component_time_series(dss, train_data[:, :n_plot], n_components=5, show=False)
            ts_path = fig_dir / f'{subject_id}_blind_timeseries.png'
            fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ts)
            plot_paths['component_time_series'] = str(ts_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in Blind DSS: {e}")

    # Transform Target (2D)
    target_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    if target_eeg.ch_names != train_eeg.ch_names:
         return raw, {'skipped': True, 'error': 'channel mismatch'}

    target_data = target_eeg.get_data()
    sources = dss.transform(target_data) # (n_comp, n_times)
    
    # Remove first component (Blink)
    if sources.shape[0] > 0:
        sources[0, :] = 0

    cleaned_data = dss.inverse_transform(sources) # Returns (n_ch, n_times)
    
    try:
        raw_out = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        raw_out._data[picks, :] = cleaned_data
    except Exception as e:
        return raw, {'error': str(e)}

    return raw_out, {'method': 'blind_dss', 'n_components_removed': 1, 'plot_paths': plot_paths}


# ------------------------------------------------------------------------------
# ECG (Heart) Removal Functions
# ------------------------------------------------------------------------------

def _remove_ecg_dss(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove heartbeat using DSS with QRS event bias."""
    
    from mne_denoise.dss import DSS, AverageBias
    from mne.preprocessing import create_ecg_epochs
    
    train_raw = raw_fit if raw_fit is not None else raw
    
    # 1. Create ECG epochs (bias = QRS timing)
    try:
        ecg_epochs = create_ecg_epochs(
            train_raw, 
            baseline=(-0.2, -0.05), 
            tmin=-0.3, 
            tmax=0.3,
            reject_by_annotation=True,
            verbose="ERROR"
        )
    except Exception as e:
        LOGGER.warning(f"ECG event detection failed ({type(e).__name__}), skipping ECG-DSS.")
        return raw, {'skipped': True, 'reason': 'no_ecg_channel'}
    
    if len(ecg_epochs) < 10:
        LOGGER.warning("Too few QRS events detected, skipping ECG-DSS")
        return raw, {'n_qrs': 0, 'skipped': True}
    
    ecg_epochs.pick_types(eeg=True, ecg=False, exclude='bads')
    
    # 2. Fit DSS
    dss = DSS(n_components=config.dss_n_components, bias=AverageBias(axis='epochs'))
    dss.fit(ecg_epochs.get_data())
    
    # PLOTTING
    plot_paths = {}
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_ecg'
            fig_dir.mkdir(parents=True, exist_ok=True)
            eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude='bads'))

            # 1. Score curve
            fig_score = plot_score_curve(dss, show=False)
            score_path = fig_dir / f'{subject_id}_ecg_score.png'
            fig_score.savefig(score_path, dpi=150, bbox_inches='tight')
            plt.close(fig_score)
            plot_paths['score_curve'] = str(score_path)

            # 2. Component Summary (Top 3)
            fig_comp = plot_component_summary(dss, ecg_epochs.get_data(), info=eeg_info, n_components=3, show=False)
            comp_path = fig_dir / f'{subject_id}_ecg_comps.png'
            fig_comp.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close(fig_comp)
            plot_paths['component_summary'] = str(comp_path)

            # 3. Spatial Patterns (topomaps)
            fig_topo = plot_spatial_patterns(dss, info=eeg_info, n_components=3, show=False)
            topo_path = fig_dir / f'{subject_id}_ecg_topo.png'
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            plt.close(fig_topo)
            plot_paths['spatial_patterns'] = str(topo_path)

            # 4. Component Time Series (stacked traces)
            fig_ts = plot_component_time_series(dss, ecg_epochs.get_data(), n_components=5, show=False)
            ts_path = fig_dir / f'{subject_id}_ecg_timeseries.png'
            fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ts)
            plot_paths['component_time_series'] = str(ts_path)

            # 5. Evoked Comparison (GFP before/after)
            sources_viz = dss.transform(ecg_epochs.get_data())
            n_remove = config.dss_ecg_n_remove or 1
            if sources_viz.shape[1] >= n_remove:
                sources_viz[:, :n_remove, :] = 0
            clean_data = dss.inverse_transform(sources_viz)
            clean_epochs = ecg_epochs.copy()
            clean_epochs._data = clean_data

            fig_evoked = plot_evoked_comparison(ecg_epochs, clean_epochs, show=False)
            evoked_path = fig_dir / f'{subject_id}_ecg_evoked.png'
            fig_evoked.savefig(evoked_path, dpi=150, bbox_inches='tight')
            plt.close(fig_evoked)
            plot_paths['evoked_comparison'] = str(evoked_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in ECG-DSS: {e}")

    # 3. Transform Target Raw
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, exclude='bads')
    
    if raw_eeg.ch_names != ecg_epochs.ch_names:
         LOGGER.warning("Channel mismatch between fit and target in ECG-DSS. Skipping.")
         return raw, {'skipped': True, 'error': 'channel mismatch'}

    data_continuous = raw_eeg.get_data() # (n_channels, n_times)
    data_reshaped = data_continuous[np.newaxis, :, :] # (1, n_ch, n_times)
    
    sources = dss.transform(data_reshaped) # (1, n_comp, n_times)
    
    n_remove = config.dss_ecg_n_remove or 1
    # Zero cardiac component
    if sources.shape[1] >= n_remove:
        sources[:, :n_remove, :] = 0  
    
    cleaned_data = dss.inverse_transform(sources)[0]
    
    # 4. Create new Raw with cleaned EEG data
    try:
        raw_out = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        raw_out._data[picks, :] = cleaned_data
    except Exception as e:
         LOGGER.error(f"Failed to merge clean data: {e}")
         return raw, {'error': str(e)}

    # Post-cleaning comparison plots (need raw_out)
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_ecg'
            fig_dir.mkdir(parents=True, exist_ok=True)

            # 6. PSD Comparison (before/after)
            fig_psd = plot_psd_comparison(raw, raw_out, fmax=50, show=False)
            psd_path = fig_dir / f'{subject_id}_ecg_psd.png'
            fig_psd.savefig(psd_path, dpi=150, bbox_inches='tight')
            plt.close(fig_psd)
            plot_paths['psd_comparison'] = str(psd_path)

            # 7. Overlay Comparison (first 5 seconds)
            fig_ov = plot_overlay_comparison(raw, raw_out, start=0.0, stop=5.0, title='ECG DSS: Signal Overlay (0-5s)', show=False)
            ov_path = fig_dir / f'{subject_id}_ecg_overlay.png'
            fig_ov.savefig(ov_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ov)
            plot_paths['overlay_comparison'] = str(ov_path)

            # 8. Time Course Comparison (first 5 channels)
            fig_tc = plot_time_course_comparison(raw, raw_out, start=0, stop=int(5 * raw.info['sfreq']), show=False)
            tc_path = fig_dir / f'{subject_id}_ecg_timecourse.png'
            fig_tc.savefig(tc_path, dpi=150, bbox_inches='tight')
            plt.close(fig_tc)
            plot_paths['time_course_comparison'] = str(tc_path)
        except Exception as e:
            LOGGER.warning(f"Post-cleaning plots failed in ECG-DSS: {e}")

    return raw_out, {
        'method': 'dss', 
        'n_qrs': len(ecg_epochs), 
        'n_components_removed': n_remove,
        'plot_paths': plot_paths
    }


def _remove_ecg_ica(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig,
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove cardiac artifacts using ICA + ICLabel."""
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    train_raw = raw_fit if raw_fit is not None else raw
    
    ica = ICA(n_components=config.ica_n_components, method='fastica', max_iter='auto', random_state=config.random_state, verbose="ERROR")
    ica.fit(train_raw, reject_by_annotation=True)
    
    labels = label_components(train_raw, ica, method='iclabel')
    
    exclude_idx = []
    probas = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'heart' and prob > config.exclude_probability:
            exclude_idx.append(i)
            probas.append(prob)
            
    ica.exclude = exclude_idx
    raw_clean = ica.apply(raw.copy())
    
    plot_paths = {}

    return raw_clean, {
        'method': 'ica', 
        'n_components_removed': len(exclude_idx),
        'probabilities': probas,
        'total_components': config.ica_n_components,
        'plot_paths': plot_paths
    }


def _remove_ecg_quasiperiodic(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove cardiac artifacts using QuasiPeriodicDenoiser template matching."""
    from mne_denoise.dss import IterativeDSS
    from mne_denoise.dss.denoisers import QuasiPeriodicDenoiser
    
    train_raw = raw_fit if raw_fit is not None else raw
    
    sfreq = train_raw.info['sfreq']
    
    qp_denoiser = QuasiPeriodicDenoiser(
        peak_distance=int(0.5 * sfreq),  # 120 BPM max
        peak_height_percentile=85,
        smooth_template=True
    )
    
    # Fit (using 2D data)
    train_eeg = train_raw.copy().pick_types(eeg=True, exclude='bads')
    idss = IterativeDSS(denoiser=qp_denoiser, n_components=config.dss_n_components, max_iter=5)
    train_data = train_eeg.get_data()
    idss.fit(train_data)
    
    # Transform Target (2D)
    target_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    if target_eeg.ch_names != train_eeg.ch_names:
         return raw, {'skipped': True, 'error': 'channel mismatch'}

    target_data = target_eeg.get_data()
    sources = idss.transform(target_data)
    
    # Zero cardiac component (first one usually, as QP maximizes periodicity)
    if sources.shape[0] > 0:
        sources[0, :] = 0
        
    cleaned_data = idss.inverse_transform(sources)
    
    try:
        raw_out = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        raw_out._data[picks, :] = cleaned_data
    except Exception as e:
         return raw, {'error': str(e)}

    # PLOTTING (after cleaning so we can do before/after)
    plot_paths = {}
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_quasiperiodic_ecg'
            fig_dir.mkdir(parents=True, exist_ok=True)
            eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude='bads'))

            # 1. Spatial Patterns (topomaps)
            fig_topo = plot_spatial_patterns(idss, info=eeg_info, n_components=3, show=False)
            if fig_topo is not None:
                topo_path = fig_dir / f'{subject_id}_qp_ecg_topo.png'
                fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
                plt.close(fig_topo)
                plot_paths['spatial_patterns'] = str(topo_path)

            # 2. Component Time Series (first 60s of 2D data)
            n_plot = min(train_data.shape[1], int(sfreq * 60))
            fig_ts = plot_component_time_series(idss, train_data[:, :n_plot], n_components=5, show=False)
            if fig_ts is not None:
                ts_path = fig_dir / f'{subject_id}_qp_ecg_timeseries.png'
                fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
                plt.close(fig_ts)
                plot_paths['component_time_series'] = str(ts_path)

            # 3. PSD Comparison (before/after cleaning)
            fig_psd = plot_psd_comparison(raw, raw_out, fmax=50, show=False)
            if fig_psd is not None:
                psd_path = fig_dir / f'{subject_id}_qp_ecg_psd.png'
                fig_psd.savefig(psd_path, dpi=150, bbox_inches='tight')
                plt.close(fig_psd)
                plot_paths['psd_comparison'] = str(psd_path)

            # 4. Overlay Comparison (first 5 seconds)
            fig_ov = plot_overlay_comparison(raw, raw_out, start=0.0, stop=5.0,
                                            title='ECG QuasiPeriodic: Signal Overlay (0-5s)', show=False)
            if fig_ov is not None:
                ov_path = fig_dir / f'{subject_id}_qp_ecg_overlay.png'
                fig_ov.savefig(ov_path, dpi=150, bbox_inches='tight')
                plt.close(fig_ov)
                plot_paths['overlay_comparison'] = str(ov_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in QuasiPeriodic DSS: {e}")

    return raw_out, {'method': 'quasiperiodic', 'n_components_removed': 1, 'plot_paths': plot_paths}


# ------------------------------------------------------------------------------
# EMG (Muscle) Removal Functions
# ------------------------------------------------------------------------------

def _remove_emg_mwf(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[Optional[mne.io.BaseRaw], Dict]:
    """Remove muscle artifacts using Multi-Channel Wiener Filter (MWF)."""
    
    # 1. Extract muscle annotations from base.py
    muscle_annot = [a for a in raw.annotations if 'muscle' in a['description']]
    if len(muscle_annot) == 0:
        LOGGER.warning("No muscle annotations from base.py, skipping MWF")
        return None, {'skipped': True, 'reason': 'no_muscle_annotations'}
    
    # Simplified implementation based on strategy doc
    try:
        clean_mask = _get_clean_to_artifact_mask(raw, muscle_annot)
        artifact_mask = ~clean_mask
        
        data = raw.get_data(picks='eeg')
        if not np.any(clean_mask) or not np.any(artifact_mask):
             return None, {'skipped': True, 'reason': 'insufficient_segments'}

        C_signal = np.cov(data[:, clean_mask])
        C_artifact = np.cov(data[:, artifact_mask])
        
        # GEVD
        try:
            eigenvalues, W = scipy.linalg.eigh(C_signal, C_signal + C_artifact)
        except scipy.linalg.LinAlgError:
             LOGGER.warning("GEVD failed (singular matrix?), skipping MWF.")
             return None, {'skipped': True, 'reason': 'linalg_error'}
        
        # Keep components
        n_keep = config.mwf_n_components
        if n_keep >= len(eigenvalues):
            n_keep = len(eigenvalues) - 1
            
        # Reconstruct (using separate W for reconstruction if needed, or simplified projection)
        # Strategy doc code:
        # data_clean = W[:, -n_keep:].T @ data
        # raw._data = W[:, -n_keep:] @ data_clean
        # This assumes W is orthogonal which satisfies reconstruction.
        
        W_keep = W[:, -n_keep:]
        data_clean = W_keep.T @ data
        reconstructed = W_keep @ data_clean
        
        raw_out = raw.copy()
        picks = mne.pick_types(raw_out.info, eeg=True, exclude=[])
        raw_out._data[picks, :] = reconstructed
        
        plot_paths = {} # No standard plot for MWF established yet
        
        return raw_out, {
            'method': 'mwf',
            'n_components_kept': n_keep,
            'n_muscle_segments': len(muscle_annot),
            'plot_paths': plot_paths
        }
        
    except Exception as e:
        LOGGER.warning(f"MWF failed: {e}")
        return None, {'error': str(e)}


def _remove_emg_wica(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove muscle artifacts using Wavelet-ICA."""
    # Placeholder for wICA
    return raw, {'skipped': True, 'reason': 'not_implemented'}


def _remove_emg_ica(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig,
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove muscle artifacts using Standard ICA + ICLabel."""
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    train_raw = raw_fit if raw_fit is not None else raw
    
    ica = ICA(n_components=config.ica_n_components, method='fastica', max_iter='auto', random_state=config.random_state, verbose="ERROR")
    ica.fit(train_raw, reject_by_annotation=True)
    
    labels = label_components(train_raw, ica, method='iclabel')
    
    exclude_idx = []
    probas = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'muscle' and prob > config.exclude_probability:
            exclude_idx.append(i)
            probas.append(prob)
            
    ica.exclude = exclude_idx
    raw_clean = ica.apply(raw.copy())
    
    plot_paths = {}
    if output_dir and 'plot_ica_components' in globals() and 'plot_ica_sources' in globals():
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'ica_emg'
            fig_dir.mkdir(parents=True, exist_ok=True)

            # Plot components
            fig_comp = plot_ica_components(ica, raw, show=False)
            comp_path = fig_dir / f'{subject_id}_emg_ica_components.png'
            fig_comp.savefig(comp_path)
            plt.close(fig_comp)
            plot_paths['ica_components'] = str(comp_path)

            # Plot sources
            fig_sources = plot_ica_sources(ica, raw, show=False)
            sources_path = fig_dir / f'{subject_id}_emg_ica_sources.png'
            fig_sources.savefig(sources_path)
            plt.close(fig_sources)
            plot_paths['ica_sources'] = str(sources_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in EMG-ICA: {e}")

    return raw_clean, {
        'method': 'ica', 
        'n_components_removed': len(exclude_idx),
        'probabilities': probas,
        'total_components': config.ica_n_components,
        'plot_paths': plot_paths
    }




def _remove_emg_dss(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    bad_segments: List[Tuple[float, float]],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Remove muscle artifacts using DSS with high-frequency power bias."""
    
    train_raw = raw_fit if raw_fit is not None else raw
    # train_eeg = train_raw.copy().pick_types(eeg=True, exclude='bads') # Not needed if we pass array to fit
    
    # Use BandpassBias to emphasize high-frequency (>30Hz) activity
    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2
    high_freq = nyquist - 1.0 # Safety margin
    
    bias = BandpassBias(freq_band=(30, high_freq), sfreq=sfreq)
    
    dss = DSS(n_components=config.dss_n_components, bias=bias)
    
    # Fit on training data (2D: n_ch, n_samples) — exclude bads for consistency
    train_eeg = train_raw.copy().pick_types(eeg=True, exclude='bads')
    train_data = train_eeg.get_data()
    dss.fit(train_data)
    
    # PLOTTING
    plot_paths = {}
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_emg'
            fig_dir.mkdir(parents=True, exist_ok=True)
            eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude='bads'))
            n_samples_plot = min(train_data.shape[1], int(sfreq * 60))

            # 1. Score curve
            fig_score = plot_score_curve(dss, show=False)
            score_path = fig_dir / f'{subject_id}_emg_score.png'
            fig_score.savefig(score_path, dpi=150, bbox_inches='tight')
            plt.close(fig_score)
            plot_paths['score_curve'] = str(score_path)

            # 2. Component Summary (Top 3, 60s slice)
            fig_comp = plot_component_summary(dss, train_data[:, :n_samples_plot], info=eeg_info, n_components=3, show=False)
            comp_path = fig_dir / f'{subject_id}_emg_comps.png'
            fig_comp.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close(fig_comp)
            plot_paths['component_summary'] = str(comp_path)

            # 3. Spatial Patterns (topomaps)
            fig_topo = plot_spatial_patterns(dss, info=eeg_info, n_components=3, show=False)
            topo_path = fig_dir / f'{subject_id}_emg_topo.png'
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            plt.close(fig_topo)
            plot_paths['spatial_patterns'] = str(topo_path)

            # 4. Component Time Series (60s slice)
            fig_ts = plot_component_time_series(dss, train_data[:, :n_samples_plot], n_components=5, show=False)
            ts_path = fig_dir / f'{subject_id}_emg_timeseries.png'
            fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ts)
            plot_paths['component_time_series'] = str(ts_path)

            # 5. Spectral PSD Comparison (component vs original PSDs)
            sources_viz = dss.transform(train_data[:, :n_samples_plot])
            fig_spsd = plot_spectral_psd_comparison(
                mne.io.RawArray(train_data[:, :n_samples_plot], eeg_info),
                sources_viz, sfreq=sfreq, fmin=1, fmax=min(80, sfreq/2 - 1), show=False
            )
            spsd_path = fig_dir / f'{subject_id}_emg_spectral_psd.png'
            fig_spsd.savefig(spsd_path, dpi=150, bbox_inches='tight')
            plt.close(fig_spsd)
            plot_paths['spectral_psd_comparison'] = str(spsd_path)

        except Exception as e:
            LOGGER.warning(f"Plotting failed in EMG-DSS: {e}")

    # 4. Transform Target
    target_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    sources = dss.transform(target_eeg.get_data()) # (n_comp, n_samples)
    
    n_remove = config.dss_emg_n_remove or 2
    if sources.shape[0] >= n_remove:
        sources[:n_remove, :] = 0  # Zero out muscle components (first ones)
    
    cleaned_data = dss.inverse_transform(sources)
    
    try:
        raw_out = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        raw_out._data[picks, :] = cleaned_data
    except Exception as e:
        return raw, {'error': str(e)}
    
    # Post-cleaning comparison plots (need raw_out)
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            fig_dir = output_dir / 'figures' / 'dss_emg'
            fig_dir.mkdir(parents=True, exist_ok=True)

            # 6. PSD Comparison (before/after) — especially useful for EMG (HF reduction)
            fig_psd = plot_psd_comparison(raw, raw_out, fmax=min(100, sfreq/2 - 1), show=False)
            psd_path = fig_dir / f'{subject_id}_emg_psd.png'
            fig_psd.savefig(psd_path, dpi=150, bbox_inches='tight')
            plt.close(fig_psd)
            plot_paths['psd_comparison'] = str(psd_path)

            # 7. Overlay Comparison (first 5 seconds)
            fig_ov = plot_overlay_comparison(raw, raw_out, start=0.0, stop=5.0, title='EMG DSS: Signal Overlay (0-5s)', show=False)
            ov_path = fig_dir / f'{subject_id}_emg_overlay.png'
            fig_ov.savefig(ov_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ov)
            plot_paths['overlay_comparison'] = str(ov_path)
        except Exception as e:
            LOGGER.warning(f"Post-cleaning plots failed in EMG-DSS: {e}")

    return raw_out, {
        'method': 'dss',
        'n_components_removed': n_remove,
        'bias_type': 'high_frequency_bandpass',
        'plot_paths': plot_paths
    }


def _get_clean_to_artifact_mask(raw, annotations):
    """Helper stub."""
    n_samples = raw.n_times
    mask_clean = np.ones(n_samples, dtype=bool)
    for annot in annotations:
        start = raw.time_as_index(annot['onset'])[0]
        end = start + int(annot['duration'] * raw.info['sfreq'])
        mask_clean[start:end] = False
    return mask_clean


# ------------------------------------------------------------------------------
# Pipeline Runner & CLI
# ------------------------------------------------------------------------------

def run_correction_pipeline(
    subject_id: str,
    bids_root: Path,
    config: ArtifactCorrectionConfig,
    output_dir: Optional[Path] = None,
    condition_name: Optional[str] = None,
    train_condition: Optional[str] = None,
    output_desc: str = "correct"
) -> bool:
    """Run the artifact correction pipeline on a subject.
    
    Args:
        subject_id: Subject ID (e.g. 'sub-001').
        bids_root: Path to BIDS dataset root.
        config: Correction configuration.
        output_dir: Directory to save results (default: BIDS derivatives).
        condition_name: Optional condition to process (e.g. 'task-rest').
        train_condition: Optional condition to use for training (e.g. 'task-rest'). 
                         If provided, segments from this condition are used to fit cleanup models.
        output_desc: BIDS desc- entity for output filename (default: 'correct').
                     Use e.g. 'correctDss' or 'correctIca' for comparison runs.
    """
    try:
        # Resolve Paths
        derivatives_root = bids_root / "derivatives" / "preproc"
        subj_deriv_dir = derivatives_root / subject_id / "eeg"
        
        # Input: Output of Base Pipeline
        # Pattern: sub-001_desc-base_eeg.fif
        input_fname = f"{subject_id}_desc-base_eeg.fif"
        input_path = subj_deriv_dir / input_fname
        
        if not input_path.exists():
            LOGGER.error(f"Input file not found: {input_path}")
            return False
            
        LOGGER.info(f"Loading base pipeline output: {input_path}")
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
        
        # Resolve Output Directory
        if output_dir is None:
            output_dir = derivatives_root # Standard BIDS structure
            
        out_subj_dir = output_dir / subject_id / "eeg"
        out_subj_dir.mkdir(parents=True, exist_ok=True)
        
        # Reports Directory
        # Match base.py structure: qc/preproc/subjects_reports_{output_desc}
        report_suffix = output_desc.replace('correct', 'correct_').rstrip('_') if output_desc != 'correct' else 'correct'
        qc_root = output_dir.parent.parent / "qc" / "preproc" / f"subjects_reports_{report_suffix}"
        qc_root.mkdir(parents=True, exist_ok=True)

        # 0. Pre-Correction PSD
        LOGGER.info("Computing pre-correction spectral metrics...")
        psd_before = (np.array([]), np.array([]))
        _, psd_pre_data, freqs_pre, _, _, _ = compute_spectral_metrics(
             raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_before = (freqs_pre, psd_pre_data)

        # 1. Resolve Train Condition (Fit Segments)
        fit_segments = None
        if train_condition:
            LOGGER.info(f"Extracting training segments from condition: {train_condition}")
            windows = _collect_block_windows(raw)
            train_blocks = [b for b in windows if b.name == train_condition]
            if train_blocks:
                fit_segments = [(b.onset, b.duration) for b in train_blocks]
                LOGGER.info(f"Found {len(fit_segments)} segments for training.")
            else:
                LOGGER.warning(f"Train condition '{train_condition}' not found. Fitting on target data.")

        # 2. Run Correction
        LOGGER.info("Starting artifact correction...")
        corrected_raw, provenance = run_source_correction(
            raw, 
            config, 
            condition_name=condition_name, 
            fit_segments=fit_segments,
            output_dir=output_dir,
            subject_id=subject_id
        )
        
        # Add basic info to provenance
        provenance["subject_id"] = subject_id
        provenance["input_file"] = str(input_path)
        provenance["train_condition"] = train_condition

        # 3. Post-Correction PSD
        LOGGER.info("Computing post-correction spectral metrics...")
        psd_after = (np.array([]), np.array([]))
        _, psd_post_data, freqs_post, _, _, _ = compute_spectral_metrics(
             corrected_raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_after = (freqs_post, psd_post_data)

        # 4. Save Outputs
        out_fname = f"{subject_id}_desc-{output_desc}_eeg.fif"
        if condition_name:
             # If processed specific condition, append to filename
             safe_cond = condition_name.lower().replace(" ", "")
             out_fname = f"{subject_id}_task-{safe_cond}_desc-{output_desc}_eeg.fif"
             
        out_path = out_subj_dir / out_fname
        prov_path = out_subj_dir / out_fname.replace("_eeg.fif", "_provenance.json")
        
        LOGGER.info(f"Saving corrected raw to {out_path}")
        corrected_raw.save(out_path, overwrite=True, verbose="ERROR")
        
        with open(prov_path, "w") as f:
            json.dump(provenance, f, cls=NumpyEncoder, indent=2)
            
        # 5. Generate Report
        create_correction_report(
            subject_id=subject_id,
            raw=corrected_raw,
            psd_before=psd_before,
            psd_after=psd_after,
            provenance=provenance,
            output_dir=qc_root
        )
        
        LOGGER.info(f"Correction pipeline completed for {subject_id}")
        return True

    except Exception as e:
        LOGGER.error(f"Failed correction for {subject_id}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 Artifact Correction")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument("--output_dir", type=str, help="Custom output directory")
    
    # Selection Arguments
    parser.add_argument("--subjects", nargs="+", help="List of specific subjects to process (e.g. sub-001 sub-002)")
    parser.add_argument("--start-from", type=str, help="Start processing from this subject ID (alphabetical)")
    parser.add_argument("--all", action="store_true", help="Process all subjects found in BIDS")
    parser.add_argument("--test", action="store_true", help="Run on a small subset (5 subjects) for testing")
    parser.add_argument("--random", action="store_true", help="Select random subjects for testing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects with existing output")
    
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Checkbox args for quick config override
    parser.add_argument("--eog-method", type=str, default="dss", choices=["dss", "ica", "none"], help="EOG removal method")
    parser.add_argument("--ecg-method", type=str, default="dss", choices=["dss", "ica", "quasiperiodic", "none"], help="ECG removal method")
    parser.add_argument("--emg-method", type=str, default="mwf", choices=["mwf", "wica", "ica", "dss", "none"], help="EMG removal method")
    parser.add_argument("--output-desc", type=str, default="correct", help="BIDS desc entity for output (e.g. correctDss, correctIca)")
    
    parser.add_argument("--condition", type=str, help="Process only specific condition")
    parser.add_argument("--train-condition", type=str, help="Condition to use for model training (e.g. Rest)")
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir) if args.output_dir else bids_root / "derivatives" / "preproc"
    
    # Setup Logging
    log_file = output_dir / "logs" / "correct_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")
    
    if not bids_root.exists():
        LOGGER.error(f"BIDS root not found: {bids_root}")
        sys.exit(1)
        
    # Load/Create Config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ArtifactCorrectionConfig(**config_dict)
    else:
        config = ArtifactCorrectionConfig(
            eog_method=args.eog_method if args.eog_method != "none" else None,
            ecg_method=args.ecg_method if args.ecg_method != "none" else None,
            emg_method=args.emg_method if args.emg_method != "none" else None
        )
    
    LOGGER.info(f"Running Correction with Config: {config}")

    # Discover files (Stage 0 Output) to find subjects
    LOGGER.info("Scanning preproc directory for available subjects...")
    preproc_dir = bids_root / "derivatives" / "preproc"
    
    if not preproc_dir.exists():
        LOGGER.error(f"Preproc directory not found: {preproc_dir}")
        sys.exit(1)
        
    # Use BIDS discovery on the derivatives folder
    # This finds sub-XXX_..._eeg.fif
    files = bids.discover_bids_files(preproc_dir, suffix="eeg", extension=".fif")
    
    if not files:
        LOGGER.error(f"No .fif files found in {preproc_dir} via BIDS discovery.")
        sys.exit(1)
        
    file_map = {}
    for f in files:
        sid = bids.parse_subject_id(f)
        file_map[f] = sid
        
    subjects_found = sorted(list(set(file_map.values())))
    LOGGER.info(f"Found {len(subjects_found)} unique subjects in BIDS directory.")
    
    # Filter Logic
    subjects_to_process = set()
    
    if args.subjects:
        subjects_to_process = set(args.subjects)
        LOGGER.info(f"Selected specific subjects: {args.subjects}")
        
    elif args.start_from:
        start_sub = args.start_from
        subjects_to_process = {s for s in subjects_found if s >= start_sub}
        if not subjects_to_process:
            LOGGER.error(f"No subjects found starting from {start_sub}.")
            sys.exit(1)
        LOGGER.info(f"Resuming from {start_sub}, selected {len(subjects_to_process)} subjects.")
            
    elif args.test:
        import random
        if args.random:
            random.seed(42)
            subjects_to_process = set(random.sample(subjects_found, min(5, len(subjects_found))))
            LOGGER.info(f"Test mode: selected 5 random subjects: {sorted(subjects_to_process)}")
        else:
            subjects_to_process = set(subjects_found[:5])
            LOGGER.info(f"Test mode: selected first 5 subjects.")
        
    elif args.all:
        subjects_to_process = set(subjects_found)
        LOGGER.info(f"Processing all {len(subjects_found)} subjects.")
        
    else:
        LOGGER.warning("No selection criteria provided (use --all, --test, --subjects, or --start-from).")
        parser.print_help()
        sys.exit(0)
        
    # Apply --skip-existing filter
    if args.skip_existing:
        LOGGER.info("Checking for existing correction output to skip...")
        # Check for _desc-correct_eeg.fif or provenance ? 
        # Usually stage 1 output is _desc-correct_eeg.fif
        
        subjects_to_skip = set()
        for sid in subjects_to_process:
            # Check if output exists
            # We assume output layout matches input: derivatives/preproc/sub-X/eeg/...
            # or custom output_dir
            out_subj_dir = output_dir / sid / "eeg"
            out_file = out_subj_dir / f"{sid}_desc-{args.output_desc}_eeg.fif"
            if out_file.exists():
                subjects_to_skip.add(sid)
        
        if subjects_to_skip:
            LOGGER.info(f"Skipping {len(subjects_to_skip)} already processed subjects.")
            subjects_to_process = subjects_to_process - subjects_to_skip
    
    subjects_sorted = sorted(list(subjects_to_process))
    
    if not subjects_sorted:
        LOGGER.warning("No subjects left to process.")
        sys.exit(0)
        
    LOGGER.info(f"Starting processing for {len(subjects_sorted)} subjects...")
    
    # Processing Loop
    success_count = 0
    fail_count = 0
    
    for sub in subjects_sorted:
        LOGGER.info(f"Processing {sub}...")
        try:
            success = run_correction_pipeline(
                subject_id=sub,
                bids_root=bids_root,
                config=config,
                output_dir=output_dir,
                condition_name=args.condition,
                train_condition=args.train_condition,
                output_desc=args.output_desc
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
                LOGGER.error(f"Failed processing {sub}")
        except Exception as e:
            LOGGER.error(f"Exception processing {sub}: {e}")
            fail_count += 1
            
    LOGGER.info(f"Batch processing complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()

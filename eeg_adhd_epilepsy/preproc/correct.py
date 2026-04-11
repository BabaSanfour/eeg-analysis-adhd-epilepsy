"""Source Correction Module (Stage 1).

Implements removal of specific physiological artifacts (EOG, ECG, EMG) using
DSS (Denoising Source Separation), ICA (Independent Component Analysis), and
other targeted methods.

This module is designed to work on MNE Raw objects, optionally leveraging
annotations from `base.py` to exclude bad segments during model fitting.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import scipy.linalg
import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from eeg_adhd_epilepsy.viz import qc as viz_qc

from .utils import benchmark_step, NumpyEncoder, _collect_block_windows
from .dss_utils import _get_dss_profile, _run_dss_artifact
from .ica_utils import fit_ica_context, apply_ica_artifact
from eeg_adhd_epilepsy.reports.correct import create_correction_report, create_correction_dataset_report
from eeg_adhd_epilepsy.features.spectral import compute_spectral_metrics, compute_lsd, compute_aperiodic_slope
from eeg_adhd_epilepsy.utils.logs import setup_logging
from eeg_adhd_epilepsy.io import bids

LOGGER = logging.getLogger(__name__)


@dataclass
class ArtifactCorrectionConfig:
    """Configuration for Stage 1 Artifact Correction."""

    # Methods
    eog_method: Optional[str] = "dss"  # 'dss', 'ica', 'blind-dss', None
    ecg_method: Optional[str] = "dss"  # 'dss', 'ica', 'quasiperiodic', None
    emg_method: Optional[str] = "mwf"  # 'mwf', 'wica', 'ica', 'dss', None

    # Shared ICA Parameters
    ica_n_components: int = 20
    ica_exclude_prob: float = 0.8

    # DSS Parameters
    dss_n_components: int = 10
    dss_n_remove_eog: int = 1
    dss_n_remove_ecg: int = 1
    dss_n_remove_emg: int = 2
    
    # Blind DSS / Adaptive Parameters
    blind_nonlinearity: str = "cube"  # 'cube', 'tanh', 'gauss', 'smooth_tanh'
    blind_alpha: float = 1.0
    blind_smooth_window: int = 10

    # MWF Parameters
    mwf_n_components: int = 30

    # wICA Parameters (Placeholder)
    wavelet_type: str = "db4"
    wavelet_level: int = 5

    # General
    random_state: int = 42


def run_source_correction(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    condition_name: Optional[str] = None,
    fit_segments: Optional[List[Tuple[float, float]]] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown",
    artifact_profile: Optional[Dict] = None
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Orchestrate Stage 1 artifact correction.
    
    Args:
        raw: The raw data object. Corrected version is returned.
        config: Configuration for artifact correction.
        condition_name: If provided, only process blocks matching this condition.
        fit_segments: Optional list of (onset, duration) tuples defining segments to use 
                      for model fitting (ICA/DSS). If None, fits on the data being corrected.
        output_dir: Directory to save plots.
        subject_id: Subject identifier for plot filenames.
        artifact_profile: Optional dictionary from Base stage provenance to guide auto-tuning.
    
    Returns:
        (corrected_raw, provenance_dict)
    """
    eog_method = config.eog_method
    ecg_method = config.ecg_method
    emg_method = config.emg_method

    provenance = {
        "steps_completed": [],
        "correction_stats": {},
        "benchmarks": {"timing": {}},
        "condition_name": condition_name,
        "fit_segments_used": fit_segments is not None,
        "methods": {
            "eog": eog_method,
            "ecg": ecg_method,
            "emg": emg_method,
        },
    }

    # Auto-Tuning Logic based on artifact_profile from Base Stage
    if artifact_profile:
        # 1. Muscle Load Tuning
        # Check if muscle was frequently detected in Base stage
        muscle_load = artifact_profile.get("autoreject_bad_fraction", 0) 
        if muscle_load > 0.15:
            LOGGER.info(f"Significant artifact load detected ({muscle_load:.1%}). Boosting correction aggression.")
            # Increase DSS removal if defaults were used
            if config.emg_method == "mwf" and config.mwf_n_components < 40:
                config.mwf_n_components = 40
            if config.eog_method == "dss" and config.dss_n_remove_eog == 1:
                config.dss_n_remove_eog = 2
                
        # 2. Add tuning influence to provenance for transparency
        provenance["base_profile_influence"] = {
            "tuning_applied": muscle_load > 0.15,
            "muscle_load": float(muscle_load),
            "adjusted_mwf_n": config.mwf_n_components,
            "adjusted_eog_n": config.dss_n_remove_eog
        }
    
    # 0. Initialize plot/snapshot tracking
    eeg_snapshots: Dict[str, str] = {}
    artifact_comparisons: Dict[str, str] = {}
    snap_dir = output_dir / "figures" if output_dir else None
    
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

    bad_segments = _extract_bad_segments(corrected_raw)
    provenance["n_bad_segments"] = len(bad_segments)

    ica_context: Optional[Dict[str, Any]] = None

    def _get_ica_context(current_raw: mne.io.BaseRaw) -> Dict[str, Any]:
        nonlocal ica_context
        if ica_context is None:
            fit_raw = raw_fit if raw_fit is not None else current_raw
            ica_context = fit_ica_context(fit_raw, config)
            LOGGER.info("Computed shared ICA + ICLabel context.")
        return ica_context
    
    provenance["eeg_snapshots"] = eeg_snapshots
    provenance["artifact_comparisons"] = artifact_comparisons

    # Artifact Correction Orchestration using Registry
    artifacts = [
        ("eog", eog_method, ("eye blink", "eye")),
        ("ecg", ecg_method, ("heart beat", "heart")),
        ("emg", emg_method, ("muscle artifact", "muscle")),
    ]

    for art_type, method, ica_labels in artifacts:
        if not method or method.lower() == "none":
            continue

        raw_before = corrected_raw.copy()
        with benchmark_step(f"{art_type}_removal", provenance):
            LOGGER.info(f"--- Running {art_type.upper()} removal using {method} ---")
            stats = {}
            
            if method == "ica":
                corrected_raw, stats = apply_ica_artifact(
                    corrected_raw,
                    _get_ica_context(corrected_raw),
                    target_labels=ica_labels,
                    exclude_probability=config.ica_exclude_prob,
                    output_dir=output_dir,
                    subject_id=subject_id,
                    artifact_label=art_type.upper()
                )
            elif method in ("dss", "blind-dss", "quasiperiodic"):
                # Profile-based DSS (EOG, ECG, or EMG)
                profile = _get_dss_profile(art_type, method, config, float(corrected_raw.info["sfreq"]))
                corrected_raw, stats = _run_dss_artifact(
                    corrected_raw,
                    config,
                    profile,
                    raw_fit=raw_fit,
                    output_dir=output_dir,
                    subject_id=subject_id,
                )
                # Specialized fallback for EOG dss
                if art_type == "eog" and method == "dss" and stats.get("skipped"):
                    LOGGER.info("DSS EOG skipped. Falling back to Blind DSS.")
                    blind_prof = _get_dss_profile("eog", "blind-dss", config, float(corrected_raw.info["sfreq"]))
                    corrected_raw, stats = _run_dss_artifact(
                        corrected_raw, config, blind_prof, raw_fit=raw_fit, output_dir=output_dir, subject_id=subject_id
                    )
                # Specialized fallback for ECG dss
                elif art_type == "ecg" and method == "dss" and stats.get("skipped"):
                     LOGGER.info("DSS ECG skipped. Falling back to QuasiPeriodic Denoiser.")
                     qp_prof = _get_dss_profile("ecg", "quasiperiodic", config, float(corrected_raw.info["sfreq"]))
                     corrected_raw, stats = _run_dss_artifact(
                         corrected_raw, config, qp_prof, raw_fit=raw_fit, output_dir=output_dir, subject_id=subject_id
                     )
            elif art_type == "emg" and method == "mwf":
                cleaned, stats = _remove_emg_mwf(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
                if cleaned is not None:
                    corrected_raw = cleaned
            elif art_type == "emg" and method == "wica":
                corrected_raw, stats = _remove_emg_wica(corrected_raw, config, bad_segments, raw_fit, output_dir, subject_id)
            else:
                LOGGER.warning(f"Unknown {art_type} method '{method}'; skipping.")
                stats = {"skipped": True, "reason": "unknown_method", "method": method}

            provenance["correction_stats"][art_type] = stats
            provenance["steps_completed"].append(f"{art_type}_removal")

        if snap_dir and not stats.get("skipped"):
            eeg_snapshots[f"after_{art_type}"] = viz_qc.save_eeg_snapshot(
                corrected_raw, snap_dir, subject_id, f"after_{art_type}")
            artifact_comparisons[art_type] = viz_qc.save_artifact_comparison(
                raw_before, corrected_raw, snap_dir, subject_id, art_type)
            
            # New: Removed Variance Topomap
            topo_path = snap_dir / f"{subject_id}_{art_type}_removed_variance_topo.png"
            fig_topo = viz_qc.plot_removed_variance_topomap(
                raw_before, corrected_raw, title=f"Removed {art_type.upper()} Variance")
            if fig_topo:
                fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
                plt.close(fig_topo)
                # Store in plot_paths for reporting
                if "plot_paths" not in stats:
                    stats["plot_paths"] = {}
                stats["plot_paths"]["removed_variance_topo"] = str(topo_path)

    return corrected_raw, provenance
        
    return corrected_raw, provenance


def _extract_bad_segments(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """Helper: Extract bad segment timestamps from base.py annotations."""
    return [
        (a['onset'], a['duration']) 
        for a in raw.annotations 
        if a['description'].startswith('BAD_')
    ]


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
    preproc_root: Optional[Path] = None,
    reports_root: Optional[Path] = None,
    condition_name: Optional[str] = None,
    train_condition: Optional[str] = None,
    output_desc: str = "correct"
) -> bool:
    """Run the artifact correction pipeline on a subject.
    
    Args:
        subject_id: Subject ID (e.g. 'sub-001').
        bids_root: Path to BIDS dataset root.
        config: Correction configuration.
        preproc_root: Root directory of stage FIF/provenance artifacts.
        reports_root: Root directory for reports/logs.
        condition_name: Optional condition to process (e.g. 'task-rest').
        train_condition: Optional condition to use for training (e.g. 'task-rest'). 
                         If provided, segments from this condition are used to fit cleanup models.
        output_desc: BIDS desc- entity for output filename (default: 'correct').
                     Use e.g. 'correctDss' or 'correctIca' for comparison runs.
    """
    try:
        subject_id = bids.normalize_subject_id(subject_id)
        output_desc = bids.validate_stage_desc(output_desc)
        bids_root = Path(bids_root).expanduser()

        preproc_root = bids.get_preproc_root(
            bids_root=bids_root,
            preproc_root=Path(preproc_root).expanduser() if preproc_root is not None else None,
        )
        reports_root = bids.get_reports_root(
            bids_root=bids_root,
            reports_root=Path(reports_root).expanduser() if reports_root is not None else None,
        )

        # Stage 0 -> Stage 1 handoff
        input_path = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc="base",
        )
        
        if not input_path.exists():
            LOGGER.error(f"Input file not found: {input_path}")
            return False
            
        LOGGER.info(f"Loading base pipeline output: {input_path}")
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")

        subject_report_path = bids.get_subject_report_path(
            reports_root=reports_root,
            stage="correct",
            subject_id=subject_id,
            create_dir=True,
        )
        figures_dir = subject_report_path.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        report_path = subject_report_path
        if output_desc != "correct":
            report_path = subject_report_path.with_name(
                f"{subject_id}_correct_{output_desc}_report.html"
            )

        # 0. Pre-Correction PSD
        LOGGER.info("Computing pre-correction spectral metrics...")
        psd_before = (np.array([]), np.array([]))
        _, psd_pre_data, freqs_pre, _, _, _ = compute_spectral_metrics(
             raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_before = (freqs_pre, psd_pre_data)

        # Signal Snapshot (Before)
        snapshot_pre_path = viz_qc.save_eeg_snapshot(
            raw, figures_dir, subject_id, "before_correction"
        )
        eeg_snapshots = {"before_correction": str(snapshot_pre_path)}
        artifact_comparisons = {}

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

        # 1b. Load Base Artifact Profile for Auto-Tuning
        base_prov = _load_base_provenance(subject_id, preproc_root)
        artifact_profile = {}
        if base_prov:
            artifact_profile = base_prov.get("integrity_stats", {})
            LOGGER.info(f"Loaded Base stage artifact profile for {subject_id}")

        # 2. Run Correction
        LOGGER.info("Starting artifact correction orchestration...")
        corrected_raw, provenance = run_source_correction(
            raw,
            config,
            condition_name=condition_name,
            fit_segments=fit_segments,
            output_dir=subject_report_path.parent,
            subject_id=subject_id,
            artifact_profile=artifact_profile,
        )
        
        # Merge initial snapshots into returned provenance
        provenance.setdefault("eeg_snapshots", {}).update(eeg_snapshots)
        provenance.setdefault("artifact_comparisons", {}).update(artifact_comparisons)
        
        task_token = condition_name if condition_name else None
        out_path = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=output_desc,
            task=task_token,
            create_dir=True,
        )
        prov_path = bids.get_stage_provenance_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=output_desc,
            task=task_token,
            create_dir=True,
        )

        # Enrich provenance schema.
        provenance["subject_id"] = subject_id
        provenance["input_file"] = str(input_path)
        provenance["output_file"] = str(out_path)
        provenance["provenance_file"] = str(prov_path)
        provenance["preproc_root"] = str(preproc_root)
        provenance["reports_root"] = str(reports_root)
        provenance["input_desc"] = "base"
        provenance["output_desc"] = output_desc
        provenance["condition_name"] = condition_name
        provenance["train_condition"] = train_condition

        # 3. Post-Correction PSD
        LOGGER.info("Computing post-correction spectral metrics...")
        _, psd_post_data, freqs_post, alpha_peak_post, _, _ = compute_spectral_metrics(
             corrected_raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_after = (freqs_post, psd_post_data)
        
        # Spectral Comparison Metrics
        lsd_val = compute_lsd(psd_post_data, psd_pre_data)
        slope_val, _, _, _ = compute_aperiodic_slope(psd_post_data, freqs_post)
        
        provenance["spectral_stats"] = {
            "alpha_peak": float(alpha_peak_post),
            "aperiodic_slope": float(slope_val),
            "lsd": float(lsd_val),
        }

        # 4. Diagnostic Plots
        # Signal Snapshot (Butterfly)
        snapshot_path = viz_qc.save_eeg_snapshot(
            corrected_raw, figures_dir, subject_id, "after_correction"
        )
        provenance.setdefault("eeg_snapshots", {})["after_correction"] = str(snapshot_path)

        # Variance Comparison
        fig_var = viz_qc.plot_channel_variance_comparison(raw, corrected_raw, subject_id)
        var_path = figures_dir / f"{subject_id}_variance_comparison.png"
        fig_var.savefig(var_path, dpi=150, bbox_inches="tight")
        plt.close(fig_var)
        provenance.setdefault("artifact_comparisons", {})["variance_comparison"] = str(var_path)

        # 4. Save Outputs
        LOGGER.info(f"Saving corrected raw to {out_path}")
        corrected_raw.save(out_path, overwrite=True, verbose="ERROR")
        
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, cls=NumpyEncoder, indent=2)
            
        # 5. Generate Report
        create_correction_report(
            subject_id=subject_id,
            raw=corrected_raw,
            psd_before=psd_before,
            psd_after=psd_after,
            provenance=provenance,
            subject_report_path=report_path,
            figures_dir=figures_dir,
        )
        
        LOGGER.info("Correction pipeline completed for %s. Output: %s", subject_id, out_path)
        return True

    except Exception as e:
        LOGGER.error(f"Failed correction for {subject_id}: {e}", exc_info=True)
        return False


def _load_base_provenance(subject_id: str, preproc_root: Path) -> Optional[Dict]:
    """Helper: Load provenance from the base stage if available."""
    # Pattern: sub-XXX_desc-base_provenance.json (rglob to handle task/no-task subdirs)
    prov_files = list(preproc_root.rglob(f"{subject_id}_*desc-base_provenance.json"))
    if not prov_files:
        LOGGER.debug(f"No base provenance found for {subject_id}")
        return None
    try:
        with open(prov_files[0], "r") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load base provenance for {subject_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 Artifact Correction")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument(
        "--preproc_root",
        type=str,
        default=None,
        help="Directory for stage FIF/provenance artifacts (default: <bids_root>/derivatives/preproc)",
    )
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Directory for reports/logs (default: <cwd>/results/reports/preproc)",
    )
    
    # Selection Arguments
    parser.add_argument("--subjects", nargs="+", help="List of specific subjects to process (e.g. sub-001 sub-002)")
    parser.add_argument("--start-from", type=str, help="Start processing from this subject ID (alphabetical)")
    parser.add_argument("--all", action="store_true", help="Process all subjects found in BIDS")
    parser.add_argument("--test", action="store_true", help="Run on a small subset (5 subjects) for testing")
    parser.add_argument("--random", action="store_true", help="Select random subjects for testing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects with existing output")
    
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Checkbox args for quick config override
    parser.add_argument(
        "--eog-method",
        type=str,
        default="dss",
        choices=["dss", "ica", "blind-dss", "none"],
        help="EOG removal method",
    )
    parser.add_argument(
        "--blind-nonlinearity",
        type=str,
        default="cube",
        choices=["cube", "tanh", "gauss", "smooth_tanh"],
        help="Nonlinearity used when --eog-method=blind-dss",
    )
    parser.add_argument(
        "--blind-alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for blind-dss tanh/gauss nonlinearities",
    )
    parser.add_argument(
        "--blind-smooth-window",
        type=int,
        default=10,
        help="Smoothing window (samples) for blind-dss smooth_tanh nonlinearity",
    )
    parser.add_argument(
        "--dss-n-remove-eog",
        type=int,
        default=1,
        help="Number of leading DSS components to remove for EOG",
    )
    parser.add_argument("--ecg-method", type=str, default="dss", choices=["dss", "ica", "quasiperiodic", "none"], help="ECG removal method")
    parser.add_argument("--emg-method", type=str, default="mwf", choices=["mwf", "wica", "ica", "dss", "none"], help="EMG removal method")
    parser.add_argument("--output-desc", type=str, default="correct", help="BIDS desc entity for output (e.g. correctDss, correctIca)")
    
    parser.add_argument("--condition", type=str, help="Process only specific condition")
    parser.add_argument("--train-condition", type=str, help="Condition to use for model training (e.g. Rest)")
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root).expanduser()
    preproc_root_arg = Path(args.preproc_root).expanduser() if args.preproc_root else None
    preproc_root = bids.get_preproc_root(
        bids_root=bids_root,
        preproc_root=preproc_root_arg,
    )
    reports_root = bids.get_reports_root(
        bids_root=bids_root,
        reports_root=Path(args.reports_root).expanduser() if args.reports_root else None,
    )
    preproc_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging
    log_file = reports_root / "logs" / "correct_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")
    
    if not bids_root.exists():
        LOGGER.error(f"BIDS root not found: {bids_root}")
        sys.exit(1)
        
    # Load/Create Config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = ArtifactCorrectionConfig(**config_dict)
    else:
        config = ArtifactCorrectionConfig(
            eog_method=args.eog_method if args.eog_method != "none" else None,
            ecg_method=args.ecg_method if args.ecg_method != "none" else None,
            emg_method=args.emg_method if args.emg_method != "none" else None,
            blind_nonlinearity=args.blind_nonlinearity,
            blind_alpha=args.blind_alpha,
            blind_smooth_window=args.blind_smooth_window,
            dss_n_remove_eog=args.dss_n_remove_eog,
        )
    
    LOGGER.info(f"Running Correction with Config: {config}")

    # Discover files (Stage 0 Output) to find subjects
    LOGGER.info("Scanning preproc directory for available subjects...")
    preproc_dir = preproc_root
    
    if not preproc_dir.exists():
        LOGGER.error(f"Preproc directory not found: {preproc_dir}")
        sys.exit(1)

    # Only use Stage 0 outputs as Stage 1 inputs.
    files = sorted(preproc_dir.rglob("*_desc-base_eeg.fif"))
    
    if not files:
        LOGGER.error(f"No Stage 0 FIF files found in {preproc_dir} (pattern: *_desc-base_eeg.fif).")
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
        normalized_subjects = [bids.normalize_subject_id(s) for s in args.subjects]
        subjects_to_process = set(normalized_subjects)
        LOGGER.info(f"Selected specific subjects: {normalized_subjects}")
        
    elif args.start_from:
        start_sub = bids.normalize_subject_id(args.start_from)
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
        
        subjects_to_skip = set()
        for sid in subjects_to_process:
            out_file = bids.get_stage_output_path(
                subject_id=sid,
                preproc_root=preproc_root,
                desc=bids.validate_stage_desc(args.output_desc),
                task=args.condition if args.condition else None,
            )
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
    success_ids: List[str] = []
    failed_ids: List[str] = []
    
    for sub in subjects_sorted:
        LOGGER.info(f"Processing {sub}...")
        try:
            success = run_correction_pipeline(
                subject_id=sub,
                bids_root=bids_root,
                config=config,
                preproc_root=preproc_root,
                reports_root=reports_root,
                condition_name=args.condition,
                train_condition=args.train_condition,
                output_desc=args.output_desc
            )
            if success:
                success_count += 1
                success_ids.append(sub)
            else:
                fail_count += 1
                failed_ids.append(sub)
                LOGGER.error(f"Failed processing {sub}")
        except Exception as e:
            LOGGER.error(f"Exception processing {sub}: {e}")
            fail_count += 1
            failed_ids.append(sub)
            
    LOGGER.info(f"Batch processing complete. Success: {success_count}, Failed: {fail_count}")
    LOGGER.info("Succeeded subjects: %s", sorted(success_ids))
    LOGGER.info("Failed subjects: %s", sorted(failed_ids))
    
    # Generate Dataset-Level Summary Report
    summary_path = bids.get_stage_summary_report_path(
        reports_root=reports_root,
        stage="correct",
        create_dir=True,
    )
    output_desc_token = bids.validate_stage_desc(args.output_desc)
    if output_desc_token != "correct":
        summary_path = summary_path.with_name(
            f"correct_{output_desc_token}_dataset_summary.html"
        )
    
    LOGGER.info("Generating dataset-level correction report...")
    create_correction_dataset_report(
        search_dir=preproc_root,
        summary_report_path=summary_path,
        output_desc=output_desc_token,
        success_subjects=success_ids,
        failed_subjects=failed_ids,
    )

if __name__ == "__main__":
    main()

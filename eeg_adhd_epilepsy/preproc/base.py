"""EEG preprocessing pipeline base module.

This module provides the core pipeline for preprocessing EEG data using MNE-Python,
PyPREP, and AutoReject. The pipeline includes:

1.  Block Annotation: Loads segment definitions from CSV.
2.  Resampling: Adjusts sampling rate.
3.  Filtering: Applies high-pass and low-pass filters.
4.  Line Noise Removal: Detects and removes power line noise (ZapLine).
5.  Global Bad Channel Detection: Identifies broken channels using RANSAC.
6.  Reference: Applies Common Average Reference (CAR).
7.  Artifact Annotation: Detects and annotates bad epochs and channels using AutoReject,
    grouping processing by experimental condition for robust threshold estimation.

The pipeline is designed to be robust to non-stationarity by handling conditions separately
during artifact rejection and using extensive logging and provenance tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json
import numpy as np
import pandas as pd
import mne

from pyprep.find_noisy_channels import NoisyChannels
from autoreject import AutoReject 

from eeg_adhd_epilepsy.preproc.utils import (
    _save_outputs,
    _resolve_segments_csv,
    PreprocConfig,
    benchmark_step,
    BlockWindow,
    DEFAULT_ARTIFACT_N_INTERPOLATE,
    _collect_block_windows,
    _get_rest_windows,
    _compute_artifact_overlap,
    _sanitize_n_interpolate,
    _group_consecutive_indices,
    _event_sample_to_onset,

)
import sys
import argparse
from eeg_adhd_epilepsy.io import bids
from eeg_adhd_epilepsy.io import csv as io_csv
from eeg_adhd_epilepsy.reports.base import create_preprocessing_report, create_dataset_report
from eeg_adhd_epilepsy.features.spectral import (
    compute_spectral_metrics,
    compute_aperiodic_slope,
    compute_lsd
)
from joblib import Parallel, delayed
from tqdm import tqdm
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib



LOGGER = logging.getLogger("preproc_base")

DEFAULT_HIGHPASS_HZ = 0.1
DEFAULT_LOWPASS_HZ = 100.0
DEFAULT_ARTIFACT_SEGMENT_S = 1.0
DEFAULT_ARTIFACT_MIN_EPOCHS = 5
DEFAULT_RANSAC_MIN_SECONDS = 1.0
DEFAULT_PSD_FMIN = 0.5
DEFAULT_PSD_FMAX = 60.0


def run_base_pipeline(
    raw: mne.io.BaseRaw,
    config: PreprocConfig,
    subject_id: str = "unknown",
    output_dir: Optional[Path] = None,
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Run the shared preprocessing trunk and return cleaned raw + provenance.

    Args:
        raw: The raw MNE object to process.
        config: Configuration dictionary defining preprocessing parameters.
        subject_id: Identifier for the subject (used in logging/filenames).
        output_dir: Directory to save intermediate checks and final output.

    Returns:
        A tuple containing:
            - The processed MNE Raw object.
            - A dictionary containing processing provenance and statistics.
    """
    output_path = Path(output_dir) if output_dir else Path("./results")
    
    # Define BIDS-like structure
    derivatives_root = output_path / "derivatives" / "preproc"
    subj_deriv_dir = derivatives_root / subject_id / "eeg"
    subj_deriv_dir.mkdir(parents=True, exist_ok=True)
    
    # QC: qc/preproc/
    qc_root = output_path / "qc" / "preproc"
    qc_root.mkdir(parents=True, exist_ok=True)
    figures_dir = qc_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Starting base pipeline for {subject_id}")

    provenance: Dict[str, Any] = {
        "subject_id": subject_id,
        "config": config,
        "steps_completed": [],
        "bad_channels_global": [],
        "artifact_stats": {},
        "block_stats": [],
        "spectral_stats": {},
        "integrity_stats": {},
    }

    # 0. Pre-Processing PSD (for Report)
    psd_before = (np.array([]), np.array([]))
    _, psd_pre_data, freqs_pre, _, _, _ = compute_spectral_metrics(
            raw, 
            picks=None, # All channels 
            fmin=DEFAULT_PSD_FMIN, 
            fmax=DEFAULT_PSD_FMAX
    )
    psd_before = (freqs_pre, psd_pre_data)

    # 1. Annotate Blocks
    segments_file = config.get("processing", {}).get("segments_file", None)
    raw = annotate_blocks_from_csv(raw, segments_file=segments_file)
    provenance["steps_completed"].append("annotate_blocks")

    # 2. Resample
    target_sfreq = config.get("processing", {}).get("resample_hz", None)
    if target_sfreq:
        with benchmark_step("resample", provenance):
            raw.resample(target_sfreq, n_jobs=int(config.get("n_jobs", 1)))
        provenance["steps_completed"].append("resample")

    # 3. Filtering and Line Noise Removal
    # ----------------------------------
    with benchmark_step("filtering_and_denosing", provenance):
        hp_hz = config.get("processing", {}).get("highpass_hz", DEFAULT_HIGHPASS_HZ)
        lp_hz = config.get("processing", {}).get("lowpass_hz", DEFAULT_LOWPASS_HZ)
        line_noise_cfg = config.get("line_noise", {})
        line_freq = line_noise_cfg.get("line_freq", 60.0)
        
        # 3a. Bandpass Filter
        # Ensure lowpass is strictly less than Nyquist (sfreq/2)
        nyquist = raw.info["sfreq"] / 2.0
        h_f = min(lp_hz, nyquist - 0.1) if lp_hz else None
        
        LOGGER.info(f"Applying Bandpass filter: {hp_hz}-{h_f} Hz")
        raw.filter(l_freq=hp_hz, h_freq=h_f, verbose="ERROR")
        provenance["steps_completed"].append("bandpass_filter")
        
        # 3b. Notch Filter
        LOGGER.info(f"Applying Notch filter at {line_freq} Hz and harmonics")
        freqs = np.arange(line_freq, raw.info["sfreq"]/2, line_freq)
        raw.notch_filter(freqs, verbose="ERROR", method='fir', phase='zero-double')
        provenance["steps_completed"].append("notch_filter")
        provenance["zapline_stats"] = {"method": "notch", "line_freq": line_freq}


    # 5. Global Bad Channel Detection
    with benchmark_step("detect_global_bads", provenance):
        provenance["bad_channels_global"] = detect_global_bads_ransac(raw, config)
        provenance["steps_completed"].append("detect_global_bads")

    LOGGER.info(f"Global bad channels ({len(raw.info['bads'])}): {raw.info['bads']}")

    # 6. Common Average Reference (CAR)
    # Applied after excluding global bads to avoid contamination.
    with benchmark_step("reference_car", provenance):
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        provenance["steps_completed"].append("car_ref")

    # 7. Block-wise Artifact Annotation (AutoReject)
    with benchmark_step("block_artifact_annotation", provenance):
        raw, artifact_stats = annotate_artifacts_blockwise(raw, config, figures_dir, subject_id)

    provenance["artifact_stats"] = artifact_stats
    provenance["block_stats"] = artifact_stats.get("by_block", [])
    provenance["steps_completed"].append("block_artifact_annotation")

    # 8. Post-Processing PSD & Report
    psd_after = (np.array([]), np.array([]))
    _, psd_post_data, freqs_post, alpha_peak, _, _ = compute_spectral_metrics(
            raw, 
            picks=None,
            fmin=DEFAULT_PSD_FMIN, 
            fmax=DEFAULT_PSD_FMAX
    )
    psd_after = (freqs_post, psd_post_data)

    # 9. Compute Additional Features (Slope, LSD, Integrity)
    slope_mean, slope_std, intercept, _ = compute_aperiodic_slope(psd_post_data, freqs_post)
    lsd_val = compute_lsd(psd_post_data, psd_pre_data) if psd_pre_data.size > 0 else float("nan")
    
    provenance["spectral_stats"] = {
        "alpha_peak": alpha_peak,
        "aperiodic_slope": slope_mean,
        "aperiodic_intercept": intercept,
        "lsd": lsd_val
    }

    # Clean Data Duration
    total_samples = raw.n_times
    bad_mask = np.zeros(total_samples, dtype=bool)
    for annot in raw.annotations:
        desc = annot['description'] 
        if desc.startswith('BAD_epoch_') or desc.startswith('BAD_ACQ_SKIP') or desc.startswith('BAD_boundary'):
            start_idx = raw.time_as_index(annot['onset'])[0]
            start_idx = max(0, start_idx)
            end_idx = raw.time_as_index(annot['onset'] + annot['duration'])[0]
            end_idx = min(total_samples, end_idx)
            if end_idx > start_idx:
                bad_mask[start_idx:end_idx] = True
    
    clean_samples = total_samples - bad_mask.sum()
    clean_duration_s = clean_samples / raw.info['sfreq']
    clean_fraction = clean_samples / total_samples if total_samples > 0 else 0.0

    provenance["integrity_stats"] = {
        "clean_duration_s": float(clean_duration_s),
        "clean_fraction": float(clean_fraction)
    }

    provenance["artifact_stats"]["artifacts_count"] = len(raw.annotations)
    
    # Save Provenance to Derivatives
    prov_fname = f"{subject_id}_desc-base_provenance.json"
    prov_path = subj_deriv_dir / prov_fname
    
    with open(prov_path, "w") as f:
        json.dump(provenance, f, default=str, indent=4) # serialize np types

    _save_outputs(raw, provenance, subj_deriv_dir, subject_id)
    
    # Generate Report
    create_preprocessing_report(
        subject_id=subject_id,
        raw=raw,
        psd_before=psd_before,
        psd_after=psd_after,
        provenance=provenance,
        output_dir=qc_root,
        figures_dir=figures_dir
    )

    LOGGER.info(f"Pipeline completed for {subject_id}")

    return raw, provenance


def annotate_blocks_from_csv(
    raw: mne.io.BaseRaw, segments_file: Optional[str] = None
) -> mne.io.BaseRaw:
    """Load segment definitions from CSV and add `BLOCK_*` annotations to raw.

    If segments_file is provided, it is used. Otherwise, the function attempts
    to infer the filename from the raw object's filename.

    Args:
        raw: The raw data object.
        segments_file: Path to the segments CSV file (optional).

    Returns:
        The annotated raw object.
    """
    csv_path = _resolve_segments_csv(raw, segments_file)

    LOGGER.info(f"Loading block definitions from {csv_path}")
    df = io_csv.load(str(csv_path))

    onsets: List[float] = []
    durations: List[float] = []
    descriptions: List[str] = []

    for _, row in df.iterrows():
        t_start = float(row["t_start"])
        t_stop = float(row["t_stop"])
        block_name = str(row["segment_type"])
        duration = t_stop - t_start
        onsets.append(t_start)
        durations.append(duration)
        descriptions.append(f"BLOCK_{block_name}")

    new_annots = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + new_annots)
    LOGGER.info(f"Added {len(new_annots)} block annotations.")
    return raw


def detect_global_bads_ransac(
    raw: mne.io.BaseRaw, config: PreprocConfig
) -> List[str]:
    """Detect global bad EEG channels with RANSAC, biased toward rest blocks.

    Uses a subset of data (rest blocks) to speed up RANSAC and focus on
    intrinsic channel quality rather than task-related artifacts.

    Args:
        raw: The raw data object (will be modified in-place to update info['bads']).
        config: Preprocessing configuration.

    Returns:
        List of newly detected bad channel names.
    """
    n_jobs = int(config.get("n_jobs", 1))
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_ch_names = [raw.ch_names[idx] for idx in eeg_picks]
    eeg_raw = raw.copy().pick(eeg_ch_names)
    rest_windows = _get_rest_windows(raw)

    raw_for_ransac = None
    if rest_windows:
        crops: List[mne.io.BaseRaw] = []
        for onset, stop in rest_windows:
            if stop <= onset:
                continue
            crop = eeg_raw.copy().crop(onset, stop, include_tmax=False)
            if crop.n_times > 0:
                crops.append(crop)

        if crops:
            raw_for_ransac = (
                crops[0]
                if len(crops) == 1
                else mne.concatenate_raws(crops, verbose="ERROR")
            )


    duration_s = raw_for_ransac.n_times / raw_for_ransac.info["sfreq"]
    LOGGER.info(f"Running RANSAC on {duration_s:.1f}s of EEG data...")

    nc = NoisyChannels(raw_for_ransac, random_state=42)
    nc.find_bad_by_ransac()
    bads = nc.get_bads(verbose=False) or []
    bads = sorted(ch for ch in bads if ch in raw.ch_names)

    # Update raw.info['bads']
    current_bads = set(raw.info.get("bads", []))
    new_bads = current_bads.union(bads)
    raw.info["bads"] = sorted(new_bads)

    return bads


def annotate_artifacts_blockwise(
    raw: mne.io.BaseRaw, 
    config: PreprocConfig, 
    figures_dir: Optional[Path] = None,
    subject_id: str = "unknown"
) -> Tuple[mne.io.BaseRaw, Dict]:
    """Run condition-wise AutoReject and add non-destructive BAD annotations.

    Groups disjoint blocks by condition (e.g., all "EO_baseline" blocks) and
    runs AutoReject on them as a single set of epochs. This ensures consistent
    thresholds across the condition.

    Args:
        raw: The raw data object.
        config: Preprocessing configuration.
        figures_dir: Directory to save AutoReject visualizations (optional).

    Returns:
        A tuple containing the annotated raw object and a statistics dictionary.
    """
    block_windows = _collect_block_windows(raw)
    stats: Dict[str, Any] = {
        "blocks_total": len(block_windows),
        "blocks_processed": 0,
        "bad_epochs": 0,
        "bad_channel_spans": 0,
        "artifacts_count": 0,
        "by_block": [],
    }

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    # Configuration
    artifacts_cfg = config.get("artifacts", {})
    bad_channels_cfg = config.get("bad_channels", {})
    seg_len = float(
        artifacts_cfg.get(
            "segment_length",
            bad_channels_cfg.get("segment_length", DEFAULT_ARTIFACT_SEGMENT_S),
        )
    )
    seg_len = seg_len if seg_len > 0 else DEFAULT_ARTIFACT_SEGMENT_S

    min_epochs = int(artifacts_cfg.get("min_epochs", DEFAULT_ARTIFACT_MIN_EPOCHS))
    min_epochs = max(1, min_epochs)

    n_interpolate = [0]

    n_jobs = int(config.get("n_jobs", 1))
    random_seed = int(config.get("random_seed", 42))
    epoch_tmax = max(seg_len - (1.0 / raw.info["sfreq"]), 0.0)

    # Group blocks by condition name
    grouped_blocks: Dict[str, List[BlockWindow]] = {}
    for block in block_windows:
        block_name = block.name
        if block_name not in grouped_blocks:
            grouped_blocks[block_name] = []
        grouped_blocks[block_name].append(block)

    LOGGER.info(
        f"Running condition-wise artifact annotation on {len(grouped_blocks)} conditions..."
    )

    new_annots: List[Tuple[float, float, str, Tuple[str, ...]]] = []

    for condition_name, blocks in grouped_blocks.items():
        # Collect epochs for this condition
        condition_events_list = []
        for block in blocks:
            # Skip if block is too short for even one segment
            if (block.stop - block.onset) < seg_len:
                LOGGER.warning(
                    f"Block {block.name} duration ({block.stop - block.onset:.2f}s) "
                    f"shorter than artifact segment length ({seg_len}s). Skipping."
                )
                continue

            events = mne.make_fixed_length_events(
                raw,
                id=1,
                start=block.onset,
                stop=block.stop,
                duration=seg_len,
                overlap=0.0,
                first_samp=True,
            )
            if len(events) > 0:
                condition_events_list.append(events)

        if not condition_events_list:
            continue

        condition_events = np.concatenate(condition_events_list)
        condition_events = condition_events[condition_events[:, 0].argsort()]


        epochs = mne.Epochs(
            raw,
            events=condition_events,
            event_id={"seg": 1},
            tmin=0.0,
            tmax=epoch_tmax,
            baseline=None,
            preload=True,
            picks=eeg_picks,
            reject_by_annotation=False,
            verbose="ERROR",
        )

        # Run AutoReject
        n_epochs = len(epochs)
        if n_epochs < 2:
            LOGGER.warning(f"Too few epochs ({n_epochs}) for AutoReject in condition {condition_name}. Skipping AR.")
            continue
            
        cv = 10
        if n_epochs < cv:
            cv = n_epochs
            LOGGER.info(f"Adjusting AutoReject CV to {cv} folds for {condition_name} ({n_epochs} epochs).")

        ar = AutoReject(
            n_interpolate=n_interpolate,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=False,
            cv=cv
        )
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)
        
        # Save AutoReject Visualization
        fig = reject_log.plot(orientation='horizontal', show=False)
        
        # Resize for better visibility with many epochs
        fig.set_size_inches(16, 10) # Wider for horizontal plot

        clean_name = condition_name.lower().replace(" ", "_")
        fig.savefig(figures_dir / f"{subject_id}_autoreject_{clean_name}.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Convert rejection log to annotations
        cond_bad_epochs = 0
        cond_bad_spans = 0

        # Bad epochs (global)
        for ep_idx, is_bad_epoch in enumerate(reject_log.bad_epochs):
            if not is_bad_epoch:
                continue
            onset_s = _event_sample_to_onset(raw, int(epochs.events[ep_idx, 0]))
            new_annots.append((onset_s, seg_len, f"BAD_epoch_{condition_name}", ()))
            cond_bad_epochs += 1

        # Bad channels (local)
        labels = np.asarray(reject_log.labels)
        if labels.ndim == 2 and labels.shape[0] == len(epochs):
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                bad_idx = np.flatnonzero(labels[:, ch_idx] != 0)
                for first_idx, last_idx in _group_consecutive_indices(bad_idx):
                    start_samp = int(epochs.events[first_idx, 0])
                    end_samp = int(epochs.events[last_idx, 0])
                    start_s = _event_sample_to_onset(raw, start_samp)
                    end_s = _event_sample_to_onset(raw, end_samp) + seg_len
                    duration_s = end_s - start_s

                    if duration_s <= 0:
                        continue
                    new_annots.append(
                        (start_s, duration_s, f"BAD_{condition_name}", (ch_name,))
                    )
                    cond_bad_spans += 1

        stats["blocks_processed"] += len(blocks)
        stats["bad_epochs"] += cond_bad_epochs
        stats["bad_channel_spans"] += cond_bad_spans
        stats["by_block"].append(
            {
                "condition": condition_name,
                "n_blocks_merged": len(blocks),
                "epochs": len(epochs),
                "bad_epochs": cond_bad_epochs,
                "bad_channel_spans": cond_bad_spans,
            }
        )

    # Add new annotations to raw
    if new_annots:
        # Calculate overlap with manual annotations before adding
        overlap_pct = _compute_artifact_overlap(raw, new_annots)
        LOGGER.info(
            f"Manual BAD Overlap: {overlap_pct:.1f}% of manual segments were re-detected."
        )
        stats["manual_overlap_pct"] = overlap_pct

        new_annots.sort(key=lambda x: x[0])
        artifact_annots = mne.Annotations(
            onset=[row[0] for row in new_annots],
            duration=[row[1] for row in new_annots],
            description=[row[2] for row in new_annots],
            orig_time=raw.annotations.orig_time,
            ch_names=[row[3] for row in new_annots],
        )
        raw.set_annotations(raw.annotations + artifact_annots)
        LOGGER.info(f"Added {len(new_annots)} artifact annotations.")

    stats["artifacts_count"] = len(new_annots)
    return raw, stats


def _process_subject(
    fpath: Path,
    output_dir: Path,
    config: Dict = None,
) -> bool:
    """Process a single subject file. Returns True on success, False on failure."""
    try:
        if config is None:
            config = {}
        sid = bids.parse_subject_id(fpath)
        LOGGER.info(f"Processing {fpath.name} (Subject: {sid})...")

        # Load Raw
        bids_root = config.get("bids_root")
        if bids_root:
            raw = bids.load_bids_raw(fpath, bids_root=Path(bids_root))
        else:
            # Fallback if bids_root not in config
            raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose="ERROR")

        # Drop A1, A2 if present (Specific User Request)
        to_drop = [ch for ch in ["A1", "A2"] if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)
            LOGGER.info(f"[{sid}] Dropped channels: {to_drop}")

        # Set Montage
        try:
            raw.set_montage("standard_1020", match_case=False)
        except Exception as e:
            LOGGER.warning(f"[{sid}] Could not set standard_1020 montage: {e}")

        # Run Pipeline
        run_base_pipeline(raw, config=config, subject_id=sid, output_dir=output_dir)
        return True

    except Exception as e:
        LOGGER.error(f"Failed processing {fpath.name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run EEG Preprocessing Pipeline on BIDS Dataset")
    parser.add_argument("--bids_root", type=str, default="/Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS", help="Path to BIDS dataset root")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)")
    
    # New Config Args
    parser.add_argument("--lowpass", type=float, default=DEFAULT_LOWPASS_HZ, help=f"Lowpass filter cutoff Hz (default: {DEFAULT_LOWPASS_HZ})")
    parser.add_argument("--highpass", type=float, default=DEFAULT_HIGHPASS_HZ, help=f"Highpass filter cutoff Hz (default: {DEFAULT_HIGHPASS_HZ})")
    parser.add_argument("--line_freq", type=float, default=60.0, help="Line noise frequency Hz (default: 60.0)")
    parser.add_argument("--resample", type=float, default=None, help="Resampling frequency Hz (optional)")

    parser.add_argument("--all", action="store_true", help="Process all available subjects")
    parser.add_argument("--test", action="store_true", help="Run on first 5 subjects for testing")
    parser.add_argument("--random", action="store_true", help="When combined with --test, select 5 random subjects instead of first 5")
    parser.add_argument("--subjects", nargs="+", help="List of specific subject IDs (e.g., sub-001 sub-002)")
    parser.add_argument("--start-from", type=str, help="Resume processing from this subject ID")
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "logs" / "preproc_base.log"
    setup_logging(log_file, "INFO")
    
    if not bids_root.exists():
        LOGGER.error(f"BIDS root not found: {bids_root}")
        sys.exit(1)
        
    # Discover files
    files = bids.discover_bids_files(bids_root, suffix="eeg", extension=".vhdr")
    
    if not files:
        LOGGER.error("No .vhdr files found in BIDS directory.")
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
            # Select 5 random subjects
            random.seed(42)  # Reproducible randomness
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
        
    # Map back to files
    files_to_process = [f for f in files if file_map[f] in subjects_to_process]
    
    if not files_to_process:
        LOGGER.warning("No files matched the selection criteria.")
        sys.exit(0)
        
    # Execution Loop
    LOGGER.info(f"Starting batch processing of {len(files_to_process)} files with {args.n_jobs} jobs...")
    
    pipeline_config = {
        "n_jobs": 1, 
        "bids_root": str(bids_root),
        "processing": {
            "highpass_hz": args.highpass,
            "lowpass_hz": args.lowpass,
            "resample_hz": args.resample,
        },
        "line_noise": {
            "line_freq": args.line_freq,
        }
    }
    
    # Run Parallel with progress bar
    with tqdm_joblib(tqdm(total=len(files_to_process), desc="Preprocessing")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_process_subject)(f, output_dir, pipeline_config)
            for f in files_to_process
        )
        
    success_count = sum(results)
    fail_count = len(results) - success_count
            
    LOGGER.info(f"Batch processing complete. Success: {success_count}, Failed: {fail_count}")
    
    # Generate Dataset Report
    qc_root = output_dir / "qc" / "preproc"
    derivatives_root = output_dir / "derivatives" / "preproc"
    
    qc_root.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Generating dataset-level report...")
    create_dataset_report(
        search_dir=derivatives_root, 
        output_dir=qc_root
    )


if __name__ == "__main__":
    main()

"""Stage 2: Residual Denoising & Final Cleanup."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

from meegkit import asr
from mne_denoise.dss import IterativeDSS, WienerMaskDenoiser
from mne_denoise.viz import plot_component_summary, plot_score_curve

from eeg_adhd_epilepsy.signal_quality.spectral import compute_spectral_metrics
from eeg_adhd_epilepsy.signal_quality.spectral import compute_lsd
from eeg_adhd_epilepsy.io import bids
from eeg_adhd_epilepsy.reports.denoise import (
    create_denoising_dataset_report,
    create_denoising_report,
)
from eeg_adhd_epilepsy.utils.logs import setup_logging

from .base import annotate_artifacts_blockwise
from .utils import NumpyEncoder, benchmark_step

LOGGER = logging.getLogger(__name__)


@dataclass
class ArtifactDenoisingConfig:
    """Configuration for Stage 2 (Denoising)."""

    transient_method: Optional[str] = "wiener"  # 'asr', 'wiener', 'dss', None

    # ASR
    asr_cutoff: float = 20.0
    asr_calibration_window: float = 10.0

    # Wiener Masking
    wiener_window_duration: float = 0.2
    wiener_noise_percentile: float = 85.0
    wiener_n_components: int = 10
    wiener_max_iter: int = 5

    # AutoReject
    autoreject_max_chunk_minutes: float = 30.0

    # General
    n_jobs: int = 1
    random_state: int = 42




def _remove_transients_wiener(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Remove transients using adaptive Wiener masking."""
    LOGGER.info("Applying Adaptive Wiener Mask Denoiser...")

    sfreq = float(raw.info["sfreq"])
    window_samples = int(config.wiener_window_duration * sfreq)

    denoiser = WienerMaskDenoiser(
        window_samples=window_samples,
        noise_percentile=config.wiener_noise_percentile,
    )

    dss = IterativeDSS(
        denoiser=denoiser,
        n_components=config.wiener_n_components,
        max_iter=config.wiener_max_iter,
        random_state=config.random_state,
        verbose="ERROR",
    )

    data_2d = raw.get_data(picks="eeg")
    dss.fit(data_2d)

    plot_paths: Dict[str, str] = {}
    if output_dir:
        import matplotlib.pyplot as plt

        fig_dir = output_dir / "figures" / "dss_wiener"
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig_score = plot_score_curve(dss, show=False)
        if fig_score is not None:
            score_path = fig_dir / f"{subject_id}_wiener_score.png"
            fig_score.savefig(score_path, dpi=150, bbox_inches="tight")
            plt.close(fig_score)
            plot_paths["score_curve"] = str(score_path)

        n_samples_plot = min(data_2d.shape[1], int(sfreq * 60))
        fig_comp = plot_component_summary(dss, data_2d[:, :n_samples_plot], n_components=3, show=False)
        if fig_comp is not None:
            comp_path = fig_dir / f"{subject_id}_wiener_comps.png"
            fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
            plt.close(fig_comp)
            plot_paths["component_summary"] = str(comp_path)

    data_denoised = dss.transform(data_2d)
    cleaned_data = dss.inverse_transform(data_denoised)

    raw_out = raw.copy()
    picks = mne.pick_types(raw_out.info, eeg=True, exclude=[])
    raw_out._data[picks, :] = cleaned_data

    return raw_out, {
        "method": "wiener_mask",
        "n_components": config.wiener_n_components,
        "plot_paths": plot_paths,
    }


def _remove_transients_asr(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Remove transients using ASR."""
    LOGGER.info("Applying ASR...")

    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])
    window_len = int(config.asr_calibration_window * sfreq)

    clean_mask = np.ones(data.shape[1], dtype=bool)
    for annot in raw.annotations:
        if str(annot["description"]).startswith("BAD_"):
            start = raw.time_as_index(float(annot["onset"]))[0]
            stop = raw.time_as_index(float(annot["onset"]) + float(annot["duration"]))[0]
            start = max(0, start)
            stop = min(data.shape[1], stop)
            clean_mask[start:stop] = False

    best_start = -1
    min_var = np.inf
    step = int(sfreq)

    for idx in range(0, max(1, data.shape[1] - window_len), step):
        if clean_mask[idx : idx + window_len].all():
            variance = float(np.var(data[:, idx : idx + window_len]))
            if variance < min_var:
                min_var = variance
                best_start = idx

    if best_start == -1:
        LOGGER.warning("No fully clean ASR calibration window found. Falling back to lowest-variance window.")
        for idx in range(0, max(1, data.shape[1] - window_len), step):
            variance = float(np.var(data[:, idx : idx + window_len]))
            if variance < min_var:
                min_var = variance
                best_start = idx

    if best_start == -1:
        return raw, {"skipped": True, "reason": "data_too_short"}

    calib_data = data[:, best_start : best_start + window_len]
    asr_model = asr.ASR(method="euclid")

    try:
        asr_model.fit(calib_data.T)
        clean_data = asr_model.transform(data.T).T

        raw_out = raw.copy()
        picks = mne.pick_types(raw_out.info, eeg=True, exclude=[])
        raw_out._data[picks, :] = clean_data
        return raw_out, {
            "method": "asr",
            "cutoff": config.asr_cutoff,
            "calibration_start": best_start / sfreq,
        }
    except Exception as exc:
        LOGGER.error("ASR failed: %s", exc)
        return raw, {"error": str(exc)}


def _calculate_recovery_stats(
    base_bad_segments: List[Tuple[float, float]],
    final_bad_segments: List[Tuple[float, float]],
    raw_times: np.ndarray,
    sfreq: float,
) -> Dict[str, Any]:
    """
    Calculate recovery metrics using boolean masks to handle overlaps correctly.
    """
    n_samples = len(raw_times)
    mask_base = np.zeros(n_samples, dtype=bool)
    mask_final = np.zeros(n_samples, dtype=bool)

    # Helper to vectorizing segment filling
    # Using time_to_indices equivalent
    def fill_mask(mask, segments):
        for onset, duration in segments:
            start_samp = int(onset * sfreq)
            end_samp = int((onset + duration) * sfreq)
            # Clip to valid range
            start_samp = max(0, start_samp)
            end_samp = min(n_samples, end_samp)
            if end_samp > start_samp:
                mask[start_samp:end_samp] = True

    fill_mask(mask_base, base_bad_segments)
    fill_mask(mask_final, final_bad_segments)

    # Metrics
    # Base Bad Time
    base_bad_samples = np.count_nonzero(mask_base)
    base_bad_seconds = base_bad_samples / sfreq

    # Final Bad Time
    final_bad_samples = np.count_nonzero(mask_final)
    final_bad_seconds = final_bad_samples / sfreq

    # Recovered: Was Bad AND Now Good (NOT Final Bad)
    mask_recovered = mask_base & (~mask_final)
    recovered_samples = np.count_nonzero(mask_recovered)
    recovered_seconds = recovered_samples / sfreq

    recovery_rate_pct = (recovered_seconds / base_bad_seconds * 100.0) if base_bad_seconds > 0 else 0.0

    # Count corrected segments (Legacy metric for report)
    # A segment is "corrected" if it has ZERO overlap with final bads
    # We can check specific segments against the final mask
    n_base_corrected = 0
    segments_to_drop = []

    for b_on, b_dur in base_bad_segments:
        start_samp = int(b_on * sfreq)
        end_samp = int((b_on + b_dur) * sfreq)
        start_samp = max(0, start_samp)
        end_samp = min(n_samples, end_samp)
        
        if end_samp > start_samp:
            # Check if any part of this segment is in final mask
            # If np.any(mask_final[start:end]), then it's not fully recovered
            if not np.any(mask_final[start_samp:end_samp]):
                n_base_corrected += 1
                segments_to_drop.append((b_on, b_dur))
        else:
            # Zero length segment? count as corrected/ignored
            pass

    return {
        "base_bad_seconds": base_bad_seconds,
        "final_bad_seconds": final_bad_seconds,
        "recovered_seconds": recovered_seconds,
        "recovery_rate_pct": recovery_rate_pct,
        "base_total": len(base_bad_segments),
        "n_base_corrected": n_base_corrected,
        "segments_to_drop": segments_to_drop # Used for dynamic updates (though explicit strip approach is safer)
    }


def _refine_autoreject(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
    base_bad_segments: List[Tuple[float, float]],
    figures_dir: Optional[Path] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Run final AutoReject refinement and compare against Stage 1 BAD segments."""
    ar_conf_dict = {
        "artifacts": {
            "segment_length": 1.0,
            "min_epochs": 1,
            "ar_max_chunk_minutes": config.autoreject_max_chunk_minutes,
        },
        "bad_channels": {},
        "n_jobs": config.n_jobs,
        "random_seed": config.random_state,
    }

    raw_for_ar = raw.copy()
    keep_indices = [
        i for i, annot in enumerate(raw_for_ar.annotations) 
        if not str(annot["description"]).startswith("BAD_")
    ]
    raw_for_ar.set_annotations(raw_for_ar.annotations[keep_indices])

    raw_refined, stats = annotate_artifacts_blockwise(
        raw_for_ar,
        config=ar_conf_dict,
        figures_dir=figures_dir,
        subject_id=subject_id,
        n_interpolate=[0, 1, 2, 4],  # Allow interpolation for Stage 2 repair
    )

    new_bad_segments = [
        (float(a["onset"]), float(a["duration"]))
        for a in raw_refined.annotations
        if str(a["description"]).startswith("BAD_")
    ]

    recovery_stats = _calculate_recovery_stats(
        base_bad_segments, 
        new_bad_segments, 
        raw_refined.times,
        raw_refined.info["sfreq"]
    )
    
    stats.update(recovery_stats)
    return raw_refined, stats


def run_residual_denoising(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
    figures_dir: Optional[Path] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Orchestrate Stage 2 denoising."""
    provenance: Dict[str, Any] = {
        "steps_completed": [],
        "transient_stats": {},
        "autoreject_stats": {},
        "benchmarks": {"timing": {}},
    }

    denoised_raw = raw.copy()

    if config.transient_method:
        with benchmark_step("transient_removal", provenance):
            if config.transient_method == "wiener":
                denoised_raw, transient_stats = _remove_transients_wiener(
                    denoised_raw,
                    config,
                    output_dir=figures_dir.parent if figures_dir else None,
                    subject_id=subject_id,
                )
            elif config.transient_method == "asr":
                denoised_raw, transient_stats = _remove_transients_asr(denoised_raw, config)
            elif config.transient_method == "dss":
                transient_stats = {"skipped": True, "reason": "dss_transient_not_implemented"}
            else:
                transient_stats = {
                    "skipped": True,
                    "reason": "unknown_method",
                    "method": config.transient_method,
                }
            provenance["transient_stats"] = transient_stats
        provenance["steps_completed"].append("transient_removal")

    with benchmark_step("final_autoreject", provenance):
        base_bad_segments = [
            (float(a["onset"]), float(a["duration"]))
            for a in raw.annotations
            if str(a["description"]).startswith("BAD_")
        ]
        denoised_raw, ar_stats = _refine_autoreject(
            denoised_raw,
            config,
            base_bad_segments,
            figures_dir=figures_dir,
            subject_id=subject_id,
        )
        provenance["autoreject_stats"] = ar_stats
    provenance["steps_completed"].append("final_autoreject")

    return denoised_raw, provenance


def run_denoising_pipeline(
    subject_id: str,
    bids_root: Path,
    config: ArtifactDenoisingConfig,
    preproc_root: Optional[Path] = None,
    reports_root: Optional[Path] = None,
    condition_name: Optional[str] = None,
    input_desc: str = "correct",
    output_desc: str = "denoise",
) -> bool:
    """Run Stage 2 on one subject (Stage 1 -> Stage 2 handoff)."""
    try:
        subject_id = bids.normalize_subject_id(subject_id)
        input_desc = bids.validate_stage_desc(input_desc)
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

        task_token = condition_name if condition_name else None
        input_path = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=input_desc,
            task=task_token,
        )
        if not input_path.exists():
            LOGGER.error("Input file not found: %s", input_path)
            return False

        LOGGER.info("Loading Stage 1 output: %s", input_path)
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")

        subject_report_path = bids.get_subject_report_path(
            reports_root=reports_root,
            stage="denoise",
            subject_id=subject_id,
            create_dir=True,
        )
        report_path = subject_report_path
        if output_desc != "denoise":
            report_path = subject_report_path.with_name(
                f"{subject_id}_denoise_{output_desc}_report.html"
            )

        figures_dir = report_path.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Computing pre-denoise spectral metrics...")
        _, psd_pre_data, freqs_pre, alpha_pre, band_pre, _ = compute_spectral_metrics(
            raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_before = (freqs_pre, psd_pre_data)

        denoised_raw, provenance = run_residual_denoising(
            raw,
            config,
            figures_dir=figures_dir,
            subject_id=subject_id,
        )

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

        provenance["subject_id"] = subject_id
        provenance["input_file"] = str(input_path)
        provenance["output_file"] = str(out_path)
        provenance["provenance_file"] = str(prov_path)
        provenance["preproc_root"] = str(preproc_root)
        provenance["reports_root"] = str(reports_root)
        provenance["input_desc"] = input_desc
        provenance["output_desc"] = output_desc
        provenance["condition_name"] = condition_name
        provenance["data_duration_s"] = raw.times[-1]

        LOGGER.info("Computing post-denoise spectral metrics...")
        _, psd_post_data, freqs_post, alpha_post, band_post, _ = compute_spectral_metrics(
            denoised_raw, picks=None, fmin=0.5, fmax=60.0
        )
        psd_after = (freqs_post, psd_post_data)

        lsd = float("nan")
        if (
            psd_pre_data.size > 0
            and psd_post_data.size > 0
            and psd_pre_data.shape == psd_post_data.shape
        ):
            lsd = compute_lsd(psd_clean=psd_post_data, psd_raw=psd_pre_data)

        band_delta_abs: Dict[str, float] = {}
        band_delta_pct: Dict[str, float] = {}
        for band_name, pre_val in band_pre.items():
            post_val = float(band_post.get(band_name, float("nan")))
            pre_val_f = float(pre_val)
            delta_abs = post_val - pre_val_f
            band_delta_abs[band_name] = delta_abs
            if np.isfinite(pre_val_f) and pre_val_f != 0:
                band_delta_pct[band_name] = (delta_abs / pre_val_f) * 100.0
            else:
                band_delta_pct[band_name] = float("nan")

        provenance["spectral_stats"] = {
            "alpha_peak_pre": float(alpha_pre),
            "alpha_peak_post": float(alpha_post),
            "alpha_peak_delta": float(alpha_post) - float(alpha_pre),
            "lsd_db": float(lsd),
            "band_power_pre": {k: float(v) for k, v in band_pre.items()},
            "band_power_post": {k: float(v) for k, v in band_post.items()},
            "band_power_delta_abs": band_delta_abs,
            "band_power_delta_pct": band_delta_pct,
        }

        LOGGER.info("Saving denoised raw to %s", out_path)
        denoised_raw.save(out_path, overwrite=True, verbose="ERROR")

        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, cls=NumpyEncoder, indent=2)

        create_denoising_report(
            subject_id=subject_id,
            raw=denoised_raw,
            psd_before=psd_before,
            psd_after=psd_after,
            provenance=provenance,
            subject_report_path=report_path,
            figures_dir=figures_dir,
        )

        LOGGER.info("Denoising pipeline completed for %s. Output: %s", subject_id, out_path)
        return True

    except Exception as exc:
        LOGGER.error("Failed denoising for %s: %s", subject_id, exc, exc_info=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 Residual Denoising")
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

    parser.add_argument("--subjects", nargs="+", help="List of specific subjects (e.g. sub-001 sub-002)")
    parser.add_argument("--start-from", type=str, help="Start processing from this subject ID")
    parser.add_argument("--all", action="store_true", help="Process all subjects found in Stage 1 outputs")
    parser.add_argument("--test", action="store_true", help="Run on a small subset (5 subjects)")
    parser.add_argument("--random", action="store_true", help="Select random subjects in test mode")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects with existing output")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--reports-only", action="store_true", help="Skip processing, only generate dataset report")
    parser.add_argument(
        "--transient-method",
        type=str,
        default="wiener",
        choices=["wiener", "asr", "dss", "none"],
        help="Transient removal method",
    )
    parser.add_argument("--condition", type=str, help="Process only specific condition/task")
    parser.add_argument(
        "--input-desc",
        type=str,
        default="correct",
        help="Input desc entity (default: correct)",
    )
    parser.add_argument(
        "--output-desc",
        type=str,
        default="denoise",
        help="Output desc entity (default: denoise)",
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_preproc_root(
        bids_root=bids_root,
        preproc_root=Path(args.preproc_root).expanduser() if args.preproc_root else None,
    )
    reports_root = bids.get_reports_root(
        bids_root=bids_root,
        reports_root=Path(args.reports_root).expanduser() if args.reports_root else None,
    )
    preproc_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    log_file = reports_root / "logs" / "denoise_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = ArtifactDenoisingConfig(**json.load(f))
    else:
        config = ArtifactDenoisingConfig(
            transient_method=args.transient_method if args.transient_method != "none" else None
        )

    input_desc = bids.validate_stage_desc(args.input_desc)
    output_desc = bids.validate_stage_desc(args.output_desc)
    task_token = args.condition if args.condition else None

    if task_token:
        pattern = f"*_task-{task_token}_desc-{input_desc}_eeg.fif"
    else:
        pattern = f"*_desc-{input_desc}_eeg.fif"

    files = sorted(preproc_root.rglob(pattern))
    if not files:
        LOGGER.error("No Stage 1 FIF files found in %s (pattern: %s)", preproc_root, pattern)
        sys.exit(1)

    subjects_found = sorted({bids.parse_subject_id(f) for f in files})
    LOGGER.info("Found %d subjects with Stage 1 inputs.", len(subjects_found))

    if args.subjects:
        subjects_to_process = {bids.normalize_subject_id(s) for s in args.subjects}
    elif args.start_from:
        start_sub = bids.normalize_subject_id(args.start_from)
        subjects_to_process = {s for s in subjects_found if s >= start_sub}
        if not subjects_to_process:
            LOGGER.error("No subjects found starting from %s.", start_sub)
            sys.exit(1)
    elif args.test:
        import random

        if args.random:
            random.seed(42)
            subjects_to_process = set(random.sample(subjects_found, min(5, len(subjects_found))))
        else:
            subjects_to_process = set(subjects_found[:5])
    elif args.all:
        subjects_to_process = set(subjects_found)
    else:
        LOGGER.warning("No selection criteria provided (use --all, --test, --subjects, or --start-from).")
        parser.print_help()
        sys.exit(0)

    if args.skip_existing:
        subjects_to_skip = set()
        for sid in subjects_to_process:
            out_file = bids.get_stage_output_path(
                subject_id=sid,
                preproc_root=preproc_root,
                desc=output_desc,
                task=args.condition if args.condition else None,
            )
            if out_file.exists():
                subjects_to_skip.add(sid)
        if subjects_to_skip:
            LOGGER.info("Skipping %d already processed subjects.", len(subjects_to_skip))
            subjects_to_process = subjects_to_process - subjects_to_skip

    subjects_sorted = sorted(subjects_to_process)
    if not subjects_sorted:
        LOGGER.warning("No subjects left to process.")
        sys.exit(0)

    success_ids: List[str] = []
    failed_ids: List[str] = []

    for sub in subjects_sorted:
        if args.reports_only:
            continue

        LOGGER.info("Processing %s...", sub)
        success = run_denoising_pipeline(
            subject_id=sub,
            bids_root=bids_root,
            config=config,
            preproc_root=preproc_root,
            reports_root=reports_root,
            condition_name=args.condition,
            input_desc=input_desc,
            output_desc=output_desc,
        )
        if success:
            success_ids.append(sub)
        else:
            failed_ids.append(sub)

    if not args.reports_only:
        LOGGER.info("Batch processing complete. Success: %d, Failed: %d", len(success_ids), len(failed_ids))
        LOGGER.info("Succeeded subjects: %s", sorted(success_ids))
        LOGGER.info("Failed subjects: %s", sorted(failed_ids))
    else:
        LOGGER.info("Skipping processing (--reports-only). Generating summary from existing files.")

    summary_path = bids.get_stage_summary_report_path(
        reports_root=reports_root,
        stage="denoise",
        create_dir=True,
    )
    if output_desc != "denoise":
        summary_path = summary_path.with_name(f"denoise_{output_desc}_dataset_summary.html")

    create_denoising_dataset_report(
        search_dir=preproc_root,
        summary_report_path=summary_path,
        output_desc=output_desc,
        success_subjects=success_ids if not args.reports_only else None,
        failed_subjects=failed_ids if not args.reports_only else None,
    )


if __name__ == "__main__":
    main()

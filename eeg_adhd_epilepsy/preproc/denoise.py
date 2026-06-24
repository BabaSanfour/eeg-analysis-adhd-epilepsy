"""Stage 2: Residual Denoising & Final Cleanup."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from meegkit import asr
from mne_denoise.dss import IterativeDSS, WienerMaskDenoiser
from mne_denoise.viz import plot_component_summary, plot_score_curve

from eeg_adhd_epilepsy.io import bids, report_paths
from eeg_adhd_epilepsy.qc import preproc_qc
from eeg_adhd_epilepsy.utils.logs import setup_logging

from .base import annotate_artifacts_blockwise
from .utils import NumpyEncoder, benchmark_step, select_subjects

LOGGER = logging.getLogger(__name__)


@dataclass
class ArtifactDenoisingConfig:
    """Configuration for Stage 2 (Denoising)."""

    transient_method: str | None = "wiener"  # 'asr', 'wiener', 'dss', None

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
    output_dir: Path | None = None,
    subject_id: str = "unknown",
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
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

    plot_paths: dict[str, str] = {}
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
        fig_comp = plot_component_summary(
            dss, data_2d[:, :n_samples_plot], n_components=3, show=False
        )
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
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
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
        LOGGER.warning(
            "No fully clean ASR calibration window found. Falling back to lowest-variance window."
        )
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
    except RuntimeError as exc:
        LOGGER.error("ASR failed: %s", exc)
        return raw, {"error": str(exc)}


def _calculate_recovery_stats(
    base_bad_segments: list[tuple[float, float]],
    final_bad_segments: list[tuple[float, float]],
    raw_times: np.ndarray,
    sfreq: float,
) -> dict[str, Any]:
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

    recovery_rate_pct = (
        (recovered_seconds / base_bad_seconds * 100.0) if base_bad_seconds > 0 else 0.0
    )

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
        # Used for dynamic updates (though explicit strip approach is safer)
        "segments_to_drop": segments_to_drop,
    }


def _refine_autoreject(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
    base_bad_segments: list[tuple[float, float]],
    figures_dir: Path | None = None,
    subject_id: str = "unknown",
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
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
        i
        for i, annot in enumerate(raw_for_ar.annotations)
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
        base_bad_segments, new_bad_segments, raw_refined.times, raw_refined.info["sfreq"]
    )

    stats.update(recovery_stats)
    return raw_refined, stats


def run_residual_denoising(
    raw: mne.io.BaseRaw,
    config: ArtifactDenoisingConfig,
    figures_dir: Path | None = None,
    subject_id: str = "unknown",
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Orchestrate Stage 2 denoising."""
    provenance: dict[str, Any] = {
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
    preproc_root: Path | None = None,
    reports_root: Path | None = None,
    input_path: Path | None = None,
    condition_name: str | None = None,
    input_desc: str = "correct",
    output_desc: str = "denoise",
    raw_lookup: dict[str, dict[str, object]] | None = None,
    previous_lookup: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run Stage 2 on one subject (Stage 1 -> Stage 2 handoff)."""
    result: dict[str, object] = {
        "success": False,
        "subject_id": bids.bids_subject_label(subject_id),
        "qc_record": None,
        "error": "",
    }
    try:
        subject = subject_id
        input_desc = bids.validate_stage_desc(input_desc)
        output_desc = bids.validate_stage_desc(output_desc)
        bids_root = Path(bids_root).expanduser()

        if preproc_root is None:
            preproc_root = bids.get_preproc_root(bids_root)
        else:
            preproc_root = Path(preproc_root).expanduser()
        if reports_root is None:
            reports_root = report_paths.default_reports_root(bids_root)
        else:
            reports_root = Path(reports_root).expanduser()

        if input_path is None:
            input_path = bids.get_stage_output_path(
                subject=subject,
                preproc_root=preproc_root,
                desc=input_desc,
                task=condition_name if condition_name else None,
            )
        input_path = Path(input_path)
        input_ids = report_paths.build_bids_report_ids(input_path)
        input_comps = bids.parse_bids_components(input_path)
        session_id = input_comps.get("session")
        run_id = input_comps.get("run")
        record_label = str(input_ids["run_prefix"])
        if not input_path.exists():
            LOGGER.error("Input file not found: %s", input_path)
            result["error"] = f"Input file not found: {input_path}"
            return result

        LOGGER.info("Loading Stage 1 output: %s", input_path)
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")

        stage_name = preproc_qc.get_preproc_qc_stage_name("denoise", output_desc)
        report_dir = report_paths.subject_report_dir(
            reports_root=reports_root,
            subject=subject,
            session=session_id or "01",
            stage=stage_name,
            create=True,
        )
        subject_report_path = report_dir / (
            f"{input_ids['subject_session_prefix']}_{stage_name.value}_report.html"
        )
        figures_dir = subject_report_path.parent / "figures" / record_label
        figures_dir.mkdir(parents=True, exist_ok=True)

        denoised_raw, provenance = run_residual_denoising(
            raw,
            config,
            figures_dir=figures_dir,
            subject_id=record_label,
        )

        task_token = condition_name if condition_name else input_comps.get("task")
        out_path = bids.get_stage_output_path(
            subject=subject,
            preproc_root=preproc_root,
            desc=output_desc,
            session=session_id,
            task=task_token,
            run=run_id,
            create_dir=True,
        )
        prov_path = out_path.with_name(out_path.name.replace("_eeg.fif", "_provenance.json"))

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

        LOGGER.info("Saving denoised raw to %s", out_path)
        denoised_raw.save(out_path, overwrite=True, verbose="ERROR")

        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, cls=NumpyEncoder, indent=2)

        result["qc_record"] = preproc_qc.build_preproc_qc_run_record(
            profile=preproc_qc.get_preproc_qc_profile("denoise"),
            reports_root=reports_root,
            current_raw=denoised_raw,
            current_filepath=out_path,
            output_desc=output_desc,
            raw_lookup=raw_lookup,
            previous_output_desc=input_desc,
            previous_lookup=previous_lookup,
        )
        result["success"] = True
        LOGGER.info("Denoising pipeline completed for %s. Output: %s", record_label, out_path)
        return result

    except Exception as exc:
        LOGGER.error("Failed denoising for %s: %s", subject_id, exc, exc_info=True)
        result["error"] = str(exc)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 Residual Denoising")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument(
        "--preproc_root",
        type=str,
        default=None,
        help=(
            "Directory for stage FIF/provenance artifacts "
            "(default: <bids_root>/derivatives/preproc)"
        ),
    )
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Directory for reports/logs (default: <cwd>/results/reports/preproc)",
    )

    parser.add_argument(
        "--subjects", nargs="+", help="List of specific subjects (e.g. sub-001 sub-002)"
    )
    parser.add_argument("--start-from", type=str, help="Start processing from this subject ID")
    parser.add_argument(
        "--all", action="store_true", help="Process all subjects found in Stage 1 outputs"
    )
    parser.add_argument("--test", action="store_true", help="Run on a small subset (5 subjects)")
    parser.add_argument("--random", action="store_true", help="Select random subjects in test mode")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip subjects with existing output"
    )

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument(
        "--reports-only", action="store_true", help="Skip processing, only generate dataset report"
    )
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
    preproc_root = bids.get_preproc_root(bids_root)
    reports_root = report_paths.default_reports_root(bids_root)
    preproc_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    log_file = reports_root / "logs" / "denoise_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    if args.config:
        with open(args.config, encoding="utf-8") as f:
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

    subjects_found = sorted({bids.parse_bids_components(f)["subject"] for f in files})
    LOGGER.info("Found %d stage-1 runs across %d subjects.", len(files), len(subjects_found))

    subjects_to_process = select_subjects(
        subjects_found,
        selected_subjects=args.subjects,
        start_from=args.start_from,
        use_test=args.test,
        use_random_test=args.random,
        use_all=args.all,
    )
    if not subjects_to_process:
        if args.start_from:
            LOGGER.error("No subjects found starting from study_id %s.", args.start_from)
            sys.exit(1)
        LOGGER.warning(
            "No selection criteria provided (use --all, --test, --subjects, or --start-from)."
        )
        parser.print_help()
        sys.exit(0)
    subjects_to_process = set(subjects_to_process)
    LOGGER.info("Selected %d subjects to process.", len(subjects_to_process))

    profile = preproc_qc.get_preproc_qc_profile("denoise")
    raw_lookup = preproc_qc.load_raw_pre_base_lookup(reports_root)
    previous_lookup = preproc_qc.load_stage_run_lookup(
        reports_root,
        preproc_qc.get_preproc_qc_stage_name("correct", output_desc=input_desc),
    )
    existing_results: list[dict[str, object]] = []

    if args.skip_existing or args.reports_only:
        existing_runs = 0
        for input_file in files:
            sid = bids.parse_bids_components(input_file)["subject"]
            if sid not in subjects_to_process:
                continue
            comps = bids.parse_bids_components(input_file)
            out_file = bids.get_stage_output_path(
                subject=sid,
                preproc_root=preproc_root,
                desc=output_desc,
                session=comps.get("session"),
                task=args.condition if args.condition else comps.get("task"),
                run=comps.get("run"),
            )
            if not out_file.exists():
                continue
            existing_runs += 1
            try:
                existing_results.append(
                    {
                        "success": True,
                        "subject_id": sid,
                        "qc_record": preproc_qc.collect_existing_preproc_qc_record(
                            profile=profile,
                            reports_root=reports_root,
                            filepath=out_file,
                            output_desc=output_desc,
                            previous_output_desc=input_desc,
                            raw_lookup=raw_lookup,
                            previous_lookup=previous_lookup,
                        ),
                        "error": "",
                    }
                )
            except Exception as exc:
                LOGGER.error(
                    "Failed rebuilding existing denoise QC record for %s (%s): %s",
                    sid,
                    input_file.name,
                    exc,
                    exc_info=True,
                )
                existing_results.append(
                    {"success": False, "subject_id": sid, "qc_record": None, "error": str(exc)}
                )
        if existing_runs:
            LOGGER.info("Skipping %d existing denoise outputs.", existing_runs)

    files_to_process = []
    existing_run_keys = set()
    for result in existing_results:
        record = result.get("qc_record")
        if isinstance(record, dict):
            existing_run_keys.add(record.get("run_key"))
    for input_file in files:
        sid = bids.parse_bids_components(input_file)["subject"]
        if sid not in subjects_to_process:
            continue
        ids = report_paths.build_bids_report_ids(input_file)
        if ids["run_key"] in existing_run_keys:
            continue
        files_to_process.append(input_file)

    if args.reports_only:
        files_to_process = []

    if not files_to_process and not existing_results:
        LOGGER.warning("No subjects left to process.")
        sys.exit(0)

    success_ids: list[str] = []
    failed_ids: list[str] = []
    qc_run_records: list[dict[str, object]] = []
    qc_subject_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for result in existing_results:
        if result.get("success") and result.get("qc_record") is not None:
            record = result["qc_record"]
            qc_run_records.append(record)
            qc_subject_groups[record["subject_session_key"]].append(record)
            preproc_qc.write_subject_preproc_qc_report(
                reports_root,
                qc_subject_groups[record["subject_session_key"]],
                profile=profile,
                output_desc=output_desc,
            )
            success_ids.append(str(record["run_prefix"]))
        else:
            failed_ids.append(str(result["subject_id"]))

    for input_file in files_to_process:
        sid = bids.parse_bids_components(input_file)["subject"]
        run_label = str(report_paths.build_bids_report_ids(input_file)["run_prefix"])
        LOGGER.info("Processing %s...", run_label)
        result = run_denoising_pipeline(
            subject_id=sid,
            bids_root=bids_root,
            config=config,
            preproc_root=preproc_root,
            reports_root=reports_root,
            input_path=input_file,
            condition_name=args.condition,
            input_desc=input_desc,
            output_desc=output_desc,
            raw_lookup=raw_lookup,
            previous_lookup=previous_lookup,
        )
        if result.get("success") and result.get("qc_record") is not None:
            record = result["qc_record"]
            success_ids.append(str(record["run_prefix"]))
            qc_run_records.append(record)
            qc_subject_groups[record["subject_session_key"]].append(record)
            preproc_qc.write_subject_preproc_qc_report(
                reports_root,
                qc_subject_groups[record["subject_session_key"]],
                profile=profile,
                output_desc=output_desc,
            )
        else:
            failed_ids.append(run_label)

    if not args.reports_only:
        LOGGER.info(
            "Batch processing complete. Success: %d, Failed: %d", len(success_ids), len(failed_ids)
        )
        LOGGER.info("Succeeded subjects: %s", sorted(success_ids))
        LOGGER.info("Failed subjects: %s", sorted(failed_ids))
    else:
        LOGGER.info("Skipping processing (--reports-only). Generating summary from existing files.")

    preproc_qc.write_preproc_qc_aggregate_reports(
        reports_root,
        qc_run_records,
        profile=profile,
        output_desc=output_desc,
    )


if __name__ == "__main__":
    main()

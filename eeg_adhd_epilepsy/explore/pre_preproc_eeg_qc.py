"""Command-line interface for raw / pre-preprocessing EEG QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from eeg_adhd_epilepsy.explore import eeg_qc
from eeg_adhd_epilepsy.utils.qc_annotations import (
    compute_special_event_counts,
    crop_raw_after_reference_event,
    summarize_annotations,
)
from eeg_adhd_epilepsy.utils.qc_config import BASIC_1020_CHANNELS, KNOWN_EVENT_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated EEG QC (no preprocessing).")
    parser.add_argument(
        "--input_dir", required=True, type=Path, help="BIDS root directory with raw EEG files."
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Directory to store QC outputs."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Jobs for parallel processing (-1 for all cores)."
    )
    parser.add_argument(
        "--generate_subject_reports", action="store_true", help="Create per-subject HTML reports."
    )
    parser.add_argument(
        "--save_json", action="store_true", help="Also save metrics to qc_report.json."
    )
    parser.add_argument(
        "--skip_figures", action="store_true", help="Skip all figure generation (CSV/JSON only)."
    )
    parser.add_argument(
        "--subjects_list", type=Path, help="File with subject IDs to include (one per line)."
    )
    parser.add_argument(
        "--amplitude_threshold", type=float, default=500.0, help="Max amplitude threshold in uV."
    )
    parser.add_argument("--min_duration", type=float, default=5.0, help="Minimum duration in minutes.")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Maximum duration in minutes.")
    parser.add_argument("--bids_session", default="01", help="BIDS session entity, e.g., '01'.")
    parser.add_argument("--bids_task", default="RESTING", help="BIDS task entity, e.g., 'RESTING'.")
    parser.add_argument("--bids_run", default="01", help="BIDS run entity, e.g., '01'.")
    parser.add_argument("--bids_acq", default=None, help="BIDS acquisition entity if any.")
    parser.add_argument("--bids_proc", default=None, help="BIDS processing label if any.")
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum frequency for PSD.")
    parser.add_argument("--fmax", type=float, default=60.0, help="Maximum frequency for PSD.")
    parser.add_argument("--line_freq", type=float, default=60.0, help="Mains frequency for line-noise metric.")
    parser.add_argument("--highpass", type=float, default=0.5, help="High-pass filter cutoff for QC (Hz).")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    return parser.parse_args()


def run_pre_qc_for_file(
    filepath: Path,
    input_dir: Path,
    standard_names: set[str],
    args: object,
    output_dirs: Dict[str, Path],
    logger: object,
) -> Dict[str, object]:
    """QC for raw/pre-preprocessing data."""
    subject_id = eeg_qc.parse_subject_id(filepath)
    metrics: Dict[str, object] = {
        "filepath": str(filepath),
        "subject_id": subject_id,
        "duration_min": float("nan"),
        "actual_signal_start_sec": float("nan"),
        "empty_start_sec": float("nan"),
        "actual_signal_end_sec": float("nan"),
        "empty_end_sec": float("nan"),
        "meas_date": "",
        "sfreq": float("nan"),
        "n_channels": 0,
        "channel_names": "",
        "n_channels_1020_match": 0,
        "non_standard_channels": "",
        "n_flat_channels": 0,
        "n_noisy_channels": 0,
        "pct_bad_channels": float("nan"),
        "amplitude_mean_uv": float("nan"),
        "amplitude_median_uv": float("nan"),
        "amplitude_std_uv": float("nan"),
        "amplitude_min_uv": float("nan"),
        "amplitude_max_uv": float("nan"),
        "alpha_peak_hz": float("nan"),
        "line_noise_ratio_mean": float("nan"),
        "line_noise_ratio_max": float("nan"),
        "hf_lf_ratio_mean": float("nan"),
        "hf_lf_ratio_max": float("nan"),
        "aperiodic_slope_mean": float("nan"),
        "aperiodic_slope_std": float("nan"),
        "flag_bad": False,
        "flag_reasons": "",
        "flag_category": "usable",
        "event_counts": "",
        "error": "",
    }
    band_power_fields = {f"band_power_{band}": float("nan") for band in eeg_qc.BAND_LIMITS}
    metrics.update(band_power_fields)
    analysis_raw = None
    basic_picks: List[str] = []
    montage_info: Dict[str, object] = {}

    analysis_start_offset = 0.0
    original_duration = float("nan")
    try:
        raw = eeg_qc.load_raw(
            filepath,
            bids_root=input_dir,
            session=getattr(args, "bids_session", None),
            task=getattr(args, "bids_task", None),
            run=getattr(args, "bids_run", None),
            acquisition=getattr(args, "bids_acq", None),
            processing=getattr(args, "bids_proc", None),
        )
        raw.load_data()
        if getattr(args, "highpass", None) is not None:
            raw.filter(args.highpass, None, fir_design="firwin", verbose="ERROR")
        else:
            raw.filter(1, None, fir_design="firwin", verbose="ERROR")
        original_duration = raw.times[-1]
        analysis_start_offset = crop_raw_after_reference_event(raw, raw.annotations, logger)
        analysis_raw, basic_picks, montage_info = eeg_qc.prepare_channel_selection(raw, standard_names, logger)
    except Exception as exc:  # pragma: no cover - defensive branch
        err_msg = f"Failed to read {filepath.name}: {exc}"
        logger.error(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True, "flag_reasons": "load_error"})
        return metrics

    try:
        meta = eeg_qc.extract_metadata(raw)
        metrics.update(meta)

        metrics["n_channels_1020_match"] = montage_info["n_channels_1020_match"]
        metrics["non_standard_channels"] = ",".join(montage_info["non_standard_channels"])
        metrics["channel_names"] = ",".join(basic_picks) if basic_picks else ",".join(meta.get("channel_names", []))
        pct_missing_1020 = montage_info["pct_missing_1020"]
        metrics["pct_missing_1020"] = pct_missing_1020

        amp_stats = eeg_qc.compute_channel_amplitude_stats(analysis_raw, basic_picks)
        metrics.update(
            {
                "amplitude_mean_uv": amp_stats["mean"],
                "amplitude_median_uv": amp_stats["median"],
                "amplitude_std_uv": amp_stats["std"],
                "amplitude_min_uv": amp_stats["min"],
                "amplitude_max_uv": amp_stats["max"],
            }
        )

        noise_info = eeg_qc.detect_flat_and_noisy_channels(analysis_raw, basic_picks)
        metrics["n_flat_channels"] = noise_info["n_flat_channels"]
        metrics["n_noisy_channels"] = noise_info["n_noisy_channels"]
        metrics["pct_bad_channels"] = noise_info["pct_bad_channels"]

        analysis_duration = raw.times[-1] if raw is not None else float("nan")
        onset_sec = analysis_start_offset
        offset_sec = analysis_start_offset + (analysis_duration if np.isfinite(analysis_duration) else 0.0)
        if np.isfinite(original_duration):
            offset_sec = min(offset_sec, original_duration)
            empty_end = max(original_duration - offset_sec, 0.0)
        else:
            empty_end = 0.0
        metrics["empty_start_sec"] = onset_sec
        metrics["actual_signal_start_sec"] = onset_sec
        metrics["empty_end_sec"] = empty_end
        metrics["actual_signal_end_sec"] = offset_sec

        spec, psd, freqs, alpha_peak, band_powers = eeg_qc.compute_psd_metrics(
            analysis_raw, basic_picks, fmin=getattr(args, "fmin", 1.0), fmax=getattr(args, "fmax", 60.0)
        )
        metrics["alpha_peak_hz"] = alpha_peak
        metrics.update({f"band_power_{k}": v for k, v in band_powers.items()})

        line_noise_mean, line_noise_ratios = eeg_qc.compute_line_noise_index(
            psd, freqs, line_freq=getattr(args, "line_freq", 60.0)
        )
        metrics["line_noise_ratio_mean"] = line_noise_mean
        metrics["line_noise_ratio_max"] = float(np.nanmax(line_noise_ratios) if line_noise_ratios.size else np.nan)
        hf_ratio_mean, hf_ratio_max = eeg_qc.compute_hf_lf_ratio(
            psd, freqs, hf_band=(30.0, 100.0), lf_band=(1.0, 30.0)
        )
        metrics["hf_lf_ratio_mean"] = hf_ratio_mean
        metrics["hf_lf_ratio_max"] = hf_ratio_max
        slope_mean, slope_std, _ = eeg_qc.compute_aperiodic_slope(psd, freqs, fmin=1.0, fmax=30.0)
        metrics["aperiodic_slope_mean"] = slope_mean
        metrics["aperiodic_slope_std"] = slope_std

        annotation_counts = summarize_annotations(raw.annotations)
        metrics["event_counts"] = (
            json.dumps(annotation_counts, ensure_ascii=False) if annotation_counts else ""
        )
        metrics["eyes_open_event_count"] = int(annotation_counts.get("Eyes Open", 0))
        metrics["eyes_closed_event_count"] = int(annotation_counts.get("Eyes Closed", 0))
        metrics["movement_event_count"] = int(annotation_counts.get("Movement", 0))
        metrics["artefact_event_count"] = int(annotation_counts.get("Artefact", 0))
        metrics["effort_event_count"] = int(annotation_counts.get("Effort", 0))
        metrics["pat_montage_event_count"] = int(annotation_counts.get("PAT Montage", 0))
        metrics["hv_event_count"] = int(annotation_counts.get("HV", 0))
        metrics["post_hv_event_count"] = int(annotation_counts.get("Post-HV", 0))
        metrics["photo_event_count"] = int(annotation_counts.get("PHOTO", 0))
        metrics["yawning_coughing_event_count"] = int(annotation_counts.get("Yawning/Coughing", 0))
        metrics["jaw_face_tension_event_count"] = int(annotation_counts.get("Jaw/Face Tension", 0))
        metrics["sleepy_event_count"] = int(annotation_counts.get("Sleepy", 0))
        metrics["sleep_event_count"] = int(annotation_counts.get("Sleep", 0))
        metrics["collaboration_event_count"] = int(annotation_counts.get("Collaboration", 0))
        metrics["emotion_behavior_event_count"] = int(annotation_counts.get("Emotion/Behavior", 0))
        metrics["oral_activity_event_count"] = int(annotation_counts.get("Oral Activity", 0))
        metrics["eye_movement_event_count"] = int(annotation_counts.get("Eye Movement", 0))
        metrics["wakefulness_event_count"] = int(annotation_counts.get("Wakefulness", 0))
        metrics["respiration_event_count"] = int(annotation_counts.get("Respiration", 0))
        special_counts = compute_special_event_counts(raw.annotations)
        metrics["sensor_action_keyword_event_count"] = special_counts["sensor_action_keyword_events"]
        metrics["eye_movement_keyword_event_count"] = special_counts["eye_movement_keyword_events"]
        metrics["clinical_comment_event_count"] = special_counts["clinical_comment_events"]

        condition_flags = eeg_qc.evaluate_condition_flags({"raw": {"pct_retained": 1.0}})
        metrics["condition_flags"] = condition_flags
        flag_category, reasons = eeg_qc.evaluate_subject_flag(metrics)
        metrics["flag_bad"] = flag_category != "usable"
        metrics["flag_category"] = flag_category
        metrics["flag_reasons"] = ";".join(reasons)

        if getattr(args, "generate_subject_reports", False):
            fig_psd_all = fig_psd_avg = fig_amp_hist = fig_var_topo = None
            fig_raw_segment_start = fig_raw_segment_end = fig_events = None

            if not getattr(args, "skip_figures", False):
                if spec is not None and psd.size > 0:
                    try:
                        fig_psd_all, fig_psd_avg = eeg_qc.plot_psd_figures(spec, freqs, psd)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("PSD plotting failed for %s: %s", filepath.name, exc)
                if amp_stats["per_channel"].size > 0:
                    try:
                        fig_amp_hist = eeg_qc.plot_amplitude_histogram(amp_stats)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Amplitude histogram failed for %s: %s", filepath.name, exc)
                if analysis_raw is not None:
                    try:
                        fig_var_topo = eeg_qc.plot_channel_variance_topomap(analysis_raw)
                        safe_onset = onset_sec if np.isfinite(onset_sec) else 0.0
                        fig_raw_segment_start = eeg_qc.plot_raw_segment(
                            analysis_raw, max(safe_onset, 0.0), title="Raw Segment - Start (10s)"
                        )
                        if np.isfinite(offset_sec):
                            last_start = max(offset_sec - 10.0, 0.0)
                        else:
                            last_start = max(analysis_raw.times[-1] - 10.0, 0.0)
                        fig_raw_segment_end = eeg_qc.plot_raw_segment(
                            analysis_raw, last_start, title="Raw Segment - End (10s)"
                        )
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Raw/variance plotting failed for %s: %s", filepath.name, exc)
                if annotation_counts:
                    try:
                        fig_events = eeg_qc.plot_events_distribution(annotation_counts)
                    except Exception as exc:  # pragma: no cover - defensive branch
                        logger.warning("Event count plotting failed for %s: %s", filepath.name, exc)

            report_path = output_dirs["subject_reports"] / f"{subject_id}_qc_report.html"
            try:
                eeg_qc.create_subject_report(
                    analysis_raw if analysis_raw is not None else raw,
                    metrics,
                    subject_id,
                    report_path,
                    fig_psd_all,
                    fig_psd_avg,
                    fig_amp_hist,
                    fig_var_topo,
                    fig_raw_segment_start,
                    fig_raw_segment_end,
                    fig_events,
                )
            except Exception as fig_exc:  # pragma: no cover - defensive branch
                logger.warning("Report generation failed for %s: %s", filepath.name, fig_exc)

    except Exception as exc:  # pragma: no cover - defensive branch
        err_msg = f"Processing failed for {filepath.name}: {exc}"
        logger.exception(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True})
    finally:
        plt.close("all")
        try:
            raw.close()
        except Exception:
            pass

    return metrics


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    subject_reports_dir = output_dir / "subject_reports"
    fig_dir = output_dir / "figures"
    log_dir = output_dir / "logs"

    logger = eeg_qc.setup_logging(log_dir / "qc_processing.log", args.log_level)
    logger.info("Starting EEG QC (pre-preprocessing)")

    subjects_filter = eeg_qc.read_subjects_list(args.subjects_list)
    files = eeg_qc.discover_bids_files(
        bids_root=args.input_dir,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        suffix="eeg",
        extension=".vhdr",
        subjects_filter=subjects_filter,
    )
    if not files:
        logger.error("No BIDS EEG (.vhdr) files found in %s with specified filters", args.input_dir)
        sys.exit(1)

    standard_names = {ch.lower() for ch in BASIC_1020_CHANNELS}
    logger.info("Found %d files to process", len(files))

    output_dirs = {"subject_reports": subject_reports_dir, "figures": fig_dir, "logs": log_dir}
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with eeg_qc.tqdm_joblib(tqdm(total=len(files), desc="Processing EEG files")):
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_pre_qc_for_file)(
                filepath=f,
                input_dir=args.input_dir,
                standard_names=standard_names,
                args=args,
                output_dirs=output_dirs,
                logger=logger,
            )
            for f in files
        )

    dataset_stats = eeg_qc.compute_dataset_stats(results)
    eeg_qc.apply_dataset_outlier_flags(results, dataset_stats)

    df = pd.DataFrame(results)
    csv_path = output_dir / "qc_report_pre_preproc.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV report to %s", csv_path)

    if args.save_json:
        json_path = output_dir / "qc_report_pre_preproc.json"
        json_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved JSON report to %s", json_path)

    flags_counter = eeg_qc.summarize_flags(results)
    unknown_events = eeg_qc.collect_unknown_events(results, KNOWN_EVENT_LABELS)
    meas_datetimes = eeg_qc.load_meas_datetimes(args.input_dir)
    fig_paths = {}
    if not args.skip_figures:
        fig_paths = eeg_qc.save_figures(df, flags_counter, fig_dir, meas_datetimes)
        summary_report_path = output_dir / "qc_summary_report_pre_preproc.html"
        eeg_qc.create_summary_report(
            df,
            fig_paths,
            summary_report_path,
            len(files),
            flags_counter,
            unknown_events,
        )
        logger.info("Saved summary HTML report to %s", summary_report_path)

    logger.info(
        "QC finished. Total files: %d, flagged: %d",
        len(files),
        int(df["flag_bad"].sum()) if "flag_bad" in df else 0,
    )


if __name__ == "__main__":
    main()

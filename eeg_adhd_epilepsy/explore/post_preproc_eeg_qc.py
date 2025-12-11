"""Command-line interface for post-preprocessing QC and pipeline comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from eeg_adhd_epilepsy.explore import eeg_qc
from eeg_adhd_epilepsy.utils.qc_config import BASIC_1020_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG QC after preprocessing/epoching.")
    parser.add_argument("--input_dir_preproc", required=True, type=Path, help="Primary preprocessed BIDS root.")
    parser.add_argument(
        "--input_dir_preproc_b",
        type=Path,
        help="Optional second preprocessed BIDS root for pipeline comparison.",
    )
    parser.add_argument(
        "--input_dir_raw",
        type=Path,
        help="Optional raw BIDS root for before/after PSD overlays.",
    )
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to store QC outputs.")
    parser.add_argument("--subjects_list", type=Path, help="File with subject IDs to include (one per line).")
    parser.add_argument("--n_jobs", type=int, default=1, help="Jobs for parallel processing (-1 for all cores).")
    parser.add_argument(
        "--generate_subject_reports", action="store_true", help="Create per-subject HTML reports."
    )
    parser.add_argument("--save_json", action="store_true", help="Also save metrics to qc_report.json.")
    parser.add_argument("--skip_figures", action="store_true", help="Skip all figure generation.")
    parser.add_argument("--preproc_suffix", default="epo", help="BIDS suffix for preprocessed files (default: epo).")
    parser.add_argument("--preproc_extension", default=".fif", help="File extension for preprocessed files.")
    parser.add_argument("--bids_session", default=None, help="BIDS session entity.")
    parser.add_argument("--bids_task", default=None, help="BIDS task entity.")
    parser.add_argument("--bids_run", default=None, help="BIDS run entity.")
    parser.add_argument("--bids_acq", default=None, help="BIDS acquisition entity.")
    parser.add_argument("--bids_proc", default=None, help="BIDS processing label.")
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum frequency for PSD.")
    parser.add_argument("--fmax", type=float, default=60.0, help="Maximum frequency for PSD.")
    parser.add_argument("--line_freq", type=float, default=60.0, help="Mains frequency for line-noise metric.")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    return parser.parse_args()


def run_post_qc_for_file(
    filepath: Path,
    standard_names: set[str],
    args: object,
    logger: object,
    before_raw: mne.io.BaseRaw | None = None,
    subject_reports_dir: Path | None = None,
    generate_report: bool = False,
    skip_figures: bool = False,
) -> Dict[str, object]:
    """QC for post-preprocessed data (Epochs or Raw)."""
    subject_id = eeg_qc.parse_subject_id(filepath)
    metrics: Dict[str, object] = {
        "filepath": str(filepath),
        "subject_id": subject_id,
        "flag_bad": False,
        "flag_reasons": "",
        "flag_category": "usable",
        "error": "",
    }
    band_power_fields = {f"band_power_{band}": float("nan") for band in eeg_qc.BAND_LIMITS}
    metrics.update(band_power_fields)

    data_obj = None
    try:
        if filepath.suffix == ".fif":
            try:
                data_obj = mne.read_epochs(filepath, preload=True, verbose="ERROR")
                metrics["data_type"] = "epochs"
            except Exception:
                data_obj = mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")
                metrics["data_type"] = "raw"
        else:
            data_obj = mne.read_epochs(filepath, preload=True, verbose="ERROR")
            metrics["data_type"] = "epochs"
    except Exception as exc:
        err_msg = f"Failed to load {filepath.name}: {exc}"
        logger.error(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True, "flag_reasons": "load_error"})
        return metrics

    try:
        analysis_raw = None
        picks_names: List[str] = []
        spec = None
        psd = np.array([])
        freqs = np.array([])

        base_info = data_obj.info
        meta = {
            "sfreq": float(base_info["sfreq"]),
            "n_channels": len(mne.pick_info(base_info, mne.pick_types(base_info, eeg=True)).ch_names),
            "channel_names": ",".join(mne.pick_info(base_info, mne.pick_types(base_info, eeg=True)).ch_names),
        }
        metrics.update(meta)

        if isinstance(data_obj, mne.Epochs):
            epoch_duration = float(data_obj.tmax - data_obj.tmin)
            metrics["n_epochs_total"] = len(data_obj.drop_log)
            metrics["n_epochs_kept"] = len(data_obj)
            metrics["usable_minutes"] = metrics["n_epochs_kept"] * epoch_duration / 60.0
            metrics["duration_min"] = metrics["usable_minutes"]
        else:
            metrics["duration_min"] = float(data_obj.times[-1] / 60.0)

        if isinstance(data_obj, mne.Epochs):
            basic_picks = mne.pick_channels(data_obj.ch_names, include=BASIC_1020_CHANNELS, ordered=False)
            picks_names = [data_obj.ch_names[idx] for idx in basic_picks]
            spec, psd, freqs, alpha_peak, band_powers = eeg_qc.compute_psd_metrics(
                data_obj, picks_names, fmin=getattr(args, "fmin", 1.0), fmax=getattr(args, "fmax", 60.0)
            )
        else:
            analysis_raw, picks_names, _ = eeg_qc.prepare_channel_selection(data_obj, standard_names, logger)
            spec, psd, freqs, alpha_peak, band_powers = eeg_qc.compute_psd_metrics(
                analysis_raw, picks_names, fmin=getattr(args, "fmin", 1.0), fmax=getattr(args, "fmax", 60.0)
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
        hurst_values, hurst_median, hurst_std = eeg_qc.compute_hurst_per_channel(
            data_obj if isinstance(data_obj, mne.Epochs) else analysis_raw, picks_names
        )
        metrics["hurst_median"] = hurst_median
        metrics["hurst_std"] = hurst_std
        metrics["hurst_values"] = hurst_values.tolist() if hurst_values.size else []

        if isinstance(data_obj, mne.Epochs):
            amp_stats = eeg_qc.compute_epoch_amplitude_stats(data_obj, picks_names)
            metrics.update(amp_stats)
            condition_retention, kept_conditions = eeg_qc.compute_condition_retention(
                data_obj, condition_map=getattr(args, "condition_map", None)
            )
            condition_amp = eeg_qc.compute_condition_amplitude_metrics(data_obj, kept_conditions, picks_names)
            metrics["condition_retention"] = condition_retention
            metrics["condition_amp"] = condition_amp
            rejection_breakdown = eeg_qc.compute_epoch_rejection_breakdown(data_obj)
            metrics["rejection_breakdown"] = rejection_breakdown
            metrics["condition_flags"] = eeg_qc.evaluate_condition_flags(condition_retention, condition_amp)
        else:
            amp_stats = eeg_qc.compute_channel_amplitude_stats(data_obj, picks_names)
            metrics["amplitude_mean_uv"] = amp_stats["mean"]
            metrics["amplitude_max_uv"] = amp_stats["max"]

        flag_category, reasons = eeg_qc.evaluate_subject_flag(metrics)
        metrics["flag_category"] = flag_category
        metrics["flag_bad"] = flag_category != "usable"
        metrics["flag_reasons"] = ";".join(reasons)

        if generate_report and subject_reports_dir is not None:
            fig_psd_all = fig_psd_avg = fig_amp_hist = fig_var_topo = None
            fig_raw_segment_start = fig_raw_segment_end = fig_events = None
            fig_overlay = None
            if not skip_figures:
                try:
                    if spec is not None and psd.size > 0:
                        fig_psd_all, fig_psd_avg = eeg_qc.plot_psd_figures(spec, freqs, psd)
                except Exception:
                    fig_psd_all = fig_psd_avg = None
                try:
                    if isinstance(data_obj, mne.Epochs):
                        data = data_obj.get_data(picks=picks_names) * 1e6
                        reshaped = data.reshape(len(data_obj), len(picks_names), -1)
                        per_channel = np.ptp(reshaped, axis=2).mean(axis=0)
                        amp_stats_fig = {
                            "mean": float(np.nanmean(per_channel)),
                            "median": float(np.nanmedian(per_channel)),
                            "std": float(np.nanstd(per_channel)),
                            "min": float(np.nanmin(per_channel)),
                            "max": float(np.nanmax(per_channel)),
                            "per_channel": per_channel,
                        }
                        fig_amp_hist = eeg_qc.plot_amplitude_histogram(amp_stats_fig)
                    elif analysis_raw is not None:
                        amp_stats_fig = eeg_qc.compute_channel_amplitude_stats(analysis_raw, picks_names)
                        fig_amp_hist = eeg_qc.plot_amplitude_histogram(amp_stats_fig)
                        fig_var_topo = eeg_qc.plot_channel_variance_topomap(analysis_raw)
                        fig_raw_segment_start = eeg_qc.plot_raw_segment(
                            analysis_raw, 0.0, title="Cleaned Segment - Start (10s)"
                        )
                        fig_raw_segment_end = eeg_qc.plot_raw_segment(
                            analysis_raw,
                            max(analysis_raw.times[-1] - 10.0, 0.0),
                            title="Cleaned Segment - End (10s)",
                        )
                except Exception:
                    fig_amp_hist = fig_var_topo = None
                if before_raw is not None and psd.size:
                    try:
                        before_analysis, before_picks, _ = eeg_qc.prepare_channel_selection(
                            before_raw, standard_names, logger
                        )
                        _, before_psd, before_freqs, _, _ = eeg_qc.compute_psd_metrics(
                            before_analysis,
                            before_picks,
                            fmin=getattr(args, "fmin", 1.0),
                            fmax=getattr(args, "fmax", 60.0),
                        )
                        fig_overlay = eeg_qc.plot_psd_overlay(before_freqs, before_psd, freqs, psd)
                    except Exception:
                        fig_overlay = None

            report_path = subject_reports_dir / f"{subject_id}_post_qc_report.html"
            try:
                eeg_qc.create_subject_report(
                    data_obj if isinstance(data_obj, mne.Epochs) else (analysis_raw or data_obj),
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
                    fig_psd_overlay_before_after=fig_overlay,
                )
            except Exception:
                logger.warning("Failed to build post-preproc report for %s", filepath.name)

    except Exception as exc:
        err_msg = f"Post-preproc processing failed for {filepath.name}: {exc}"
        logger.exception(err_msg)
        metrics.update({"error": err_msg, "flag_bad": True})
    finally:
        plt.close("all")

    return metrics


def compute_pipeline_deltas(metrics_a: Dict[str, object], metrics_b: Dict[str, object]) -> Dict[str, float]:
    keys = [
        "usable_minutes",
        "hf_lf_ratio_mean",
        "line_noise_ratio_mean",
        "aperiodic_slope_mean",
        "amplitude_mean_uv",
        "alpha_peak_hz",
    ]
    deltas: Dict[str, float] = {}
    for key in keys:
        a_val = metrics_a.get(key)
        b_val = metrics_b.get(key)
        if a_val is None or b_val is None:
            deltas[key] = float("nan")
            continue
        try:
            deltas[key] = float(b_val) - float(a_val)
        except Exception:
            deltas[key] = float("nan")
    return deltas


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    subject_reports_dir = output_dir / "subject_reports"
    fig_dir = output_dir / "figures"
    log_dir = output_dir / "logs"

    logger = eeg_qc.setup_logging(log_dir / "qc_processing.log", args.log_level)
    logger.info("Starting EEG QC (post-preprocessing)")

    subjects_filter = eeg_qc.read_subjects_list(args.subjects_list)
    files_a = eeg_qc.discover_bids_files(
        bids_root=args.input_dir_preproc,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        suffix=args.preproc_suffix,
        extension=args.preproc_extension,
        subjects_filter=subjects_filter,
    )
    if not files_a:
        logger.error("No preprocessed files found in %s", args.input_dir_preproc)
        sys.exit(1)

    files_b = []
    if args.input_dir_preproc_b:
        files_b = eeg_qc.discover_bids_files(
            bids_root=args.input_dir_preproc_b,
            session=args.bids_session,
            task=args.bids_task,
            run=args.bids_run,
            acquisition=args.bids_acq,
            processing=args.bids_proc,
            suffix=args.preproc_suffix,
            extension=args.preproc_extension,
            subjects_filter=subjects_filter,
        )
    map_b = {eeg_qc.parse_subject_id(p): p for p in files_b}

    raw_map = {}
    if args.input_dir_raw:
        raw_candidates = eeg_qc.discover_bids_files(
            bids_root=args.input_dir_raw,
            session=args.bids_session,
            task=args.bids_task,
            run=args.bids_run,
            acquisition=args.bids_acq,
            processing=args.bids_proc,
            suffix="eeg",
            extension=".vhdr",
            subjects_filter=subjects_filter,
        )
        raw_map = {eeg_qc.parse_subject_id(p): p for p in raw_candidates}

    output_dirs = {"subject_reports": subject_reports_dir, "figures": fig_dir, "logs": log_dir}
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    standard_names = {ch.lower() for ch in BASIC_1020_CHANNELS}

    def _process_file(file_a: Path) -> List[Dict[str, object]]:
        subj = eeg_qc.parse_subject_id(file_a)
        before_raw = None
        if subj in raw_map:
            try:
                before_raw = eeg_qc.load_raw(
                    raw_map[subj],
                    bids_root=args.input_dir_raw,
                    session=args.bids_session,
                    task=args.bids_task,
                    run=args.bids_run,
                    acquisition=args.bids_acq,
                    processing=args.bids_proc,
                )
                before_raw.load_data()
            except Exception:
                before_raw = None
        metrics_list: List[Dict[str, object]] = []
        metrics_a = run_post_qc_for_file(
            filepath=file_a,
            standard_names=standard_names,
            args=args,
            logger=logger,
            before_raw=before_raw,
            subject_reports_dir=subject_reports_dir if args.generate_subject_reports else None,
            generate_report=args.generate_subject_reports,
            skip_figures=args.skip_figures,
        )
        metrics_a["pipeline"] = "A"
        metrics_list.append(metrics_a)

        file_b = map_b.get(subj)
        if file_b:
            metrics_b = run_post_qc_for_file(
                filepath=file_b,
                standard_names=standard_names,
                args=args,
                logger=logger,
                before_raw=before_raw,
                subject_reports_dir=subject_reports_dir if args.generate_subject_reports else None,
                generate_report=args.generate_subject_reports,
                skip_figures=args.skip_figures,
            )
            metrics_b["pipeline"] = "B"
            metrics_list.append(metrics_b)
            deltas = compute_pipeline_deltas(metrics_a, metrics_b)
            metrics_a.update({f"delta_{k}": v for k, v in deltas.items()})

        if before_raw is not None:
            try:
                before_raw.close()
            except Exception:
                pass
        return metrics_list

    records: List[Dict[str, object]] = []
    with eeg_qc.tqdm_joblib(tqdm(total=len(files_a), desc="Processing preprocessed files")):
        nested_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_process_file)(file_a) for file_a in files_a
        )
    for item in nested_results:
        records.extend(item)

    dataset_stats = eeg_qc.compute_dataset_stats(records)
    eeg_qc.apply_dataset_outlier_flags(records, dataset_stats)
    df = pd.DataFrame(records)
    csv_path = output_dir / "qc_report_post_preproc.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV report to %s", csv_path)

    if args.save_json:
        json_path = output_dir / "qc_report_post_preproc.json"
        json_path.write_text(json.dumps(records, indent=2))
        logger.info("Saved JSON report to %s", json_path)

    flags_counter = eeg_qc.summarize_flags(records)
    fig_paths = {}
    if not args.skip_figures:
        fig_paths = eeg_qc.save_figures(df, flags_counter, fig_dir, meas_datetimes=None)
        summary_report_path = output_dir / "qc_summary_report_post_preproc.html"
        eeg_qc.create_summary_report(
            df,
            fig_paths,
            summary_report_path,
            len(files_a),
            flags_counter,
            unknown_events=None,
        )
        logger.info("Saved summary HTML report to %s", summary_report_path)

    logger.info(
        "Post-preproc QC finished. Files: %d (pipeline A) + %d (pipeline B), flagged: %d",
        len(files_a),
        len(files_b),
        int(df["flag_bad"].sum()) if "flag_bad" in df else 0,
    )


if __name__ == "__main__":
    main()

"""Unified command-line interface for cleaned (post-preprocessing) EEG QC."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import mne
from joblib import Parallel, delayed
from tqdm import tqdm

import eeg_adhd_epilepsy.io.bids as bids_io
import eeg_adhd_epilepsy.qc.metrics as qc_metrics
import eeg_adhd_epilepsy.utils.stats as qc_stats
import eeg_adhd_epilepsy.reports.qc as qc_reports
import eeg_adhd_epilepsy.viz.qc as qc_viz
import eeg_adhd_epilepsy.features.spectral as feat_spectral
import eeg_adhd_epilepsy.features.time as feat_time
from eeg_adhd_epilepsy.preproc.utils import load_segments_for_raw
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib
from eeg_adhd_epilepsy.utils.config import ANNOTATION_INTEREST_MAP

KNOWN_EVENT_LABELS = set(ANNOTATION_INTEREST_MAP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated EEG QC for cleaned data.")
    parser.add_argument("--input_dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="QC output directory.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (-1 for all).")
    parser.add_argument(
        "--analyze_segments", action="store_true", help="Enable segment-level analysis."
    )
    # File filters
    parser.add_argument("--subjects_list", type=Path, help="Filter to subjects in file.")
    parser.add_argument("--bids_session", help="BIDS session.")
    parser.add_argument("--bids_task", default="RESTING", help="BIDS task (default: RESTING).")
    parser.add_argument("--bids_run", help="BIDS run.")
    parser.add_argument("--bids_acq", help="BIDS acquisition.")
    parser.add_argument("--bids_proc", help="BIDS processing (e.g. clean).")
    
    # Parameters
    parser.add_argument("--amplitude_threshold", type=float, default=500.0, help="Amp thresh (uV).")
    parser.add_argument("--line_freq", type=float, default=60.0, help="Line frequency.")
    parser.add_argument("--min_segment_duration", type=float, default=5.0, help="Min segment length (s).")
    
    parser.add_argument("--skip_reports", action="store_true", help="Skip HTML reports.")
    parser.add_argument("--save_json", action="store_true", help="Save metrics JSON.")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _process_file(
    filepath: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
    output_dirs: Dict[str, Path],
) -> Dict[str, object]:

    subject_id = bids_io.parse_subject_id(filepath)
    result = {"subject_id": subject_id, "file_metrics": {}, "segment_metrics": [], "error": ""}
    
    try:
        raw = bids_io.load_bids_raw(
            filepath,
            bids_root=args.input_dir,
            session=args.bids_session,
            task=args.bids_task,
            run=args.bids_run,
            acquisition=args.bids_acq,
            processing=args.bids_proc,
        )
        raw.load_data()
        
        # Prepare Analysis Picks (All EEG)
        analysis_raw = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                
        # Whole File Spectral
        spec, psd, freqs, alpha_peak, band_powers, per_channel_powers = feat_spectral.compute_spectral_metrics(
             analysis_raw, picks=picks
        )
        line_noise_mean, line_noise_ratios = feat_spectral.compute_line_noise_index(psd, freqs)
        hf_lf_mean, _, hf_lf_ratios = feat_spectral.compute_hf_lf_ratio(psd, freqs)
        slope_mean, _, _, slope_per_channel = feat_spectral.compute_aperiodic_slope(psd, freqs)
        
        amp_stats = feat_time.compute_channel_amplitude_stats(analysis_raw, picks)
        noise_info = feat_time.detect_flat_and_noisy_channels(analysis_raw, picks)

        file_metrics = {
            "filepath": str(filepath),
            "subject_id": subject_id,
            "duration_min": raw.n_times / raw.info["sfreq"] / 60.0,
            "sfreq": raw.info["sfreq"],
            "amplitude_mean_uv": amp_stats["mean"],
            "amplitude_max_uv": amp_stats["max"],
            "pct_bad_channels": noise_info["pct_bad_channels"],
            "alpha_peak_hz": alpha_peak,
            "line_noise_ratio_mean": line_noise_mean,
            "hf_lf_ratio_mean": hf_lf_mean,
            "aperiodic_slope_mean": slope_mean,
        }
        # Add band powers
        file_metrics.update({f"band_power_{b}": v for b, v in band_powers.items()})
        
        # Flagging
        flag_status, reasons = qc_stats.evaluate_subject_flag(file_metrics)
        file_metrics["subject_flag"] = flag_status
        file_metrics["subject_flag_reasons"] = ";".join(reasons)
        
        result["file_metrics"] = file_metrics
        
        # --- Segment Analysis (Optional) ---
        if args.analyze_segments:
             segments_df = load_segments_for_raw(raw)
             if segments_df is not None and not segments_df.empty:
                 for _, row in segments_df.iterrows():
                     dur = float(row.get("duration", 0))
                     if dur < args.min_segment_duration:
                         continue
                     seg_start = float(row.get("t_start", 0))
                     seg_stop = float(row.get("t_stop", 0))
                     segment = qc_metrics.crop_segment(analysis_raw, seg_start, seg_stop, picks=picks)
                     if segment:
                         seg_metrics = qc_metrics.compute_segment_qc(
                             segment, picks=picks, logger=logger, line_freq=args.line_freq,
                             include_channel_metrics=not args.skip_reports
                         )
                         seg_rec = {
                             "subject_id": subject_id,
                             "segment_type": row.get("segment_type"),
                             "t_start": seg_start,
                             "duration": dur,
                         }
                         seg_rec.update(seg_metrics)
                         result["segment_metrics"].append(seg_rec)

    except Exception as e:
        logger.error(f"Error processing {subject_id}: {e}")
        result["error"] = str(e)
    
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_log = args.output_dir / "logs" / "qc_clean.log"
    setup_log.parent.mkdir(exist_ok=True)
    logger = setup_logging(setup_log, args.log_level)
    
    subjects_filter = bids_io.read_subjects_list(args.subjects_list)
    files = bids_io.discover_bids_files(
        args.input_dir,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc, # usually 'clean'
        subjects_filter=subjects_filter
    )
    
    if not files:
        logger.warning("No files found.")
        return

    output_dirs = {
        "reports": args.output_dir / "reports",
        "figures": args.output_dir / "figures"
    }
    for d in output_dirs.values(): 
        d.mkdir(exist_ok=True)

    with tqdm_joblib(tqdm(total=len(files), desc="QC Clean")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_process_file)(f, args, logger, output_dirs)
            for f in files
        )
    
    # Aggregation
    file_records = []
    segment_records = []
    
    for res in results:
        if res.get("file_metrics"):
            file_records.append(res["file_metrics"])
        if res.get("segment_metrics"):
            segment_records.extend(res["segment_metrics"])

    if file_records:
        df_files = pd.DataFrame(file_records)
        df_files.to_csv(args.output_dir / "qc_clean_files.csv", index=False)
        qc_reports.create_summary_report(df_files, args.output_dir / "qc_clean_summary.html")

    if segment_records:
        df_segments = pd.DataFrame(segment_records)
        df_segments.to_csv(args.output_dir / "qc_clean_segments.csv", index=False)
        if not args.skip_reports:
             fig_paths = qc_viz.save_segment_dataset_figures(df_segments, output_dirs["figures"])
             qc_reports.create_segment_dataset_report(
                 df_segments, fig_paths, args.output_dir / "qc_clean_segments_report.html"
             )

if __name__ == "__main__":
    main()

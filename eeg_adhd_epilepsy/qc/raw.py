"""Unified command-line interface for raw (pre-preprocessing) EEG QC."""

from __future__ import annotations

import argparse
import logging
import traceback
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
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib
from eeg_adhd_epilepsy.utils.events import crop_raw_to_recording_start
import numpy as np
from collections import defaultdict




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated EEG QC for raw data (pre-preprocessing).")
    parser.add_argument("--input_dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="QC output directory.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (-1 for all).")
    parser.add_argument(
        "--analysis_level", 
        choices=["whole", "segments", "both"], 
        default="whole",
        help="Analysis level: 'whole' (file), 'segments' (via existing CSV), or 'both'."
    )
    parser.add_argument("--segment_types", help="Comma-separated segment types to include (default: all).")
    # File filters
    parser.add_argument("--subjects_list", type=Path, help="Filter to subjects in file.")
    parser.add_argument("--bids_task", default=None, help="BIDS task (filter).")
    
    # Parameters
    parser.add_argument("--min_duration", type=float, default=5.0, help="Min file duration (min).")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max file duration (min).")
    parser.add_argument("--amplitude_threshold", type=float, default=500.0, help="Amp thresh (uV).")
    parser.add_argument("--highpass", type=float, default=0.5, help="Highpass filter (Hz).")
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
    comps = bids_io.parse_bids_components(filepath)
    subject_id = f"sub-{comps.get('subject', 'unknown')}"
    session_id = comps.get('session')

    result = {"subject_id": subject_id, "file_metrics": {}, "segment_metrics": [], "error": "", "segment_csv_path": None}
    
    subject_out_dir = output_dirs["reports"].parent / "subjects" / f"sub-{comps.get('subject', 'unknown')}"
    if session_id:
         subject_out_dir = subject_out_dir / f"ses-{session_id}"
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    
    subject_fig_dir = subject_out_dir / "figures"
    subject_fig_dir.mkdir(parents=True, exist_ok=True)

    raw = None
    file_metrics = None

    try:
        raw = bids_io.load_bids_raw(
            filepath,
            bids_root=args.input_dir,
        )
        raw.load_data()
        
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")
        except Exception as e:
            logger.warning(f"Could not set standard montage: {e}")
            
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks) == 0:
             logger.error(f"Critical: No EEG channels found even after fallback for {subject_id}.")
             result["error"] = "No EEG channels found."
             return result
        
        # --- Shared Preprocessing ---
        # 1. Basic Metadata
        meas_date = raw.info.get("meas_date")
        duration_sec = raw.n_times / raw.info["sfreq"]
        
        file_metrics = {
            "filepath": str(filepath),
            "subject_id": subject_id,
            "meas_date": meas_date.isoformat() if meas_date else "",
            "duration_min": duration_sec / 60.0,
            "sfreq": raw.info["sfreq"],
            "offset_seconds": 0.0,
        }
        
        # 2. Crop & Filter
        analysis_raw = raw.copy()
        cropped_raw = crop_raw_to_recording_start(analysis_raw)
        
        if cropped_raw is not None:
             file_metrics["offset_seconds"] = cropped_raw.first_time - analysis_raw.first_time
             if args.highpass and args.highpass > 0:
                 cropped_raw.filter(args.highpass, None, fir_design="firwin", verbose="ERROR")
        else:
             # Fallback if cropping fails (should rarely happen if raw is valid)
             cropped_raw = analysis_raw
        
        # --- Analysis Level: WHOLE ---
        if args.analysis_level in ["whole", "both"]:
             computed_metrics = qc_metrics.compute_segment_qc(
                 cropped_raw,
                 picks=picks,
                 line_freq=args.line_freq,
                 include_channel_metrics=not args.skip_reports
             )
             
             # Flagging
             file_metrics.update(computed_metrics)
             flag_status, reasons = qc_stats.evaluate_subject_flag(file_metrics)
             file_metrics["subject_flag"] = flag_status
             file_metrics["subject_flag_reasons"] = ";".join(reasons)
             
             result["file_metrics"] = file_metrics
             
             # Add as a segment record for comparison
             whole_rec = {
                 "subject_id": subject_id,
                 "segment_type": "Whole",
                 "t_start": 0.0,
                 "duration": duration_sec,
                 "flag_bad": flag_status,
                 "flag_reasons": ";".join(reasons)
             }
             whole_rec.update(computed_metrics)
             result["segment_metrics"].append(whole_rec)
        else:
             # Even if skipping "whole" QC, we should populate result["file_metrics"] with basic metadata
             result["file_metrics"] = file_metrics

        # --- Analysis Level: SEGMENTS ---
        if args.analysis_level in ["segments", "both"]:
             base_name = filepath.stem # e.g. sub-01_ses-01_task-rest_run-01_eeg
             if base_name.endswith("_eeg"):
                 base_name = base_name[:-4]
             seg_csv_path = filepath.parent / f"{base_name}_segments.csv"
             
             if seg_csv_path.exists():
                 segments_df = pd.read_csv(seg_csv_path)
                 
                 # Filter types if requested
                 if args.segment_types:
                     allowed = [t.strip() for t in args.segment_types.split(",")]
                     segments_df = segments_df[segments_df["segment_type"].isin(allowed)]
                 
                 for _, row in segments_df.iterrows():
                     dur = float(row.get("duration", 0))
                     if dur < args.min_segment_duration:
                         continue
                     seg_start = float(row.get("t_start", 0))
                     seg_stop = float(row.get("t_stop", 0))
                     
                     segment = qc_metrics.crop_segment(cropped_raw, seg_start, seg_stop, picks=picks)
                     if segment:
                         seg_metrics = qc_metrics.compute_segment_qc(
                             segment, picks=picks, line_freq=args.line_freq,
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
             else:
                 logger.warning(f"No segment definitions found at {seg_csv_path} (run qc/conditions.py first).")

    except Exception as e:
        logger.error(f"Error processing {subject_id}: {e}\n{traceback.format_exc()}")
        result["error"] = str(e)
    
    # Create per-subject report
    report_filename = f"{subject_id}_qc_report.html"
    if session_id:
        report_filename = f"{subject_id}_ses-{session_id}_qc_report.html"

    report_path = output_dirs["reports"] / report_filename
    
    if not args.skip_reports and file_metrics and raw:         
         fig_output_dir = output_dirs["figures"] / subject_id
         if session_id:
             fig_output_dir = fig_output_dir / f"ses-{session_id}"
         fig_output_dir.mkdir(parents=True, exist_ok=True)
         
         fig_paths = qc_viz.save_subject_figures(
             file_metrics, 
             raw, 
             fig_output_dir
         )
         
         qc_reports.create_subject_report(
             raw,
             file_metrics, 
             subject_id,
             report_path,
             fig_paths
         )
    
    # Process collected segments for this subject
    if result.get("segment_metrics"):         
         subj_fig_paths = {}
         
         from collections import defaultdict
         type_groups = defaultdict(list)
         for rec in result["segment_metrics"]:
             stype = rec.get("segment_type", "Unknown")
             if "per_channel_metrics" in rec:
                 type_groups[stype].append(rec["per_channel_metrics"])
                 
         # For each type, compute mean of each metric
         for stype, metrics_list in type_groups.items():
             if not metrics_list: continue
             
             # Aggregate
             agg_metrics = defaultdict(list)
             valid_ch_names = raw.ch_names # Assumption: all segments have same channels
             
             for m in metrics_list:
                 for key, arr in m.items():
                     if len(arr) > 0:
                         agg_metrics[key].append(arr)
                         
             # Compute means
             mean_metrics = {}
             for key, arrays in agg_metrics.items():
                 # Stack and mean
                 try:
                     stack = np.vstack(arrays)
                     mean_metrics[key] = np.nanmean(stack, axis=0)
                 except ValueError:
                     pass
             
             # Generate Grid Plot for this type
             # We want: Band Powers (Alpha, etc.)
             band_means = {}
             for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                 k = f"band_power_{band}"
                 if k in mean_metrics:
                     band_means[f"{band.capitalize()} Power"] = mean_metrics[k]
            
             if band_means:
                 fig = qc_viz.plot_topomap_grid(
                     band_means, 
                     raw.info, 
                     title=f"{stype} - Average Band Powers", 
                     cmap="viridis",
                     unit="uV^2",
                     ncols=3
                 )
                 if fig:
                     out_name = f"{stype.replace(' ', '')}_band_topomaps.png"
                     out_path = subject_fig_dir / out_name
                     fig.savefig(out_path, dpi=100)
                     import matplotlib.pyplot as plt
                     plt.close(fig)
                     subj_fig_paths[f"{stype}_band_topomaps"] = out_path

         df_subj_segments = pd.DataFrame(result["segment_metrics"])
         
         if "per_channel_metrics" in df_subj_segments.columns:
             df_subj_segments = df_subj_segments.drop(columns=["per_channel_metrics"])
         
         # 1. Save CSV immediately
         csv_name = f"{subject_id}_qc_segments.csv" 
         if session_id:
             csv_name = f"{subject_id}_ses-{session_id}_qc_segments.csv"
         
         csv_path = subject_out_dir / csv_name
         df_subj_segments.to_csv(csv_path, index=False)
         result["segment_csv_path"] = str(csv_path)
         
         # 2. Generate Subject Segment Report immediately
         if not args.skip_reports:
             seg_rep_name = f"{subject_id}_segments_report.html"
             if session_id:
                 seg_rep_name = f"{subject_id}_ses-{session_id}_segments_report.html"
                 
             qc_reports.create_segment_subject_report(
                 df_subj_segments,
                 subject_id=subject_id if not session_id else f"{subject_id}_ses-{session_id}",
                 output_path=subject_out_dir / seg_rep_name,
                 fig_paths=subj_fig_paths
             )
         
         # Clear from result to save memory
         del result["segment_metrics"]
    
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_log = args.output_dir / "logs" / "qc_raw.log"
    setup_log.parent.mkdir(exist_ok=True)
    logger = setup_logging(setup_log, args.log_level)
    
    subjects_filter = bids_io.read_subjects_list(args.subjects_list)
    files = bids_io.discover_bids_files(
        args.input_dir,
        task=args.bids_task,
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

    with tqdm_joblib(tqdm(total=len(files), desc="QC Raw")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_process_file)(f, args, logger, output_dirs)
            for f in files
        )
    
    # Aggregation
    file_records = []
    
    # Discovery of segment files (Done out of the loop to avoid OOM)
    segment_files = list(args.output_dir.glob("**/subjects/**/*_qc_segments.csv"))
    
    for res in results:
        if res.get("file_metrics"):
            file_records.append(res["file_metrics"])

    if file_records:
        df_files = pd.DataFrame(file_records)
        df_files.to_csv(args.output_dir / "qc_raw_files.csv", index=False)
        
        # Generate figures and compute flags counter for summary report
        from collections import Counter
        # Build flags counter from flag reasons (Compute BEFORE figure generation)
        flags_counter = Counter()
        if "subject_flag_reasons" in df_files.columns:
            for reasons_str in df_files["subject_flag_reasons"].dropna():
                if reasons_str:
                    for reason in reasons_str.split(";"):
                        reason = reason.strip()
                        if reason:
                            flags_counter[reason] += 1

        # Generate figures
        fig_paths = qc_reports.save_figures(
            df_files, 
            flags_counter, 
            output_dirs["figures"]
        )
        total_files = len(files)
        
        qc_reports.create_summary_report(
            df_files, 
            fig_paths, 
            args.output_dir / "qc_raw_summary.html",
            total_files,
            flags_counter
        )
        
    if segment_files:
        logger.info(f"Aggregating {len(segment_files)} segment files...")
        # Read and concat efficiently
        df_segments = pd.concat((pd.read_csv(f) for f in segment_files), ignore_index=True)
        
        df_segments.to_csv(args.output_dir / "qc_raw_segments.csv", index=False)
        # Generate segment dataset report
        if not args.skip_reports:
             fig_paths = qc_reports.save_segment_dataset_figures(df_segments, output_dirs["figures"])
             qc_reports.create_segment_dataset_report(
                 df_segments, fig_paths, args.output_dir / "qc_raw_segments_report.html"
             )

if __name__ == "__main__":
    main()

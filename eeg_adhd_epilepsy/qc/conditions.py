"""CLI for condition segment analysis and reporting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from tqdm import tqdm

import eeg_adhd_epilepsy.io.bids as io_bids
import eeg_adhd_epilepsy.reports.conditions as report_cond
import eeg_adhd_epilepsy.viz.conditions as viz_cond
import eeg_adhd_epilepsy.utils.events as utils_events
import eeg_adhd_epilepsy.qc.segmentation as qc_segmentation
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Condition analysis and reporting tools.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Path to BIDS dataset root.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Path to write output reports.")
    parser.add_argument("--bids_session", help="Process specific session.")
    parser.add_argument("--bids_task", help="Process specific task.")
    parser.add_argument("--subjects_list", type=Path, help="Path to text file with subject IDs to process.")
    parser.add_argument("--n_jobs", type=int, default="-1", help="Number of parallel jobs.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    return parser.parse_args()


def _process_subject(
    filepath: Path,
    bids_root: Path,
    output_dir: Path,
    session: str | None = None,
    task: str | None = None,
) -> Dict[str, object] | None:
    """Process a single subject file and generate a report."""
    subject_id = io_bids.parse_subject_id(filepath)
    subject_out_dir = output_dir / "subjects" / subject_id
    fig_dir = subject_out_dir / "figures"
    
    try:
        raw = io_bids.load_bids_raw(
            filepath, 
            bids_root, 
            session=session, 
            task=task
        )
        
        # 1. Extract Condition Segments
        df_segments = qc_segmentation.extract_condition_segments(raw)
        
        seg_filename = filepath.stem.replace("_eeg", "") + "_segments.csv"
        bids_out_path = filepath.parent / seg_filename
        df_segments.to_csv(bids_out_path, index=False)
        
        # 2. Compute Segment Stats
        summary = qc_segmentation.summarize_condition_segments(df_segments)
        
        # 3. Compute Event Counts
        # Use raw counts for Bad/Clinical events
        
        raw_counts = utils_events.summarize_annotations(raw)
        event_counts = {}

        # 1. Populate Standard Conditions from Segmentation Summary
        # HV/Photo/Post-HV counts are reliability computed in summary
        event_counts["HV Start"] = summary.get("hv_block_count", 0)
        event_counts["HV End"] = summary.get("hv_block_count", 0)
        event_counts["Photo"] = summary.get("photo_block_count", 0)
        event_counts["Post-HV"] = summary.get("post_hv_block_count", 0)

        # 2. Populate Eye States from Segments DataFrame
        # Count number of segments where EO/EC is active
        if not df_segments.empty:
            n_eo = len(df_segments[pd.to_numeric(df_segments["eyes_open_duration"], errors='coerce') > 0])
            n_ec = len(df_segments[pd.to_numeric(df_segments["eyes_closed_duration"], errors='coerce') > 0])
            event_counts["Eyes Open"] = n_eo
            event_counts["Eyes Closed"] = n_ec
        else:
            event_counts["Eyes Open"] = 0
            event_counts["Eyes Closed"] = 0

        # 3. Add Clinical/Other events from Raw Counts
        # Filter out the condition events we just populated from segmentation
        for desc, count in raw_counts.items():
            clean_desc = qc_segmentation.normalize_label(desc)
            
            # Skip if it refers to a condition we already handled
            if clean_desc == "eyes_open" or clean_desc == "eyes_closed":
                continue
            if qc_segmentation.is_hv_start(clean_desc):
                continue
            if qc_segmentation.is_hv_end(clean_desc):
                continue
            if qc_segmentation.is_photo(clean_desc):
                continue
            if qc_segmentation.is_post_hv(clean_desc):
                continue
                
            # It is likely a Clinical, Bad, or other event -> Add it
            event_counts[desc] = event_counts.get(desc, 0) + count
                
        # 4. Generate Figures
        fig_paths = viz_cond.save_condition_segment_figures(df_segments, fig_dir)
        
        # 5. Generate Individual Report
        report_path = subject_out_dir / f"{subject_id}_condition_report.html"
        raw_duration = raw.times[-1] if raw.n_times > 0 else 0.0
                
        report_cond.create_condition_segments_report(
            summary=summary,
            figure_paths=fig_paths,
            output_path=report_path,
            subject_id=subject_id,
            raw_duration=raw_duration,
            event_counts=event_counts,
        )
        
        return {
            "subject_id": subject_id,
            "filepath": str(filepath),
            "summary": summary,
            "event_counts": event_counts,
            "raw_duration": raw_duration,
            "report_path": str(report_path),
        }

    except Exception as e:
        logging.error(f"Failed to process {subject_id}: {e}", exc_info=True)
        return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_log = args.output_dir / "logs" / "conditions_qc.log"
    setup_log.parent.mkdir(exist_ok=True)
    logger = setup_logging(setup_log, args.log_level)
    
    subjects_filter = io_bids.read_subjects_list(args.subjects_list)
    
    logger.info("Discovering files...")
    files = io_bids.discover_bids_files(
        bids_root=args.input_dir,
        session=args.bids_session,
        task=args.bids_task,
        subjects_filter=subjects_filter,
    )
    
    if not files:
        logger.warning("No files found matching criteria.")
        return

    logger.info(f"files found: {len(files)}")
    logger.info(f"Processing in {args.n_jobs} parallel jobs...")
    
    with tqdm_joblib(tqdm(total=len(files), desc="Conditions QC")):
        results = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(_process_subject)(
                filepath=f,
                bids_root=args.input_dir,
                output_dir=args.output_dir,
                session=args.bids_session,
                task=args.bids_task,
            ) 
            for f in files
        )
    
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        summaries_df = pd.DataFrame([r["summary"] for r in valid_results])
        event_counts_list = [r["event_counts"] for r in valid_results]
        
        logger.info("Generating dataset-level figures...")
        figure_paths = viz_cond.plot_dataset_durations(summaries_df, args.output_dir)
        
        event_dist_path = viz_cond.save_dataset_events_distribution(event_counts_list, args.output_dir)
        
        p_cond = args.output_dir / "dataset_event_distributions_conditions.png"
        if p_cond.exists():
             figure_paths["dataset_event_distributions_conditions.png"] = p_cond
        
        p_clin = args.output_dir / "dataset_event_distributions_clinical.png"
        if p_clin.exists():
             figure_paths["dataset_event_distributions_clinical.png"] = p_clin
             
        if event_dist_path and event_dist_path not in figure_paths.values():
             figure_paths["dataset_event_distributions.png"] = event_dist_path
            
        global_report_path = args.output_dir / "dataset_conditions_summary.html"
        logger.info(f"Generating global report at {global_report_path}")
        report_cond.create_dataset_conditions_report(
            subjects_data=valid_results,
            output_path=global_report_path,
            figure_paths=figure_paths,
        )
    else:
        logger.error("No valid results produced.")


if __name__ == "__main__":
    main()

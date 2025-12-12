"""Segment-level QC on raw (pre-preprocessing) EEG."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from eeg_adhd_epilepsy.explore import eeg_qc
from eeg_adhd_epilepsy.explore.condition_segments_summary import extract_condition_segments
from eeg_adhd_epilepsy.utils.qc_config import BASIC_1020_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment-level EEG QC before preprocessing.")
    parser.add_argument("--input_dir", required=True, type=Path, help="BIDS root directory with raw EEG files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to store QC outputs.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Jobs for parallel processing (-1 for all cores).")
    parser.add_argument("--subjects_list", type=Path, help="File with subject IDs to include (one per line).")
    parser.add_argument("--bids_session", default=None, help="BIDS session entity.")
    parser.add_argument("--bids_task", default=None, help="BIDS task entity.")
    parser.add_argument("--bids_run", default=None, help="BIDS run entity.")
    parser.add_argument("--bids_acq", default=None, help="BIDS acquisition entity.")
    parser.add_argument("--bids_proc", default=None, help="BIDS processing label if any.")
    parser.add_argument("--min_segment_duration", type=float, default=5.0, help="Minimum segment duration to analyze (s).")
    parser.add_argument(
        "--highpass",
        type=float,
        default=0.5,
        help="High-pass filter cutoff for QC (Hz). Use 0 or a negative value to skip filtering.",
    )
    parser.add_argument("--line_freq", type=float, default=60.0, help="Mains frequency for line-noise metric.")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    parser.add_argument(
        "--skip_reports",
        action="store_true",
        help="Disable HTML segment reports (subject-level and dataset-level).",
    )
    return parser.parse_args()


def summarize_dataset_segments(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {}
    summary: Dict[str, object] = {
        "total_segments": int(len(df)),
        "total_duration_sec": float(pd.to_numeric(df["duration"], errors="coerce").sum()),
    }
    if "segment_type" in df:
        summary["total_duration_by_segment_type"] = {
            str(k): float(v) for k, v in df.groupby("segment_type")["duration"].sum().items()
        }
        summary["median_duration_by_segment_type"] = {
            str(k): float(v) for k, v in df.groupby("segment_type")["duration"].median().items()
        }
    for metric in [
        "segment_hf_lf_ratio",
        "segment_aperiodic_slope",
        "segment_amplitude_mean_uv",
        "segment_line_noise_ratio",
    ]:
        if metric in df:
            summary[f"median_{metric}"] = float(pd.to_numeric(df[metric], errors="coerce").median())
    return summary


def process_file(
    filepath: Path,
    args: argparse.Namespace,
    standard_names: set[str],
    logger: object,
    subjects_dir: Path,
    reports_dir: Path | None,
) -> Dict[str, object]:
    subject_id = eeg_qc.parse_subject_id(filepath)
    records: List[Dict[str, object]] = []
    error = ""
    segment_df: pd.DataFrame | None = None
    agg_df: pd.DataFrame | None = None
    segment_csv = None
    agg_csv = None
    report_path = None
    per_channel_accum: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_channel_maps: Dict[str, Dict[str, float]] = {}
    try:
        raw = eeg_qc.load_raw(
            filepath,
            bids_root=args.input_dir,
            session=args.bids_session,
            task=args.bids_task,
            run=args.bids_run,
            acquisition=args.bids_acq,
            processing=args.bids_proc,
        )
        raw.load_data()
        if getattr(args, "highpass", None) is not None and args.highpass > 0:
            raw.filter(args.highpass, None, fir_design="firwin", verbose="ERROR")
        analysis_raw, basic_picks, montage_info = eeg_qc.prepare_channel_selection(raw, standard_names, logger)
        segments_df = extract_condition_segments(raw)
        if segments_df is None or segments_df.empty or analysis_raw is None:
            return {"subject_id": subject_id, "records": records, "error": error}

        for _, row in segments_df.iterrows():
            duration = float(row.get("duration", 0.0))
            if duration < args.min_segment_duration:
                continue
            segment = eeg_qc.crop_segment(
                analysis_raw,
                float(row.get("t_start", 0.0)),
                float(row.get("t_stop", 0.0)),
                picks=basic_picks,
            )
            if segment is None:
                continue
            qc_metrics = eeg_qc.compute_segment_qc(
                segment,
                picks=basic_picks,
                logger=logger,
                line_freq=getattr(args, "line_freq", 60.0),
                include_channel_metrics=not args.skip_reports,
            )
            channel_metrics = qc_metrics.pop("per_channel_metrics", None)
            if channel_metrics:
                for name, arr in channel_metrics.items():
                    arr_np = np.asarray(arr, dtype=float)
                    if arr_np.size == len(basic_picks):
                        per_channel_accum[name].append(arr_np)
            record = {
                "subject_id": subject_id,
                "segment_type": row.get("segment_type", ""),
                "t_start": float(row.get("t_start", np.nan)),
                "t_stop": float(row.get("t_stop", np.nan)),
                "duration": duration,
                "freq_hz": float(row.get("freq_hz", np.nan)),
                "hv_index": float(row.get("hv_index", np.nan)),
                "post_hv_index": float(row.get("post_hv_index", np.nan)),
                "eyes_open_duration": float(row.get("eyes_open_duration", np.nan)),
                "eyes_closed_duration": float(row.get("eyes_closed_duration", np.nan)),
                "n_channels_1020_match": montage_info.get("n_channels_1020_match", 0),
                "pct_missing_1020": montage_info.get("pct_missing_1020", float("nan")),
            }
            record.update(qc_metrics)
            records.append(record)
        if records:
            subject_dir = subjects_dir / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)
            segment_df = pd.DataFrame(records)
            segment_csv = subject_dir / f"{subject_id}_segment_qc_pre.csv"
            segment_df.to_csv(segment_csv, index=False)
            if reports_dir is not None:
                topomap_figs: Dict[str, object] = {}
                if per_channel_accum and analysis_raw is not None:
                    topo_arrays: Dict[str, np.ndarray] = {}
                    expected_len = len(basic_picks)
                    for name, arr_list in per_channel_accum.items():
                        valid = [
                            np.asarray(arr, dtype=float) for arr in arr_list if np.asarray(arr, dtype=float).size == expected_len
                        ]
                        if not valid:
                            continue
                        stacked = np.vstack(valid)
                        topo_arrays[name] = np.nanmean(stacked, axis=0)
                    for metric_key, values in topo_arrays.items():
                        arr = np.asarray(values, dtype=float)
                        if arr.size == 0 or not np.isfinite(arr).any():
                            continue
                        if metric_key.startswith("band_power_"):
                            label = f"{metric_key.replace('band_power_', '').title()} band power (uV^2)"
                        elif metric_key == "amplitude_ptp_uv":
                            label = "Amplitude (uV ptp)"
                        elif metric_key == "line_noise_ratio":
                            label = "Line-noise ratio"
                        elif metric_key == "hf_lf_ratio":
                            label = "HF/LF ratio"
                        elif metric_key == "aperiodic_slope":
                            label = "Aperiodic slope"
                        else:
                            label = metric_key
                        cmap = "RdBu_r" if metric_key in {"line_noise_ratio", "hf_lf_ratio", "aperiodic_slope"} else "viridis"
                        fig_topo = eeg_qc.plot_metric_topomap(
                            arr, analysis_raw, basic_picks, title=f"{label} Topomap", cmap=cmap, unit=None
                        )
                        if fig_topo is not None:
                            topomap_figs[label] = fig_topo
                        per_channel_maps[metric_key] = {ch: float(val) for ch, val in zip(basic_picks, arr)}
                report_path = reports_dir / f"{subject_id}_segment_qc_pre_report.html"
                eeg_qc.create_segment_subject_report(segment_df, subject_id, report_path, topomap_figs=topomap_figs)
            agg_df = eeg_qc.aggregate_segment_qc(records)
            if not agg_df.empty:
                agg_csv = subject_dir / f"{subject_id}_segment_qc_pre_aggregated.csv"
                agg_df.to_csv(agg_csv, index=False)
    except Exception as exc:  # pragma: no cover - defensive branch
        error = str(exc)
        logger.error("Failed processing %s: %s", subject_id, exc, exc_info=True)
    return {
        "subject_id": subject_id,
        "records": records,
        "segment_df": segment_df,
        "agg_df": agg_df,
        "segment_csv": segment_csv,
        "agg_csv": agg_csv,
        "report_path": report_path,
        "topomap_channel_maps": per_channel_maps,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "subject_reports"
    fig_dir = output_dir / "figures"
    log_file = output_dir / "logs" / "pre_preproc_segment_qc.log"
    logger = eeg_qc.setup_logging(log_file, args.log_level)

    standard_names = {name.lower() for name in BASIC_1020_CHANNELS}
    subjects_filter = eeg_qc.read_subjects_list(args.subjects_list)
    files = eeg_qc.discover_bids_files(
        args.input_dir,
        session=args.bids_session,
        task=args.bids_task,
        run=args.bids_run,
        acquisition=args.bids_acq,
        processing=args.bids_proc,
        subjects_filter=subjects_filter,
    )
    if not files:
        logger.warning("No EEG files found under %s", args.input_dir)
        return
    logger.info("Found %d files to process", len(files))

    subjects_dir = output_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True) if not args.skip_reports else None

    with eeg_qc.tqdm_joblib(tqdm(total=len(files), desc="Segment QC (pre)")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_file)(
                filepath, args, standard_names, logger, subjects_dir, None if args.skip_reports else reports_dir
            )
            for filepath in files
        )

    all_segment_frames: List[pd.DataFrame] = []
    aggregated_frames: List[pd.DataFrame] = []
    errors = []
    dataset_topo_payloads: List[Dict[str, Dict[str, float]]] = []

    for res in results:
        if res.get("segment_df") is not None:
            all_segment_frames.append(res["segment_df"])
        if res.get("agg_df") is not None and not res["agg_df"].empty:
            aggregated_frames.append(res["agg_df"])
        if res.get("error"):
            errors.append((res["subject_id"], res["error"]))
        topo_map = res.get("topomap_channel_maps")
        if topo_map:
            dataset_topo_payloads.append(topo_map)

    if all_segment_frames:
        dataset_segments = pd.concat(all_segment_frames, ignore_index=True)
        dataset_segments.to_csv(output_dir / "segment_qc_pre_all_segments.csv", index=False)
        summary = summarize_dataset_segments(dataset_segments)
        (output_dir / "segment_qc_pre_dataset_summary.json").write_text(json.dumps(summary, indent=2))
        topomap_aggregates = eeg_qc.aggregate_topomap_metrics(dataset_topo_payloads) if not args.skip_reports else {}
        if not args.skip_reports:
            fig_paths = eeg_qc.save_segment_dataset_figures(
                dataset_segments, fig_dir, topomap_aggregates=topomap_aggregates
            )
            report_path = output_dir / "segment_qc_pre_summary_report.html"
            eeg_qc.create_segment_dataset_report(dataset_segments, fig_paths, report_path)
            logger.info("Saved dataset summary HTML report to %s", report_path)

    if aggregated_frames:
        summary_df = pd.concat(aggregated_frames, ignore_index=True)
        summary_df.to_csv(output_dir / "segment_qc_pre_summary_by_subject_and_type.csv", index=False)

    if errors:
        logger.warning("Completed with %d errors", len(errors))
    else:
        logger.info("Completed without errors")


if __name__ == "__main__":
    main()

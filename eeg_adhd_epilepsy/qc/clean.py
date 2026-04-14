"""Post-preprocessing EEG QC focused on comparison, residual artifacts, and retention."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import eeg_adhd_epilepsy.io.bids as bids_io
import eeg_adhd_epilepsy.signal_quality.metrics as signal_quality
import eeg_adhd_epilepsy.reports.clean_qc as qc_reports
import eeg_adhd_epilepsy.reports.eeg_report as report_eeg
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-preprocessing EEG QC with pre/post comparison and retention metrics."
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="QC output directory.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (-1 for all).")
    parser.add_argument(
        "--analyze_segments", action="store_true", help="Enable segment-level analysis on cleaned data."
    )
    parser.add_argument("--subjects_list", type=Path, help="Filter to subjects in file.")
    parser.add_argument("--bids_session", help="BIDS session.")
    parser.add_argument("--bids_task", default="RESTING", help="BIDS task (default: RESTING).")
    parser.add_argument("--bids_run", help="BIDS run.")
    parser.add_argument("--bids_acq", help="BIDS acquisition.")
    parser.add_argument("--bids_proc", help="BIDS processing (e.g. clean).")
    parser.add_argument(
        "--pre_base_qc_csv",
        type=Path,
        default=None,
        help="Optional pre-base raw_qc_runs.csv path. Defaults to sibling reports/summary/raw_qc_pre_base/raw_qc_runs.csv.",
    )
    parser.add_argument("--line_freq", type=float, default=60.0, help="Line frequency.")
    parser.add_argument("--min_segment_duration", type=float, default=5.0, help="Min segment length (s).")
    parser.add_argument("--skip_reports", action="store_true", help="Skip HTML reports.")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _resolve_pre_base_qc_csv(input_dir: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    reports_root = bids_io.get_reports_root(bids_root=input_dir)
    candidate = reports_root / "summary" / "raw_qc_pre_base" / "raw_qc_runs.csv"
    return candidate if candidate.exists() else None


def _load_pre_base_qc_runs(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for key in ("run_prefix", "subject_session_prefix", "subject_id", "session_id", "run_id"):
        if key not in df:
            df[key] = ""
    return df


def _evaluate_post_clean_flag(metrics: Dict[str, object]) -> tuple[bool, str]:
    reasons: list[str] = []
    n_bad = float(metrics.get("n_flat_channels", 0) or 0) + float(metrics.get("n_noisy_channels", 0) or 0)
    if n_bad >= 7:
        reasons.append("too_many_bad_channels")
    elif n_bad >= 4:
        reasons.append("many_bad_channels")

    amplitude_max = pd.to_numeric(metrics.get("amplitude_max_uv"), errors="coerce")
    if np.isfinite(amplitude_max) and float(amplitude_max) > 800.0:
        reasons.append("amplitude_above_threshold")

    line_noise_ratio = pd.to_numeric(metrics.get("line_noise_ratio"), errors="coerce")
    if np.isfinite(line_noise_ratio) and float(line_noise_ratio) > 5.0:
        reasons.append("line_noise_residual")

    hf_lf_ratio = pd.to_numeric(metrics.get("hf_lf_ratio"), errors="coerce")
    if np.isfinite(hf_lf_ratio) and float(hf_lf_ratio) > 0.5:
        reasons.append("high_hf_ratio")

    duration_retention_pct = pd.to_numeric(metrics.get("duration_retention_pct"), errors="coerce")
    if np.isfinite(duration_retention_pct) and float(duration_retention_pct) < 50.0:
        reasons.append("low_duration_retention")

    coverage_retention_pct = pd.to_numeric(metrics.get("coverage_retention_pct"), errors="coerce")
    if np.isfinite(coverage_retention_pct) and float(coverage_retention_pct) < 80.0:
        reasons.append("low_condition_retention")

    flag_bad = any(
        reason in reasons
        for reason in ("too_many_bad_channels", "amplitude_above_threshold", "low_duration_retention")
    )
    return flag_bad, ";".join(reasons)


def _process_file(
    filepath: Path,
    args: argparse.Namespace,
    logger,
    pre_base_lookup: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    ids = bids_io.build_bids_report_ids(filepath)
    subject_id = str(ids["subject_id"])
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
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks) == 0:
            raise RuntimeError("No EEG channels found in cleaned recording.")

        metrics = signal_quality.compute_signal_qc_metrics(
            raw,
            picks=picks,
            line_freq=args.line_freq,
            include_channel_metrics=False,
        )

        segments_df = bids_io.load_segments_for_raw(raw)
        condition_summary = report_eeg.summarize_condition_segments(segments_df)
        pre_base = pre_base_lookup.get(str(ids["run_prefix"]), {})

        raw_duration_sec = float(raw.n_times / raw.info["sfreq"]) if raw.n_times > 0 else float("nan")
        pre_raw_duration_sec = pd.to_numeric(pre_base.get("raw_duration"), errors="coerce")
        duration_retention_pct = (
            float(raw_duration_sec / pre_raw_duration_sec * 100.0)
            if np.isfinite(pre_raw_duration_sec) and pre_raw_duration_sec > 0
            else np.nan
        )

        cleaned_condition_duration = float(condition_summary.get("total_duration", 0.0) or 0.0)
        pre_condition_duration = pd.to_numeric(pre_base.get("total_duration"), errors="coerce")
        coverage_retention_pct = (
            float(cleaned_condition_duration / pre_condition_duration * 100.0)
            if np.isfinite(pre_condition_duration) and pre_condition_duration > 0
            else np.nan
        )

        file_metrics = {
            "filepath": str(filepath),
            "subject_id": subject_id,
            "session_id": str(ids["session_id"]),
            "run_id": str(ids["run_id"]),
            "subject_session_prefix": str(ids["subject_session_prefix"]),
            "run_prefix": str(ids["run_prefix"]),
            "duration_min": raw_duration_sec / 60.0 if np.isfinite(raw_duration_sec) else np.nan,
            "amplitude_mean_uv": metrics.get("amplitude_mean_uv"),
            "amplitude_max_uv": metrics.get("amplitude_max_uv"),
            "n_flat_channels": int(metrics.get("n_flat_channels", 0) or 0),
            "n_noisy_channels": int(metrics.get("n_noisy_channels", 0) or 0),
            "pct_bad_channels": metrics.get("pct_bad_channels"),
            "alpha_peak_hz": metrics.get("alpha_peak_hz"),
            "line_noise_ratio": metrics.get("line_noise_ratio"),
            "hf_lf_ratio": metrics.get("hf_lf_ratio"),
            "aperiodic_slope": metrics.get("aperiodic_slope"),
            "pre_amplitude_mean_uv": pd.to_numeric(pre_base.get("amplitude_mean_uv"), errors="coerce"),
            "pre_amplitude_max_uv": pd.to_numeric(pre_base.get("amplitude_max_uv"), errors="coerce"),
            "pre_pct_bad_channels": pd.to_numeric(pre_base.get("pct_bad_channels"), errors="coerce"),
            "pre_line_noise_ratio": pd.to_numeric(pre_base.get("line_noise_ratio"), errors="coerce"),
            "pre_hf_lf_ratio": pd.to_numeric(pre_base.get("hf_lf_ratio"), errors="coerce"),
            "pre_alpha_peak_hz": pd.to_numeric(pre_base.get("alpha_peak_hz"), errors="coerce"),
            "pre_aperiodic_slope": pd.to_numeric(pre_base.get("aperiodic_slope"), errors="coerce"),
            "raw_duration_pre_sec": pre_raw_duration_sec,
            "condition_duration_pre_sec": pre_condition_duration,
            "condition_duration_post_sec": cleaned_condition_duration,
            "duration_retention_pct": duration_retention_pct,
            "coverage_retention_pct": coverage_retention_pct,
        }
        for post_col, pre_col, delta_col in (
            ("amplitude_mean_uv", "pre_amplitude_mean_uv", "amplitude_mean_delta_uv"),
            ("amplitude_max_uv", "pre_amplitude_max_uv", "amplitude_max_delta_uv"),
            ("pct_bad_channels", "pre_pct_bad_channels", "pct_bad_channels_delta"),
            ("line_noise_ratio", "pre_line_noise_ratio", "line_noise_ratio_delta"),
            ("hf_lf_ratio", "pre_hf_lf_ratio", "hf_lf_ratio_delta"),
            ("alpha_peak_hz", "pre_alpha_peak_hz", "alpha_peak_delta_hz"),
            ("aperiodic_slope", "pre_aperiodic_slope", "aperiodic_slope_delta"),
        ):
            post_value = pd.to_numeric(file_metrics.get(post_col), errors="coerce")
            pre_value = pd.to_numeric(file_metrics.get(pre_col), errors="coerce")
            file_metrics[delta_col] = (
                float(post_value - pre_value)
                if np.isfinite(post_value) and np.isfinite(pre_value)
                else np.nan
            )

        flag_bad, flag_reasons = _evaluate_post_clean_flag(file_metrics)
        file_metrics["flag_bad"] = flag_bad
        file_metrics["flag_reasons"] = flag_reasons

        result["file_metrics"] = file_metrics

        if args.analyze_segments and segments_df is not None and not segments_df.empty:
            for row in segments_df.itertuples(index=False):
                duration = float(getattr(row, "duration", 0.0) or 0.0)
                if duration < args.min_segment_duration:
                    continue
                segment = signal_quality.crop_segment(
                    raw,
                    float(getattr(row, "t_start", 0.0) or 0.0),
                    float(getattr(row, "t_stop", 0.0) or 0.0),
                    picks=picks,
                )
                if segment is None:
                    continue
                seg_metrics = signal_quality.compute_signal_qc_metrics(
                    segment,
                    picks=picks,
                    line_freq=args.line_freq,
                    include_channel_metrics=False,
                )
                result["segment_metrics"].append(
                    {
                        "subject_id": subject_id,
                        "run_id": str(ids["run_id"]),
                        "segment_type": getattr(row, "segment_type", ""),
                        "t_start": float(getattr(row, "t_start", np.nan)),
                        "duration": duration,
                        "segment_duration_sec": seg_metrics.get("duration_sec"),
                        "segment_n_channels": seg_metrics.get("n_channels"),
                        "segment_amplitude_mean_uv": seg_metrics.get("amplitude_mean_uv"),
                        "segment_amplitude_median_uv": seg_metrics.get("amplitude_median_uv"),
                        "segment_amplitude_std_uv": seg_metrics.get("amplitude_std_uv"),
                        "segment_amplitude_min_uv": seg_metrics.get("amplitude_min_uv"),
                        "segment_amplitude_max_uv": seg_metrics.get("amplitude_max_uv"),
                        "segment_flat_channels": seg_metrics.get("flat_channels"),
                        "segment_noisy_channels": seg_metrics.get("noisy_channels"),
                        "segment_n_flat_channels": seg_metrics.get("n_flat_channels"),
                        "segment_n_noisy_channels": seg_metrics.get("n_noisy_channels"),
                        "segment_pct_bad_channels": seg_metrics.get("pct_bad_channels"),
                        "segment_alpha_peak_hz": seg_metrics.get("alpha_peak_hz"),
                        "segment_hf_lf_ratio": seg_metrics.get("hf_lf_ratio"),
                        "segment_line_noise_ratio": seg_metrics.get("line_noise_ratio"),
                        "segment_aperiodic_slope": seg_metrics.get("aperiodic_slope"),
                        "segment_flag_bad": seg_metrics.get("flag_bad"),
                        "segment_flag_reasons": seg_metrics.get("flag_reasons"),
                    }
                )

    except Exception as exc:
        logger.error("Error processing %s: %s", subject_id, exc)
        result["error"] = str(exc)

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
        processing=args.bids_proc,
        subjects_filter=subjects_filter,
    )
    if not files:
        logger.warning("No cleaned files found.")
        return

    pre_base_qc_csv = _resolve_pre_base_qc_csv(args.input_dir, args.pre_base_qc_csv)
    pre_base_df = _load_pre_base_qc_runs(pre_base_qc_csv)
    pre_base_lookup = (
        pre_base_df.drop_duplicates("run_prefix").set_index("run_prefix").to_dict("index")
        if not pre_base_df.empty and "run_prefix" in pre_base_df
        else {}
    )
    if pre_base_qc_csv is not None:
        logger.info("Using pre-base QC reference: %s", pre_base_qc_csv)
    else:
        logger.warning("No pre-base QC reference found; comparison columns will be empty.")

    output_dirs = {"figures": args.output_dir / "figures"}
    output_dirs["figures"].mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm(total=len(files), desc="QC Clean")):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_process_file)(f, args, logger, pre_base_lookup)
            for f in files
        )

    file_records = [res["file_metrics"] for res in results if res.get("file_metrics")]
    segment_records = []
    for res in results:
        if res.get("segment_metrics"):
            segment_records.extend(res["segment_metrics"])

    if not file_records:
        logger.warning("No cleaned QC records were generated.")
        return

    df_files = pd.DataFrame(file_records).sort_values(
        ["subject_id", "session_id", "run_id", "filepath"],
        na_position="last",
    )
    df_files.to_csv(args.output_dir / "qc_clean_files.csv", index=False)

    flags_counter = Counter()
    for reasons in df_files.get("flag_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(reasons).split(";"):
            reason = reason.strip()
            if reason:
                flags_counter[reason] += 1

    fig_paths = qc_reports.save_figures(df_files, flags_counter, output_dirs["figures"])

    segment_df = pd.DataFrame(segment_records) if segment_records else pd.DataFrame()
    segment_fig_paths = {}
    if not segment_df.empty:
        segment_df.to_csv(args.output_dir / "qc_clean_segments.csv", index=False)
        if not args.skip_reports:
            segment_fig_paths = qc_reports.save_segment_dataset_figures(segment_df, output_dirs["figures"] / "segments")
            qc_reports.create_segment_dataset_report(
                segment_df,
                segment_fig_paths,
                args.output_dir / "qc_clean_segments_report.html",
            )

    if not args.skip_reports:
        qc_reports.create_summary_report(
            df_files,
            fig_paths,
            args.output_dir / "qc_clean_summary.html",
            total_files=len(files),
            flags_counter=flags_counter,
            unknown_events=None,
            report_title="Post-Preprocessing EEG QC Dataset Summary",
            summary_heading="Post-Preprocessing QC Summary",
            total_label="Total cleaned files processed",
            segment_df=segment_df if not segment_df.empty else None,
            segment_fig_paths=segment_fig_paths if segment_fig_paths else None,
        )


if __name__ == "__main__":
    main()

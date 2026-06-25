"""Raw signal QC builders for the pre-base stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

import eeg_adhd_epilepsy.io.report_paths as report_paths
import eeg_adhd_epilepsy.reports.eeg_report as report_eeg
import eeg_adhd_epilepsy.reports.raw_qc as report_raw_qc
import eeg_adhd_epilepsy.signal_quality.metrics as signal_quality
import eeg_adhd_epilepsy.viz.raw_qc as viz_raw_qc
from eeg_adhd_epilepsy.qc.utils import (
    _BASE_WEIGHTED_METRICS,
    DEFAULT_SIGNAL_THRESHOLDS,
    MAX_METRICS,
    SignalQCThresholds,
    _build_channel_diagnostics,
    _build_topomap_aggregates,
    _clean_scalar,
    _combine_weighted_topomaps,
    compute_channel_failure_rates,
    compute_qc_score,
    evaluate_signal_qc_flag,
)
from eeg_adhd_epilepsy.utils.events import crop_raw_to_recording_start
from eeg_adhd_epilepsy.utils.constants import BASIC_1020_CHANNELS
WEIGHTED_METRICS = (*_BASE_WEIGHTED_METRICS, "coverage_pct")


def _prepare_analysis_raw(
    raw: mne.io.BaseRaw,
    *,
    highpass: float,
) -> tuple[mne.io.BaseRaw, list[int]]:
    analysis_raw = raw.copy().load_data()
    target_channels = [
        channel
        for channel in BASIC_1020_CHANNELS
        if channel in analysis_raw.ch_names
    ]
    if target_channels:
        analysis_raw.pick(target_channels)
    picks = list(mne.pick_types(analysis_raw.info, eeg=True, exclude=[]))
    if not picks:
        raise RuntimeError("No EEG channels found.")
    cropped_raw = crop_raw_to_recording_start(analysis_raw)
    if cropped_raw is None:
        cropped_raw = analysis_raw
    if highpass > 0:
        cropped_raw.filter(highpass, None, fir_design="firwin", verbose="ERROR")
    return cropped_raw, picks


def _build_run_metric_row(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "amplitude_mean_uv": metrics.get("amplitude_mean_uv"),
        "amplitude_max_uv": metrics.get("amplitude_max_uv"),
        "n_flat_channels": int(metrics.get("n_flat_channels", 0) or 0),
        "n_noisy_channels": int(metrics.get("n_noisy_channels", 0) or 0),
        "pct_bad_channels": metrics.get("pct_bad_channels"),
        "line_noise_ratio": metrics.get("line_noise_ratio"),
        "hf_lf_ratio": metrics.get("hf_lf_ratio"),
        "alpha_peak_hz": metrics.get("alpha_peak_hz"),
        "aperiodic_slope": metrics.get("aperiodic_slope"),
    }


def _build_segment_qc_rows(
    cropped_raw: mne.io.BaseRaw,
    *,
    picks: Sequence[int],
    ids: Mapping[str, object],
    filepath: str,
    condition_segments_df: pd.DataFrame,
    analysis_level: str,
    line_freq: float,
    min_segment_duration: float,
) -> pd.DataFrame:
    if analysis_level not in {"segments", "both"}:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in condition_segments_df.itertuples(index=False):
        duration = float(getattr(row, "duration", 0.0) or 0.0)
        if duration < min_segment_duration:
            continue
        t_start = float(getattr(row, "t_start", 0.0) or 0.0)
        t_stop = float(getattr(row, "t_stop", 0.0) or 0.0)
        segment = signal_quality.crop_segment(cropped_raw, t_start, t_stop, picks=list(picks))
        if segment is None:
            continue
        metrics = signal_quality.compute_signal_qc_metrics(
            segment,
            picks=list(picks),
            line_freq=line_freq,
            include_channel_metrics=False,
        )
        rows.append(
            {
                "subject_id": ids["subject_id"],
                "session_id": ids["session_id"],
                "run_id": ids["run_id"],
                "subject_session_prefix": ids["subject_session_prefix"],
                "run_prefix": ids["run_prefix"],
                "filepath": filepath,
                "segment_type": getattr(row, "segment_type", None),
                "eye_state": getattr(row, "eye_state", None),
                "t_start": t_start,
                "duration": duration,
                "segment_alpha_power": (metrics.get("band_powers") or {}).get("alpha", np.nan),
                "segment_amplitude_mean_uv": metrics.get("amplitude_mean_uv"),
                "segment_amplitude_max_uv": metrics.get("amplitude_max_uv"),
                "segment_pct_bad_channels": metrics.get("pct_bad_channels"),
                "segment_line_noise_ratio": metrics.get("line_noise_ratio"),
                "segment_hf_lf_ratio": metrics.get("hf_lf_ratio"),
                "segment_aperiodic_slope": metrics.get("aperiodic_slope"),
                "segment_flag_bad": bool(metrics.get("flag_bad", False)),
                "segment_flag_reasons": metrics.get("flag_reasons", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_run_summary_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "subject_id": record["subject_id"],
        "session_id": record["session_id"],
        "run_id": record["run_id"],
        "subject_session_prefix": record["subject_session_prefix"],
        "run_prefix": record["run_prefix"],
        "filepath": record["filepath"],
        "source_dataset": record["source_dataset"],
        "record_date": record["record_date"],
        "meas_datetime": record["meas_datetime"],
        "raw_duration": float(record["raw_duration"]),
        "age_group": record["age_group"],
        "sex": record["sex"],
        "combined_diagnosis": record["combined_diagnosis"],
        "subject_flag": record["subject_flag"],
        "subject_flag_reasons": record["subject_flag_reasons"],
        "amplitude_mean_uv": record["amplitude_mean_uv"],
        "amplitude_max_uv": record["amplitude_max_uv"],
        "n_flat_channels": record["n_flat_channels"],
        "n_noisy_channels": record["n_noisy_channels"],
        "pct_bad_channels": record["pct_bad_channels"],
        "line_noise_ratio": record["line_noise_ratio"],
        "hf_lf_ratio": record["hf_lf_ratio"],
        "alpha_peak_hz": record["alpha_peak_hz"],
        "aperiodic_slope": record["aperiodic_slope"],
        "coverage_pct": record["coverage_pct"],
        **record["condition_summary"],
    }


def _whole_recording_metrics(
    cropped_raw: mne.io.BaseRaw,
    *,
    picks: Sequence[int],
    channel_names: Sequence[str],
    raw_duration: float,
    line_freq: float,
    thresholds: SignalQCThresholds | None,
    base_metrics: Mapping[str, object],
) -> tuple[dict[str, object], dict, dict]:
    """Whole-recording signal-QC metrics, usability flag, topomaps, and diagnostics."""
    computed_metrics = signal_quality.compute_signal_qc_metrics(
        cropped_raw,
        picks=list(picks),
        line_freq=line_freq,
        include_channel_metrics=True,
    )
    run_metrics = {**base_metrics, **_build_run_metric_row(computed_metrics)}
    subject_flag, reasons = evaluate_signal_qc_flag(run_metrics, thresholds)
    run_metrics["subject_flag"] = subject_flag
    run_metrics["subject_flag_reasons"] = ";".join(reasons)
    file_topomaps = _build_topomap_aggregates(
        computed_metrics,
        channel_names=list(channel_names),
        weight=max(raw_duration, 1.0),
    )
    channel_diagnostics = _build_channel_diagnostics(
        computed_metrics,
        channel_names=list(channel_names),
    )
    return run_metrics, file_topomaps, channel_diagnostics


def build_raw_qc_run_record(
    *,
    raw: mne.io.BaseRaw,
    bids_path: BIDSPath,
    condition_segments_df: pd.DataFrame,
    condition_summary: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    analysis_level: str = "both",
    line_freq: float = 60.0,
    highpass: float = 0.5,
    min_segment_duration: float = 5.0,
    thresholds: SignalQCThresholds | None = None,
) -> dict[str, object]:
    metadata = metadata or {}
    filepath = str(bids_path.fpath)
    ids = report_paths.build_bids_report_ids(bids_path.fpath)
    ids["filepath"] = filepath
    cropped_raw, picks = _prepare_analysis_raw(raw, highpass=highpass)
    raw_duration = float(raw.times[-1]) if raw.n_times > 0 else 0.0
    channel_names = [cropped_raw.ch_names[index] for index in picks]

    base_metrics: dict[str, object] = {
        "subject_id": ids["subject_id"],
        "session_id": ids["session_id"],
        "run_id": ids["run_id"],
        "subject_session_prefix": ids["subject_session_prefix"],
        "run_prefix": ids["run_prefix"],
        "filepath": filepath,
        "raw_duration": raw_duration,
    }
    if analysis_level in {"whole", "both"}:
        run_metrics, file_topomaps, channel_diagnostics = _whole_recording_metrics(
            cropped_raw,
            picks=picks,
            channel_names=channel_names,
            raw_duration=raw_duration,
            line_freq=line_freq,
            thresholds=thresholds,
            base_metrics=base_metrics,
        )
    else:
        run_metrics = {
            **base_metrics,
            "amplitude_mean_uv": np.nan,
            "amplitude_max_uv": np.nan,
            "n_flat_channels": 0,
            "n_noisy_channels": 0,
            "pct_bad_channels": np.nan,
            "line_noise_ratio": np.nan,
            "hf_lf_ratio": np.nan,
            "alpha_peak_hz": np.nan,
            "aperiodic_slope": np.nan,
            "subject_flag": "",
            "subject_flag_reasons": "",
        }
        file_topomaps = {}
        channel_diagnostics = {}

    coverage_pct = (
        float(condition_summary.get("total_duration", 0.0) or 0.0) / raw_duration * 100.0
        if raw_duration > 0
        else np.nan
    )
    run_metrics["coverage_pct"] = coverage_pct
    segment_df = _build_segment_qc_rows(
        cropped_raw,
        picks=picks,
        ids=ids,
        filepath=filepath,
        condition_segments_df=condition_segments_df,
        analysis_level=analysis_level,
        line_freq=line_freq,
        min_segment_duration=min_segment_duration,
    )

    record = {
        **ids,
        "filepath": filepath,
        "source_dataset": _clean_scalar(metadata.get("source_dataset")),
        "record_date": _clean_scalar(
            raw.info.get("meas_date").date().isoformat() if raw.info.get("meas_date") else ""
        ),
        "meas_datetime": _clean_scalar(
            raw.info.get("meas_date").isoformat() if raw.info.get("meas_date") else ""
        ),
        "raw_duration": raw_duration,
        "age_group": _clean_scalar(metadata.get("age_group")),
        "sex": _clean_scalar(metadata.get("sex")),
        "combined_diagnosis": _clean_scalar(metadata.get("combined_diagnosis")),
        "condition_summary": dict(condition_summary),
        "condition_segments_df": condition_segments_df,
        "segment_df": segment_df,
        "file_topomaps": file_topomaps,
        "channel_diagnostics": channel_diagnostics,
        **run_metrics,
    }
    record["qc_score"] = compute_qc_score(run_metrics, thresholds=thresholds)
    active = thresholds or DEFAULT_SIGNAL_THRESHOLDS
    record["thresholds"] = {
        "n_bad_borderline": active.n_bad_borderline,
        "n_bad_unusable": active.n_bad_unusable,
        "amplitude_max_uv": active.amplitude_max_uv,
        "line_noise_ratio": active.line_noise_ratio,
        "hf_lf_ratio": active.hf_lf_ratio,
    }
    record["summary_row"] = _build_run_summary_row(record)
    return record


def _aggregate_subject_metrics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    first = records[0]
    weights = np.asarray(
        [float(record.get("raw_duration", 0.0) or 0.0) for record in records], dtype=float
    )

    def weighted_mean(field: str) -> float:
        values = pd.to_numeric(
            pd.Series([record.get(field) for record in records]), errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return float("nan")
        return float(np.average(values[valid], weights=weights[valid]))

    def max_value(field: str) -> float:
        values = pd.to_numeric(
            pd.Series([record.get(field) for record in records]), errors="coerce"
        )
        return float(values.max(skipna=True)) if not values.isna().all() else float("nan")

    status_order = {"usable": 0, "borderline": 1, "unusable": 2}
    status = "usable"
    reasons: set[str] = set()
    for record in records:
        record_status = str(record.get("subject_flag") or "usable")
        if status_order.get(record_status, 0) > status_order.get(status, 0):
            status = record_status
        reasons.update(
            reason.strip()
            for reason in str(record.get("subject_flag_reasons") or "").split(";")
            if reason.strip()
        )

    output = {
        "subject_id": first["subject_id"],
        "session_id": first["session_id"],
        "subject_session_prefix": first["subject_session_prefix"],
        "source_dataset": first.get("source_dataset"),
        "raw_duration": float(sum(weights)),
        "n_runs": len(records),
        "age_group": first.get("age_group"),
        "sex": first.get("sex"),
        "combined_diagnosis": first.get("combined_diagnosis"),
        "subject_flag": status,
        "subject_flag_reasons": ";".join(sorted(reasons)),
        "filepath": ";".join(
            str(record.get("filepath") or "") for record in records if record.get("filepath")
        ),
    }
    for field in WEIGHTED_METRICS:
        output[field] = weighted_mean(field)
    for field in MAX_METRICS:
        output[field] = max_value(field)
    return output


def _aggregate_channel_diagnostics(
    records: Sequence[dict[str, object]],
    topomap_aggregates: Mapping[str, tuple[list[str], np.ndarray]],
) -> dict[str, object]:
    flat_channels = sorted(
        {
            channel
            for record in records
            for channel in record.get("channel_diagnostics", {}).get("flat_channels", [])
        }
    )
    noisy_channels = sorted(
        {
            channel
            for record in records
            for channel in record.get("channel_diagnostics", {}).get("noisy_channels", [])
        }
    )
    top_amplitude = []
    top_line_noise = []
    amplitude_payload = topomap_aggregates.get("amplitude_ptp_uv")
    if amplitude_payload:
        channels, values = amplitude_payload
        amp_pairs = sorted(zip(channels, values), key=lambda item: item[1], reverse=True)
        top_amplitude = [
            (channel, float(value)) for channel, value in amp_pairs[:5] if np.isfinite(value)
        ]
    line_payload = topomap_aggregates.get("line_noise_ratio")
    if line_payload:
        channels, values = line_payload
        line_pairs = sorted(zip(channels, values), key=lambda item: item[1], reverse=True)
        top_line_noise = [
            (channel, float(value)) for channel, value in line_pairs[:5] if np.isfinite(value)
        ]
    return {
        "flat_channels": flat_channels,
        "noisy_channels": noisy_channels,
        "top_amplitude_channels": top_amplitude,
        "top_line_noise_channels": top_line_noise,
    }


def _build_subject_summary_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "subject_id": record["subject_id"],
        "session_id": record["session_id"],
        "subject_session_prefix": record["subject_session_prefix"],
        "source_dataset": record["source_dataset"],
        "raw_duration": float(record["raw_duration"]),
        "n_runs": int(record["n_runs"]),
        "age_group": record["age_group"],
        "sex": record["sex"],
        "combined_diagnosis": record["combined_diagnosis"],
        "subject_flag": record["subject_flag"],
        "subject_flag_reasons": record["subject_flag_reasons"],
        "amplitude_mean_uv": record["amplitude_mean_uv"],
        "amplitude_max_uv": record["amplitude_max_uv"],
        "n_flat_channels": record["n_flat_channels"],
        "n_noisy_channels": record["n_noisy_channels"],
        "pct_bad_channels": record["pct_bad_channels"],
        "line_noise_ratio": record["line_noise_ratio"],
        "hf_lf_ratio": record["hf_lf_ratio"],
        "alpha_peak_hz": record["alpha_peak_hz"],
        "aperiodic_slope": record["aperiodic_slope"],
        "coverage_pct": record["coverage_pct"],
        **record["condition_summary"],
    }


def _alpha_reactivity(segment_df: pd.DataFrame) -> dict[str, float]:
    """Eyes-closed vs eyes-open resting alpha power (Berger effect).

    Uses baseline EO/EC segments only. Returns the mean alpha band power for
    each state and their ratio (EC/EO); a ratio > 1 is the expected
    physiological pattern in awake resting EEG and a sensitive validity check.
    """
    result = {
        "alpha_power_eo": float("nan"),
        "alpha_power_ec": float("nan"),
        "alpha_reactivity": float("nan"),
    }
    if (
        segment_df is None
        or segment_df.empty
        or "segment_alpha_power" not in segment_df.columns
        or "segment_type" not in segment_df.columns
    ):
        return result
    power = pd.to_numeric(segment_df["segment_alpha_power"], errors="coerce")
    seg_type = segment_df["segment_type"].astype(str)
    eo = power[seg_type.eq("EO_baseline")].mean()
    ec = power[seg_type.eq("EC_baseline")].mean()
    result["alpha_power_eo"] = float(eo) if np.isfinite(eo) else float("nan")
    result["alpha_power_ec"] = float(ec) if np.isfinite(ec) else float("nan")
    if np.isfinite(eo) and np.isfinite(ec) and eo > 0:
        result["alpha_reactivity"] = float(ec / eo)
    return result


def write_subject_raw_qc_report(
    reports_root: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    ids = records[0]
    subject_prefix = str(ids["subject_session_prefix"])
    subject_dir = report_paths.subject_report_dir(
        reports_root,
        str(ids["subject"]),
        str(ids["session"]),
        report_paths.ReportStage.RAW_QC_PRE_BASE,
        create=True,
    )
    fig_dir = subject_dir / "figures"
    topomap_aggregates = _combine_weighted_topomaps(record["file_topomaps"] for record in records)
    subject_segment_df = (
        pd.concat(
            [record["segment_df"] for record in records if not record["segment_df"].empty],
            ignore_index=True,
        )
        if any(not record["segment_df"].empty for record in records)
        else pd.DataFrame()
    )
    condition_segments_df = pd.concat(
        [record["condition_segments_df"].assign(run_id=record["run_id"]) for record in records],
        ignore_index=True,
    )
    condition_summary = report_eeg.summarize_condition_segments(condition_segments_df)
    subject_metrics = _aggregate_subject_metrics(records)
    subject_record = {
        **subject_metrics,
        "condition_summary": condition_summary,
        "thresholds": records[0].get("thresholds"),
        **_alpha_reactivity(subject_segment_df),
    }
    channel_diagnostics = _aggregate_channel_diagnostics(records, topomap_aggregates)
    figure_paths = viz_raw_qc.save_subject_raw_qc_figures(
        subject_segment_df,
        topomap_aggregates,
        fig_dir,
    )
    run_summary_df = (
        report_raw_qc.build_run_summary_table(records) if len(records) > 1 else pd.DataFrame()
    )
    report_raw_qc.generate_raw_qc_subject_report(
        record=subject_record,
        run_summary_df=run_summary_df,
        channel_diagnostics=channel_diagnostics,
        figure_paths=figure_paths,
        output_path=subject_dir / f"{subject_prefix}_raw_qc_pre_base_report.html",
    )
    return subject_record


def write_raw_qc_aggregate_reports(
    reports_root: Path,
    run_records: list[dict[str, object]],
) -> None:
    if not run_records:
        return

    summary_dir = report_paths.summary_report_dir(
        reports_root, report_paths.ReportStage.RAW_QC_PRE_BASE, create=True
    )

    runs_df = pd.DataFrame([record["summary_row"] for record in run_records]).sort_values(
        ["subject_id", "session_id", "run_id", "filepath"],
        na_position="last",
    )
    runs_df.to_csv(summary_dir / "raw_qc_runs.csv", index=False)

    segment_frames = [
        record["segment_df"] for record in run_records if not record["segment_df"].empty
    ]
    segments_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    segments_df.to_csv(summary_dir / "raw_qc_segments.csv", index=False)

    subject_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in run_records:
        subject_groups[record["subject_session_key"]].append(record)

    subject_rows: list[dict[str, object]] = []
    for (_subject_id, _session_id), records in sorted(subject_groups.items()):
        subject_record = write_subject_raw_qc_report(reports_root, records)
        subject_rows.append(_build_subject_summary_row(subject_record))

    subjects_df = pd.DataFrame(subject_rows).sort_values(
        ["subject_id", "session_id"],
        na_position="last",
    )
    subjects_df.to_csv(summary_dir / "raw_qc_subjects.csv", index=False)

    topomap_aggregates = _combine_weighted_topomaps(
        record["file_topomaps"] for record in run_records
    )
    figure_paths = viz_raw_qc.save_dataset_raw_qc_figures(
        runs_df,
        segments_df,
        topomap_aggregates,
        summary_dir / "figures",
    )
    dataset_tables = report_raw_qc.build_dataset_report_tables(runs_df, subjects_df)
    dataset_tables["channel_failure_df"] = report_raw_qc.build_channel_failure_table(
        compute_channel_failure_rates(run_records)
    )
    dataset_tables["outlier_df"] = report_raw_qc.build_cohort_outlier_table(runs_df)
    report_raw_qc.generate_raw_qc_dataset_report(
        tables=dataset_tables,
        figure_paths=figure_paths,
        output_path=summary_dir / "raw_qc_pre_base_dataset_summary.html",
    )

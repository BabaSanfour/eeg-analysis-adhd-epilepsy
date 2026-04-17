"""Shared stage-owned post-preprocessing QC framework."""

from __future__ import annotations

import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

import eeg_adhd_epilepsy.io.bids as bids_io
import eeg_adhd_epilepsy.reports.preproc_qc as report_preproc_qc
import eeg_adhd_epilepsy.signal_quality.metrics as signal_quality
import eeg_adhd_epilepsy.viz.preproc_qc as viz_preproc_qc

LOGGER = logging.getLogger(__name__)


TOPOMAP_METRIC_KEYS = ("amplitude_ptp_uv", "line_noise_ratio", "hf_lf_ratio")
WEIGHTED_METRICS = (
    "amplitude_mean_uv",
    "pct_bad_channels",
    "line_noise_ratio",
    "hf_lf_ratio",
    "alpha_peak_hz",
    "aperiodic_slope",
    "duration_retention_pct",
    "condition_coverage_retention_pct",
    # Amplitude delta keys keep _uv suffix — consistent with signal_quality output naming.
    "amplitude_mean_uv_delta_prev",
    "amplitude_mean_uv_delta_raw",
    "amplitude_max_uv_delta_prev",
    "amplitude_max_uv_delta_raw",
    "pct_bad_channels_delta_prev",
    "pct_bad_channels_delta_raw",
    "line_noise_ratio_delta_prev",
    "line_noise_ratio_delta_raw",
    "hf_lf_ratio_delta_prev",
    "hf_lf_ratio_delta_raw",
    "alpha_peak_hz_delta_prev",
    "alpha_peak_hz_delta_raw",
    "aperiodic_slope_delta_prev",
    "aperiodic_slope_delta_raw",
)
MAX_METRICS = ("amplitude_max_uv", "n_flat_channels", "n_noisy_channels")

# Metrics to compute deltas for — must match keys returned by compute_signal_qc_metrics.
_DELTA_METRICS = (
    "amplitude_mean_uv",
    "amplitude_max_uv",
    "pct_bad_channels",
    "line_noise_ratio",
    "hf_lf_ratio",
    "alpha_peak_hz",
    "aperiodic_slope",
)


@dataclass(frozen=True)
class PreprocQCProfile:
    stage: str
    display_name: str
    default_output_desc: str
    previous_stage: str | None
    previous_stage_label: str
    raw_reference_label: str = "Raw Pre-Base"


PREPROC_QC_PROFILES = {
    "base": PreprocQCProfile(
        stage="base",
        display_name="Base",
        default_output_desc="base",
        previous_stage=None,
        previous_stage_label="Raw Pre-Base",
    ),
    "correct": PreprocQCProfile(
        stage="correct",
        display_name="Correct",
        default_output_desc="correct",
        previous_stage="base",
        previous_stage_label="Base",
    ),
    "denoise": PreprocQCProfile(
        stage="denoise",
        display_name="Denoise",
        default_output_desc="denoise",
        previous_stage="correct",
        previous_stage_label="Correct",
    ),
}


def get_preproc_qc_profile(stage: str) -> PreprocQCProfile:
    try:
        return PREPROC_QC_PROFILES[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported preproc QC stage: {stage!r}") from exc


def get_preproc_qc_stage_name(stage: str, output_desc: str | None = None) -> str:
    profile = get_preproc_qc_profile(stage)
    if output_desc and output_desc != profile.default_output_desc:
        return bids_io.normalize_stage_name(f"{stage}_{output_desc}_qc")
    return bids_io.normalize_stage_name(f"{stage}_qc")


def _clean_scalar(value: object) -> object:
    return None if pd.isna(value) else value


def load_stage_run_lookup(
    reports_root: Path,
    stage_name: str,
    *,
    csv_name: str | None = None,
) -> dict[str, dict[str, object]]:
    summary_path = bids_io.get_stage_summary_dir(reports_root, stage_name, create_dir=False) / (
        csv_name or f"{stage_name}_runs.csv"
    )
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    lookup: dict[str, dict[str, object]] = {}
    for row in df.to_dict(orient="records"):
        for key in (
            row.get("run_prefix"),
            row.get("subject_session_prefix"),
            row.get("filepath"),
        ):
            key_str = str(key or "").strip()
            if key_str:
                lookup[key_str] = row
    return lookup


def load_raw_pre_base_lookup(reports_root: Path) -> dict[str, dict[str, object]]:
    return load_stage_run_lookup(reports_root, "raw_qc_pre_base", csv_name="raw_qc_runs.csv")


def update_run_lookup(lookup: dict[str, dict[str, object]], record: Mapping[str, object]) -> None:
    row = dict(record)
    for key in (
        row.get("run_prefix"),
        row.get("subject_session_prefix"),
        row.get("filepath"),
    ):
        key_str = str(key or "").strip()
        if key_str:
            lookup[key_str] = row


def _resolve_reference_row(
    lookup: Mapping[str, Mapping[str, object]],
    *,
    subject_id: str,
    subject_session_prefix: str,
    run_prefix: str,
    filepath: str | None = None,
) -> dict[str, object]:
    candidates = [
        run_prefix,
        subject_session_prefix,
        str(filepath or ""),
        f"{subject_id}_ses-01",
        subject_id,
    ]
    for key in candidates:
        row = lookup.get(key)
        if row:
            return dict(row)
    return {}


def _prepare_signal(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, list[str]]:
    """Return a loaded copy of raw and the EEG channel names to use for QC."""
    prepared = raw.copy().load_data()
    pick_idx = list(mne.pick_types(prepared.info, eeg=True, exclude=[]))
    if not pick_idx:
        raise RuntimeError("No EEG channels found in stage output.")
    picks = [prepared.ch_names[i] for i in pick_idx]
    return prepared, picks


def _annotation_intervals(raw: mne.io.BaseRaw) -> list[tuple[float, float]]:
    intervals = []
    for annot in raw.annotations:
        if str(annot["description"]).startswith("BAD_"):
            onset = float(annot["onset"])
            duration = float(annot["duration"])
            stop = onset + duration
            if stop > onset:
                intervals.append((onset, stop))
    return bids_io.merge_intervals(intervals)


def _interval_overlap(start: float, stop: float, overlaps: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for bad_start, bad_stop in overlaps:
        overlap_start = max(start, bad_start)
        overlap_stop = min(stop, bad_stop)
        if overlap_stop > overlap_start:
            total += overlap_stop - overlap_start
    return total


def compute_clean_duration(raw: mne.io.BaseRaw) -> float:
    total_duration = float(raw.times[-1]) if raw.times.size else 0.0
    bad_duration = sum(stop - start for start, stop in _annotation_intervals(raw))
    return max(total_duration - bad_duration, 0.0)


def compute_usable_condition_coverage(raw: mne.io.BaseRaw) -> float:
    segments_df = bids_io.load_segments_for_raw(raw)
    if segments_df is None or segments_df.empty:
        return 0.0
    bad_intervals = _annotation_intervals(raw)
    total = 0.0
    for row in segments_df.itertuples(index=False):
        start = float(getattr(row, "t_start", 0.0) or 0.0)
        stop = float(getattr(row, "t_stop", 0.0) or 0.0)
        if stop <= start:
            continue
        overlap = _interval_overlap(start, stop, bad_intervals)
        total += max((stop - start) - overlap, 0.0)
    return total


def _evaluate_preproc_qc_flag(metrics_row: Mapping[str, object]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    n_bad = float(metrics_row.get("n_flat_channels", 0) or 0) + float(metrics_row.get("n_noisy_channels", 0) or 0)
    if n_bad >= 7:
        reasons.append("too_many_bad_channels")
    elif n_bad >= 4:
        reasons.append("many_bad_channels")
    amp_max = pd.to_numeric(metrics_row.get("amplitude_max_uv"), errors="coerce")
    if np.isfinite(amp_max) and float(amp_max) > 800.0:
        reasons.append("amplitude_above_threshold")
    line_noise = pd.to_numeric(metrics_row.get("line_noise_ratio"), errors="coerce")
    if np.isfinite(line_noise) and float(line_noise) > 5.0:
        reasons.append("line_noise_residual")
    hf_lf_ratio = pd.to_numeric(metrics_row.get("hf_lf_ratio"), errors="coerce")
    if np.isfinite(hf_lf_ratio) and float(hf_lf_ratio) > 0.5:
        reasons.append("high_hf_ratio")
    duration_retention = pd.to_numeric(metrics_row.get("duration_retention_pct"), errors="coerce")
    if np.isfinite(duration_retention) and float(duration_retention) < 50.0:
        reasons.append("low_duration_retention")
    coverage_retention = pd.to_numeric(metrics_row.get("condition_coverage_retention_pct"), errors="coerce")
    if np.isfinite(coverage_retention) and float(coverage_retention) < 80.0:
        reasons.append("low_condition_retention")

    if "too_many_bad_channels" in reasons or "amplitude_above_threshold" in reasons or "low_duration_retention" in reasons:
        return "unusable", reasons
    if reasons:
        return "borderline", reasons
    return "usable", []


def _build_topomap_aggregates(metrics: Mapping[str, object], *, channel_names: Sequence[str], weight: float) -> dict[str, tuple[list[str], np.ndarray, float]]:
    per_channel_metrics = metrics.get("per_channel_metrics") or {}
    topomaps: dict[str, tuple[list[str], np.ndarray, float]] = {}
    for metric_key in TOPOMAP_METRIC_KEYS:
        values = per_channel_metrics.get(metric_key)
        arr = np.asarray(values, dtype=float) if values is not None else np.array([])
        if arr.size == 0 or len(channel_names) != arr.size:
            continue
        topomaps[metric_key] = (list(channel_names), arr, float(weight))
    return topomaps


def _combine_weighted_topomaps(mappings: Iterable[Mapping[str, tuple[Sequence[str], np.ndarray, float]]]) -> dict[str, tuple[list[str], np.ndarray]]:
    combined: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for mapping in mappings:
        for metric, (channels, values, weight) in mapping.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0 or len(channels) != arr.size or weight <= 0:
                continue
            metric_store = combined.setdefault(metric, {})
            for channel, value in zip(channels, arr):
                if not np.isfinite(value):
                    continue
                total, total_weight = metric_store.get(channel, (0.0, 0.0))
                metric_store[channel] = (total + float(value) * float(weight), total_weight + float(weight))

    output: dict[str, tuple[list[str], np.ndarray]] = {}
    for metric, channel_store in combined.items():
        channels = sorted(channel_store)
        if not channels:
            continue
        values = []
        for channel in channels:
            total, total_weight = channel_store[channel]
            values.append(total / total_weight if total_weight > 0 else np.nan)
        output[metric] = (channels, np.asarray(values, dtype=float))
    return output


def _build_channel_diagnostics(metrics: Mapping[str, object], *, channel_names: Sequence[str]) -> dict[str, object]:
    per_channel_metrics = metrics.get("per_channel_metrics") or {}
    amplitude = np.asarray(per_channel_metrics.get("amplitude_ptp_uv", np.array([])), dtype=float)
    line_noise = np.asarray(per_channel_metrics.get("line_noise_ratio", np.array([])), dtype=float)
    top_amplitude = []
    top_line_noise = []
    if amplitude.size == len(channel_names):
        amp_pairs = sorted(zip(channel_names, amplitude), key=lambda item: item[1], reverse=True)
        top_amplitude = [(channel, float(value)) for channel, value in amp_pairs[:5] if np.isfinite(value)]
    if line_noise.size == len(channel_names):
        line_pairs = sorted(zip(channel_names, line_noise), key=lambda item: item[1], reverse=True)
        top_line_noise = [(channel, float(value)) for channel, value in line_pairs[:5] if np.isfinite(value)]
    return {
        "flat_channels": list(metrics.get("flat_channels", [])),
        "noisy_channels": list(metrics.get("noisy_channels", [])),
        "top_amplitude_channels": top_amplitude,
        "top_line_noise_channels": top_line_noise,
    }


def _delta(current_value: object, reference_value: object) -> float:
    current = pd.to_numeric(current_value, errors="coerce")
    reference = pd.to_numeric(reference_value, errors="coerce")
    if np.isfinite(current) and np.isfinite(reference):
        return float(current - reference)
    return float("nan")


def _reference_metric(raw_reference: Mapping[str, object], key: str) -> object:
    return raw_reference.get(key)


def build_preproc_qc_run_record(
    *,
    profile: PreprocQCProfile,
    reports_root: Path,
    current_raw: mne.io.BaseRaw,
    current_filepath: Path,
    output_desc: str | None = None,
    previous_stage_label: str | None = None,
    previous_output_desc: str | None = None,
    raw_lookup: Mapping[str, Mapping[str, object]] | None = None,
    previous_lookup: Mapping[str, Mapping[str, object]] | None = None,
    pipeline_warnings: Sequence[str] | None = None,
) -> dict[str, object]:
    ids = bids_io.build_bids_report_ids(current_filepath)
    prepared_raw, picks = _prepare_signal(current_raw)
    metrics = signal_quality.compute_signal_qc_metrics(
        prepared_raw,
        picks=picks,
        line_freq=60.0,
        include_channel_metrics=True,
    )
    retained_duration_sec = compute_clean_duration(prepared_raw)
    usable_condition_coverage_sec = compute_usable_condition_coverage(prepared_raw)
    subject_session_prefix = str(ids["subject_session_prefix"])

    if raw_lookup is None:
        raw_lookup = load_raw_pre_base_lookup(reports_root)
    raw_reference = _resolve_reference_row(
        raw_lookup,
        subject_id=str(ids["subject_id"]),
        subject_session_prefix=subject_session_prefix,
        run_prefix=str(ids["run_prefix"]),
        filepath=str(current_filepath),
    )
    raw_duration_sec = pd.to_numeric(raw_reference.get("raw_duration"), errors="coerce")
    raw_condition_duration = pd.to_numeric(raw_reference.get("total_duration"), errors="coerce")

    if profile.previous_stage is None:
        prev_metrics: Mapping[str, object] = raw_reference
    else:
        if previous_lookup is None:
            previous_stage_name = get_preproc_qc_stage_name(
                profile.previous_stage,
                previous_output_desc or get_preproc_qc_profile(profile.previous_stage).default_output_desc,
            )
            previous_lookup = load_stage_run_lookup(reports_root, previous_stage_name)
        prev_metrics = _resolve_reference_row(
            previous_lookup,
            subject_id=str(ids["subject_id"]),
            subject_session_prefix=subject_session_prefix,
            run_prefix=str(ids["run_prefix"]),
            filepath=str(current_filepath),
        )

    duration_retention_pct = (
        float(retained_duration_sec / raw_duration_sec * 100.0)
        if np.isfinite(raw_duration_sec) and raw_duration_sec > 0
        else np.nan
    )
    condition_coverage_retention_pct = (
        float(usable_condition_coverage_sec / raw_condition_duration * 100.0)
        if np.isfinite(raw_condition_duration) and raw_condition_duration > 0
        else np.nan
    )

    run_metrics = {
        "filepath": str(current_filepath),
        "subject_id": ids["subject_id"],
        "session_id": ids["session_id"],
        "run_id": ids["run_id"],
        "subject_session_prefix": ids["subject_session_prefix"],
        "run_prefix": ids["run_prefix"],
        "subject_session_key": ids["subject_session_key"],
        "run_key": ids["run_key"],
        "stage": profile.stage,
        "output_desc": output_desc or profile.default_output_desc,
        "source_stage": previous_stage_label or profile.previous_stage_label,
        "reference_stage": profile.previous_stage_label,
        "raw_duration_sec": _clean_scalar(raw_duration_sec),
        "retained_duration_sec": retained_duration_sec,
        "usable_condition_coverage_sec": usable_condition_coverage_sec,
        "duration_retention_pct": duration_retention_pct,
        "condition_coverage_retention_pct": condition_coverage_retention_pct,
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

    for metric in _DELTA_METRICS:
        run_metrics[f"{metric}_delta_prev"] = _delta(metrics.get(metric), prev_metrics.get(metric))
        run_metrics[f"{metric}_delta_raw"] = _delta(metrics.get(metric), _reference_metric(raw_reference, metric))

    qc_flag, qc_reasons = _evaluate_preproc_qc_flag(run_metrics)
    run_metrics["qc_flag"] = qc_flag
    run_metrics["qc_flag_reasons"] = ";".join(qc_reasons)
    
    warnings = list(pipeline_warnings or [])
    if pd.isna(metrics.get("aperiodic_slope")):
        warnings.append("Spectral slope fitting skipped (insufficient clean/finite data).")
    run_metrics["pipeline_warnings"] = "; ".join(warnings)

    segment_comparison, segments_df = _compute_post_clean_segment_metrics(
        prepared_raw,
        picks=picks,
        reports_root=reports_root,
        run_prefix=str(ids["run_prefix"]),
    )
    return {
        **run_metrics,
        "channel_diagnostics": _build_channel_diagnostics(metrics, channel_names=picks),
        "topomap_aggregates": _build_topomap_aggregates(metrics, channel_names=picks, weight=max(retained_duration_sec, 1.0)),
        "segment_comparison": segment_comparison,
        "segments_df": segments_df,
    }


def _compute_post_clean_segment_metrics(
    raw: mne.io.BaseRaw,
    *,
    picks: Sequence[str],
    reports_root: Path,
    run_prefix: str,
    min_duration_sec: float = 5.0,
    line_freq: float = 60.0,
) -> pd.DataFrame:
    """Compute per-condition segment QC metrics on the cleaned output and compare to pre-base.

    Delegates per-segment computation to raw_metrics._build_segment_qc_rows — the same
    function used for the pre-base stage — so pre vs post values are on the same basis
    (full segment window, no BAD sub-interval splitting).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            1. Aggregated Summary: one row per segment_type with mean stats.
            2. Raw Segments: one row per individual segment with timing and metrics.
    """
    segments_df = bids_io.load_segments_for_raw(raw)
    if segments_df is None or segments_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    post_rows: list[dict[str, object]] = []
    for row in segments_df.itertuples(index=False):
        duration = float(getattr(row, "duration", 0.0) or 0.0)
        if duration < min_duration_sec:
            continue
        t_start = float(getattr(row, "t_start", 0.0) or 0.0)
        t_stop = float(getattr(row, "t_stop", 0.0) or 0.0)
        seg_type = str(getattr(row, "segment_type", ""))
        seg = signal_quality.crop_segment(raw, t_start, t_stop, picks=list(picks))
        if seg is None:
            continue
        try:
            m = signal_quality.compute_signal_qc_metrics(
                seg, picks=list(picks), line_freq=line_freq, include_channel_metrics=False
            )
        except ZeroDivisionError:
            LOGGER.debug(
                "Skipping segment %s [%.1f–%.1f s] for %s: all Welch windows rejected.",
                seg_type, t_start, t_stop, run_prefix,
            )
            continue
        post_rows.append({
            "segment_type": seg_type,
            "t_start": t_start,
            "t_stop": t_stop,
            "duration": duration,
            "segment_amplitude_mean_uv": m.get("amplitude_mean_uv"),
            "segment_amplitude_max_uv": m.get("amplitude_max_uv"),
            "segment_pct_bad_channels": m.get("pct_bad_channels"),
            "segment_line_noise_ratio": m.get("line_noise_ratio"),
            "segment_hf_lf_ratio": m.get("hf_lf_ratio"),
            "segment_aperiodic_slope": m.get("aperiodic_slope"),
        })
    post_rows_df = pd.DataFrame(post_rows)

    if post_rows_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Load pre-base segment rows from the raw_qc_segments CSV (written by raw_metrics stage).
    raw_qc_csv = bids_io.get_stage_summary_dir(reports_root, "raw_qc_pre_base", create_dir=False) / "raw_qc_segments.csv"
    if raw_qc_csv.exists():
        try:
            pre_all = pd.read_csv(raw_qc_csv)
            pre_df = pre_all[pre_all.get("run_prefix", pd.Series(dtype=str)) == run_prefix] if "run_prefix" in pre_all.columns else pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Could not load raw QC segments CSV: %s", exc)
            pre_df = pd.DataFrame()
    else:
        pre_df = pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return pd.to_numeric(df[col], errors="coerce").groupby(df["segment_type"]).mean()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        post_agg = post_rows_df.groupby("segment_type").agg(
            n_segments_post=("segment_type", "count"),
            total_duration_post_sec=("duration", "sum"),
            mean_amplitude_post=("segment_amplitude_mean_uv", "mean"),
            mean_line_noise_post=("segment_line_noise_ratio", "mean"),
            mean_hf_lf_post=("segment_hf_lf_ratio", "mean"),
            mean_pct_bad_channels_post=("segment_pct_bad_channels", "mean"),
            mean_aperiodic_slope_post=("segment_aperiodic_slope", "mean"),
        ).reset_index()

    if not pre_df.empty and "segment_type" in pre_df.columns:
        pre_agg = pre_df.groupby("segment_type").agg(
            n_segments_pre=("segment_type", "count"),
            mean_amplitude_pre=("segment_amplitude_mean_uv", "mean"),
            mean_line_noise_pre=("segment_line_noise_ratio", "mean"),
            mean_hf_lf_pre=("segment_hf_lf_ratio", "mean"),
            mean_pct_bad_channels_pre=("segment_pct_bad_channels", "mean"),
            mean_aperiodic_slope_pre=("segment_aperiodic_slope", "mean"),
        ).reset_index()
        result = post_agg.merge(pre_agg, on="segment_type", how="left")
    else:
        result = post_agg.copy()
        for col in ("n_segments_pre", "mean_amplitude_pre", "mean_line_noise_pre",
                    "mean_hf_lf_pre", "mean_pct_bad_channels_pre", "mean_aperiodic_slope_pre"):
            result[col] = float("nan")

    cols = [
        "segment_type",
        "n_segments_pre", "n_segments_post",
        "total_duration_post_sec",
        "mean_amplitude_pre", "mean_amplitude_post",
        "mean_line_noise_pre", "mean_line_noise_post",
        "mean_hf_lf_pre", "mean_hf_lf_post",
        "mean_pct_bad_channels_pre", "mean_pct_bad_channels_post",
        "mean_aperiodic_slope_pre", "mean_aperiodic_slope_post",
    ]
    summary_df = result[[c for c in cols if c in result.columns]].sort_values("segment_type").reset_index(drop=True)
    return summary_df, post_rows_df



def collect_existing_preproc_qc_record(
    *,
    profile: PreprocQCProfile,
    reports_root: Path,
    filepath: Path,
    output_desc: str | None = None,
    previous_output_desc: str | None = None,
    raw_lookup: Mapping[str, Mapping[str, object]] | None = None,
    previous_lookup: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    current_raw = mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")
    return build_preproc_qc_run_record(
        profile=profile,
        reports_root=reports_root,
        current_raw=current_raw,
        current_filepath=filepath,
        output_desc=output_desc,
        previous_output_desc=previous_output_desc,
        raw_lookup=raw_lookup,
        previous_lookup=previous_lookup,
        pipeline_warnings=None,
    )


def _aggregate_subject_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("records must be non-empty")
    first = records[0]
    weights = pd.Series(
        [pd.to_numeric(record.get("retained_duration_sec"), errors="coerce") for record in records],
        dtype=float,
    ).fillna(0.0)
    total_weight = float(weights.sum())

    aggregate = {key: value for key, value in first.items() if key not in {"topomap_aggregates", "channel_diagnostics", "segment_comparison", "segments_df"}}
    aggregate["n_runs"] = len(records)
    aggregate["retained_duration_sec"] = float(sum(float(record.get("retained_duration_sec", 0.0) or 0.0) for record in records))
    aggregate["usable_condition_coverage_sec"] = float(sum(float(record.get("usable_condition_coverage_sec", 0.0) or 0.0) for record in records))

    for metric in WEIGHTED_METRICS:
        series = pd.to_numeric([record.get(metric) for record in records], errors="coerce")
        valid = np.isfinite(series) & np.isfinite(weights.to_numpy(dtype=float))
        if valid.any() and float(weights.to_numpy(dtype=float)[valid].sum()) > 0:
            aggregate[metric] = float(np.average(series[valid], weights=weights.to_numpy(dtype=float)[valid]))
        else:
            aggregate[metric] = np.nan

    for metric in MAX_METRICS:
        series = pd.to_numeric([record.get(metric) for record in records], errors="coerce")
        aggregate[metric] = float(np.nanmax(series)) if np.isfinite(series).any() else np.nan

    status_order = {"usable": 0, "borderline": 1, "unusable": 2}
    aggregate["qc_flag"] = max((record.get("qc_flag", "usable") for record in records), key=lambda status: status_order.get(str(status), -1))
    reasons: list[str] = []
    for record in records:
        reasons.extend([reason for reason in str(record.get("qc_flag_reasons", "")).split(";") if reason])
    aggregate["qc_flag_reasons"] = ";".join(sorted(set(reasons)))
    aggregate["topomap_aggregates"] = _combine_weighted_topomaps(record.get("topomap_aggregates", {}) for record in records)
    aggregate["channel_diagnostics"] = records[0].get("channel_diagnostics", {})

    # Combine per-condition segment comparisons across runs: average numeric columns
    # per segment_type so multi-run subjects don't get duplicate condition rows.
    seg_frames = [
        r.get("segment_comparison") for r in records
        if isinstance(r.get("segment_comparison"), pd.DataFrame)
        and not r["segment_comparison"].empty
    ]
    if seg_frames:
        combined = pd.concat(seg_frames, ignore_index=True)
        numeric_cols = [c for c in combined.columns if c != "segment_type"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggregate["segment_comparison"] = (
                combined.groupby("segment_type", as_index=False)[numeric_cols]
                .mean(numeric_only=True)
                .sort_values("segment_type")
                .reset_index(drop=True)
            )
    else:
        aggregate["segment_comparison"] = pd.DataFrame()

    # Combine individual segments for temporal plotting (concatenation is fine here)
    seg_dfs = [r.get("segments_df") for r in records if isinstance(r.get("segments_df"), pd.DataFrame) and not r["segments_df"].empty]
    aggregate["segments_df"] = pd.concat(seg_dfs, ignore_index=True) if seg_dfs else pd.DataFrame()

    return aggregate


def write_subject_preproc_qc_report(
    reports_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    profile: PreprocQCProfile,
    output_desc: str | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must be non-empty")
    aggregate = _aggregate_subject_metrics(records)
    stage_name = get_preproc_qc_stage_name(profile.stage, output_desc or str(aggregate.get("output_desc") or profile.default_output_desc))
    output_path = bids_io.get_subject_session_stage_report_path(
        reports_root=reports_root,
        subject_id=str(aggregate["subject_id"]),
        session_id=str(aggregate.get("session_id") or ""),
        stage=stage_name,
        report_stem=str(aggregate["subject_session_prefix"]),
        create_dir=True,
    )
    figures_dir = output_path.parent / "figures"
    figure_paths = viz_preproc_qc.save_subject_preproc_qc_figures(
        record=aggregate,
        topomap_aggregates=aggregate.get("topomap_aggregates"),
        segments_df=aggregate.get("segments_df"),
        output_dir=figures_dir,
    )

    # Reconstruct AutoReject figures dir deterministically: during run_base_pipeline() the AR
    # PNGs are saved to <stage_report_dir>/figures/<run_prefix>/ for each run.
    autoreject_figures: dict[str, Path] = {}
    for record in records:
        run_prefix = str(record.get("run_prefix") or "")
        if not run_prefix:
            continue
        ar_dir = output_path.parent / "figures" / run_prefix
        if ar_dir.is_dir():
            for png in sorted(ar_dir.glob("*_autoreject_*.png")):
                # Key: "<run_prefix>/<condition>" so multiple runs don't collide
                stem = png.stem  # e.g. sub-01_ses-01_run-01_autoreject_EO_baseline
                autoreject_figures[f"{run_prefix}/{stem}"] = png

    run_summary_df = report_preproc_qc.build_run_summary_table(records)
    report_preproc_qc.generate_subject_report(
        record=aggregate,
        previous_stage_label=profile.previous_stage_label,
        raw_reference_label=profile.raw_reference_label,
        stage_display_name=profile.display_name,
        figures=figure_paths,
        run_summary_df=run_summary_df,
        output_path=output_path,
        channel_diagnostics=aggregate.get("channel_diagnostics") or {},
        autoreject_figures=autoreject_figures,
        segment_comparison=aggregate.get("segment_comparison"),
    )
    aggregate["report_path"] = str(output_path)
    return aggregate


def write_preproc_qc_aggregate_reports(
    reports_root: Path,
    run_records: Sequence[Mapping[str, object]],
    *,
    profile: PreprocQCProfile,
    output_desc: str | None = None,
) -> Path | None:
    if not run_records:
        return None
    output_desc = output_desc or str(run_records[0].get("output_desc") or profile.default_output_desc)
    stage_name = get_preproc_qc_stage_name(profile.stage, output_desc)
    summary_dir = bids_io.get_stage_summary_dir(reports_root, stage_name, create_dir=True)
    runs_df = pd.DataFrame([{k: v for k, v in record.items() if k not in {"topomap_aggregates", "channel_diagnostics"}} for record in run_records])
    runs_df.to_csv(summary_dir / f"{stage_name}_runs.csv", index=False)

    subject_groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in run_records:
        subject_groups[record["subject_session_key"]].append(record)
    subject_rows = []
    for records in subject_groups.values():
        subject_record = write_subject_preproc_qc_report(
            reports_root,
            records,
            profile=profile,
            output_desc=output_desc,
        )
        subject_rows.append({k: v for k, v in subject_record.items() if k not in {"topomap_aggregates", "channel_diagnostics", "segment_comparison"}})
    subjects_df = pd.DataFrame(subject_rows)
    subjects_df.to_csv(summary_dir / f"{stage_name}_subjects.csv", index=False)

    topomap_aggregates = _combine_weighted_topomaps(record.get("topomap_aggregates", {}) for record in run_records)
    figure_paths = viz_preproc_qc.save_dataset_preproc_qc_figures(
        runs_df=runs_df,
        topomap_aggregates=topomap_aggregates,
        output_dir=summary_dir / "figures",
    )
    # Build cross-subject per-condition summary by aggregating all run segment_comparisons.
    seg_frames = [
        r.get("segment_comparison") for r in run_records
        if isinstance(r.get("segment_comparison"), pd.DataFrame)
        and not r["segment_comparison"].empty
    ]
    if seg_frames:
        combined_segs = pd.concat(seg_frames, ignore_index=True)
        numeric_cols = [c for c in combined_segs.columns if c != "segment_type"]
        
        counts = combined_segs.groupby("segment_type").size().reset_index(name="n_usable_runs")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            condition_summary_df = (
                combined_segs.groupby("segment_type", as_index=False)[numeric_cols]
                .mean(numeric_only=True)
                .merge(counts, on="segment_type")
                .sort_values("segment_type")
                .reset_index(drop=True)
            )
    else:
        condition_summary_df = pd.DataFrame()

    report_preproc_qc.generate_dataset_report(
        runs_df=runs_df,
        subjects_df=subjects_df,
        stage_display_name=profile.display_name,
        previous_stage_label=profile.previous_stage_label,
        raw_reference_label=profile.raw_reference_label,
        figures=figure_paths,
        condition_summary_df=condition_summary_df,
        output_path=summary_dir / f"{stage_name}_dataset_summary.html",
    )
    return summary_dir

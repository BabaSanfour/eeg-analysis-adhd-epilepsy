"""Dataset-level statistics, aggregation, and flagging utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, MutableMapping, Tuple, Set, Sequence

import numpy as np
import pandas as pd
import mne


def compute_dataset_stats(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(records)

    metric_cols = [
        "segment_amplitude_mean_uv",
        "segment_amplitude_max_uv",
        "segment_pct_bad_channels",
        "duration_min",
        "segment_alpha_peak_hz",
        "segment_band_power_delta",
        "segment_band_power_theta",
        "segment_band_power_alpha",
        "segment_band_power_beta",
        "segment_band_power_gamma",
        "segment_hf_lf_ratio",
        "segment_line_noise_ratio",
        "segment_aperiodic_slope",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)
        stats[col] = {"mean": float(series.mean(skipna=True)), "std": float(series.std(skipna=True))}
    return stats


def apply_dataset_outlier_flags(
    records: List[Dict[str, object]], dataset_stats: Dict[str, Dict[str, float]]
) -> None:
    for rec in records:
        if rec.get("error"):
            continue
        reasons = rec.get("flag_reasons", "").split(";") if rec.get("flag_reasons") else []
        for col, stats in dataset_stats.items():
            value = rec.get(col)
            if value is None or not np.isfinite(value):
                continue
            std = stats["std"]
            mean = stats["mean"]
            if std > 0 and abs(value - mean) > 3 * std:
                reasons.append(f"{col}_outlier")
        if reasons:
            rec["flag_bad"] = True
            rec["flag_reasons"] = ";".join(sorted(set(filter(None, reasons))))
        else:
            rec["flag_bad"] = False
            rec["flag_reasons"] = ""


def summarize_flags(records: List[Dict[str, object]]) -> Counter:
    counter: Counter = Counter()
    for rec in records:
        reasons = rec.get("flag_reasons", "")
        if not reasons:
            continue
        for reason in reasons.split(";"):
            if reason:
                counter[reason] += 1
    return counter


def collect_unknown_events(
    records: List[Dict[str, object]], known_labels: Set[str]
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, object]] = {}
    for rec in records:
        payload = rec.get("event_counts")
        if not payload:
            continue
        try:
            counts = json.loads(payload)
        except Exception:
            continue
        subject_id = rec.get("subject_id", "unknown")
        for label, value in counts.items():
            if label in known_labels:
                continue
            try:
                occurrences = int(value)
            except Exception:
                continue
            if occurrences <= 0:
                continue
            entry = summary.setdefault(label, {"occurrences": 0, "subjects": set()})
            entry["occurrences"] += occurrences
            entry["subjects"].add(subject_id)
    formatted: Dict[str, Dict[str, int]] = {}
    for label, data in summary.items():
        formatted[label] = {
            "occurrences": int(data["occurrences"]),
            "n_subjects": len(data["subjects"]),
        }
    return formatted


def _flag_artifact_metric(zscore: float) -> str:
    ARTIFACT_Z_OK = 2.0
    ARTIFACT_Z_BAD = 3.0
    
    if not np.isfinite(zscore):
        return "unknown"
    if abs(zscore) <= ARTIFACT_Z_OK:
        return "good"
    if abs(zscore) <= ARTIFACT_Z_BAD:
        return "borderline"
    return "unusable"


def _flag_retention(pct: float) -> str:
    # Constants from eeg_qc (DATA_RETENTION_GOOD = 0.50 etc)
    DATA_RETENTION_GOOD = 0.50
    DATA_RETENTION_BORDERLINE = 0.30
    
    if not np.isfinite(pct):
        return "unknown"
    if pct >= DATA_RETENTION_GOOD:
        return "good"
    if pct >= DATA_RETENTION_BORDERLINE:
        return "borderline"
    return "unusable"


def _flag_bad_channels(n_bad: float) -> str:
    BAD_CHANNEL_GOOD = 3
    BAD_CHANNEL_BORDERLINE = 6
    if n_bad <= BAD_CHANNEL_GOOD:
        return "good"
    if n_bad <= BAD_CHANNEL_BORDERLINE:
        return "borderline"
    return "unusable"


def evaluate_condition_flags(
    condition_retention: Dict[str, Dict[str, float]],
    condition_amp: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Assign flags per condition using simple thresholds."""
    flags: Dict[str, Dict[str, object]] = {}
    for cond, stats in condition_retention.items():
        reasons: List[str] = []
        status = _flag_retention(stats.get("pct_retained", float("nan")))
        if status == "borderline":
            reasons.append("low_retention")
        elif status == "unusable":
            reasons.append("very_low_retention")
        if condition_amp and cond in condition_amp:
            max_ptp = condition_amp[cond].get("max_ptp_uv", float("nan"))
            if np.isfinite(max_ptp) and max_ptp > 500.0:
                status = "unusable"
                reasons.append("extreme_amplitude")
        flags[cond] = {"flag": status, "flag_reasons": ";".join(reasons)}
    return flags


def evaluate_subject_flag(
    metrics: MutableMapping[str, object],
    dataset_stats: Mapping[str, Dict[str, float]] | None = None,
) -> Tuple[str, List[str]]:
    """Aggregate subject-level usability flag."""
    reasons: List[str] = []
    # Hardcoded thresholds
    HF_RATIO_FLAG = 0.50
    APERIODIC_SLOPE_MIN = 0.50
    APERIODIC_SLOPE_MAX = 3.0
    LINE_NOISE_RATIO_FLAG = 5.0
    
    # Basic duration / amplitude / bad channels
    duration = float(metrics.get("duration_min", float("nan")))
    if np.isfinite(duration) and duration < 5:
        reasons.append("short_duration")
    if np.isfinite(duration) and duration > 60:
        reasons.append("long_duration")
    n_bad = float(metrics.get("segment_n_flat_channels", 0)) + float(metrics.get("segment_n_noisy_channels", 0))
    bad_flag = _flag_bad_channels(n_bad)
    if bad_flag == "borderline":
        reasons.append("many_bad_channels")
    elif bad_flag == "unusable":
        reasons.append("too_many_bad_channels")

    if np.isfinite(metrics.get("segment_amplitude_max_uv", float("nan"))) and metrics["segment_amplitude_max_uv"] > 800:
        reasons.append("amplitude_above_threshold")
    hf_ratio = metrics.get("segment_hf_lf_ratio", float("nan"))
    if np.isfinite(hf_ratio) and hf_ratio > HF_RATIO_FLAG:
        reasons.append("high_hf_ratio")
    slope = metrics.get("segment_aperiodic_slope", float("nan"))
    if np.isfinite(slope) and (slope < APERIODIC_SLOPE_MIN or slope > APERIODIC_SLOPE_MAX):
        reasons.append("extreme_aperiodic_slope")
    line_noise = metrics.get("segment_line_noise_ratio", float("nan"))
    if np.isfinite(line_noise) and line_noise > LINE_NOISE_RATIO_FLAG:
        reasons.append("line_noise_residual")

    if dataset_stats:
        for col, stats in dataset_stats.items():
            value = metrics.get(col)
            if value is None or not np.isfinite(value):
                continue
            std = stats.get("std", 0.0)
            mean = stats.get("mean", 0.0)
            if std > 0 and abs(value - mean) > 3 * std:
                reasons.append(f"{col}_outlier")

    condition_flags = metrics.get("condition_flags")
    if isinstance(condition_flags, dict):
        worst = "good"
        order = {"good": 0, "borderline": 1, "unusable": 2, "unknown": 1}
        for cond_info in condition_flags.values():
            flag = cond_info.get("flag", "good")
            if order.get(flag, 0) > order.get(worst, 0):
                worst = flag
        if worst == "borderline":
            reasons.append("borderline_condition")
        elif worst == "unusable":
            reasons.append("unusable_condition")

    if not reasons:
        return "usable", []
    if any(reason in {"unusable_condition", "too_many_bad_channels"} for reason in reasons):
        return "unusable", reasons
    if len(reasons) >= 2:
        return "borderline", reasons
    return "borderline", reasons


def compute_condition_retention(
    epochs: mne.Epochs,
    condition_map: Mapping[str, str] | None = None,
    condition_key: str = "condition",
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Return retention stats per condition and labels for kept epochs."""
    id_to_name = {v: k for k, v in (epochs.event_id or {}).items()}
    epoch_duration = float(epochs.tmax - epochs.tmin)

    metadata_labels: Dict[int, str] = {}
    if epochs.metadata is not None and condition_key in epochs.metadata.columns:
        for meta_idx, event_idx in enumerate(epochs.selection):
             if meta_idx < len(epochs.metadata):
                metadata_labels[int(event_idx)] = str(epochs.metadata.iloc[meta_idx][condition_key])

    stats: Dict[str, Dict[str, float]] = {}
    reasons_per_condition: Dict[str, Counter] = defaultdict(Counter)
    kept_conditions: List[str] = []

    drop_log = epochs.drop_log
    
    for idx, (log, event) in enumerate(zip(drop_log, epochs.events)):
        event_id = int(event[2])
        event_name = id_to_name.get(event_id)
        # Use metadata label if available, otherwise just event name (normalized)
        # Removed _label_condition fallback as events are standardized.
        cond = metadata_labels.get(idx) or (condition_map.get(event_name) if condition_map else event_name) or "UNKNOWN"
            
        cond_stats = stats.setdefault(
            cond,
            {"total": 0, "kept": 0, "rejected": 0, "pct_retained": float("nan"), "usable_minutes": float("nan")},
        )
        cond_stats["total"] += 1
        if len(log) == 0:
            cond_stats["kept"] += 1
            kept_conditions.append(cond)
        else:
            cond_stats["rejected"] += 1
            for reason in log:
                reasons_per_condition[cond][reason] += 1

    for cond, cond_stats in stats.items():
        cond_total = cond_stats["total"]
        cond_stats["pct_retained"] = cond_stats["kept"] / max(cond_total, 1)
        cond_stats["usable_minutes"] = cond_stats["kept"] * epoch_duration / 60.0
        cond_stats["rejection_reasons"] = dict(reasons_per_condition.get(cond, {}))

    return stats, kept_conditions


def compute_condition_amplitude_metrics(
    epochs: mne.Epochs,
    kept_conditions: Sequence[str],
    picks: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Mean/max peak-to-peak amplitude per condition."""
    if epochs is None or not kept_conditions:
        return {}
    data = epochs.get_data(picks=picks) * 1e6
    ptp = np.ptp(data, axis=2).mean(axis=1)  # (n_epochs,)
    per_condition: Dict[str, Dict[str, float]] = {}
    for cond in set(kept_conditions):
        mask = [c == cond for c in kept_conditions]
        if not any(mask):
            continue
        values = ptp[mask]
        per_condition[cond] = {
            "mean_ptp_uv": float(np.nanmean(values)),
            "max_ptp_uv": float(np.nanmax(values)),
        }
    return per_condition


def compute_epoch_rejection_breakdown(epochs: mne.Epochs) -> Dict[str, float]:
    """Percentage of epochs rejected by each criterion (if drop_log is populated)."""
    if epochs is None:
        return {}
    breakdown: Counter = Counter()
    total = len(epochs.drop_log)
    for log in epochs.drop_log:
        for reason in log:
            breakdown[reason] += 1
    return {reason: count / max(total, 1) for reason, count in breakdown.items()}


def _channel_sort_key(name: str) -> Tuple[int, str]:
    # Need access to config.BASIC_1020_CHANNELS to sort?
    # Or import it.
    from eeg_adhd_epilepsy.utils.config import BASIC_1020_CHANNELS
    order = {ch: idx for idx, ch in enumerate(BASIC_1020_CHANNELS)}
    return (order.get(name, len(order)), name)


def aggregate_topomap_metrics(
    metric_maps: Sequence[Mapping[str, Mapping[str, float]]]
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Aggregate per-channel metric dicts into channel-aligned arrays (mean across subjects)."""
    aggregated: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for metric_map in metric_maps:
        if not metric_map:
            continue
        for metric, channel_values in metric_map.items():
            if not channel_values:
                continue
            for ch, val in channel_values.items():
                try:
                    v = float(val)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                aggregated[metric][ch].append(v)

    results: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for metric, channel_dict in aggregated.items():
        if not channel_dict:
            continue
        channels = sorted(channel_dict.keys(), key=_channel_sort_key)

def aggregate_segment_qc(
    segment_qc_rows: List[Mapping[str, object]], group_cols: List[str] | None = None
) -> pd.DataFrame:
    """Aggregate segment QC metrics per subject/segment_type."""
    if not segment_qc_rows:
        return pd.DataFrame()
    df = pd.DataFrame(segment_qc_rows)
    group_cols = group_cols or ["subject_id", "segment_type"]
    for col in group_cols:
        if col not in df.columns:
            return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return df[group_cols]
    grouped = df.groupby(group_cols, dropna=False)[numeric_cols]
    agg_df = grouped.agg(["mean", "median"])
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    return agg_df.reset_index()

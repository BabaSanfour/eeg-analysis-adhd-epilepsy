"""Shared QC utilities — signal metrics, topomap aggregation, thresholds.

This module is the single source of truth for:
- QC threshold dataclasses (``SignalQCThresholds``, ``DescriptorQCThresholds``)
- Shared helper functions used by both ``raw_metrics`` and ``preproc_qc``
- Cross-run electrode failure rate computation
- Continuous QC confidence score
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Threshold dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalQCThresholds:
    """Thresholds for raw and preprocessed signal QC flagging.

    All values are configurable at instantiation so downstream code can
    override them without touching source.  The class-level defaults match
    the original hard-coded values.

    Parameters
    ----------
    n_bad_unusable:
        Number of bad (flat + noisy) channels that triggers ``unusable``.
    n_bad_borderline:
        Number of bad channels that triggers ``borderline``.
    amplitude_max_uv:
        Peak-to-peak amplitude (µV) above which the record is ``unusable``.
    line_noise_ratio:
        Line noise ratio above which the record is flagged.
    hf_lf_ratio:
        HF/LF power ratio above which the record is flagged.
    duration_retention_pct:
        Clean-duration retention (%) below which the record is ``unusable``.
        Only evaluated at preproc stages.
    condition_coverage_retention_pct:
        Condition coverage retention (%) below which the record is
        ``borderline``.  Only evaluated at preproc stages.
    """

    n_bad_unusable: int = 7
    n_bad_borderline: int = 4
    amplitude_max_uv: float = 800.0
    line_noise_ratio: float = 5.0
    hf_lf_ratio: float = 0.5
    duration_retention_pct: float = 50.0
    condition_coverage_retention_pct: float = 80.0


@dataclass(frozen=True)
class DescriptorQCThresholds:
    """Thresholds for descriptor-extraction QC flagging.

    Parameters
    ----------
    warn_nan_rate:
        Per-subject average NaN rate that triggers a ``warn`` flag.
    fail_nan_rate:
        Per-subject average NaN rate that triggers a ``fail`` flag.
    warn_feature_missingness:
        Per-feature missing rate that triggers ``warn``.
    fail_feature_missingness:
        Per-feature missing rate that triggers ``fail``.
    warn_family_failure_rate:
        Per-family extractor failure rate that triggers ``warn``.
    fail_family_failure_rate:
        Per-family extractor failure rate that triggers ``fail``.
    near_constant_std_tol:
        Standard deviation at or below which a feature is constant.
    warn_zero_variance_fraction:
        Fraction of constant features that triggers ``warn``.
    fail_zero_variance_fraction:
        Fraction of constant features that triggers ``fail``.
    warn_subject_outlier_fraction:
        Per-subject outlier feature fraction that triggers ``warn``.
    fail_subject_outlier_fraction:
        Per-subject outlier feature fraction that triggers ``fail``.
    outlier_z_threshold:
        MAD-based robust z-score threshold for outlier detection.
    """

    warn_nan_rate: float = 0.01
    fail_nan_rate: float = 0.20
    warn_feature_missingness: float = 0.20
    fail_feature_missingness: float = 0.50
    warn_family_failure_rate: float = 0.05
    fail_family_failure_rate: float = 0.25
    near_constant_std_tol: float = 1e-12
    warn_zero_variance_fraction: float = 0.01
    fail_zero_variance_fraction: float = 0.05
    warn_subject_outlier_fraction: float = 0.10
    fail_subject_outlier_fraction: float = 0.25
    outlier_z_threshold: float = 5.0


# Module-level default instances — used by all QC functions when no explicit
# thresholds are passed.
DEFAULT_SIGNAL_THRESHOLDS = SignalQCThresholds()
DEFAULT_DESCRIPTOR_THRESHOLDS = DescriptorQCThresholds()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPOMAP_METRIC_KEYS: tuple[str, ...] = (
    "amplitude_ptp_uv",
    "line_noise_ratio",
    "hf_lf_ratio",
)

_BASE_WEIGHTED_METRICS: tuple[str, ...] = (
    "amplitude_mean_uv",
    "pct_bad_channels",
    "line_noise_ratio",
    "hf_lf_ratio",
    "alpha_peak_hz",
    "aperiodic_slope",
)
"""Weighted-average metric keys shared by the raw-QC and preproc-QC stages."""

MAX_METRICS: tuple[str, ...] = (
    "amplitude_max_uv",
    "n_flat_channels",
    "n_noisy_channels",
)
"""Max-aggregate metric keys shared by the raw-QC and preproc-QC stages."""


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _clean_scalar(value: object) -> object:
    """Return ``None`` for NaN/NA, otherwise the original value."""
    return None if pd.isna(value) else value


# ---------------------------------------------------------------------------
# Topomap aggregation
# ---------------------------------------------------------------------------


def _build_topomap_aggregates(
    metrics: Mapping[str, object],
    *,
    channel_names: Sequence[str],
    weight: float,
) -> dict[str, tuple[list[str], np.ndarray, float]]:
    """Extract per-channel metric arrays ready for weighted aggregation.

    Parameters
    ----------
    metrics:
        Dict returned by ``compute_signal_qc_metrics`` (must contain
        ``"per_channel_metrics"``).
    channel_names:
        Channel names corresponding to the per-channel arrays.
    weight:
        Weight for this observation (typically recording duration in seconds).

    Returns
    -------
    ``{metric_key: (channel_names, values_array, weight)}``
    """
    per_channel_metrics = metrics.get("per_channel_metrics") or {}
    topomaps: dict[str, tuple[list[str], np.ndarray, float]] = {}
    for metric_key in TOPOMAP_METRIC_KEYS:
        values = per_channel_metrics.get(metric_key)
        arr = np.asarray(values, dtype=float) if values is not None else np.array([])
        if arr.size == 0 or len(channel_names) != arr.size:
            continue
        topomaps[metric_key] = (list(channel_names), arr, float(weight))
    return topomaps


def _combine_weighted_topomaps(
    mappings: Iterable[Mapping[str, tuple[Sequence[str], np.ndarray, float]]],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Combine per-file topomap aggregates into a single weighted average.

    Parameters
    ----------
    mappings:
        Iterable of ``{metric: (channels, values, weight)}`` dicts as
        returned by :func:`_build_topomap_aggregates`.

    Returns
    -------
    ``{metric: (sorted_channel_names, weighted_mean_values)}``
    """
    combined: dict[str, dict[str, tuple[float, float]]] = {}
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
                metric_store[channel] = (
                    total + float(value) * float(weight),
                    total_weight + float(weight),
                )

    output: dict[str, tuple[list[str], np.ndarray]] = {}
    for metric, channel_store in combined.items():
        channels = sorted(channel_store)
        if not channels:
            continue
        values = [
            total / total_weight if total_weight > 0 else np.nan
            for total, total_weight in (channel_store[ch] for ch in channels)
        ]
        output[metric] = (channels, np.asarray(values, dtype=float))
    return output


# ---------------------------------------------------------------------------
# Channel diagnostics
# ---------------------------------------------------------------------------


def _build_channel_diagnostics(
    metrics: Mapping[str, object],
    *,
    channel_names: Sequence[str],
) -> dict[str, object]:
    """Summarise per-channel amplitude and line-noise rankings.

    Parameters
    ----------
    metrics:
        Dict returned by ``compute_signal_qc_metrics``.
    channel_names:
        Channel names corresponding to per-channel metric arrays.

    Returns
    -------
    Dict with keys:
        ``flat_channels``, ``noisy_channels``,
        ``top_amplitude_channels``, ``top_line_noise_channels``.
    """
    per_channel_metrics = metrics.get("per_channel_metrics") or {}
    amplitude = np.asarray(per_channel_metrics.get("amplitude_ptp_uv", np.array([])), dtype=float)
    line_noise = np.asarray(per_channel_metrics.get("line_noise_ratio", np.array([])), dtype=float)
    top_amplitude: list[tuple[str, float]] = []
    top_line_noise: list[tuple[str, float]] = []
    if amplitude.size == len(channel_names):
        amp_pairs = sorted(zip(channel_names, amplitude), key=lambda x: x[1], reverse=True)
        top_amplitude = [(ch, float(v)) for ch, v in amp_pairs[:5] if np.isfinite(v)]
    if line_noise.size == len(channel_names):
        line_pairs = sorted(zip(channel_names, line_noise), key=lambda x: x[1], reverse=True)
        top_line_noise = [(ch, float(v)) for ch, v in line_pairs[:5] if np.isfinite(v)]
    return {
        "flat_channels": list(metrics.get("flat_channels", [])),
        "noisy_channels": list(metrics.get("noisy_channels", [])),
        "top_amplitude_channels": top_amplitude,
        "top_line_noise_channels": top_line_noise,
    }


# ---------------------------------------------------------------------------
# Cross-run electrode failure rates  (state-of-the-art: consensus bad-channel)
# ---------------------------------------------------------------------------


def compute_channel_failure_rates(
    records: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute per-electrode failure rate across a collection of run records.

    Aggregates ``flat_channels`` and ``noisy_channels`` from each record's
    ``channel_diagnostics`` dict and returns the fraction of records in which
    each channel was marked bad.  Useful for identifying systematic equipment
    or placement problems at the electrode level.

    Parameters
    ----------
    records:
        Sequence of QC run records, each expected to have a
        ``"channel_diagnostics"`` key produced by
        :func:`_build_channel_diagnostics`.

    Returns
    -------
    ``{channel_name: failure_rate}`` where ``failure_rate`` ∈ ``[0, 1]``.
    An empty dict is returned when ``records`` is empty.
    """
    if not records:
        return {}
    flat_counter: Counter[str] = Counter()
    noisy_counter: Counter[str] = Counter()
    for record in records:
        diag = record.get("channel_diagnostics") or {}
        for ch in diag.get("flat_channels", []):
            flat_counter[ch] += 1
        for ch in diag.get("noisy_channels", []):
            noisy_counter[ch] += 1
    total = len(records)
    all_channels = set(flat_counter) | set(noisy_counter)
    return {ch: (flat_counter[ch] + noisy_counter[ch]) / total for ch in sorted(all_channels)}


# ---------------------------------------------------------------------------
# Continuous QC score  (state-of-the-art: soft ranking vs. binary flag)
# ---------------------------------------------------------------------------


def compute_qc_score(
    metrics_row: Mapping[str, object],
    thresholds: SignalQCThresholds | None = None,
) -> float:
    """Compute a continuous quality score in ``[0, 1]`` (lower = better).

    Each metric is normalised against twice its ``unusable`` threshold so
    that a record right at the threshold scores ~0.5 and a perfect record
    scores ~0.

    Parameters
    ----------
    metrics_row:
        Dict of QC metrics as produced by ``build_raw_qc_run_record`` or
        ``build_preproc_qc_run_record``.
    thresholds:
        :class:`SignalQCThresholds` instance.  Defaults to
        :data:`DEFAULT_SIGNAL_THRESHOLDS`.

    Returns
    -------
    Float in ``[0, 1]``, or ``nan`` if no valid metrics are available.
    """
    t = thresholds or DEFAULT_SIGNAL_THRESHOLDS
    components: list[float] = []

    n_bad = float(metrics_row.get("n_flat_channels", 0) or 0) + float(
        metrics_row.get("n_noisy_channels", 0) or 0
    )
    components.append(min(n_bad / max(t.n_bad_unusable, 1), 1.0))

    amp = pd.to_numeric(metrics_row.get("amplitude_max_uv"), errors="coerce")
    if np.isfinite(amp):
        components.append(min(float(amp) / (t.amplitude_max_uv * 2.0), 1.0))

    ln = pd.to_numeric(metrics_row.get("line_noise_ratio"), errors="coerce")
    if np.isfinite(ln):
        components.append(min(float(ln) / (t.line_noise_ratio * 2.0), 1.0))

    hf = pd.to_numeric(metrics_row.get("hf_lf_ratio"), errors="coerce")
    if np.isfinite(hf):
        components.append(min(float(hf) / (t.hf_lf_ratio * 2.0), 1.0))

    return float(np.mean(components)) if components else float("nan")


# ---------------------------------------------------------------------------
# Shared signal QC flag evaluation
# ---------------------------------------------------------------------------


def evaluate_signal_qc_flag(
    metrics_row: Mapping[str, object],
    thresholds: SignalQCThresholds | None = None,
    *,
    check_retention: bool = False,
) -> tuple[str, list[str]]:
    """Evaluate a run's signal QC flag from its metric row.

    Checks bad-channel count, peak amplitude, line-noise ratio, and HF/LF
    ratio against ``thresholds``.  When ``check_retention=True`` the
    preprocessed-stage retention metrics (``duration_retention_pct`` and
    ``condition_coverage_retention_pct``) are also evaluated — these fields
    are only present in preproc-stage records and are silently skipped when
    absent or non-finite.

    Parameters
    ----------
    metrics_row:
        Dict of QC metrics as produced by ``build_raw_qc_run_record`` or
        ``build_preproc_qc_run_record``.
    thresholds:
        :class:`SignalQCThresholds` instance.  Defaults to
        :data:`DEFAULT_SIGNAL_THRESHOLDS`.
    check_retention:
        Pass ``True`` for preproc stages to include duration/coverage
        retention checks.

    Returns
    -------
    ``(flag, reasons)`` where ``flag`` ∈ ``{"usable", "borderline",
    "unusable"}`` and ``reasons`` is the list of triggered flag codes.
    """
    t = thresholds or DEFAULT_SIGNAL_THRESHOLDS
    reasons: list[str] = []

    n_bad = float(metrics_row.get("n_flat_channels", 0) or 0) + float(
        metrics_row.get("n_noisy_channels", 0) or 0
    )
    if n_bad >= t.n_bad_unusable:
        reasons.append("too_many_bad_channels")
    elif n_bad >= t.n_bad_borderline:
        reasons.append("many_bad_channels")

    amp_max = pd.to_numeric(metrics_row.get("amplitude_max_uv"), errors="coerce")
    if np.isfinite(amp_max) and float(amp_max) > t.amplitude_max_uv:
        reasons.append("amplitude_above_threshold")

    line_noise = pd.to_numeric(metrics_row.get("line_noise_ratio"), errors="coerce")
    if np.isfinite(line_noise) and float(line_noise) > t.line_noise_ratio:
        reasons.append("line_noise_residual")

    hf_lf = pd.to_numeric(metrics_row.get("hf_lf_ratio"), errors="coerce")
    if np.isfinite(hf_lf) and float(hf_lf) > t.hf_lf_ratio:
        reasons.append("high_hf_ratio")

    if check_retention:
        duration_ret = pd.to_numeric(metrics_row.get("duration_retention_pct"), errors="coerce")
        if np.isfinite(duration_ret) and float(duration_ret) < t.duration_retention_pct:
            reasons.append("low_duration_retention")

        coverage_ret = pd.to_numeric(
            metrics_row.get("condition_coverage_retention_pct"), errors="coerce"
        )
        if np.isfinite(coverage_ret) and float(coverage_ret) < t.condition_coverage_retention_pct:
            reasons.append("low_condition_retention")

    unusable_codes = {"too_many_bad_channels", "amplitude_above_threshold"}
    if check_retention:
        unusable_codes.add("low_duration_retention")

    if unusable_codes & set(reasons):
        return "unusable", reasons
    if reasons:
        return "borderline", reasons
    return "usable", []

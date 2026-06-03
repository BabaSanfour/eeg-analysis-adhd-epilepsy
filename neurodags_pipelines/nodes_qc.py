"""Base-stage QC nodes for the neurodags preprocessing pipeline.

Three nodes:
  compute_raw_qc_metrics  – signal QC on the unprocessed source file
  build_base_qc_record    – full QC on CleanedPrepRaw, deltas vs raw, channel diagnostics
  generate_base_qc_report – HTML report for a single run from the QC record JSON

AGGREGATOR NODE NOTE
--------------------
``generate_base_qc_report``, ``generate_correct_qc_report``, and ``generate_denoise_qc_report``
are partial aggregator nodes: they receive a QC record as primary input but also glob for AR
plot PNGs from the parent directory of ``reference_base`` (see ``_reconstruct_qc_report_inputs``).
This is a workaround for neurodags lacking a gather/fan-in primitive — ideally the AR plot paths
would be explicit DAG inputs declared in the YAML rather than discovered at runtime via glob.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node

LOGGER = logging.getLogger(__name__)
_LINE_FREQ = 60.0
_MIN_SEGMENT_SEC = 5.0

_DELTA_METRICS = (
    "amplitude_mean_uv",
    "amplitude_max_uv",
    "pct_bad_channels",
    "line_noise_ratio",
    "hf_lf_ratio",
    "alpha_peak_hz",
    "aperiodic_slope",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mne_raw(obj: Any):
    from neurodags.loaders import load_meeg
    if isinstance(obj, NodeResult):
        for k, art in obj.artifacts.items():
            if k == ".fif" or k.endswith(".fif"):
                return art.item
        return next(iter(obj.artifacts.values())).item
    if isinstance(obj, (str, os.PathLike)):
        return load_meeg(obj)
    return obj


def _load_json(obj: Any) -> dict:
    if isinstance(obj, (str, os.PathLike)):
        with open(obj, encoding="utf-8") as fh:
            return json.load(fh)
    if isinstance(obj, NodeResult):
        arts = list(obj.artifacts.values())
        if len(arts) == 1:
            v = arts[0].item
            if isinstance(v, (str, os.PathLike)):
                return _load_json(v)
            return v
        raise ValueError("Cannot unwrap multi-artifact NodeResult as JSON")
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(float(obj)) else float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _delta(current: Any, reference: Any) -> float:
    c = pd.to_numeric(current, errors="coerce")
    r = pd.to_numeric(reference, errors="coerce")
    return float(c - r) if np.isfinite(c) and np.isfinite(r) else float("nan")


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _strip_bad_annotations(raw: Any) -> Any:
    """Return a copy of raw with BAD_ annotations removed.

    Per-segment QC metrics should reflect the full window quality, not be
    skipped because autoreject marked sub-intervals inside the window as bad.
    MNE's compute_psd omits BAD-annotated regions by default, which causes
    ZeroDivisionError when all Welch windows inside a condition are rejected.
    """
    import mne
    raw_clean = raw.copy()
    keep = [
        i for i, desc in enumerate(raw_clean.annotations.description)
        if not str(desc).startswith("BAD")
    ]
    raw_clean.set_annotations(raw_clean.annotations[keep])
    return raw_clean


# ---------------------------------------------------------------------------
# Node 1: compute_raw_qc_metrics
# ---------------------------------------------------------------------------

@register_node
def compute_raw_qc_metrics(mne_object) -> NodeResult:
    """Signal QC metrics on the unprocessed source recording.

    Parses BIDS identifiers from raw.filenames[0] and stores them together
    with scalar quality metrics and per-condition segment metrics in a JSON
    artifact.  These values become the denominator and baseline for delta
    calculations in build_base_qc_record.
    """
    from eeg_adhd_epilepsy.io import bids as bids_io
    from eeg_adhd_epilepsy.qc import preproc_qc
    from eeg_adhd_epilepsy.signal_quality import metrics as signal_quality

    raw = _load_mne_raw(mne_object)

    source_path = (
        Path(raw.filenames[0])
        if raw.filenames and raw.filenames[0]
        else Path("unknown")
    )
    ids = bids_io.build_bids_report_ids(source_path)

    prepared, picks = preproc_qc._prepare_signal(raw)
    metrics = signal_quality.compute_signal_qc_metrics(
        prepared, picks=picks, line_freq=_LINE_FREQ, include_channel_metrics=False
    )

    total_duration = float(prepared.times[-1]) if prepared.times.size else float("nan")

    segments_df = bids_io.load_segments_for_raw(prepared)
    if segments_df is not None and not segments_df.empty and "t_start" in segments_df and "t_stop" in segments_df:
        total_condition = float(
            (pd.to_numeric(segments_df["t_stop"], errors="coerce") - pd.to_numeric(segments_df["t_start"], errors="coerce"))
            .clip(lower=0)
            .sum()
        )
    else:
        total_condition = 0.0

    # Per-condition segment metrics on the raw recording — used as pre-base baseline
    # when build_base_qc_record computes the Per-Condition Pre vs Post comparison.
    raw_segment_rows: list[dict] = []
    if segments_df is not None and not segments_df.empty:
        run_prefix = str(ids["run_prefix"])
        prepared_no_bad = _strip_bad_annotations(prepared)
        for row in segments_df.itertuples(index=False):
            t_start = float(getattr(row, "t_start", 0.0) or 0.0)
            t_stop = float(getattr(row, "t_stop", 0.0) or 0.0)
            duration = float(getattr(row, "duration", t_stop - t_start) or 0.0)
            if duration < _MIN_SEGMENT_SEC:
                continue
            seg = signal_quality.crop_segment(prepared_no_bad, t_start, t_stop, picks=list(picks))
            if seg is None:
                continue
            try:
                m = signal_quality.compute_signal_qc_metrics(
                    seg, picks=list(picks), line_freq=_LINE_FREQ, include_channel_metrics=False
                )
            except Exception:
                continue
            raw_segment_rows.append({
                "run_prefix": run_prefix,
                "segment_type": str(getattr(row, "segment_type", "")),
                "t_start": t_start,
                "t_stop": t_stop,
                "duration": duration,
                "segment_amplitude_mean_uv": _safe_float(m.get("amplitude_mean_uv")),
                "segment_amplitude_max_uv": _safe_float(m.get("amplitude_max_uv")),
                "segment_pct_bad_channels": _safe_float(m.get("pct_bad_channels")),
                "segment_line_noise_ratio": _safe_float(m.get("line_noise_ratio")),
                "segment_hf_lf_ratio": _safe_float(m.get("hf_lf_ratio")),
                "segment_aperiodic_slope": _safe_float(m.get("aperiodic_slope")),
            })

    record: dict = {
        "subject_id": str(ids["subject_id"]),
        "session_id": str(ids.get("session_id") or ""),
        "run_id": str(ids.get("run_id") or ""),
        "subject_session_prefix": str(ids["subject_session_prefix"]),
        "run_prefix": str(ids["run_prefix"]),
        "subject_session_key": list(ids["subject_session_key"]),
        "raw_duration": _safe_float(total_duration),
        "total_duration": total_condition,
        "amplitude_mean_uv": _safe_float(metrics.get("amplitude_mean_uv")),
        "amplitude_max_uv": _safe_float(metrics.get("amplitude_max_uv")),
        "pct_bad_channels": _safe_float(metrics.get("pct_bad_channels")),
        "line_noise_ratio": _safe_float(metrics.get("line_noise_ratio")),
        "hf_lf_ratio": _safe_float(metrics.get("hf_lf_ratio")),
        "alpha_peak_hz": _safe_float(metrics.get("alpha_peak_hz")),
        "aperiodic_slope": _safe_float(metrics.get("aperiodic_slope")),
        "n_flat_channels": int(metrics.get("n_flat_channels", 0) or 0),
        "n_noisy_channels": int(metrics.get("n_noisy_channels", 0) or 0),
        "raw_segments_df": raw_segment_rows,
    }
    return NodeResult(
        artifacts={"._raw_qc.json": Artifact(item=record, writer=lambda p, d=record: _write_json(p, d))}
    )


# ---------------------------------------------------------------------------
# Shared QC record builder (used by base / correct / denoise nodes)
# ---------------------------------------------------------------------------

def _build_stage_qc_record(
    mne_object: Any,
    raw_metrics: Any,
    prev_stage_metrics: Any,  # None for base (prev == raw)
    *,
    stage: str,
    source_stage_label: str,
    previous_stage_label: str,
    output_artifact_key: str,
) -> NodeResult:
    """Compute a full QC record for one preprocessing stage.

    raw_metrics        – RawQCMetrics JSON (scalar baseline + raw_segments_df).
    prev_stage_metrics – previous stage's QC JSON for delta_prev; None means
                         use raw_metrics (base stage where prev == raw).
    """
    from eeg_adhd_epilepsy.io import bids as bids_io
    from eeg_adhd_epilepsy.qc import preproc_qc
    from eeg_adhd_epilepsy.signal_quality import metrics as signal_quality

    raw = _load_mne_raw(mne_object)
    raw_ref = _load_json(raw_metrics)
    prev_ref = _load_json(prev_stage_metrics) if prev_stage_metrics is not None else raw_ref

    prepared, picks = preproc_qc._prepare_signal(raw)
    qc_metrics = signal_quality.compute_signal_qc_metrics(
        prepared, picks=picks, line_freq=_LINE_FREQ, include_channel_metrics=True
    )

    retained_sec = preproc_qc.compute_clean_duration(prepared)
    usable_coverage_sec = preproc_qc.compute_usable_condition_coverage(prepared)

    raw_duration = pd.to_numeric(raw_ref.get("raw_duration"), errors="coerce")
    raw_condition = pd.to_numeric(raw_ref.get("total_duration"), errors="coerce")

    duration_retention_pct = _safe_float(
        retained_sec / raw_duration * 100.0
        if np.isfinite(raw_duration) and raw_duration > 0
        else float("nan")
    )
    condition_coverage_retention_pct = _safe_float(
        usable_coverage_sec / raw_condition * 100.0
        if np.isfinite(raw_condition) and raw_condition > 0
        else float("nan")
    )

    run_metrics: dict = {
        "subject_id": str(raw_ref.get("subject_id", "")),
        "session_id": str(raw_ref.get("session_id", "")),
        "run_id": str(raw_ref.get("run_id", "")),
        "subject_session_prefix": str(raw_ref.get("subject_session_prefix", "")),
        "run_prefix": str(raw_ref.get("run_prefix", "")),
        "stage": stage,
        "output_desc": stage,
        "source_stage": source_stage_label,
        "reference_stage": previous_stage_label,
        "raw_duration_sec": _safe_float(raw_duration),
        "retained_duration_sec": _safe_float(retained_sec),
        "usable_condition_coverage_sec": _safe_float(usable_coverage_sec),
        "duration_retention_pct": duration_retention_pct,
        "condition_coverage_retention_pct": condition_coverage_retention_pct,
        "amplitude_mean_uv": _safe_float(qc_metrics.get("amplitude_mean_uv")),
        "amplitude_max_uv": _safe_float(qc_metrics.get("amplitude_max_uv")),
        "n_flat_channels": int(qc_metrics.get("n_flat_channels", 0) or 0),
        "n_noisy_channels": int(qc_metrics.get("n_noisy_channels", 0) or 0),
        "pct_bad_channels": _safe_float(qc_metrics.get("pct_bad_channels")),
        "line_noise_ratio": _safe_float(qc_metrics.get("line_noise_ratio")),
        "hf_lf_ratio": _safe_float(qc_metrics.get("hf_lf_ratio")),
        "alpha_peak_hz": _safe_float(qc_metrics.get("alpha_peak_hz")),
        "aperiodic_slope": _safe_float(qc_metrics.get("aperiodic_slope")),
    }

    for metric in _DELTA_METRICS:
        run_metrics[f"{metric}_delta_prev"] = _safe_float(
            _delta(qc_metrics.get(metric), prev_ref.get(metric))
        )
        run_metrics[f"{metric}_delta_raw"] = _safe_float(
            _delta(qc_metrics.get(metric), raw_ref.get(metric))
        )

    warnings_list: list[str] = []
    if pd.isna(qc_metrics.get("aperiodic_slope")):
        warnings_list.append("Spectral slope fitting skipped (insufficient data).")
    run_metrics["pipeline_warnings"] = "; ".join(warnings_list)

    qc_flag, qc_reasons = preproc_qc._evaluate_preproc_qc_flag(run_metrics)
    run_metrics["qc_flag"] = qc_flag
    run_metrics["qc_flag_reasons"] = ";".join(qc_reasons)

    channel_diagnostics = preproc_qc._build_channel_diagnostics(qc_metrics, channel_names=picks)

    topomap_raw = preproc_qc._build_topomap_aggregates(
        qc_metrics, channel_names=picks, weight=max(retained_sec, 1.0)
    )
    topomap_serializable = {
        metric: {
            "channels": list(channels),
            "values": [_safe_float(v) for v in values],
        }
        for metric, (channels, values, _weight) in topomap_raw.items()
    }

    # Per-segment post-clean metrics for temporal plots and pre/post comparison.
    # BAD_ annotations are stripped before PSD so MNE does not omit all Welch
    # windows — we want whole-window quality, not annotation-filtered quality.
    segments_df_raw = bids_io.load_segments_for_raw(prepared)
    prepared_no_bad = _strip_bad_annotations(prepared)
    post_rows: list[dict] = []
    if segments_df_raw is not None and not segments_df_raw.empty:
        for row in segments_df_raw.itertuples(index=False):
            t_start = float(getattr(row, "t_start", 0.0) or 0.0)
            t_stop = float(getattr(row, "t_stop", 0.0) or 0.0)
            duration = float(getattr(row, "duration", 0.0) or 0.0)
            if duration < _MIN_SEGMENT_SEC:
                continue
            seg = signal_quality.crop_segment(prepared_no_bad, t_start, t_stop, picks=list(picks))
            if seg is None:
                continue
            try:
                m = signal_quality.compute_signal_qc_metrics(
                    seg, picks=list(picks), line_freq=_LINE_FREQ, include_channel_metrics=False
                )
            except Exception:
                continue
            post_rows.append({
                "segment_type": str(getattr(row, "segment_type", "")),
                "t_start": t_start,
                "t_stop": t_stop,
                "duration": duration,
                # Unprefixed names match temporal viz plot expectations
                "amplitude_mean_uv": _safe_float(m.get("amplitude_mean_uv")),
                "amplitude_max_uv": _safe_float(m.get("amplitude_max_uv")),
                "pct_bad_channels": _safe_float(m.get("pct_bad_channels")),
                "line_noise_ratio": _safe_float(m.get("line_noise_ratio")),
                "hf_lf_ratio": _safe_float(m.get("hf_lf_ratio")),
                "aperiodic_slope": _safe_float(m.get("aperiodic_slope")),
            })

    # Per-condition pre (raw baseline) vs post comparison — all stages use raw as "pre"
    segment_comparison_rows: list[dict] = []
    if post_rows:
        post_df = pd.DataFrame(post_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            post_agg = post_df.groupby("segment_type").agg(
                n_segments_post=("segment_type", "count"),
                total_duration_post_sec=("duration", "sum"),
                mean_amplitude_post=("amplitude_mean_uv", "mean"),
                mean_line_noise_post=("line_noise_ratio", "mean"),
                mean_hf_lf_post=("hf_lf_ratio", "mean"),
                mean_pct_bad_channels_post=("pct_bad_channels", "mean"),
                mean_aperiodic_slope_post=("aperiodic_slope", "mean"),
            ).reset_index()

        pre_rows_raw = raw_ref.get("raw_segments_df") or []
        if pre_rows_raw:
            pre_df = pd.DataFrame(pre_rows_raw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pre_agg = pre_df.groupby("segment_type").agg(
                    n_segments_pre=("segment_type", "count"),
                    mean_amplitude_pre=("segment_amplitude_mean_uv", "mean"),
                    mean_line_noise_pre=("segment_line_noise_ratio", "mean"),
                    mean_hf_lf_pre=("segment_hf_lf_ratio", "mean"),
                    mean_pct_bad_channels_pre=("segment_pct_bad_channels", "mean"),
                    mean_aperiodic_slope_pre=("segment_aperiodic_slope", "mean"),
                ).reset_index()
            merged = post_agg.merge(pre_agg, on="segment_type", how="left")
        else:
            merged = post_agg.copy()
            for col in ("n_segments_pre", "mean_amplitude_pre", "mean_line_noise_pre",
                        "mean_hf_lf_pre", "mean_pct_bad_channels_pre", "mean_aperiodic_slope_pre"):
                merged[col] = float("nan")

        ordered_cols = [
            "segment_type",
            "n_segments_pre", "n_segments_post",
            "total_duration_post_sec",
            "mean_amplitude_pre", "mean_amplitude_post",
            "mean_line_noise_pre", "mean_line_noise_post",
            "mean_hf_lf_pre", "mean_hf_lf_post",
            "mean_pct_bad_channels_pre", "mean_pct_bad_channels_post",
            "mean_aperiodic_slope_pre", "mean_aperiodic_slope_post",
        ]
        merged = merged[[c for c in ordered_cols if c in merged.columns]].sort_values("segment_type").reset_index(drop=True)
        segment_comparison_rows = [
            {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
             for k, v in row.items()}
            for row in merged.to_dict(orient="records")
        ]

    record: dict = {
        **run_metrics,
        "channel_diagnostics": channel_diagnostics,
        "topomap_aggregates": topomap_serializable,
        "segments_df": post_rows,
        "segment_comparison": segment_comparison_rows,
    }
    return NodeResult(
        artifacts={output_artifact_key: Artifact(item=record, writer=lambda p, d=record: _write_json(p, d))}
    )


def _reconstruct_qc_report_inputs(
    record: dict,
    reference_base: Any,
    ar_glob_suffix: str,
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None, dict, dict[str, Path]]:
    """Shared reconstruction of topomap, segments, segment_comparison, channel_diagnostics, AR figures."""
    topomap_agg: dict = {}
    for metric, data in (record.get("topomap_aggregates") or {}).items():
        if data and data.get("channels") and data.get("values"):
            values_arr = np.array(
                [float("nan") if v is None else float(v) for v in data["values"]],
                dtype=float,
            )
            topomap_agg[metric] = (data["channels"], values_arr)

    segs_raw = record.get("segments_df") or []
    segments_df = pd.DataFrame(segs_raw) if segs_raw else None

    seg_cmp_raw = record.get("segment_comparison") or []
    if seg_cmp_raw:
        seg_cmp_df = pd.DataFrame(seg_cmp_raw)
        for col in seg_cmp_df.columns:
            if col != "segment_type":
                seg_cmp_df[col] = pd.to_numeric(seg_cmp_df[col], errors="coerce")
        segment_comparison = seg_cmp_df if not seg_cmp_df.empty else None
    else:
        segment_comparison = None

    channel_diagnostics = record.get("channel_diagnostics") or {}

    ar_figures: dict[str, Path] = {}
    if reference_base is not None:
        ref = Path(reference_base)
        for png in sorted(ref.parent.glob(f"{ref.name}@{ar_glob_suffix}_ar_plot_*.png")):
            cond = png.stem.split("_ar_plot_", 1)[-1]
            ar_figures[f"run/{cond}"] = png

    return topomap_agg, segments_df, segment_comparison, channel_diagnostics, ar_figures


# ---------------------------------------------------------------------------
# Nodes 2a–2c: build_*_qc_record
# ---------------------------------------------------------------------------

@register_node
def build_base_qc_record(mne_object, raw_metrics) -> NodeResult:
    """Full QC record for the base-preprocessed raw (deltas vs raw baseline)."""
    return _build_stage_qc_record(
        mne_object, raw_metrics, None,
        stage="base",
        source_stage_label="Raw Pre-Base",
        previous_stage_label="Raw Pre-Base",
        output_artifact_key="._base_qc.json",
    )


@register_node
def build_correct_qc_record(mne_object, raw_metrics, base_qc_record) -> NodeResult:
    """Full QC record for the correct stage (deltas vs base and vs raw)."""
    return _build_stage_qc_record(
        mne_object, raw_metrics, base_qc_record,
        stage="correct",
        source_stage_label="Base",
        previous_stage_label="Base",
        output_artifact_key="._correct_qc.json",
    )


@register_node
def build_denoise_qc_record(mne_object, raw_metrics, correct_qc_record) -> NodeResult:
    """Full QC record for the denoise stage (deltas vs correct and vs raw)."""
    return _build_stage_qc_record(
        mne_object, raw_metrics, correct_qc_record,
        stage="denoise",
        source_stage_label="Correct",
        previous_stage_label="Correct",
        output_artifact_key="._denoise_qc.json",
    )


# ---------------------------------------------------------------------------
# Shared QC report generator
# ---------------------------------------------------------------------------

def _generate_stage_qc_report(
    qc_record: Any,
    reference_base: Any,
    *,
    stage_display_name: str,
    previous_stage_label: str,
    raw_reference_label: str,
    ar_glob_suffix: str,
    report_html_suffix: str,
    figures_suffix: str,
) -> NodeResult:
    """Shared HTML report generator for any preprocessing QC stage."""
    import matplotlib
    matplotlib.use("Agg")

    from eeg_adhd_epilepsy.reports import preproc_qc as report_preproc_qc
    from eeg_adhd_epilepsy.viz import preproc_qc as viz_preproc_qc

    record = _load_json(qc_record)
    topomap_agg, segments_df, segment_comparison, channel_diagnostics, ar_figures = (
        _reconstruct_qc_report_inputs(record, reference_base, ar_glob_suffix)
    )

    def _writer(
        html_path: str,
        rec: dict = record,
        topos: dict = topomap_agg,
        segs: pd.DataFrame | None = segments_df,
        ch_diag: dict = channel_diagnostics,
        ar_figs: dict = ar_figures,
        seg_cmp: pd.DataFrame | None = segment_comparison,
        _prev_label: str = previous_stage_label,
        _raw_label: str = raw_reference_label,
        _stage_name: str = stage_display_name,
        _fig_sfx: str = figures_suffix,
        _html_sfx: str = report_html_suffix,
    ) -> None:
        output_path = Path(html_path)
        figures_dir = output_path.parent / output_path.name.replace(_html_sfx, "_figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig_paths = viz_preproc_qc.save_subject_preproc_qc_figures(
            record=rec,
            topomap_aggregates=topos or None,
            segments_df=segs if (segs is not None and not segs.empty) else None,
            output_dir=figures_dir,
        )

        scalar_record = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        run_summary_df = pd.DataFrame([scalar_record])

        report_preproc_qc.generate_subject_report(
            record=rec,
            previous_stage_label=_prev_label,
            raw_reference_label=_raw_label,
            stage_display_name=_stage_name,
            figures=fig_paths,
            run_summary_df=run_summary_df,
            output_path=output_path,
            channel_diagnostics=ch_diag,
            autoreject_figures=ar_figs if ar_figs else None,
            segment_comparison=seg_cmp,
        )

    return NodeResult(
        artifacts={report_html_suffix: Artifact(item=record, writer=_writer)}
    )


# ---------------------------------------------------------------------------
# Nodes 3a–3c: generate_*_qc_report
# ---------------------------------------------------------------------------

@register_node
def generate_base_qc_report(qc_record, reference_base=None) -> NodeResult:
    """Generate a single-run base QC HTML report from the QC record JSON."""
    return _generate_stage_qc_report(
        qc_record, reference_base,
        stage_display_name="Base",
        previous_stage_label="Raw Pre-Base",
        raw_reference_label="Raw Pre-Base",
        ar_glob_suffix="CleanedPrepRaw",
        report_html_suffix="._base_qc_report.html",
        figures_suffix="._base_qc_report_figures",
    )


@register_node
def generate_correct_qc_report(qc_record, reference_base=None) -> NodeResult:
    """Generate a single-run correct QC HTML report from the QC record JSON."""
    return _generate_stage_qc_report(
        qc_record, reference_base,
        stage_display_name="Correct",
        previous_stage_label="Base",
        raw_reference_label="Raw Pre-Base",
        ar_glob_suffix="CorrectRaw",
        report_html_suffix="._correct_qc_report.html",
        figures_suffix="._correct_qc_report_figures",
    )


@register_node
def generate_denoise_qc_report(qc_record, reference_base=None) -> NodeResult:
    """Generate a single-run denoise QC HTML report from the QC record JSON."""
    return _generate_stage_qc_report(
        qc_record, reference_base,
        stage_display_name="Denoise",
        previous_stage_label="Correct",
        raw_reference_label="Raw Pre-Base",
        ar_glob_suffix="DenoiseRaw",
        report_html_suffix="._denoise_qc_report.html",
        figures_suffix="._denoise_qc_report_figures",
    )

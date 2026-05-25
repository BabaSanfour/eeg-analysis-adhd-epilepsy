"""Base-stage QC nodes for the neurodags preprocessing pipeline.

Three nodes:
  compute_raw_qc_metrics  – signal QC on the unprocessed source file
  build_base_qc_record    – full QC on CleanedPrepRaw, deltas vs raw, channel diagnostics
  generate_base_qc_report – HTML report for a single run from the QC record JSON
"""

from __future__ import annotations

import json
import logging
import os
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
    # neurodags passes {"cached": [path, ...]} when parent has overwrite=True but sub-derivative is cached
    if isinstance(obj, dict) and "cached" in obj:
        paths = obj["cached"]
        fif_paths = [p for p in paths if str(p).endswith(".fif")]
        path = fif_paths[0] if fif_paths else (paths[0] if paths else None)
        if path:
            return load_meeg(path)
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
    # neurodags passes {"cached": [path, ...]} when parent has overwrite=True but sub-derivative is cached
    if isinstance(obj, dict) and "cached" in obj:
        paths = [p for p in obj["cached"] if str(p).endswith(".json")]
        if paths:
            with open(paths[0], encoding="utf-8") as fh:
                return json.load(fh)
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


# ---------------------------------------------------------------------------
# Node 1: compute_raw_qc_metrics
# ---------------------------------------------------------------------------

@register_node
def compute_raw_qc_metrics(mne_object) -> NodeResult:
    """Signal QC metrics on the unprocessed source recording.

    Parses BIDS identifiers from raw.filenames[0] and stores them together
    with scalar quality metrics in a JSON artifact.  These values become the
    denominator and baseline for delta calculations in build_base_qc_record.
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
    }
    return NodeResult(
        artifacts={"._raw_qc.json": Artifact(item=record, writer=lambda p, d=record: _write_json(p, d))}
    )


# ---------------------------------------------------------------------------
# Node 2: build_base_qc_record
# ---------------------------------------------------------------------------

@register_node
def build_base_qc_record(mne_object, raw_metrics) -> NodeResult:
    """Full QC record for the base-preprocessed raw.

    Computes signal quality metrics, duration retention, deltas vs the raw
    baseline from raw_metrics, QC flag, channel diagnostics, topomap data,
    and per-segment post-clean metrics for temporal plots.
    All results are stored in a single JSON artifact.
    """
    from eeg_adhd_epilepsy.io import bids as bids_io
    from eeg_adhd_epilepsy.qc import preproc_qc
    from eeg_adhd_epilepsy.signal_quality import metrics as signal_quality

    raw = _load_mne_raw(mne_object)
    raw_ref = _load_json(raw_metrics)

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
        "stage": "base",
        "output_desc": "base",
        "source_stage": "Raw Pre-Base",
        "reference_stage": "Raw Pre-Base",
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
        d = _delta(qc_metrics.get(metric), raw_ref.get(metric))
        run_metrics[f"{metric}_delta_prev"] = _safe_float(d)
        run_metrics[f"{metric}_delta_raw"] = _safe_float(d)

    warnings_list = []
    if pd.isna(qc_metrics.get("aperiodic_slope")):
        warnings_list.append("Spectral slope fitting skipped (insufficient data).")
    run_metrics["pipeline_warnings"] = "; ".join(warnings_list)

    qc_flag, qc_reasons = preproc_qc._evaluate_preproc_qc_flag(run_metrics)
    run_metrics["qc_flag"] = qc_flag
    run_metrics["qc_flag_reasons"] = ";".join(qc_reasons)

    # Channel diagnostics — already JSON-serializable (lists, floats)
    channel_diagnostics = preproc_qc._build_channel_diagnostics(qc_metrics, channel_names=picks)

    # Topomap aggregates — serialize numpy arrays as lists
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

    # Per-segment post-clean metrics for temporal plots
    segments_df_raw = bids_io.load_segments_for_raw(prepared)
    post_rows: list[dict] = []
    if segments_df_raw is not None and not segments_df_raw.empty:
        for row in segments_df_raw.itertuples(index=False):
            t_start = float(getattr(row, "t_start", 0.0) or 0.0)
            t_stop = float(getattr(row, "t_stop", 0.0) or 0.0)
            duration = float(getattr(row, "duration", 0.0) or 0.0)
            if duration < _MIN_SEGMENT_SEC:
                continue
            seg = signal_quality.crop_segment(prepared, t_start, t_stop, picks=list(picks))
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
                "amplitude_mean_uv": _safe_float(m.get("amplitude_mean_uv")),
                "line_noise_ratio": _safe_float(m.get("line_noise_ratio")),
                "hf_lf_ratio": _safe_float(m.get("hf_lf_ratio")),
            })

    record: dict = {
        **run_metrics,
        "channel_diagnostics": channel_diagnostics,
        "topomap_aggregates": topomap_serializable,
        "segments_df": post_rows,
    }
    return NodeResult(
        artifacts={"._base_qc.json": Artifact(item=record, writer=lambda p, d=record: _write_json(p, d))}
    )


# ---------------------------------------------------------------------------
# Node 3: generate_base_qc_report
# ---------------------------------------------------------------------------

@register_node
def generate_base_qc_report(qc_record, reference_base=None) -> NodeResult:
    """Generate a single-run base QC HTML report from the QC record JSON.

    Reconstructs topomap arrays and the per-segment DataFrame from the JSON,
    saves figures alongside the HTML, and collects any AutoReject plot PNGs
    saved by autoreject_annotate_blockwise for the same reference.
    """
    import matplotlib
    matplotlib.use("Agg")

    from eeg_adhd_epilepsy.reports import preproc_qc as report_preproc_qc
    from eeg_adhd_epilepsy.viz import preproc_qc as viz_preproc_qc

    record = _load_json(qc_record)

    # Reconstruct topomap_aggregates: {metric: (channels, np.ndarray)}
    topomap_agg: dict = {}
    for metric, data in (record.get("topomap_aggregates") or {}).items():
        if data and data.get("channels") and data.get("values"):
            values_arr = np.array(
                [float("nan") if v is None else float(v) for v in data["values"]],
                dtype=float,
            )
            topomap_agg[metric] = (data["channels"], values_arr)

    # Reconstruct segments_df for temporal plots
    segs_raw = record.get("segments_df") or []
    segments_df = pd.DataFrame(segs_raw) if segs_raw else None

    channel_diagnostics = record.get("channel_diagnostics") or {}

    # Collect AutoReject figure PNGs saved by autoreject_annotate_blockwise
    ar_figures: dict[str, Path] = {}
    if reference_base is not None:
        ref = Path(reference_base)
        for png in sorted(ref.parent.glob(f"{ref.name}@CleanedPrepRaw_ar_plot_*.png")):
            cond = png.stem.split("_ar_plot_", 1)[-1]
            ar_figures[f"run/{cond}"] = png

    def _writer(
        html_path: str,
        rec: dict = record,
        topos: dict = topomap_agg,
        segs: pd.DataFrame | None = segments_df,
        ch_diag: dict = channel_diagnostics,
        ar_figs: dict = ar_figures,
    ) -> None:
        output_path = Path(html_path)
        figures_dir = output_path.parent / (
            output_path.name.replace("._base_qc_report.html", "_figures")
        )
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
            previous_stage_label="Raw Pre-Base",
            raw_reference_label="Raw Pre-Base",
            stage_display_name="Base",
            figures=fig_paths,
            run_summary_df=run_summary_df,
            output_path=output_path,
            channel_diagnostics=ch_diag,
            autoreject_figures=ar_figs if ar_figs else None,
            segment_comparison=None,
        )

    return NodeResult(
        artifacts={
            "._base_qc_report.html": Artifact(item=record, writer=_writer)
        }
    )

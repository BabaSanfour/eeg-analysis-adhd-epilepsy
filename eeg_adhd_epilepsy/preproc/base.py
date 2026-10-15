"""EEG preprocessing pipeline base module.

This module provides the core pipeline for preprocessing EEG data using MNE-Python,
PyPREP, and AutoReject. The pipeline includes:

1.  Block Awareness: Uses embedded `BLOCK_*` annotations already present in BIDS.
2.  Filtering: Applies high-pass and low-pass filters.
3.  Line Noise Removal: ZapLine with at most ``ZAPLINE_MAX_COMPONENTS`` components, then
    spectrum interpolation where line noise remains.
4.  Resampling: Adjusts sampling rate (downsampled last, after band-limiting).
5.  Global Bad Channel Detection: Identifies broken channels using RANSAC.
6.  Reference: Applies Common Average Reference (CAR).
7.  Artifact Annotation: Annotates bad epochs and channels with AutoReject, fitted per
    experimental condition.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne
import numpy as np
from autoreject import AutoReject
from joblib import Parallel, delayed
from mne_bids import get_entities_from_fname
from pyprep.find_noisy_channels import NoisyChannels
from tqdm import tqdm

from eeg_adhd_epilepsy.io import bids, readers, report_paths
from eeg_adhd_epilepsy.preproc.epochs import build_block_events_by_condition
from eeg_adhd_epilepsy.preproc.utils import (
    NumpyEncoder,
    _compute_artifact_overlap,
    benchmark_step,
    inflate_bad_annotations,
)
from eeg_adhd_epilepsy.qc import preproc_qc
from eeg_adhd_epilepsy.signal_quality.spectral import compute_line_noise_index
from eeg_adhd_epilepsy.utils import events
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib

LOGGER = logging.getLogger("preproc_base")

DEFAULT_HIGHPASS_HZ = 0.1
DEFAULT_LOWPASS_HZ = 100.0
# AutoReject: fitted on 1 s segments, per condition, only when a condition has at least
# AR_MIN_EPOCHS of them; longer conditions are fitted in AR_CHUNK_MINUTES chunks (73
# conditions on 2026-10-14). It only rejects: no channel is interpolated.
AR_SEGMENT_S = 1.0
AR_MIN_EPOCHS = 5
AR_CHUNK_MINUTES = 30.0
RANDOM_SEED = 42
# Line noise: ZapLine keeps at most this many components, then spectrum interpolation
# removes what is left when the QC line-noise index (median over channels) is above
# LINE_NOISE_RATIO_MAX. Line noise from many sources otherwise takes ZapLine 13-19 of 19
# components and 5-6 dB of 30-55 Hz power. Measured on all 1,237 recordings (2026-10-15):
# 259 are interpolated, 11 keep a 60 Hz peak above 3 dB over the floor, and the others
# stay as ZapLine made them.
ZAPLINE_MAX_COMPONENTS = 3
LINE_NOISE_RATIO_MAX = 1.25


def _group_consecutive_indices(indices: list[int]) -> list[tuple[int, int]]:
    """Group consecutive integers into inclusive (start, end) tuples."""
    if len(indices) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def _event_sample_to_onset(raw: mne.io.BaseRaw, event_sample: int) -> float:
    """Convert an event sample index into onset seconds in raw time."""
    return max(0.0, (event_sample - raw.first_samp) / raw.info["sfreq"])


def _prepare_condition_epoch_inputs(
    raw: mne.io.BaseRaw,
    segment_duration: float,
) -> list[tuple[str, list[events.BlockWindow], np.ndarray]]:
    """Return condition-grouped blocks with their fixed-length events."""
    block_windows = events.collect_block_windows(raw)
    grouped_blocks: dict[str, list[events.BlockWindow]] = {}
    for block in block_windows:
        grouped_blocks.setdefault(block.name, []).append(block)

    events_by_condition = build_block_events_by_condition(
        raw,
        segment_duration=segment_duration,
        overlap=0.0,
    )
    prepared: list[tuple[str, list[events.BlockWindow], np.ndarray]] = []
    for condition_name, blocks in grouped_blocks.items():
        condition_events = events_by_condition.get(condition_name)
        if condition_events is None or len(condition_events) == 0:
            continue
        prepared.append((condition_name, blocks, condition_events))
    return prepared


def _iter_autoreject_chunks(
    epochs: mne.Epochs, condition_name: str
) -> list[tuple[int, mne.Epochs]]:
    """Split a condition's epochs into AutoReject-sized chunks."""
    n_epochs_total = len(epochs)
    if n_epochs_total < AR_MIN_EPOCHS:
        LOGGER.warning(
            "Too few epochs (%d < %d) for AutoReject in condition %s. Skipping AR.",
            n_epochs_total,
            AR_MIN_EPOCHS,
            condition_name,
        )
        return []

    n_epochs_chunk_max = max(1, int((AR_CHUNK_MINUTES * 60.0) / AR_SEGMENT_S))
    if n_epochs_total <= n_epochs_chunk_max:
        return [(0, epochs)]

    n_chunks = int(np.ceil(n_epochs_total / n_epochs_chunk_max))
    LOGGER.info(
        "Condition '%s' too long (%d epochs). "
        "Splitting into %d chunks of ~%d epochs (~%.1f min each).",
        condition_name,
        n_epochs_total,
        n_chunks,
        n_epochs_chunk_max,
        AR_CHUNK_MINUTES,
    )
    chunks: list[tuple[int, mne.Epochs]] = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * n_epochs_chunk_max
        end_idx = min((chunk_idx + 1) * n_epochs_chunk_max, n_epochs_total)
        chunk = epochs[start_idx:end_idx]
        if len(chunk) >= AR_MIN_EPOCHS:
            chunks.append((chunk_idx, chunk))
    return chunks


def _reject_log_to_annotations(
    raw: mne.io.BaseRaw,
    epochs_chunk: mne.Epochs,
    reject_log,
    condition_name: str,
) -> tuple[list[tuple[float, float, str, tuple[str, ...]]], int, int]:
    """Convert an AutoReject reject log into raw-time annotations."""
    new_annots: list[tuple[float, float, str, tuple[str, ...]]] = []
    bad_epoch_count = 0
    bad_span_count = 0

    for ep_idx, is_bad_epoch in enumerate(reject_log.bad_epochs):
        if not is_bad_epoch:
            continue
        onset_s = _event_sample_to_onset(raw, int(epochs_chunk.events[ep_idx, 0]))
        new_annots.append((onset_s, AR_SEGMENT_S, f"BAD_epoch_{condition_name}", ()))
        bad_epoch_count += 1

    labels = np.asarray(reject_log.labels)
    if labels.ndim == 2 and labels.shape[0] == len(epochs_chunk):
        for ch_idx, ch_name in enumerate(epochs_chunk.ch_names):
            bad_idx = np.flatnonzero(labels[:, ch_idx] != 0)
            for first_idx, last_idx in _group_consecutive_indices(bad_idx):
                start_samp = int(epochs_chunk.events[first_idx, 0])
                end_samp = int(epochs_chunk.events[last_idx, 0])
                start_s = _event_sample_to_onset(raw, start_samp)
                end_s = _event_sample_to_onset(raw, end_samp) + AR_SEGMENT_S
                duration_s = end_s - start_s
                if duration_s <= 0:
                    continue
                new_annots.append((start_s, duration_s, f"BAD_{condition_name}", (ch_name,)))
                bad_span_count += 1

    return new_annots, bad_epoch_count, bad_span_count


def _run_autoreject_chunk(
    raw: mne.io.BaseRaw,
    epochs_chunk: mne.Epochs,
    condition_name: str,
    chunk_suffix: str,
    n_jobs: int,
) -> tuple[list[tuple[float, float, str, tuple[str, ...]]], int, int]:
    """Fit AutoReject on one chunk and convert the reject log to annotations."""
    n_epochs_chunk = len(epochs_chunk)
    LOGGER.info(
        "Processing AutoReject for %s%s (%d epochs)", condition_name, chunk_suffix, n_epochs_chunk
    )
    ar = AutoReject(
        n_interpolate=np.array([0]),
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        verbose=False,
        cv=min(10, n_epochs_chunk),
    )
    ar.fit(epochs_chunk)
    return _reject_log_to_annotations(
        raw, epochs_chunk, ar.get_reject_log(epochs_chunk), condition_name=condition_name
    )


def _compute_clean_stats(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Compute clean-data fraction and per-type bad fractions from raw annotations.

    Considers only global (channel-less) BAD_ annotations and categorises them as
    autoreject-generated (`BAD_epoch_*`) or manual (`BAD_movement`, `BAD_yawn`,
    etc.); technical spans (`BAD_ACQ_SKIP`, `BAD_boundary`) reduce clean time but
    are counted as neither manual nor autoreject. Channel-specific spans are
    excluded because they do not remove the whole-recording time window.

    Bad time per category is computed by merging overlapping intervals (shared
    ``events.merge_intervals`` helper, the same primitive used by the QC stage).

    Returns:
        Dict with keys ``clean_duration_s``, ``clean_fraction``,
        ``manual_bad_fraction``, ``autoreject_bad_fraction``.
    """
    total_duration_s = float(raw.times[-1]) if raw.times.size else 0.0
    if total_duration_s <= 0:
        return {
            "clean_duration_s": 0.0,
            "clean_fraction": 0.0,
            "manual_bad_fraction": 0.0,
            "autoreject_bad_fraction": 0.0,
        }

    all_bad: list[tuple[float, float]] = []
    manual_bad: list[tuple[float, float]] = []
    autoreject_bad: list[tuple[float, float]] = []
    for annot in raw.annotations:
        desc = str(annot["description"])
        if not desc.startswith("BAD_") or annot["ch_names"]:
            continue  # non-BAD or channel-specific span
        start = max(float(annot["onset"]), 0.0)
        stop = min(float(annot["onset"]) + float(annot["duration"]), total_duration_s)
        if stop <= start:
            continue
        all_bad.append((start, stop))
        if desc.startswith("BAD_epoch_"):
            autoreject_bad.append((start, stop))
        elif not desc.startswith(("BAD_ACQ_SKIP", "BAD_boundary")):
            manual_bad.append((start, stop))

    def _summed(intervals: list[tuple[float, float]]) -> float:
        return sum(stop - start for start, stop in events.merge_intervals(intervals))

    clean_duration_s = max(total_duration_s - _summed(all_bad), 0.0)
    return {
        "clean_duration_s": clean_duration_s,
        "clean_fraction": clean_duration_s / total_duration_s,
        "manual_bad_fraction": _summed(manual_bad) / total_duration_s,
        "autoreject_bad_fraction": _summed(autoreject_bad) / total_duration_s,
    }


def _line_noise_ratio(raw: mne.io.BaseRaw, line_freq: float) -> float:
    """The QC line-noise index (``compute_line_noise_index``), median over EEG channels."""
    psd, freqs = raw.compute_psd(picks="eeg", fmin=0.5, fmax=99.5, verbose="ERROR").get_data(
        return_freqs=True
    )
    return float(np.median(compute_line_noise_index(psd, freqs, line_freq=line_freq)[1]))


def _remove_line_noise(
    raw: mne.io.BaseRaw, line_freq: float, adaptive: bool
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """ZapLine with at most ``ZAPLINE_MAX_COMPONENTS``, then spectrum interpolation if needed."""
    from mne_denoise.spectrum_interpolation import SpectrumInterpolation
    from mne_denoise.zapline import ZapLine

    sfreq = raw.info["sfreq"]
    zapline = ZapLine(sfreq=sfreq, line_freq=line_freq, adaptive=adaptive)
    cleaned = zapline.fit_transform(raw)
    n_auto = int(zapline.n_removed_)
    # Adaptive mode bounds its components per segment itself.
    if not adaptive and n_auto > ZAPLINE_MAX_COMPONENTS:
        zapline = ZapLine(sfreq=sfreq, line_freq=line_freq, n_select=ZAPLINE_MAX_COMPONENTS)
        cleaned = zapline.fit_transform(raw)
    line_ratio = _line_noise_ratio(cleaned, line_freq)
    interpolated = line_ratio > LINE_NOISE_RATIO_MAX
    if interpolated:
        cleaned = SpectrumInterpolation(line_freq=line_freq, n_harmonics=1).fit_transform(cleaned)
    return cleaned, {
        "method": "zapline",
        "line_freq": line_freq,
        "adaptive": adaptive,
        "n_selected_auto": n_auto,
        "n_removed": int(zapline.n_removed_),
        "line_noise_ratio_after_zapline": line_ratio,
        "spectrum_interpolation": interpolated,
    }


def run_base_pipeline(
    raw: mne.io.BaseRaw,
    config: dict[str, Any],
    subject_id: str = "unknown",
    record_label: str | None = None,
) -> tuple[mne.io.BaseRaw, dict]:
    """Run the shared preprocessing trunk and return cleaned raw + provenance.

    Pure transform — applies preprocessing steps and returns ``(cleaned_raw,
    provenance)``.  All I/O (path construction, file saving, report directory
    setup) is the responsibility of the caller (see :func:`run_base_record`).

    Args:
        raw: The raw MNE object to process.
        config: ``n_jobs`` and the ``processing`` and ``line_noise`` blocks of
            ``_build_pipeline_config``.
        subject_id: Identifier for the subject (used in logging/filenames).

    Returns:
        A tuple containing:
            - The processed MNE Raw object.
            - A dictionary containing processing provenance and statistics.
    """
    subject_id = bids.bids_subject_label(subject_id)
    record_label = record_label or subject_id

    LOGGER.info("Starting base pipeline for %s", record_label)

    provenance: dict[str, Any] = {
        "subject_id": subject_id,
        "config": config,
        "steps_completed": [],
        "pipeline_warnings": [],
        "bad_channels_global": [],
        "artifact_stats": {},
        "block_stats": [],
        "integrity_stats": {},
    }

    # 1. Embedded block annotations
    n_blocks = len(events.collect_block_windows(raw))
    if n_blocks == 0:
        LOGGER.warning(
            "No embedded BLOCK_* annotations found; block-aware steps will have limited context."
        )

    # 1b. Inflate Manual Annotations (Major -> 5s, Common -> 3s)
    raw = inflate_bad_annotations(raw)
    LOGGER.info("Inflated manual annotations: %d", len(raw.annotations))

    provenance["steps_completed"].append("embedded_blocks")

    # 2-3. Filtering and Line Noise Removal
    # --------------------------------------
    # Run on the full-rate signal, before any downsampling, so the high-pass and
    # line-noise estimates are computed at the original temporal resolution.
    with benchmark_step("filtering_and_denoising", provenance):
        hp_hz = config.get("processing", {}).get("highpass_hz", DEFAULT_HIGHPASS_HZ)
        lp_hz = config.get("processing", {}).get("lowpass_hz", DEFAULT_LOWPASS_HZ)
        line_noise_cfg = config.get("line_noise", {})
        line_freq = line_noise_cfg.get("line_freq", 60.0)

        # 2. Bandpass Filter
        # Ensure lowpass is strictly less than Nyquist (sfreq/2)
        nyquist = raw.info["sfreq"] / 2.0
        h_f = min(lp_hz, nyquist - 0.1) if lp_hz else None
        n_jobs = int(config.get("n_jobs", 1))

        LOGGER.info("Applying Bandpass filter: %s-%s Hz (n_jobs=%d)", hp_hz, h_f, n_jobs)
        raw.load_data()
        raw.filter(l_freq=hp_hz, h_freq=h_f, verbose="ERROR", n_jobs=n_jobs)
        provenance["steps_completed"].append("bandpass_filter")

        # 3. Line Noise Removal
        adaptive = line_noise_cfg.get("adaptive", False)
        LOGGER.info("Applying ZapLine (%s Hz, adaptive=%s)...", line_freq, adaptive)
        raw, provenance["zapline_stats"] = _remove_line_noise(raw, line_freq, adaptive)
        provenance["steps_completed"].append("zapline")

    # 4. Resample
    # Downsample last among the band-limiting steps (filter-then-downsample
    # convention). MNE applies its own anti-alias low-pass during resampling.
    target_sfreq = config.get("processing", {}).get("resample_hz", None)
    if target_sfreq:
        with benchmark_step("resample", provenance):
            raw.resample(target_sfreq, n_jobs=int(config.get("n_jobs", 1)))
        provenance["steps_completed"].append("resample")

    # 5. Global Bad Channel Detection
    with benchmark_step("detect_global_bads", provenance):
        bads, ransac_warning = detect_global_bads_ransac(raw)
        provenance["bad_channels_global"] = bads
        if ransac_warning:
            provenance["pipeline_warnings"].append(ransac_warning)
        provenance["steps_completed"].append("detect_global_bads")

    LOGGER.info("Global bad channels (%d): %s", len(raw.info["bads"]), raw.info["bads"])

    # 6. Common Average Reference (CAR)
    # Applied after excluding global bads to avoid contamination.
    with benchmark_step("reference_car", provenance):
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        provenance["steps_completed"].append("car_ref")

    # 7. Block-wise Artifact Annotation (AutoReject)
    with benchmark_step("block_artifact_annotation", provenance):
        raw, artifact_stats = annotate_artifacts_blockwise(raw, n_jobs=int(config.get("n_jobs", 1)))

    provenance["artifact_stats"] = artifact_stats
    provenance["block_stats"] = artifact_stats.get("by_block", [])
    provenance["steps_completed"].append("block_artifact_annotation")

    # Clean Data Duration & Detailed Artifact Stats
    provenance["integrity_stats"] = _compute_clean_stats(raw)
    provenance["artifact_stats"]["artifacts_count"] = len(raw.annotations)

    LOGGER.info("Base pipeline completed for %s.", record_label)
    return raw, provenance


def detect_global_bads_ransac(raw: mne.io.BaseRaw) -> tuple[list[str], str | None]:
    """Detect global bad EEG channels with RANSAC, biased toward rest blocks.

    Uses a subset of data (rest blocks) to speed up RANSAC and focus on
    intrinsic channel quality rather than task-related artifacts.

    Args:
        raw: Input raw data.

    Returns:
        tuple[List[str], str | None]:
            1. List of newly detected bad channel names.
            2. Warning message if RANSAC was skipped/failed.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        LOGGER.warning("No EEG channels available for RANSAC bad channel detection.")
        return [], "No EEG channels available for RANSAC bad channel detection."

    eeg_ch_names = [raw.ch_names[idx] for idx in eeg_picks]
    eeg_raw = raw.copy().pick(eeg_ch_names)
    rest_windows = events.collect_baseline_windows(raw)

    raw_for_ransac = eeg_raw
    if rest_windows:
        crops: list[mne.io.BaseRaw] = []
        for onset, stop in rest_windows:
            if stop <= onset:
                continue
            crop = eeg_raw.copy().crop(onset, stop, include_tmax=False)
            if crop.n_times > 0:
                crops.append(crop)

        if crops:
            raw_for_ransac = (
                crops[0] if len(crops) == 1 else mne.concatenate_raws(crops, verbose="ERROR")
            )

    duration_s = raw_for_ransac.n_times / raw_for_ransac.info["sfreq"]
    LOGGER.info("Running RANSAC on %.1fs of EEG data...", duration_s)

    nc = NoisyChannels(raw_for_ransac, random_state=RANDOM_SEED)
    try:
        nc.find_bad_by_ransac()
        bads = nc.get_bads(verbose=False) or []
    except (ValueError, OSError) as exc:
        msg = f"RANSAC bad channel detection skipped: {exc}"
        LOGGER.warning(msg)
        return [], msg

    bads = sorted(ch for ch in bads if ch in raw.ch_names)

    # Update raw.info['bads']
    current_bads = set(raw.info.get("bads", []))
    new_bads = current_bads.union(bads)
    raw.info["bads"] = sorted(new_bads)

    return bads, None


def annotate_artifacts_blockwise(
    raw: mne.io.BaseRaw, n_jobs: int = 1
) -> tuple[mne.io.BaseRaw, dict]:
    """Run condition-wise AutoReject and add non-destructive BAD annotations.

    Groups disjoint blocks by condition (e.g., all "EO_baseline" blocks) and
    runs AutoReject on them as a single set of epochs. This ensures consistent
    thresholds across the condition.

    Args:
        raw: The raw data object.
        n_jobs: Cores AutoReject may use.

    Returns:
        A tuple containing the annotated raw object and a statistics dictionary.
    """
    block_windows = events.collect_block_windows(raw)
    stats: dict[str, Any] = {
        "blocks_total": len(block_windows),
        "blocks_processed": 0,
        "bad_epochs": 0,
        "bad_channel_spans": 0,
        "artifacts_count": 0,
        "by_block": [],
    }

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        LOGGER.warning("No EEG channels available for AutoReject annotation.")
        return raw, stats

    epoch_tmax = max(AR_SEGMENT_S - (1.0 / raw.info["sfreq"]), 0.0)
    condition_inputs = _prepare_condition_epoch_inputs(raw, segment_duration=AR_SEGMENT_S)

    LOGGER.info(
        "Running condition-wise artifact annotation on %d conditions...", len(condition_inputs)
    )

    new_annots: list[tuple[float, float, str, tuple[str, ...]]] = []

    for condition_name, blocks, condition_events in condition_inputs:
        epochs = mne.Epochs(
            raw,
            condition_events,
            event_id={"seg": 1},
            tmin=0.0,
            tmax=epoch_tmax,
            baseline=None,
            reject=None,
            verbose="ERROR",
            preload=True,
            proj=False,
            picks=eeg_picks,
            reject_by_annotation=False,
        )

        n_epochs_total = len(epochs)
        epoch_chunks = _iter_autoreject_chunks(epochs, condition_name)
        if not epoch_chunks:
            continue

        for chunk_idx, epochs_chunk in epoch_chunks:
            chunk_suffix = f"_chunk{chunk_idx + 1}" if len(epoch_chunks) > 1 else ""
            chunk_annots, chunk_bad_epochs, chunk_bad_spans = _run_autoreject_chunk(
                raw, epochs_chunk, condition_name, chunk_suffix, n_jobs
            )
            new_annots.extend(chunk_annots)
            stats["bad_epochs"] += chunk_bad_epochs
            stats["bad_channel_spans"] += chunk_bad_spans

        stats["blocks_processed"] += len(blocks)
        stats["by_block"].append(
            {
                "condition": condition_name,
                "n_blocks_merged": len(blocks),
                "epochs_total": n_epochs_total,
                "chunks_processed": len(epoch_chunks),
            }
        )

    if not new_annots:
        stats["artifacts_count"] = 0
        return raw, stats

    overlap_pct = _compute_artifact_overlap(raw, new_annots)
    LOGGER.info(
        "Manual BAD Overlap: %.1f%% of manual segments were re-detected.",
        overlap_pct,
    )
    stats["manual_overlap_pct"] = overlap_pct

    new_annots.sort(key=lambda x: x[0])
    artifact_annots = mne.Annotations(
        onset=[row[0] for row in new_annots],
        duration=[row[1] for row in new_annots],
        description=[row[2] for row in new_annots],
        orig_time=raw.annotations.orig_time,
        ch_names=[row[3] for row in new_annots],
    )
    raw.set_annotations(raw.annotations + artifact_annots)
    LOGGER.info("Added %d artifact annotations.", len(new_annots))
    stats["artifacts_count"] = len(new_annots)
    return raw, stats


def run_base_record(
    subject_id: str,
    source_path: Path,
    bids_root: Path,
    config: dict | None = None,
    reports_root: Path | None = None,
    raw_lookup: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Process one BIDS run through the base stage and build the shared QC record."""
    result: dict[str, object] = {
        "success": False,
        "subject_id": subject_id,
        "qc_record": None,
        "error": "",
    }
    try:
        config = dict(config or {})
        comps = get_entities_from_fname(source_path)
        record_label = str(report_paths.build_bids_report_ids(source_path)["run_prefix"])
        LOGGER.info("Processing %s (Record: %s)...", source_path.name, record_label)

        preproc_root = bids.get_derivative_root(
            Path(bids_root).expanduser(), bids.DerivativeStage.PREPROC
        )
        if reports_root is None:
            reports_root = report_paths.default_reports_root(Path(bids_root).expanduser())
        else:
            reports_root = Path(reports_root).expanduser()

        raw = readers.read_bids_raw(
            bids_root=Path(bids_root),
            subject=comps["subject"],
            task=comps.get("task"),
            session=comps.get("session"),
            run=comps.get("run"),
        )

        cleaned_raw, _provenance = run_base_pipeline(
            raw, config=config, subject_id=subject_id, record_label=record_label
        )

        # Build output paths and persist results
        out_path = bids.get_stage_output_path(source_path, preproc_root, "base", create_dir=True)
        prov_path = out_path.with_name(out_path.name.replace("_eeg.fif", "_provenance.json"))
        with open(prov_path, "w") as f:
            json.dump(_provenance, f, cls=NumpyEncoder, indent=4)
        cleaned_raw.save(out_path, overwrite=True, verbose="ERROR")
        LOGGER.info("Base pipeline output saved: %s", out_path)

        result["qc_record"] = preproc_qc.build_preproc_qc_run_record(
            profile=preproc_qc.get_preproc_qc_profile("base"),
            reports_root=reports_root,
            current_raw=cleaned_raw,
            current_filepath=out_path,
            output_desc="base",
            raw_lookup=raw_lookup,
            pipeline_warnings=_provenance.get("pipeline_warnings", []),
        )
        result["success"] = True
        return result
    except Exception as exc:
        LOGGER.error("Failed processing %s: %s", source_path.name, exc, exc_info=True)
        result["error"] = str(exc)
        return result


def _build_pipeline_config(args: argparse.Namespace) -> dict[str, Any]:
    """The per-record config read by run_base_pipeline; each worker uses one core."""
    return {
        "n_jobs": 1,
        "processing": {
            "highpass_hz": args.highpass,
            "lowpass_hz": args.lowpass,
            "resample_hz": args.resample,
        },
        "line_noise": {
            "line_freq": args.line_freq,
            "adaptive": args.adaptive,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run EEG Preprocessing Pipeline on BIDS Dataset")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Recordings processed in parallel (default: 1)"
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=DEFAULT_LOWPASS_HZ,
        help=f"Lowpass filter cutoff Hz (default: {DEFAULT_LOWPASS_HZ})",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=DEFAULT_HIGHPASS_HZ,
        help=f"Highpass filter cutoff Hz (default: {DEFAULT_HIGHPASS_HZ})",
    )
    parser.add_argument(
        "--line_freq", type=float, default=60.0, help="Line noise frequency Hz (default: 60.0)"
    )
    parser.add_argument(
        "--resample", type=float, default=None, help="Resampling frequency Hz (optional)"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive line noise removal (for ZapLine)"
    )
    parser.add_argument(
        "--subjects", nargs="+", help="List of specific subject IDs (e.g., sub-001 sub-002)"
    )
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root)",
    )
    args = parser.parse_args()

    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_derivative_root(bids_root, bids.DerivativeStage.PREPROC)
    reports_root = (
        Path(args.reports_root).expanduser()
        if args.reports_root
        else report_paths.default_reports_root(bids_root)
    )
    reports_root.mkdir(parents=True, exist_ok=True)
    setup_logging(reports_root / "logs" / "preproc_base.log", "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    files_found = bids.discover_raw_eeg(bids_root)
    if not files_found:
        LOGGER.error("No .vhdr files found in BIDS directory.")
        sys.exit(1)

    bids.write_preproc_dataset_description(preproc_root)

    subjects = {get_entities_from_fname(path)["subject"] for path in files_found}
    if args.subjects:
        subjects &= {bids.bids_subject(s) for s in args.subjects}
    # Largest recording first: the long ones start early and short ones fill the tail (on
    # 2026-10-14 this matched the shortest possible makespan; discovery order took 26 % longer).
    files = sorted(
        (path for path in files_found if get_entities_from_fname(path)["subject"] in subjects),
        key=lambda path: path.with_suffix(".eeg").stat().st_size,
        reverse=True,
    )
    if not files:
        LOGGER.warning("No files matched the final selection criteria.")
        sys.exit(0)

    LOGGER.info(
        "Processing %d recordings of %d subjects on %d workers.",
        len(files),
        len(subjects),
        args.n_jobs,
    )
    raw_lookup = preproc_qc.load_raw_pre_base_lookup(reports_root)
    with tqdm_joblib(tqdm(total=len(files), desc="Processing recordings")):
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_base_record)(
                subject_id=get_entities_from_fname(f)["subject"],
                source_path=f,
                bids_root=bids_root,
                config=_build_pipeline_config(args),
                reports_root=reports_root,
                raw_lookup=raw_lookup,
            )
            for f in files
        )

    profile = preproc_qc.get_preproc_qc_profile("base")
    qc_run_records = [r["qc_record"] for r in results if r["success"]]
    by_subject: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in qc_run_records:
        by_subject[record["subject_session_key"]].append(record)
    for records in by_subject.values():
        preproc_qc.write_subject_preproc_qc_report(
            reports_root, records, profile=profile, output_desc="base"
        )

    failed = sorted({r["subject_id"] for r in results if not r["success"]})
    LOGGER.info(
        "Batch processing complete. Succeeded: %d recordings; failed subjects: %d %s",
        len(qc_run_records),
        len(failed),
        failed,
    )
    # Every recording failing points at a wiring or environment error, not at the data.
    if failed and not qc_run_records:
        LOGGER.error("All recordings failed the base stage — aborting (likely a systemic error).")
        sys.exit(1)

    LOGGER.info("Generating shared base QC dataset report...")
    preproc_qc.write_preproc_qc_aggregate_reports(
        reports_root, qc_run_records, profile=profile, output_desc="base"
    )


if __name__ == "__main__":
    main()

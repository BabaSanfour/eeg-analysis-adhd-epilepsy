"""EEG preprocessing pipeline base module.

This module provides the core pipeline for preprocessing EEG data using MNE-Python,
PyPREP, and AutoReject. The pipeline includes:

1.  Block Awareness: Uses embedded `BLOCK_*` annotations already present in BIDS.
2.  Resampling: Adjusts sampling rate.
3.  Filtering: Applies high-pass and low-pass filters.
4.  Line Noise Removal: Detects and removes power line noise (ZapLine).
5.  Global Bad Channel Detection: Identifies broken channels using RANSAC.
6.  Reference: Applies Common Average Reference (CAR).
7.  Artifact Annotation: Detects and annotates bad epochs and channels using AutoReject,
    grouping processing by experimental condition for robust threshold estimation.

The pipeline is designed to be robust to non-stationarity by handling conditions separately
during artifact rejection and using extensive logging and provenance tracking.
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
from pyprep.find_noisy_channels import NoisyChannels
from tqdm import tqdm

from eeg_adhd_epilepsy.io import bids, readers, report_paths
from eeg_adhd_epilepsy.preproc.epochs import build_block_events_by_condition
from eeg_adhd_epilepsy.preproc.utils import (
    NumpyEncoder,
    PreprocConfig,
    _compute_artifact_overlap,
    benchmark_step,
    inflate_bad_annotations,
)
from eeg_adhd_epilepsy.qc import preproc_qc
from eeg_adhd_epilepsy.utils import events
from eeg_adhd_epilepsy.utils.logs import setup_logging, tqdm_joblib

LOGGER = logging.getLogger("preproc_base")

DEFAULT_HIGHPASS_HZ = 0.1
DEFAULT_LOWPASS_HZ = 100.0
DEFAULT_ARTIFACT_SEGMENT_S = 1.0
DEFAULT_ARTIFACT_MIN_EPOCHS = 5


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
    epochs: mne.Epochs,
    segment_duration: float,
    chunk_minutes: float,
    min_epochs: int,
    condition_name: str,
) -> list[tuple[int, mne.Epochs]]:
    """Split a condition's epochs into AutoReject-sized chunks."""
    n_epochs_total = len(epochs)
    if n_epochs_total < min_epochs:
        LOGGER.warning(
            "Too few epochs (%d < %d) for AutoReject in condition %s. Skipping AR.",
            n_epochs_total,
            min_epochs,
            condition_name,
        )
        return []

    n_epochs_chunk_max = max(1, int((chunk_minutes * 60.0) / segment_duration))
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
        chunk_minutes,
    )
    chunks: list[tuple[int, mne.Epochs]] = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * n_epochs_chunk_max
        end_idx = min((chunk_idx + 1) * n_epochs_chunk_max, n_epochs_total)
        chunk = epochs[start_idx:end_idx]
        if len(chunk) >= min_epochs:
            chunks.append((chunk_idx, chunk))
    return chunks


def _save_autoreject_plot(
    reject_log,
    *,
    figures_dir: Path | None,
    record_label: str,
    condition_name: str,
    chunk_suffix: str,
) -> None:
    """Save the AutoReject reject-log plot when a figures dir is provided."""
    fig = reject_log.plot(orientation="horizontal", show=False)
    fig.set_size_inches(16, 10)
    if figures_dir is not None:
        clean_name = condition_name.lower().replace(" ", "_")
        fig.savefig(
            figures_dir / f"{record_label}_autoreject_{clean_name}{chunk_suffix}.png",
            dpi=150,
        )
    import matplotlib.pyplot as plt

    plt.close(fig)


def _reject_log_to_annotations(
    raw: mne.io.BaseRaw,
    epochs_chunk: mne.Epochs,
    reject_log,
    condition_name: str,
    segment_duration: float,
) -> tuple[list[tuple[float, float, str, tuple[str, ...]]], int, int]:
    """Convert an AutoReject reject log into raw-time annotations."""
    new_annots: list[tuple[float, float, str, tuple[str, ...]]] = []
    bad_epoch_count = 0
    bad_span_count = 0

    for ep_idx, is_bad_epoch in enumerate(reject_log.bad_epochs):
        if not is_bad_epoch:
            continue
        onset_s = _event_sample_to_onset(raw, int(epochs_chunk.events[ep_idx, 0]))
        new_annots.append((onset_s, segment_duration, f"BAD_epoch_{condition_name}", ()))
        bad_epoch_count += 1

    labels = np.asarray(reject_log.labels)
    if labels.ndim == 2 and labels.shape[0] == len(epochs_chunk):
        for ch_idx, ch_name in enumerate(epochs_chunk.ch_names):
            bad_idx = np.flatnonzero(labels[:, ch_idx] != 0)
            for first_idx, last_idx in _group_consecutive_indices(bad_idx):
                start_samp = int(epochs_chunk.events[first_idx, 0])
                end_samp = int(epochs_chunk.events[last_idx, 0])
                start_s = _event_sample_to_onset(raw, start_samp)
                end_s = _event_sample_to_onset(raw, end_samp) + segment_duration
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
    segment_duration: float,
    n_interpolate: list[int],
    random_seed: int,
    n_jobs: int,
    figures_dir: Path | None,
    record_label: str,
) -> tuple[list[tuple[float, float, str, tuple[str, ...]]], int, int] | None:
    """Fit AutoReject on one chunk and convert the reject log to annotations."""
    n_epochs_chunk = len(epochs_chunk)
    LOGGER.info(
        "Processing AutoReject for %s%s (%d epochs)",
        condition_name,
        chunk_suffix,
        n_epochs_chunk,
    )

    cv = min(10, n_epochs_chunk)
    if n_epochs_chunk < 10:
        LOGGER.info(
            "Adjusting AutoReject CV to %d folds for %s%s.",
            cv,
            condition_name,
            chunk_suffix,
        )

    ar = AutoReject(
        n_interpolate=np.asarray(n_interpolate, dtype=int),
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=False,
        cv=cv,
    )
    ar.fit(epochs_chunk)
    reject_log = ar.get_reject_log(epochs_chunk)
    _save_autoreject_plot(
        reject_log,
        figures_dir=figures_dir,
        record_label=record_label,
        condition_name=condition_name,
        chunk_suffix=chunk_suffix,
    )
    return _reject_log_to_annotations(
        raw,
        epochs_chunk,
        reject_log,
        condition_name=condition_name,
        segment_duration=segment_duration,
    )


def _compute_clean_stats(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Compute clean-data fraction and per-type bad fractions from raw annotations.

    Iterates over global (channel-less) BAD_ annotations and categorises them as
    autoreject-generated (`BAD_epoch_*`) or manual (`BAD_movement`, `BAD_yawn`,
    etc.), ignoring technical ones (`BAD_ACQ_SKIP`, `BAD_boundary`).

    Returns:
        Dict with keys ``clean_duration_s``, ``clean_fraction``,
        ``manual_bad_fraction``, ``autoreject_bad_fraction``.
    """
    total_samples = raw.n_times
    mask_all_bad = np.zeros(total_samples, dtype=bool)
    mask_manual = np.zeros(total_samples, dtype=bool)
    mask_autoreject = np.zeros(total_samples, dtype=bool)

    for annot in raw.annotations:
        desc = annot["description"]
        if not desc.startswith("BAD_"):
            continue
        if annot.get("ch_names", []):
            continue  # channel-specific span — skip for global stats

        start_idx = max(0, raw.time_as_index(annot["onset"])[0])
        end_idx = min(total_samples, raw.time_as_index(annot["onset"] + annot["duration"])[0])
        if end_idx <= start_idx:
            continue

        mask_all_bad[start_idx:end_idx] = True
        if desc.startswith("BAD_epoch_"):
            mask_autoreject[start_idx:end_idx] = True
        elif desc.startswith(("BAD_ACQ_SKIP", "BAD_boundary")):
            pass  # technical — don't count as manual or autoreject
        else:
            mask_manual[start_idx:end_idx] = True

    clean_samples = total_samples - int(mask_all_bad.sum())
    return {
        "clean_duration_s": float(clean_samples / raw.info["sfreq"]),
        "clean_fraction": float(clean_samples / total_samples) if total_samples > 0 else 0.0,
        "manual_bad_fraction": float(mask_manual.sum() / total_samples)
        if total_samples > 0
        else 0.0,
        "autoreject_bad_fraction": float(mask_autoreject.sum() / total_samples)
        if total_samples > 0
        else 0.0,
    }


def run_base_pipeline(
    raw: mne.io.BaseRaw,
    config: PreprocConfig,
    subject_id: str = "unknown",
    session_id: str | None = None,
    task: str | None = None,
    run_id: str | None = None,
    record_label: str | None = None,
    figures_dir: Path | None = None,
) -> tuple[mne.io.BaseRaw, dict]:
    """Run the shared preprocessing trunk and return cleaned raw + provenance.

    Pure transform — applies preprocessing steps and returns ``(cleaned_raw,
    provenance)``.  All I/O (path construction, file saving, report directory
    setup) is the responsibility of the caller (see :func:`run_base_record`).

    Args:
        raw: The raw MNE object to process.
        config: Configuration dictionary defining preprocessing parameters.
        subject_id: Identifier for the subject (used in logging/filenames).
        figures_dir: Optional directory for saving per-stage diagnostic figures.

    Returns:
        A tuple containing:
            - The processed MNE Raw object.
            - A dictionary containing processing provenance and statistics.
    """
    subject = subject_id
    subject_id = bids.bids_subject_label(subject)
    record_label = record_label or subject_id

    LOGGER.info("Starting base pipeline for %s", record_label)

    provenance: dict[str, Any] = {
        "subject_id": bids.bids_subject_label(subject_id),
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

    # 2. Resample
    target_sfreq = config.get("processing", {}).get("resample_hz", None)
    if target_sfreq:
        with benchmark_step("resample", provenance):
            raw.resample(target_sfreq, n_jobs=int(config.get("n_jobs", 1)))
        provenance["steps_completed"].append("resample")

    # 3. Filtering and Line Noise Removal
    # ----------------------------------
    with benchmark_step("filtering_and_denoising", provenance):
        hp_hz = config.get("processing", {}).get("highpass_hz", DEFAULT_HIGHPASS_HZ)
        lp_hz = config.get("processing", {}).get("lowpass_hz", DEFAULT_LOWPASS_HZ)
        line_noise_cfg = config.get("line_noise", {})
        line_freq = line_noise_cfg.get("line_freq", 60.0)

        # 3a. Bandpass Filter
        # Ensure lowpass is strictly less than Nyquist (sfreq/2)
        nyquist = raw.info["sfreq"] / 2.0
        h_f = min(lp_hz, nyquist - 0.1) if lp_hz else None
        n_jobs = int(config.get("n_jobs", 1))

        LOGGER.info("Applying Bandpass filter: %s-%s Hz (n_jobs=%d)", hp_hz, h_f, n_jobs)
        raw.filter(l_freq=hp_hz, h_freq=h_f, verbose="ERROR", n_jobs=n_jobs)
        provenance["steps_completed"].append("bandpass_filter")

        # 3b. Line Noise Removal
        adaptive = line_noise_cfg.get("adaptive", False)

        from mne_denoise.zapline import ZapLine

        LOGGER.info("Applying ZapLine (%s Hz, adaptive=%s)...", line_freq, adaptive)

        # ZapLine Class Usage
        zapline_obj = ZapLine(sfreq=raw.info["sfreq"], line_freq=line_freq, adaptive=adaptive)
        # fit_transform works for both adaptive and standard modes
        raw = zapline_obj.fit_transform(raw)

        provenance["steps_completed"].append("zapline")
        provenance["zapline_stats"] = {
            "method": "zapline",
            "line_freq": line_freq,
            "adaptive": adaptive,
            "n_removed": int(zapline_obj.n_removed_),
        }

    # 5. Global Bad Channel Detection
    with benchmark_step("detect_global_bads", provenance):
        bads, ransac_warning = detect_global_bads_ransac(raw, config, record_label=record_label)
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
        raw, artifact_stats = annotate_artifacts_blockwise(
            raw,
            config,
            figures_dir=figures_dir,
            record_label=record_label,
        )

    provenance["artifact_stats"] = artifact_stats
    provenance["block_stats"] = artifact_stats.get("by_block", [])
    provenance["steps_completed"].append("block_artifact_annotation")

    # Clean Data Duration & Detailed Artifact Stats
    provenance["integrity_stats"] = _compute_clean_stats(raw)
    provenance["artifact_stats"]["artifacts_count"] = len(raw.annotations)

    LOGGER.info("Base pipeline completed for %s.", record_label)
    return raw, provenance


def detect_global_bads_ransac(
    raw: mne.io.BaseRaw, config: PreprocConfig, record_label: str = "run"
) -> tuple[list[str], str | None]:
    """Detect global bad EEG channels with RANSAC, biased toward rest blocks.

    Uses a subset of data (rest blocks) to speed up RANSAC and focus on
    intrinsic channel quality rather than task-related artifacts.

    Args:
        raw: Input raw data.
        config: Preprocessing configuration.
        record_label: Label for logging.

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

    nc = NoisyChannels(raw_for_ransac, random_state=42)
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
    raw: mne.io.BaseRaw,
    config: PreprocConfig,
    figures_dir: Path | None = None,
    record_label: str = "record",
    n_interpolate: list[int] = None,
) -> tuple[mne.io.BaseRaw, dict]:
    """Run condition-wise AutoReject and add non-destructive BAD annotations.

    Groups disjoint blocks by condition (e.g., all "EO_baseline" blocks) and
    runs AutoReject on them as a single set of epochs. This ensures consistent
    thresholds across the condition.

    Args:
        raw: The raw data object.
        config: Preprocessing configuration.
        figures_dir: Directory to save AutoReject visualizations (optional).
        record_label: Run-scoped label used for figure filenames.

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

    # Configuration
    artifacts_cfg = config.get("artifacts", {})
    bad_channels_cfg = config.get("bad_channels", {})
    seg_len = float(
        artifacts_cfg.get(
            "segment_length",
            bad_channels_cfg.get("segment_length", DEFAULT_ARTIFACT_SEGMENT_S),
        )
    )
    seg_len = seg_len if seg_len > 0 else DEFAULT_ARTIFACT_SEGMENT_S

    min_epochs = int(artifacts_cfg.get("min_epochs", DEFAULT_ARTIFACT_MIN_EPOCHS))
    min_epochs = max(1, min_epochs)

    if n_interpolate is None:
        n_interpolate = np.array([0], dtype=int)

    n_jobs = int(config.get("n_jobs", 1))
    random_seed = int(config.get("random_seed", 42))
    epoch_tmax = max(seg_len - (1.0 / raw.info["sfreq"]), 0.0)
    chunk_minutes = float(artifacts_cfg.get("ar_max_chunk_minutes", 30.0))
    chunk_minutes = max(1.0, chunk_minutes)
    condition_inputs = _prepare_condition_epoch_inputs(raw, segment_duration=seg_len)

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
        epoch_chunks = _iter_autoreject_chunks(
            epochs,
            segment_duration=seg_len,
            chunk_minutes=chunk_minutes,
            min_epochs=min_epochs,
            condition_name=condition_name,
        )
        if not epoch_chunks:
            continue

        for chunk_idx, epochs_chunk in epoch_chunks:
            chunk_suffix = f"_chunk{chunk_idx + 1}" if len(epoch_chunks) > 1 else ""
            result = _run_autoreject_chunk(
                raw,
                epochs_chunk,
                condition_name=condition_name,
                chunk_suffix=chunk_suffix,
                segment_duration=seg_len,
                n_interpolate=n_interpolate,
                random_seed=random_seed,
                n_jobs=n_jobs,
                figures_dir=figures_dir,
                record_label=record_label,
            )
            if result is None:
                continue
            chunk_annots, chunk_bad_epochs, chunk_bad_spans = result
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
        "skipped": False,
        "subject_id": subject_id,
        "qc_record": None,
        "error": "",
    }
    try:
        config = dict(config or {})
        subject = subject_id
        ids = report_paths.build_bids_report_ids(source_path)
        comps = bids.parse_bids_components(source_path)
        session_id = comps.get("session")
        task = comps.get("task")
        run_id = comps.get("run")
        record_label = str(ids["run_prefix"])
        LOGGER.info("Processing %s (Record: %s)...", source_path.name, record_label)

        # Resolve roots and setup report/figures directories
        preproc_root = bids.get_preproc_root(Path(bids_root).expanduser())
        if reports_root is None:
            reports_root = report_paths.default_reports_root(Path(bids_root).expanduser())
        else:
            reports_root = Path(reports_root).expanduser()

        stage_name = preproc_qc.get_preproc_qc_stage_name("base", "base")
        report_dir = report_paths.subject_report_dir(
            reports_root=reports_root,
            subject=subject,
            session=session_id or "01",
            stage=stage_name,
            create=True,
        )
        subject_report_path = report_dir / f"{record_label}_{stage_name.value}_report.html"
        figures_dir = subject_report_path.parent / "figures" / record_label
        figures_dir.mkdir(parents=True, exist_ok=True)

        source_entities = bids.parse_bids_components(source_path)
        raw = readers.read_bids_raw(
            bids_root=Path(bids_root),
            subject=source_entities["subject"],
            task=source_entities.get("task", task),
            session=source_entities.get("session", session_id),
            run=source_entities.get("run", run_id),
        )

        cleaned_raw, _provenance = run_base_pipeline(
            raw,
            config=config,
            subject=subject,
            session_id=session_id,
            task=task,
            run_id=run_id,
            record_label=record_label,
            figures_dir=figures_dir,
        )

        # Build output paths and persist results
        out_path = bids.get_stage_output_path(
            subject=subject,
            preproc_root=preproc_root,
            desc="base",
            session=session_id,
            task=task,
            run=run_id,
            create_dir=True,
        )
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


def main():
    parser = argparse.ArgumentParser(description="Run EEG Preprocessing Pipeline on BIDS Dataset")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)"
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage outputs instead of skipping them",
    )
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Custom root directory for reports (defaults to sibling of bids_root)",
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_preproc_root(bids_root)
    reports_root = (
        Path(args.reports_root).expanduser()
        if args.reports_root
        else report_paths.default_reports_root(bids_root)
    )
    reports_root.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = reports_root / "logs" / "preproc_base.log"
    setup_logging(log_file, "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    files_found = bids.discover_bids_files(bids_root, suffix="eeg", extension=".vhdr")

    if not files_found:
        LOGGER.error("No .vhdr files found in BIDS directory.")
        sys.exit(1)

    subjects_found = sorted({bids.parse_bids_components(path)["subject"] for path in files_found})
    LOGGER.info(
        "Found %d EEG runs across %d subjects in BIDS directory.",
        len(files_found),
        len(subjects_found),
    )

    if args.subjects:
        normalized_subjects = [bids.study_id_to_bids_subject(s) for s in args.subjects]
        subjects_to_process = set(normalized_subjects)
        LOGGER.info("Selected specific subjects: %s", normalized_subjects)
    else:
        subjects_to_process = set(subjects_found)
        LOGGER.info("Processing all %d subjects.", len(subjects_found))

    profile = preproc_qc.get_preproc_qc_profile("base")
    qc_run_records: list[dict[str, object]] = []
    qc_subject_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    subject_status: dict[str, bool] = {}
    subject_skipped: dict[str, bool] = {}  # Tracks if ALL runs for a subject were skipped

    def consume_result(result: dict[str, object], *, subject_id: str) -> None:
        ok = bool(result.get("success"))
        skipped = bool(result.get("skipped", False))

        subject_status[subject_id] = subject_status.get(subject_id, True) and ok
        # Subject is considered skipped only if ALL its runs consumed so far are skipped
        subject_skipped[subject_id] = subject_skipped.get(subject_id, True) and skipped
        if not ok:
            return
        record = result.get("qc_record")
        if not isinstance(record, dict):
            return
        qc_run_records.append(record)
        qc_subject_groups[record["subject_session_key"]].append(record)
        preproc_qc.write_subject_preproc_qc_report(
            reports_root,
            qc_subject_groups[record["subject_session_key"]],
            profile=profile,
            output_desc="base",
        )

    existing_results: list[dict[str, object]] = []
    raw_lookup = preproc_qc.load_raw_pre_base_lookup(reports_root)

    if not args.overwrite:
        LOGGER.info("Checking for existing base outputs to skip...")

        existing_runs = 0
        for fpath in files_found:
            sid = bids.parse_bids_components(fpath)["subject"]
            if sid not in subjects_to_process:
                continue
            comps = bids.parse_bids_components(fpath)
            out_path = bids.get_stage_output_path(
                subject=sid,
                preproc_root=preproc_root,
                desc="base",
                session=comps.get("session"),
                task=comps.get("task"),
                run=comps.get("run"),
            )
            if not out_path.exists():
                continue
            existing_runs += 1
            try:
                existing_results.append(
                    {
                        "success": True,
                        "skipped": True,
                        "subject_id": sid,
                        "qc_record": preproc_qc.collect_existing_preproc_qc_record(
                            profile=preproc_qc.get_preproc_qc_profile("base"),
                            reports_root=reports_root,
                            filepath=out_path,
                            output_desc="base",
                            raw_lookup=raw_lookup,
                        ),
                        "error": "",
                    }
                )
                consume_result(existing_results[-1], subject_id=sid)
            except Exception as exc:
                LOGGER.error(
                    "Failed rebuilding existing base QC record for %s (%s): %s",
                    sid,
                    fpath.name,
                    exc,
                    exc_info=True,
                )
                existing_results.append(
                    {
                        "success": False,
                        "skipped": False,
                        "subject_id": sid,
                        "qc_record": None,
                        "error": str(exc),
                    }
                )
                consume_result(existing_results[-1], subject_id=sid)
        if existing_runs:
            LOGGER.info("Skipping %d existing base outputs.", existing_runs)
        else:
            LOGGER.info("No existing runs found to skip.")

    files_to_process = []
    existing_run_keys = set()
    for result in existing_results:
        record = result.get("qc_record")
        if isinstance(record, dict):
            existing_run_keys.add(record.get("run_key"))
    for fpath in files_found:
        sid = bids.parse_bids_components(fpath)["subject"]
        if sid not in subjects_to_process:
            continue
        ids = report_paths.build_bids_report_ids(fpath)
        if ids["run_key"] in existing_run_keys:
            continue
        files_to_process.append(fpath)

    if not files_to_process and not existing_results:
        LOGGER.warning("No files matched the final selection criteria.")
        sys.exit(0)

    LOGGER.info("Scanning file durations to optimize parallelization...")
    short_files = []
    long_files = []

    for f in tqdm(files_to_process, desc="Checking Durations"):
        try:
            raw_info = mne.io.read_raw_brainvision(f, preload=False, verbose="ERROR")
            duration_min = (raw_info.n_times / raw_info.info["sfreq"]) / 60.0

            if duration_min >= 30.0:
                long_files.append(f)
            else:
                short_files.append(f)
        except Exception as e:
            LOGGER.warning(
                "Could not read duration for %s, treating as long file. Error: %s", f.name, e
            )
            long_files.append(f)

    LOGGER.info(
        "Optimization Strategy: %d short files (<30m), %d long files (>=30m)",
        len(short_files),
        len(long_files),
    )

    # ---------------------------------------------------------
    # Phase 1: Process Short Files (Parallel Subjects)
    # ---------------------------------------------------------
    if short_files:
        LOGGER.info(
            "--- Phase 1: Processing %d short files in parallel (n_jobs=%d) ---",
            len(short_files),
            args.n_jobs,
        )

        pipeline_config_short = {
            "n_jobs": 1,  # 1 core per subject internally
            "bids_root": str(bids_root),
            "preproc_root": str(preproc_root),
            "reports_root": str(reports_root),
            "pre_base_raw_lookup": raw_lookup,
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

        with tqdm_joblib(tqdm(total=len(short_files), desc="Processing Short Files")):
            results_short = Parallel(n_jobs=args.n_jobs)(
                delayed(run_base_record)(
                    subject_id=bids.parse_bids_components(f)["subject"],
                    source_path=f,
                    bids_root=bids_root,
                    config=pipeline_config_short,
                    reports_root=reports_root,
                    raw_lookup=raw_lookup,
                )
                for f in short_files
            )
        for fpath, result in zip(short_files, results_short):
            consume_result(result, subject_id=bids.parse_bids_components(fpath)["subject"])

    # ---------------------------------------------------------
    # Phase 2: Process Long Files (Sequential Subjects, Parallel Internal)
    # ---------------------------------------------------------
    if long_files:
        LOGGER.info(
            "--- Phase 2: Processing %d long files sequentially (internal n_jobs=%d) ---",
            len(long_files),
            args.n_jobs,
        )

        pipeline_config_long = {
            "n_jobs": args.n_jobs,  # Full power per subject
            "bids_root": str(bids_root),
            "preproc_root": str(preproc_root),
            "reports_root": str(reports_root),
            "pre_base_raw_lookup": raw_lookup,
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

        # Simple loop, no Parallel (or Parallel(n_jobs=1))
        # We use a loop to ensure strictly sequential execution to save memory
        for f in tqdm(long_files, desc="Processing Long Files"):
            res = run_base_record(
                subject_id=bids.parse_bids_components(f)["subject"],
                source_path=f,
                bids_root=bids_root,
                config=pipeline_config_long,
                reports_root=reports_root,
                raw_lookup=raw_lookup,
            )
            consume_result(res, subject_id=bids.parse_bids_components(f)["subject"])

    success_ids = sorted(
        [sid for sid, ok in subject_status.items() if ok and not subject_skipped.get(sid, False)]
    )
    skipped_ids = sorted(
        [sid for sid, ok in subject_status.items() if ok and subject_skipped.get(sid, False)]
    )
    failed_ids = sorted([sid for sid, ok in subject_status.items() if not ok])

    success_count = len(success_ids)
    skipped_count = len(skipped_ids)
    fail_count = len(failed_ids)

    LOGGER.info(
        "Batch processing complete. Success: %d, Skipped: %d, Failed: %d",
        success_count,
        skipped_count,
        fail_count,
    )
    if success_ids:
        LOGGER.info("Succeeded subjects: %s", success_ids)
    if skipped_ids:
        LOGGER.info("Skipped (already processed) subjects: %s", skipped_ids)
    if failed_ids:
        LOGGER.info("Failed subjects: %s", failed_ids)

    LOGGER.info("Generating shared base QC dataset report...")
    preproc_qc.write_preproc_qc_aggregate_reports(
        reports_root,
        qc_run_records,
        profile=profile,
        output_desc="base",
    )


if __name__ == "__main__":
    main()

"""Utility classes and functions for EEG preprocessing."""
from __future__ import annotations
import time
import logging
from contextlib import contextmanager
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from typing import Any, Dict, List, Literal, TypedDict, Optional, Sequence, Tuple, Union


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AdaptiveParams(TypedDict, total=False):
    fmin: float
    fmax: float
    process_harmonics: bool
    max_harmonics: Optional[int]
    hybrid_fallback: bool
    min_chunk_len: float
    n_remove_params: Dict[str, Union[float, int]]
    qa_params: Dict[str, float]

class LineNoiseConfig(TypedDict):
    method: Literal["zapline", "spectrum_fit", "notch"]
    line_freq: float
    zapline_n_remove: int
    spectrum_fit_bandwidth: float
    notch_width: float
    adaptive: bool
    adaptive_params: AdaptiveParams

class SegmentRejectionConfig(TypedDict):
    enabled: bool
    bad_segments_annotation: Union[str, List[str]]
    segment_buffer: float
    # "reject" (drop), "correct" (keep), "interpolate" (reconstruct), "ignore"
    segment_handling: Literal["reject", "correct", "interpolate", "ignore"]

class ArtifactConfig(TypedDict):
    ica_enable: bool
    ica_method: str
    ica_n_components: Optional[Union[int, float]]
    ica_exclude_annotated: bool
    dss_enable: bool
    dss_supervised: bool
    dss_supervised: bool
    autoreject_enable: bool

class SeizureConfig(TypedDict):
    mode: Literal["preserve", "suppress"]
    exclusion_margin: float

class ProcessingConfig(TypedDict):
    level: Literal["easy", "severe"]
    mode: Literal["continuous", "epochs"]
    highpass_hz: float
    lowpass_hz: float
    resample_hz: Optional[float]
    epoch_type: Literal["fixed", "events", "segments"]
    epoch_length: float      # For fixed
    epoch_tmin: float        # For events/segments
    epoch_tmax: float        # For events/segments
    epoch_event_id: Optional[Dict[str, int]] # Optional mapping
    segments_file: Optional[str] # Path to segmentation CSV

class BadChannelsConfig(TypedDict):
    use_ransac: bool
    segment_based: bool  # New: Check bads in chunks
    segment_length: float # Chunk length in seconds
    max_bad_fraction: float # If channel bad in > X fraction of chunks, mark global bad

class ReportingConfig(TypedDict):
    html_report: bool
    report_title: str

class PreprocConfig(TypedDict):
    line_noise: LineNoiseConfig
    segment_rejection: SegmentRejectionConfig
    artifacts: ArtifactConfig
    seizure: SeizureConfig
    processing: ProcessingConfig
    bad_channels: BadChannelsConfig
    reporting: ReportingConfig
    n_jobs: int
    random_seed: int
    verbose: str
    output_dir: str


@contextmanager
def benchmark_step(name: str, provenance: Dict):
    """Context manager to measure wall-clock time and log it."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        provenance.setdefault("benchmarks", {}).setdefault("timing", {})[name] = duration
        logging.getLogger(__name__).info("Step '%s' finished in %.2f sec", name, duration)


DEFAULT_ARTIFACT_N_INTERPOLATE = (1, 2, 4)
DEFAULT_REST_KEYWORDS = (
    "raw_baseline",
    "eo_baseline",
    "ec_baseline",
    "rest",
    "baseline",
)


@dataclass
class BlockWindow:
    """Represents a continuous block of time defined by an annotation."""

    onset: float
    duration: float
    description: str

    @property
    def stop(self) -> float:
        """Return end time of the block in seconds."""
        return self.onset + self.duration

    @property
    def name(self) -> str:
        """Return block name stripped of the 'BLOCK_' prefix."""
        if self.description.startswith("BLOCK_"):
            return self.description[6:]
        return self.description


def _resolve_segments_csv(
    raw: mne.io.BaseRaw, segments_file: Optional[str]
) -> Optional[Path]:
    """Resolve the path to the segments CSV file."""
    if segments_file:
        segments_path = Path(segments_file).expanduser()
        if segments_path.exists():
            return segments_path
        if raw.filenames and raw.filenames[0]:
            candidate = Path(raw.filenames[0]).parent / segments_path
            if candidate.exists():
                return candidate
        return segments_path  # Return even if not exists, let caller fail

    if not raw.filenames or not raw.filenames[0]:
        return None

    raw_path = Path(raw.filenames[0])
    stem = raw_path.stem
    # Handle common BIDS-like suffixes
    for suffix in ("_eeg", "_meg", "_ieeg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return raw_path.parent / f"{stem}_segments.csv"


def _collect_block_windows(raw: mne.io.BaseRaw) -> List[BlockWindow]:
    """Parse annotations to collect all BLOCK_* segments."""
    if raw.n_times == 0:
        return []

    max_t = float(raw.times[-1])
    windows: List[BlockWindow] = []
    for annot in raw.annotations:
        desc = str(annot["description"])
        if not desc.startswith("BLOCK_"):
            continue

        onset = float(annot["onset"])
        duration = float(annot["duration"])
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            continue

        start = max(0.0, onset)
        stop = min(max_t, onset + duration)
        if stop <= start:
            continue

        windows.append(
            BlockWindow(onset=start, duration=stop - start, description=desc)
        )

    windows.sort(key=lambda block: block.onset)
    return windows


def _get_rest_windows(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """Identify windows belonging to resting state (baseline) blocks."""
    rest_windows: List[Tuple[float, float]] = []
    for block in _collect_block_windows(raw):
        block_name = block.name.lower()
        if any(key in block_name for key in DEFAULT_REST_KEYWORDS):
            rest_windows.append((block.onset, block.stop))
    return rest_windows


def _compute_artifact_overlap(
    raw: mne.io.BaseRaw,
    new_annots: List[Tuple[float, float, str, Tuple[str, ...]]],
) -> float:
    """Calculate percentage of existing manual BAD segments covered by detected artifacts.

    Assumes any pre-existing annotation starting with 'BAD_' is a ground truth
    manual annotation. Treats zero-duration Point annotations as 5.0s segments.
    """
    existing_bads = []
    for annot in raw.annotations:
        desc = str(annot["description"])
        if desc.startswith("BAD_"):
            start = float(annot["onset"])
            duration = float(annot.get("duration", 0))
            if duration <= 0:
                duration = 5.0
            existing_bads.append((start, start + duration))

    if not existing_bads:
        return 0.0

    if not new_annots:
        return 0.0

    detected_intervals = []
    for onset, dur, _, _ in new_annots:
        detected_intervals.append((onset, onset + dur))

    max_time = raw.times[-1]
    res = 0.1  # 100ms resolution
    n_points = int(max_time / res) + 1

    if n_points <= 0:
        return 0.0

    mask_existing = np.zeros(n_points, dtype=bool)
    mask_detected = np.zeros(n_points, dtype=bool)

    for s, e in existing_bads:
        idx_s = int(s / res)
        idx_e = int(e / res) + 1
        idx_s = max(0, min(idx_s, n_points))
        idx_e = max(0, min(idx_e, n_points))
        if idx_e > idx_s:
            mask_existing[idx_s:idx_e] = True

    for s, e in detected_intervals:
        idx_s = int(s / res)
        idx_e = int(e / res) + 1
        idx_s = max(0, min(idx_s, n_points))
        idx_e = max(0, min(idx_e, n_points))
        if idx_e > idx_s:
            mask_detected[idx_s:idx_e] = True

    n_existing = np.sum(mask_existing)
    if n_existing == 0:
        return 0.0

    n_overlap = np.sum(mask_existing & mask_detected)
    return (n_overlap / n_existing) * 100.0


def _save_outputs(
    raw: mne.io.BaseRaw, provenance: Dict[str, Any], output_dir: Path, subject_id: str
) -> None:
    """Save processed raw data and provenance to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename: sub-X_desc-base_eeg.fif
    fname = f"{subject_id}_desc-base_eeg.fif"
    out_file = output_dir / fname
    prov_fname = output_dir / f"{subject_id}_provenance.json"

    logging.getLogger(__name__).info(f"Saving preprocessed raw to {out_file}")
    raw.save(out_file, overwrite=True, verbose="ERROR")
    with prov_fname.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, cls=NumpyEncoder, indent=2)

    logging.getLogger(__name__).info(f"Saved base output: {out_file}")
    logging.getLogger(__name__).info(f"Saved provenance: {prov_fname}")


def _sanitize_n_interpolate(raw_value: Any) -> List[int]:
    """Ensure n_interpolate is a valid list of integers."""
    if isinstance(raw_value, (list, tuple)):
        clean = set()
        for value in raw_value:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                clean.add(parsed)
        clean = sorted(clean)
        if clean:
            return list(clean)
    return list(DEFAULT_ARTIFACT_N_INTERPOLATE)


def _group_consecutive_indices(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Group consecutive integers into (start, end) tuples."""
    if len(indices) == 0:
        return []

    groups: List[Tuple[int, int]] = []
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
    """Convert a sample index (considering file offset) to onset time in seconds."""
    return max(0.0, (event_sample - raw.first_samp) / raw.info["sfreq"])

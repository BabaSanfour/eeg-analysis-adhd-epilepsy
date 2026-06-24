"""Type definitions and JSON serialization helpers for the preprocessing pipeline."""

from __future__ import annotations

import json
from typing import Literal

import numpy as np

try:
    from typing import TypedDict
except ImportError:  # Python 3.7
    from typing_extensions import TypedDict


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AdaptiveParams(TypedDict, total=False):
    fmin: float
    fmax: float
    process_harmonics: bool
    max_harmonics: int | None
    hybrid_fallback: bool
    min_chunk_len: float
    n_remove_params: dict[str, float | int]
    qa_params: dict[str, float]


class LineNoiseConfig(TypedDict):
    line_freq: float
    adaptive: bool
    adaptive_params: AdaptiveParams


class SegmentRejectionConfig(TypedDict):
    enabled: bool
    bad_segments_annotation: str | list[str]
    segment_buffer: float
    segment_handling: Literal["reject", "correct", "interpolate", "ignore"]


class ArtifactConfig(TypedDict):
    ica_enable: bool
    ica_method: str
    ica_n_components: int | float | None
    ica_exclude_annotated: bool
    dss_enable: bool
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
    resample_hz: float | None
    epoch_type: Literal["fixed", "events"]
    epoch_length: float
    epoch_tmin: float
    epoch_tmax: float
    epoch_event_id: dict[str, int] | None


class BadChannelsConfig(TypedDict):
    use_ransac: bool
    segment_based: bool
    segment_length: float
    max_bad_fraction: float


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

"""Preprocessing utility helpers.

Re-exports all public names so existing ``from .utils import ...`` calls in
the pipeline modules continue to work without changes.  New code is encouraged
to import from the specific submodule directly, e.g.::

    from eeg_adhd_epilepsy.preproc.utils.subjects import select_subjects
"""

from eeg_adhd_epilepsy.io.bids import load_stage_artifacts

from . import thresholds  # expose sub-module for callers that do `from .utils import thresholds`
from .artifacts import _compute_artifact_overlap, inflate_bad_annotations
from .subjects import _normalize_subject_list, select_subjects
from .timing import benchmark_step
from .types import (
    AdaptiveParams,
    ArtifactConfig,
    BadChannelsConfig,
    LineNoiseConfig,
    NumpyEncoder,
    PreprocConfig,
    ProcessingConfig,
    ReportingConfig,
    SegmentRejectionConfig,
    SeizureConfig,
)

__all__ = [
    # types
    "NumpyEncoder",
    "AdaptiveParams",
    "LineNoiseConfig",
    "SegmentRejectionConfig",
    "ArtifactConfig",
    "SeizureConfig",
    "ProcessingConfig",
    "BadChannelsConfig",
    "ReportingConfig",
    "PreprocConfig",
    # io
    "load_stage_artifacts",
    # artifacts
    "_compute_artifact_overlap",
    "inflate_bad_annotations",
    # subjects
    "_normalize_subject_list",
    "select_subjects",
    # timing
    "benchmark_step",
    # sub-modules
    "thresholds",
]

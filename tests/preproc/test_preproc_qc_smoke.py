"""Smoke tests for the preprocessing / raw-QC visualization and metric paths.

These exercise the ``viz.preproc_qc`` figure builders and the
``signal_quality`` metric entry points on a small synthetic recording. They
are deliberately shallow: the goal is to catch import-time and call-time
breakage (e.g. a missing ``import mne``) in code that the analysis-focused
suite never touches, not to assert pixel-level figure content.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pytest

import eeg_adhd_epilepsy.signal_quality.metrics as signal_quality
import eeg_adhd_epilepsy.viz.preproc_qc as viz_preproc_qc


@pytest.mark.integration
def test_save_artifact_comparison_runs(synthetic_raw, tmp_path):
    """Regression: this path calls ``mne.viz.plot_topomap`` at runtime and used
    to raise ``NameError`` because ``mne`` was not imported in the module."""
    raw_before = synthetic_raw(seed=0)
    raw_after = synthetic_raw(seed=1)

    out = viz_preproc_qc.save_artifact_comparison(
        raw_before,
        raw_after,
        fig_dir=tmp_path,
        subject_id="sub-01",
        artifact_type="eog",
        window=5.0,
        search_start=5.0,
    )

    assert Path(out).exists()


@pytest.mark.integration
def test_plot_compare_variance_topomaps_runs(synthetic_raw, tmp_path):
    raw_orig = synthetic_raw(seed=0)
    raw_dss = synthetic_raw(seed=1)
    raw_ica = synthetic_raw(seed=2)

    out = viz_preproc_qc.plot_compare_variance_topomaps(
        raw_orig, raw_dss, raw_ica, subject_id="sub-01", fig_dir=tmp_path
    )

    assert Path(out).exists()


@pytest.mark.integration
def test_plot_channel_variance_comparison_returns_figure(synthetic_raw):
    fig = viz_preproc_qc.plot_channel_variance_comparison(
        synthetic_raw(seed=0), synthetic_raw(seed=1), subject_id="sub-01"
    )

    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.integration
def test_save_eeg_snapshot_runs(synthetic_raw, tmp_path):
    out = viz_preproc_qc.save_eeg_snapshot(
        synthetic_raw(seed=0),
        fig_dir=tmp_path,
        subject_id="sub-01",
        label="pre",
        start=5.0,
        duration=10.0,
    )

    assert Path(out).exists()


@pytest.mark.integration
def test_compute_signal_qc_metrics_on_synthetic_raw(synthetic_raw):
    metrics = signal_quality.compute_signal_qc_metrics(synthetic_raw(seed=0))

    assert isinstance(metrics, dict)
    assert metrics  # non-empty for a real recording
    # An alpha bump was injected, so a finite alpha peak should be detectable.
    assert "segment_alpha_peak_hz" in metrics or "alpha_peak_hz" in metrics


def test_compute_signal_qc_metrics_handles_none():
    assert signal_quality.compute_signal_qc_metrics(None) == {}

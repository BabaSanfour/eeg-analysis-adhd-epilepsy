"""Stage 03 line-noise removal: ZapLine with a component cap, then interpolation if needed."""

from __future__ import annotations

import numpy as np

from eeg_adhd_epilepsy.preproc.base import (
    LINE_NOISE_RATIO_MAX,
    ZAPLINE_MAX_COMPONENTS,
    _line_noise_ratio,
    _remove_line_noise,
)


def test_line_from_many_sources_is_capped_then_interpolated(synthetic_raw):
    """Line noise from as many sources as channels; a clean recording is left alone."""
    raw = synthetic_raw(seed=0, sfreq=200.0)
    rng = np.random.default_rng(1)
    t = raw.times
    # One 60 Hz source per channel. Each phase drifts on its own: pure tones at one
    # frequency span only two dimensions, whatever their phases.
    drift = np.cumsum(rng.standard_normal((len(raw.ch_names), len(t))), axis=1) * 1e-2
    noisy = raw.copy()
    noisy._data += 6e-5 * np.sin(2 * np.pi * 60.0 * t + drift)  # ~25 dB over the floor

    cleaned, stats = _remove_line_noise(noisy, 60.0, adaptive=False)
    assert stats["n_selected_auto"] > ZAPLINE_MAX_COMPONENTS
    assert stats["n_removed"] == ZAPLINE_MAX_COMPONENTS
    assert stats["spectrum_interpolation"]
    assert _line_noise_ratio(cleaned, 60.0) < LINE_NOISE_RATIO_MAX

    _, stats = _remove_line_noise(raw, 60.0, adaptive=False)
    assert not stats["spectrum_interpolation"]

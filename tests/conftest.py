"""Shared test configuration, fixtures, and automatic marker assignment.

Layout
------
Tests mirror the package layout::

    tests/analysis/   -> eeg_adhd_epilepsy.analysis.*  (decoding, dim-reduction, descriptors)
    tests/io/         -> eeg_adhd_epilepsy.io.*        (metadata builder, re-epoching)
    tests/reports/    -> eeg_adhd_epilepsy.reports.*   (decoding report composition)
    tests/preproc/    -> eeg_adhd_epilepsy.preproc.* / viz.preproc_qc / qc.*

Markers
-------
Tests are tagged automatically by :func:`pytest_collection_modifyitems` so the
suite can be sliced without hand-annotating every test:

* ``unit``         - fast, isolated (the default).
* ``integration``  - multi-component, exercises a real pipeline path.
* ``cli``          - drives a console-script ``main()`` entry point.
* ``slow``         - long-running (CLI / end-to-end / smoke).

Run subsets with e.g. ``pytest -m unit`` or ``pytest -m "not slow"``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

# Standard 10-20 channels that resolve to positions in MNE's ``standard_1020``
# montage, so topomap-based QC figures have sensor coordinates to plot.
_SYNTHETIC_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, isolated unit test")
    config.addinivalue_line("markers", "integration: multi-component / pipeline test")
    config.addinivalue_line("markers", "cli: drives a console-script entry point")
    config.addinivalue_line("markers", "slow: long-running test")


def pytest_collection_modifyitems(config, items):
    """Auto-tag collected tests by their name and location."""
    for item in items:
        name = item.name
        path = str(item.fspath)
        is_cli = "_cli" in name or "command_line" in name
        is_e2e = "end_to_end" in name or "_smoke" in path
        if is_cli:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.cli)
            item.add_marker(pytest.mark.slow)
        elif is_e2e:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def synthetic_raw():
    """Factory for a montaged synthetic EEG :class:`mne.io.RawArray`.

    Returns a callable ``make(seed=0, sfreq=250.0, duration=60.0)`` so a test
    can build several independent recordings (e.g. before/after correction)
    with controllable content. The recording carries the ``standard_1020``
    montage, so topomap-based QC figures have channel positions.
    """
    import mne

    def make(seed: int = 0, sfreq: float = 250.0, duration: float = 60.0):
        rng = np.random.default_rng(seed)
        n_samples = int(sfreq * duration)
        # Microvolt-scale noise + an alpha bump, expressed in volts for MNE.
        data = rng.standard_normal((len(_SYNTHETIC_CHANNELS), n_samples)) * 20e-6
        t = np.arange(n_samples) / sfreq
        data += 10e-6 * np.sin(2 * np.pi * 10.0 * t)
        info = mne.create_info(list(_SYNTHETIC_CHANNELS), sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_montage("standard_1020", verbose="ERROR")
        return raw

    return make

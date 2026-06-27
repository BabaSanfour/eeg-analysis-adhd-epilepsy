from pathlib import Path

import pytest

from eeg_adhd_epilepsy.analysis.dataset import build_container


def test_build_container_rejects_bandpass_without_reepoch():
    # Filtering happens on the continuous recording inside the re-epoch path;
    # the saved epoch derivatives are already windowed, so bandpass there is
    # ill-posed and must raise rather than silently no-op.
    with pytest.raises(ValueError, match="bandpass is only supported"):
        build_container(
            bids_root=Path("/nonexistent"),
            use_derivatives=True,
            window_source="derivative",
            bandpass=(0.5, 40.0),
        )

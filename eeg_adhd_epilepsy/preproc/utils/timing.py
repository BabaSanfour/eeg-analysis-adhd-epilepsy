"""Pipeline step timing context manager."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict

LOGGER = logging.getLogger(__name__)


@contextmanager
def benchmark_step(name: str, provenance: Dict):
    """Measure wall-clock time for a named step and record it in provenance.

    Usage::

        with benchmark_step("bandpass_filter", provenance):
            raw.filter(...)

    The elapsed seconds are stored under
    ``provenance["benchmarks"]["timing"][name]``.
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        provenance.setdefault("benchmarks", {}).setdefault("timing", {})[name] = elapsed
        LOGGER.info("Step '%s' finished in %.2f sec", name, elapsed)

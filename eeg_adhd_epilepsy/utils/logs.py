"""Logging and progress bar utilities."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from joblib import parallel
from tqdm import tqdm


def setup_logging(log_file: Path | None, level: str) -> logging.Logger:
    """Configure a logger that writes to both file and stdout."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("eeg_qc")


@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallBack(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

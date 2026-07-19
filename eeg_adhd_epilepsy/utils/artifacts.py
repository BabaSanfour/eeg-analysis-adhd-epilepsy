"""Shared persistence helpers for project-level derivative artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import yaml


def write_text_atomic(path: str | Path, text: str) -> None:
    """Atomically replace a text file using a process-unique sibling temporary."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def freeze_config_used(
    config: Mapping[str, Any],
    derivative_root: str | Path,
    *,
    volatile_keys: Collection[str] = (),
    sanitize: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    overwrite: bool = False,
    mismatch_message: str = "Existing derivative root uses a different configuration.",
) -> Path:
    """Write or verify the canonical ``config_used.yaml`` snapshot for a root."""
    snapshot = {key: value for key, value in config.items() if key not in volatile_keys}
    serialized = yaml.safe_dump(
        dict(sanitize(snapshot)) if sanitize is not None else snapshot,
        sort_keys=True,
    )
    path = Path(derivative_root) / "config_used.yaml"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return path
        if not overwrite:
            raise ValueError(mismatch_message)
    write_text_atomic(path, serialized)
    return path

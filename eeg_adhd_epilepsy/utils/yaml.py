"""Shared YAML configuration loading.

Single home for reading a YAML config file into a validated mapping, so every
config-driven script gets the same ``expanduser`` handling and type check
instead of rolling its own ``yaml.safe_load``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its top-level mapping.

    Raises ``ValueError`` if the document is not a mapping. Use plain
    ``yaml.safe_load`` directly when a non-mapping payload (e.g. a list) is
    expected.
    """
    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


__all__ = ["load_yaml_config"]

"""Portable scientific identities for analysis inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SCIENTIFIC_PATH_KEYS = (
    "metadata",
    "descriptor_table_path",
    "descriptor_feature_columns_path",
    "embedding_derivative_root",
)


def _portable_reference(value: Any, *, bids_root: Any) -> Any:
    """Return a mount-independent reference for one configured path."""
    if value in (None, ""):
        return value

    path = Path(value).expanduser()
    if not path.is_absolute():
        return f"relative:///{path.as_posix().lstrip('/')}"

    if bids_root not in (None, ""):
        root = Path(bids_root).expanduser()
        try:
            relative = path.resolve(strict=False).relative_to(root.resolve(strict=False))
        except ValueError:
            pass
        else:
            return f"bids:///{relative.as_posix()}"

    # External inputs are identified by their dataset-local filename. This is
    # deliberately a path identity, not a content fingerprint: the old hashes
    # also did not notice content changes at an unchanged path.
    return f"external:///{path.name}"


def normalize_scientific_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize known input paths while preserving all non-path values."""
    normalized = dict(config)
    bids_root = normalized.get("bids_root")
    if bids_root not in (None, ""):
        normalized["bids_root"] = "bids:///"
    for key in _SCIENTIFIC_PATH_KEYS:
        if key in normalized:
            normalized[key] = _portable_reference(normalized[key], bids_root=bids_root)
    return normalized

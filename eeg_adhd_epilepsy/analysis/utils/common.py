"""Cross-pipeline shared helpers for the analysis entry points.

Functions used by two or more of: classical decoding, foundation decoding,
dimensionality reduction.  Nothing in this module is pipeline-specific.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from coco_pipe.io import DataContainer
from coco_pipe.io.quality import group_labels

# ---------------------------------------------------------------------------
# Config guard
# ---------------------------------------------------------------------------


def require_config(
    config: dict[str, Any],
    key: str,
    expected_type: type = list,
    cast_str: bool = False,
) -> Any:
    """Require an explicitly configured, non-empty item of the expected type.

    Raises :class:`ValueError` if the key is absent, falsy, or the wrong type.
    """
    items = config.get(key)
    if not items or not isinstance(items, expected_type):
        raise ValueError(
            f"{key} must be explicitly configured as a non-empty {expected_type.__name__}."
        )
    if expected_type is list and cast_str:
        return [str(x) for x in items]
    return items


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def base_layout_mode(input_mode: str) -> str:
    """Analysis mode used to *load* the shared base container for a scope.

    Descriptor containers are loaded in ``sensor`` layout (``obs × sensor ×
    feature``) so a single load can be re-sliced into every descriptor analysis
    unit (flat, family, sensor, descriptor, …) without re-reading the table.
    Everything else loads flat. Shared by the dim-reduction and decoding loaders
    so the rule lives in one place.
    """
    return "sensor" if input_mode == "descriptors" else "flat"


# ---------------------------------------------------------------------------
# Family QC helpers
# ---------------------------------------------------------------------------


def families_for_analysis_unit(
    source: DataContainer,
    unit: dict,
    descriptor_families: Sequence[str] | None = None,
) -> list[str]:
    """QC grouping labels covered by one analysis unit (see coco_pipe.group_labels)."""
    group_by = (source.meta or {}).get("family_qc_group_by", "family")
    labels = group_labels(unit["container"], group_by)
    if descriptor_families and group_by == "family":
        wanted = {str(v) for v in descriptor_families}
        labels = [lbl for lbl in labels if lbl in wanted]
    return labels or ([str(unit["family"])] if unit.get("family") else [])


def apply_family_qc_mask(
    container: DataContainer,
    families: Sequence[str],
) -> tuple[DataContainer, np.ndarray]:
    """Filter observations using only the requested families' QC decisions."""
    bad_ids_by_family = dict(container.meta.get("family_qc_bad_ids", {}))
    if not bad_ids_by_family:
        indices = np.arange(container.X.shape[0], dtype=int)
        return container, indices
    if container.ids is None:
        raise ValueError("Family-scoped QC requires observation IDs.")
    bad_ids: set[str] = set()
    for family in families:
        bad_ids.update(str(value) for value in bad_ids_by_family.get(str(family), []))
    ids = np.asarray(container.ids, dtype=object).astype(str)
    keep_indices = np.flatnonzero(~np.isin(ids, list(bad_ids)))
    return container.isel(obs=keep_indices), keep_indices


# ---------------------------------------------------------------------------
# Container pooling
# ---------------------------------------------------------------------------


def pool_containers(containers: list[DataContainer]) -> DataContainer:
    """Concatenate conditions while preserving deferred family-QC semantics."""
    if not containers:
        raise ValueError("No containers to pool.")

    dims = containers[0].dims
    common_coords: dict[str, list] = {}
    for dim in dims:
        if dim == "obs":
            continue
        if dim in containers[0].coords:
            common = set(containers[0].coords[dim])
            for c in containers[1:]:
                common &= set(c.coords[dim])
            # Preserve original order from the first container
            common_coords[dim] = [x for x in containers[0].coords[dim] if x in common]

    aligned = []
    for c in containers:
        c_aligned = c
        for dim, values in common_coords.items():
            if len(values) < len(c_aligned.coords[dim]):
                c_aligned = c_aligned.select(**{dim: values})
        aligned.append(c_aligned)

    pooled = DataContainer.concat(aligned)
    group_by_values = {
        str(container.meta["family_qc_group_by"])
        for container in containers
        if container.meta.get("family_qc_group_by") is not None
    }
    if len(group_by_values) > 1:
        raise ValueError(
            "Cannot pool containers with different family-QC grouping levels: "
            f"{sorted(group_by_values)}"
        )
    pooled.meta = {
        **dict(pooled.meta),
        "family_qc_bad_ids": {
            group: sorted(
                {
                    str(obs_id)
                    for item in containers
                    for obs_id in item.meta.get("family_qc_bad_ids", {}).get(group, [])
                }
            )
            for group in {
                group for item in containers for group in item.meta.get("family_qc_bad_ids", {})
            }
        },
    }
    if group_by_values:
        pooled.meta["family_qc_group_by"] = next(iter(group_by_values))
    return pooled


__all__ = [
    "apply_family_qc_mask",
    "base_layout_mode",
    "families_for_analysis_unit",
    "pool_containers",
    "require_config",
]

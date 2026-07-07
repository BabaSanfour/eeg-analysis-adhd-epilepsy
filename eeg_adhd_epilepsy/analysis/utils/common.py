"""Cross-pipeline shared helpers for the analysis entry points.

Functions used by two or more of: classical decoding, foundation decoding,
dimensionality reduction.  Nothing in this module is pipeline-specific.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
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


def _common_non_obs_coords(
    dims: tuple[str, ...],
    coords_list: list[dict[str, Any]],
) -> dict[str, list]:
    """Per non-obs dim, the coord values shared by every input (first-input order)."""
    common: dict[str, list] = {}
    for dim in dims:
        if dim == "obs" or dim not in coords_list[0]:
            continue
        shared = set(coords_list[0][dim])
        for coords in coords_list[1:]:
            shared &= set(coords[dim])
        common[dim] = [value for value in coords_list[0][dim] if value in shared]
    return common


def _align_non_obs(container: DataContainer, common: dict[str, list]) -> DataContainer:
    """Subset ``container`` down to the shared non-obs coordinates."""
    aligned = container
    for dim, values in common.items():
        if len(values) < len(aligned.coords[dim]):
            aligned = aligned.select(**{dim: values})
    return aligned


def _pooled_family_qc_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge deferred family-QC bad-id sets across pooled inputs."""
    group_by_values = {
        str(meta["family_qc_group_by"])
        for meta in metas
        if meta.get("family_qc_group_by") is not None
    }
    if len(group_by_values) > 1:
        raise ValueError(
            "Cannot pool containers with different family-QC grouping levels: "
            f"{sorted(group_by_values)}"
        )
    all_groups = {group for meta in metas for group in meta.get("family_qc_bad_ids", {})}
    merged: dict[str, Any] = {
        "family_qc_bad_ids": {
            group: sorted(
                {
                    str(obs_id)
                    for meta in metas
                    for obs_id in meta.get("family_qc_bad_ids", {}).get(group, [])
                }
            )
            for group in all_groups
        }
    }
    if group_by_values:
        merged["family_qc_group_by"] = next(iter(group_by_values))
    return merged


def pool_containers(containers: list[DataContainer]) -> DataContainer:
    """Concatenate conditions while preserving deferred family-QC semantics."""
    if not containers:
        raise ValueError("No containers to pool.")
    common = _common_non_obs_coords(containers[0].dims, [c.coords for c in containers])
    pooled = DataContainer.concat([_align_non_obs(c, common) for c in containers])
    pooled.meta = {**dict(pooled.meta), **_pooled_family_qc_meta([c.meta for c in containers])}
    return pooled


def container_pool_spec(container: DataContainer) -> dict[str, Any]:
    """Capture what :func:`pool_containers_streaming` needs *except* the X payload.

    Lets a caller release a scope container's (large) data array while keeping the
    small coords/ids/meta used to size and label the pooled result, so pooling can
    reload each condition's X one at a time instead of holding them all at once.
    """
    obs_axis = container.dims.index("obs")
    return {
        "dims": container.dims,
        "shape": tuple(container.X.shape),
        "dtype": container.X.dtype,
        "n_obs": int(container.X.shape[obs_axis]),
        "coords": dict(container.coords),
        "y": container.y,
        "ids": container.ids,
        "meta": container.meta,
    }


def _pool_shell(spec: dict[str, Any], common: dict[str, list]) -> DataContainer:
    """A zero-storage stand-in: obs coords + shared non-obs dim coords, no payload."""
    dims = spec["dims"]
    n_obs = spec["n_obs"]
    coords_spec = spec["coords"]
    shape = list(spec["shape"])
    for dim, values in common.items():
        # Mirror _align_non_obs: the payload is only subset (and thus resized)
        # when the shared set is strictly smaller than this input's dim coord.
        if dim in coords_spec and len(values) < len(coords_spec[dim]):
            shape[dims.index(dim)] = len(values)
    coords: dict[str, Any] = {
        key: np.asarray(values)
        for key, values in spec["coords"].items()
        if key not in dims and np.ndim(values) == 1 and len(values) == n_obs
    }
    for dim, values in common.items():
        coords[dim] = np.asarray(values)
    # broadcast_to gives a correct-shape, zero-storage view, so concat allocates
    # exactly one pooled-size array below — which we then fill in place.
    shell_x = np.broadcast_to(np.zeros((), spec["dtype"]), tuple(shape))
    return DataContainer(
        X=shell_x, dims=dims, coords=coords, y=spec["y"], ids=spec["ids"], meta=spec["meta"]
    )


def pool_containers_streaming(
    specs: list[dict[str, Any]],
    loaders: list[Callable[[], DataContainer]],
) -> DataContainer:
    """Memory-frugal :func:`pool_containers`: fill one buffer via per-condition reloads.

    *specs* (from :func:`container_pool_spec`) size and label the pooled container
    without its X; *loaders* re-read each condition's full data one at a time to
    fill the single buffer that :meth:`DataContainer.concat` allocates from the
    zero-storage shells. Peak host memory is ~one condition plus the pooled result
    rather than every condition at once. The output equals
    ``pool_containers([...])`` over the same conditions.
    """
    if not specs or len(specs) != len(loaders):
        raise ValueError("pool_containers_streaming needs matching, non-empty specs and loaders.")
    common = _common_non_obs_coords(specs[0]["dims"], [spec["coords"] for spec in specs])
    pooled = DataContainer.concat([_pool_shell(spec, common) for spec in specs])

    obs_axis = specs[0]["dims"].index("obs")
    offset = 0
    for index, loader in enumerate(loaders):
        aligned = _align_non_obs(loader(), common)
        n_obs = aligned.X.shape[obs_axis]
        pooled.X[offset : offset + n_obs] = aligned.X
        if index == 0:
            # Refresh non-obs coords from the first *aligned, real* container so any
            # auxiliary (e.g. per-channel) coords match the subset payload exactly.
            for key, values in aligned.coords.items():
                array = np.asarray(values)
                if not (array.ndim == 1 and len(array) == n_obs):
                    pooled.coords[key] = array
        offset += n_obs
        del aligned
        gc.collect()

    pooled.meta = {**dict(pooled.meta), **_pooled_family_qc_meta([spec["meta"] for spec in specs])}
    return pooled


__all__ = [
    "apply_family_qc_mask",
    "base_layout_mode",
    "container_pool_spec",
    "families_for_analysis_unit",
    "pool_containers",
    "pool_containers_streaming",
    "require_config",
]

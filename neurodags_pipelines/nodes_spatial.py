"""Spatial pooling nodes: average over channel groups."""

from __future__ import annotations

import os

import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


def _to_nc_writer(da_or_ds):
    return lambda path, obj=da_or_ds: obj.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _resolve_xr(obj):
    if isinstance(obj, (str, os.PathLike)):
        loaded = xr.open_dataset(str(obj))
        if len(loaded.data_vars) == 1:
            return next(iter(loaded.data_vars.values()))
        return loaded
    if isinstance(obj, NodeResult):
        if ".nc" in obj.artifacts:
            return _resolve_xr(obj.artifacts[".nc"].item)
        raise ValueError("NodeResult has no .nc artifact")
    if isinstance(obj, xr.Dataset) and len(obj.data_vars) == 1:
        return next(iter(obj.data_vars.values()))
    return obj


@register_node
def pool_channels(
    xarray_data,
    channel_groups: dict[str, list[str]],
    spaces_dim: str = "spaces",
) -> NodeResult:
    """Average over named channel groups, producing a *regions* dimension.

    Parameters
    ----------
    xarray_data
        DataArray with a *spaces_dim* dimension (channel names as coords).
    channel_groups
        Mapping of ``{region_name: [ch1, ch2, ...]}``.
    spaces_dim
        Name of the spatial / channel dimension (default ``"spaces"``).
    """
    da = _resolve_xr(xarray_data)
    if not isinstance(da, xr.DataArray):
        raise ValueError("pool_channels expects a DataArray, got Dataset")
    if spaces_dim not in da.dims:
        raise ValueError(f"'{spaces_dim}' not in dims {list(da.dims)}")

    avail: list[str] = (
        [str(v) for v in da.coords[spaces_dim].values]
        if spaces_dim in da.coords
        else []
    )

    region_das: list[xr.DataArray] = []
    region_names: list[str] = []

    for region_name, ch_list in channel_groups.items():
        present = [c for c in ch_list if c in avail]
        if not present:
            continue
        region_da = da.sel({spaces_dim: present}).mean(dim=spaces_dim)
        region_das.append(region_da.expand_dims({"regions": [region_name]}))
        region_names.append(region_name)

    if not region_das:
        raise ValueError(
            f"None of the channel_groups channels were found in '{spaces_dim}'. "
            f"Available: {avail[:10]}{'...' if len(avail) > 10 else ''}"
        )

    pooled = xr.concat(region_das, dim="regions").assign_coords(regions=region_names)
    return NodeResult(artifacts={".nc": Artifact(item=pooled, writer=_to_nc_writer(pooled))})

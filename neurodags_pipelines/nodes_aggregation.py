"""Multi-stat aggregation nodes: iqr, mad across xarray dimensions."""

from __future__ import annotations

import os

import numpy as np
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
def iqr_across_dimension(xarray_data, dim: str) -> NodeResult:
    """Inter-quartile range across *dim*."""
    try:
        from scipy.stats import iqr as scipy_iqr
    except ImportError as exc:
        raise ImportError("scipy required for iqr_across_dimension") from exc

    da = _resolve_xr(xarray_data)
    if not isinstance(da, xr.DataArray):
        raise ValueError("iqr_across_dimension expects a DataArray, got Dataset")

    result = xr.apply_ufunc(
        scipy_iqr,
        da,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        dask="parallelized",
    )
    result = result.assign_coords({k: v for k, v in da.coords.items() if k != dim})
    return NodeResult(artifacts={".nc": Artifact(item=result, writer=_to_nc_writer(result))})


@register_node
def mad_across_dimension(xarray_data, dim: str) -> NodeResult:
    """Median absolute deviation across *dim*."""
    da = _resolve_xr(xarray_data)
    if not isinstance(da, xr.DataArray):
        raise ValueError("mad_across_dimension expects a DataArray, got Dataset")

    def _mad(arr: np.ndarray) -> np.ndarray:
        med = np.median(arr, axis=-1, keepdims=True)
        return np.median(np.abs(arr - med), axis=-1)

    result = xr.apply_ufunc(
        _mad,
        da,
        input_core_dims=[[dim]],
        dask="parallelized",
    )
    result = result.assign_coords({k: v for k, v in da.coords.items() if k != dim})
    return NodeResult(artifacts={".nc": Artifact(item=result, writer=_to_nc_writer(result))})

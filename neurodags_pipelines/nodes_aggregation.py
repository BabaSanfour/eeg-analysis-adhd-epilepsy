"""Multi-stat aggregation nodes: iqr, mad across xarray dimensions."""

from __future__ import annotations

import numpy as np
import xarray as xr
import nodes_utils

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node

_to_nc_writer = nodes_utils._to_nc_writer
_resolve_xr = nodes_utils._resolve_xr


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
    result_dims = set(result.dims)
    result = result.assign_coords({
        k: v for k, v in da.coords.items()
        if k != dim and set(v.dims).issubset(result_dims)
    })
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
    result_dims = set(result.dims)
    result = result.assign_coords({
        k: v for k, v in da.coords.items()
        if k != dim and set(v.dims).issubset(result_dims)
    })
    return NodeResult(artifacts={".nc": Artifact(item=result, writer=_to_nc_writer(result))})

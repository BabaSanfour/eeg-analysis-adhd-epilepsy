"""Shared helpers and utility nodes."""

from __future__ import annotations

import os
import sys

import xarray as xr

# Make this directory importable so other definition files can do:
#   import nodes_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


def _to_nc_writer(da_or_ds):
    return lambda path, obj=da_or_ds: obj.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _resolve_xr(obj):
    """Coerce NodeResult / path / Dataset to DataArray where possible."""
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
def extract_sfreq_from_xarray(data_like) -> NodeResult:
    """Return sfreq as a scalar NodeResult from any MNE/xarray input.

    Handles: file path string (.fif), MNE Raw/Epochs, xarray DataArray/Dataset.
    """
    import json
    from pathlib import Path as _Path

    import xarray as xr

    sfreq: float | None = None

    if isinstance(data_like, (str, _Path)):
        import mne
        obj = mne.read_epochs(str(data_like), preload=False, verbose="ERROR")
        sfreq = float(obj.info["sfreq"])
    elif hasattr(data_like, "info"):
        sfreq = float(data_like.info["sfreq"])
    elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
        meta = json.loads(data_like.attrs.get("metadata", "{}"))
        sfreq = meta.get("sfreq")
        if sfreq is not None:
            sfreq = float(sfreq)
    else:
        raise TypeError(f"extract_sfreq_from_xarray: unsupported type {type(data_like)}")

    if sfreq is None:
        raise ValueError("extract_sfreq_from_xarray: could not determine sfreq")
    return NodeResult(artifacts={"": Artifact(item=sfreq, writer=lambda _: None)})

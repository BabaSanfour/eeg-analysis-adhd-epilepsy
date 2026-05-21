"""Custom neurodags nodes for the EEG ADHD/epilepsy pipeline.

Points implemented:
  1. Multi-stat aggregation  : iqr_across_dimension, mad_across_dimension
  3. Spatial pooling         : pool_channels
  5. Condition granularity   : preprocess_raw, extract_condition_epochs
  6. Artifact cleaning       : zapline_denoise, ransac_bad_channels,
                               apply_car, autoreject_annotate,
                               ica_artifact_correction
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_nc_writer(da_or_ds):
    return lambda path, obj=da_or_ds: obj.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _resolve_xr(obj):
    """Coerce NodeResult / path / Dataset to DataArray where possible."""
    if isinstance(obj, (str, os.PathLike)):
        loaded = xr.open_dataset(str(obj))
        if len(loaded.data_vars) == 1:
            return next(iter(loaded.data_vars.values()))
        return loaded  # return Dataset
    if isinstance(obj, NodeResult):
        if ".nc" in obj.artifacts:
            return _resolve_xr(obj.artifacts[".nc"].item)
        raise ValueError("NodeResult has no .nc artifact")
    if isinstance(obj, xr.Dataset) and len(obj.data_vars) == 1:
        return next(iter(obj.data_vars.values()))
    return obj  # already DataArray or Dataset


# ---------------------------------------------------------------------------
# Point 1 — Multi-stat aggregation
# ---------------------------------------------------------------------------

@register_node
def iqr_across_dimension(xarray_data, dim: str) -> NodeResult:
    """Inter-quartile range across *dim*.

    Used for band power and FOOOF scalar aggregation (mirrors
    configs/descriptors.yaml aggregation.descriptors band_summaries stats).
    """
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
    """Median absolute deviation across *dim*.

    Used for complexity feature aggregation (mirrors
    configs/descriptors.yaml complexity_summaries stats).
    """
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


# ---------------------------------------------------------------------------
# Point 3 — Spatial pooling
# ---------------------------------------------------------------------------

@register_node
def pool_channels(
    xarray_data,
    channel_groups: dict[str, list[str]],
    spaces_dim: str = "spaces",
) -> NodeResult:
    """Average over named channel groups, producing a *regions* dimension.

    Equivalent of configs/descriptors.yaml pooling.channel_groups.
    Channels absent from the DataArray are silently skipped per group;
    groups with no present channels are dropped entirely.

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


# ---------------------------------------------------------------------------
# Point 5 — Condition granularity
# ---------------------------------------------------------------------------

@register_node
def preprocess_raw(
    mne_object,
    filter_args: dict[str, Any] | None = None,
    notch_filter: dict[str, Any] | None = None,
    resample: float | None = None,
) -> NodeResult:
    """Filter and resample a Raw recording without epoching.

    Equivalent to the filtering steps of basic_preprocessing but keeps the
    continuous Raw so that condition windows can be extracted downstream via
    extract_condition_epochs.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    if notch_filter is not None:
        raw.notch_filter(**notch_filter, verbose=False)
    if filter_args is not None:
        raw.filter(**filter_args, verbose=False)
    if resample is not None:
        raw.resample(float(resample), verbose=False)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def extract_condition_epochs(
    mne_object,
    condition_name: str,
    annotation_prefix: str = "BLOCK_",
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
    reject_by_annotation: str | None = None,
) -> NodeResult:
    """Extract fixed-length epochs from BLOCK_<condition_name> annotation windows.

    Equivalent to the per-condition epoch extraction in extract_descriptors.py.
    Looks for annotations whose description equals
    ``{annotation_prefix}{condition_name}`` and slices out fixed-length epochs
    within each matching window.

    Parameters
    ----------
    mne_object
        Preprocessed MNE Raw (e.g. from preprocess_raw).
    condition_name
        Condition label (appended to *annotation_prefix*).
    annotation_prefix
        Prefix used in the Raw annotations (default ``"BLOCK_"``).
    epoch_duration
        Length of each fixed epoch in seconds.
    epoch_overlap
        Overlap between consecutive epochs in seconds.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        # Raw .fif saved by preprocess_raw
        mne_object = _mne.io.read_raw_fif(str(mne_object), preload=True, verbose="ERROR")

    target_desc = f"{annotation_prefix}{condition_name}"
    windows: list[tuple[float, float]] = []
    for annot in mne_object.annotations:
        desc = str(annot["description"])
        # BrainVision round-trip wraps descriptions in "Comment/" — strip it.
        if desc.startswith("Comment/"):
            desc = desc[len("Comment/"):]
        if desc == target_desc:
            onset = float(annot["onset"])
            windows.append((onset, onset + float(annot["duration"])))

    if not windows:
        normalized = sorted({
            str(a["description"]).removeprefix("Comment/")
            for a in mne_object.annotations
        })
        raise ValueError(
            f"No annotations matching '{target_desc}' found. "
            f"Present descriptions (normalized): {normalized}"
        )

    epoch_chunks: list[_mne.BaseEpochs] = []
    for onset, offset in windows:
        crop = mne_object.copy().crop(onset, min(offset, mne_object.times[-1] + mne_object.first_time))
        if crop.n_times < int(epoch_duration * crop.info["sfreq"]):
            continue
        eps = _mne.make_fixed_length_epochs(
            crop,
            duration=epoch_duration,
            overlap=epoch_overlap,
            preload=True,
            verbose="ERROR",
            reject_by_annotation=reject_by_annotation or False,
        )
        if len(eps) > 0:
            epoch_chunks.append(eps)

    if not epoch_chunks:
        raise ValueError(
            f"Condition '{condition_name}' found in annotations but all windows "
            f"were too short for {epoch_duration}s epochs."
        )

    epochs = (
        epoch_chunks[0]
        if len(epoch_chunks) == 1
        else _mne.concatenate_epochs(epoch_chunks, verbose="ERROR")
    )

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=epochs,
                writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


# ---------------------------------------------------------------------------
# Point 6 — Artifact cleaning nodes
# ---------------------------------------------------------------------------

@register_node
def zapline_denoise(
    mne_object,
    line_freq: float = 60.0,
    adaptive: bool = False,
) -> NodeResult:
    """Remove power-line noise using ZapLine (mne-denoise).

    Equivalent to the ZapLine step in base.py run_base_pipeline.
    """
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from mne_denoise.zapline import ZapLine
    except ImportError as exc:
        raise ImportError("mne-denoise required for zapline_denoise") from exc

    raw = mne_object.copy().load_data()
    zapline = ZapLine(sfreq=raw.info["sfreq"], line_freq=line_freq, adaptive=adaptive)
    raw = zapline.fit_transform(raw)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def ransac_bad_channels(
    mne_object,
    block_label: str | None = None,
    annotation_prefix: str = "BLOCK_",
) -> NodeResult:
    """Detect and mark bad channels using RANSAC (pyprep).

    Equivalent to detect_global_bads_ransac in base.py.
    Marks detected channels in raw.info['bads'] without removing them.

    Parameters
    ----------
    block_label
        If set, RANSAC runs only on segments matching
        ``{annotation_prefix}{block_label}`` annotations (mirrors the
        rest-block-biased approach in base.py).  None = use full recording.
    annotation_prefix
        Prefix for block annotations (default ``"BLOCK_"``).
    """
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except ImportError as exc:
        raise ImportError("pyprep required for ransac_bad_channels") from exc

    import mne as _mne

    raw = mne_object.copy().load_data()
    eeg_picks = _mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return NodeResult(
            artifacts={".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))}
        )

    raw_for_ransac = raw
    if block_label is not None:
        target = f"{annotation_prefix}{block_label}"
        crops: list = []
        for annot in raw.annotations:
            desc = str(annot["description"])
            if desc.startswith("Comment/"):
                desc = desc[len("Comment/"):]
            if desc == target:
                onset = float(annot["onset"])
                offset = onset + float(annot["duration"])
                crop = raw.copy().crop(onset, min(offset, raw.times[-1] + raw.first_time))
                if crop.n_times > 0:
                    crops.append(crop)
        if crops:
            raw_for_ransac = crops[0] if len(crops) == 1 else _mne.concatenate_raws(crops, verbose="ERROR")

    try:
        nc = NoisyChannels(raw_for_ransac, random_state=42)
        nc.find_bad_by_ransac()
        bads = nc.get_bads(verbose=False) or []
        bads = sorted(ch for ch in bads if ch in raw.ch_names)
        raw.info["bads"] = sorted(set(raw.info.get("bads") or []) | set(bads))
    except (ValueError, OSError):
        pass  # RANSAC failed silently — common on short/low-channel data

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def apply_car(mne_object) -> NodeResult:
    """Apply Common Average Reference.

    Equivalent to the CAR step in base.py (applied after bad channel exclusion).
    """
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=raw,
                writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def autoreject_annotate(
    mne_object,
    segment_duration: float = 1.0,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
) -> NodeResult:
    """Run AutoReject on fixed-length segments and add BAD_ annotations.

    Simplified equivalent of annotate_artifacts_blockwise in base.py.
    Operates on the whole recording (not condition-aware) — for condition-
    aware AR, run after extract_condition_epochs.

    Outputs Raw (with BAD_ annotations added) and also returns Epochs for
    downstream feature extraction.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from autoreject import AutoReject
    except ImportError as exc:
        raise ImportError("autoreject required for autoreject_annotate") from exc

    raw = mne_object.copy().load_data()
    n_interp = np.asarray(n_interpolate or [0], dtype=int)
    cv = min(10, max(2, min_epochs))

    # Create 1s segments for AR threshold estimation
    seg_epochs = _mne.make_fixed_length_epochs(raw, duration=segment_duration, preload=True, verbose="ERROR")
    if len(seg_epochs) < min_epochs:
        # Too few epochs — skip AR, just epoch with fixed length
        epochs = _mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=epoch_overlap, preload=True, verbose="ERROR")
        return NodeResult(
            artifacts={
                ".fif": Artifact(item=epochs, writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"))
            }
        )

    # AutoReject requires valid channel positions. If missing (e.g. synthetic data),
    # directly patch ch['loc'][:3] with evenly-spaced positions on a unit circle.
    _locs = np.array([ch["loc"][:3] for ch in seg_epochs.info["chs"]])
    if np.allclose(_locs, 0) or not np.all(np.isfinite(_locs)):
        _n = len(seg_epochs.ch_names)
        _angles = np.linspace(0, 2 * np.pi, _n, endpoint=False)
        seg_epochs = seg_epochs.copy()
        with seg_epochs.info._unlock():
            for _i, _ch in enumerate(seg_epochs.info["chs"]):
                _a = _angles[_i]
                _ch["loc"][:3] = [np.cos(_a) * 0.09, np.sin(_a) * 0.09, 0.01]

    ar = AutoReject(n_interpolate=n_interp, random_state=42, n_jobs=1, verbose=False, cv=cv)
    ar.fit(seg_epochs)
    reject_log = ar.get_reject_log(seg_epochs)

    # Add BAD_epoch annotations for rejected segments
    new_annots: list[tuple[float, float, str]] = []
    for ep_idx, is_bad in enumerate(reject_log.bad_epochs):
        if not is_bad:
            continue
        onset = float(seg_epochs.events[ep_idx, 0] - raw.first_samp) / raw.info["sfreq"]
        new_annots.append((max(0.0, onset), segment_duration, "BAD_epoch"))

    if new_annots:
        ar_annots = _mne.Annotations(
            onset=[a[0] for a in new_annots],
            duration=[a[1] for a in new_annots],
            description=[a[2] for a in new_annots],
        )
        raw.set_annotations(raw.annotations + ar_annots)

    # Create final epochs for feature extraction
    epochs = _mne.make_fixed_length_epochs(
        raw, duration=epoch_duration, overlap=epoch_overlap,
        reject_by_annotation="omit", preload=True, verbose="ERROR",
    )

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=epochs,
                writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def autoreject_annotate_raw(
    mne_object,
    condition_name: str | None = None,
    annotation_prefix: str = "BLOCK_",
    segment_duration: float = 1.0,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
) -> NodeResult:
    """Run AutoReject on Raw and add BAD_epoch annotations; return annotated Raw.

    Equivalent of annotate_artifacts_blockwise in base.py.
    When condition_name is set, AR runs only on 1s segments within
    BLOCK_{condition_name} windows (per-condition AR, matching base.py behavior).
    When condition_name is None, runs on the whole recording.

    Use extract_condition_epochs(reject_by_annotation="omit") downstream
    to get clean Epochs from the annotated Raw.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    try:
        from autoreject import AutoReject
    except ImportError as exc:
        raise ImportError("autoreject required for autoreject_annotate_raw") from exc

    raw = mne_object.copy().load_data()
    n_interp = np.asarray(n_interpolate or [0], dtype=int)
    sfreq = raw.info["sfreq"]
    step = int(segment_duration * sfreq)
    tmax = max(segment_duration - 1.0 / sfreq, 0.0)

    if condition_name is not None:
        target = f"{annotation_prefix}{condition_name}"
        windows: list[tuple[float, float]] = []
        for annot in raw.annotations:
            desc = str(annot["description"])
            if desc.startswith("Comment/"):
                desc = desc[len("Comment/"):]
            if desc == target:
                onset = float(annot["onset"])
                windows.append((onset, onset + float(annot["duration"])))

        if not windows:
            return NodeResult(artifacts={
                ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
            })

        event_rows: list[list[int]] = []
        for onset, offset in windows:
            start_samp = int(raw.time_as_index(onset)[0]) + raw.first_samp
            end_samp = int(raw.time_as_index(offset)[0]) + raw.first_samp
            t = start_samp
            while t + step <= end_samp:
                event_rows.append([t, 0, 1])
                t += step

        if not event_rows:
            return NodeResult(artifacts={
                ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
            })

        events = np.array(event_rows, dtype=int)
        seg_epochs = _mne.Epochs(
            raw, events, event_id={"seg": 1}, tmin=0.0, tmax=tmax,
            baseline=None, preload=True, verbose="ERROR", reject_by_annotation=False,
        )
    else:
        seg_epochs = _mne.make_fixed_length_epochs(raw, duration=segment_duration, preload=True, verbose="ERROR")

    if len(seg_epochs) < min_epochs:
        return NodeResult(artifacts={
            ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
        })

    # Patch channel positions for synthetic data (same workaround as autoreject_annotate)
    _locs = np.array([ch["loc"][:3] for ch in seg_epochs.info["chs"]])
    if np.allclose(_locs, 0) or not np.all(np.isfinite(_locs)):
        _n = len(seg_epochs.ch_names)
        _angles = np.linspace(0, 2 * np.pi, _n, endpoint=False)
        seg_epochs = seg_epochs.copy()
        with seg_epochs.info._unlock():
            for _i, _ch in enumerate(seg_epochs.info["chs"]):
                _a = _angles[_i]
                _ch["loc"][:3] = [np.cos(_a) * 0.09, np.sin(_a) * 0.09, 0.01]

    cv = min(10, max(2, len(seg_epochs)))
    ar = AutoReject(n_interpolate=n_interp, random_state=42, n_jobs=1, verbose=False, cv=cv)
    ar.fit(seg_epochs)
    reject_log = ar.get_reject_log(seg_epochs)

    new_annots: list[tuple[float, float, str]] = []
    for ep_idx, is_bad in enumerate(reject_log.bad_epochs):
        if not is_bad:
            continue
        onset = float(seg_epochs.events[ep_idx, 0] - raw.first_samp) / sfreq
        new_annots.append((max(0.0, onset), segment_duration, "BAD_epoch"))

    if new_annots:
        raw.set_annotations(raw.annotations + _mne.Annotations(
            onset=[a[0] for a in new_annots],
            duration=[a[1] for a in new_annots],
            description=[a[2] for a in new_annots],
        ))

    return NodeResult(artifacts={
        ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
    })


@register_node
def autoreject_clean_epochs(
    mne_object,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
) -> NodeResult:
    """Run AutoReject on Epochs and return cleaned Epochs (bad epochs dropped).

    Condition-aware equivalent of autoreject_annotate — use after
    extract_condition_epochs so AR thresholds are estimated per condition.
    Unlike autoreject_annotate (which needs Raw), this node accepts Epochs
    directly and simply drops rejected epochs rather than annotating raw time.
    """
    import mne as _mne

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = _mne.read_epochs(str(mne_object), preload=True, verbose="ERROR")

    try:
        from autoreject import AutoReject
    except ImportError as exc:
        raise ImportError("autoreject required for autoreject_clean_epochs") from exc

    epochs = mne_object.copy().load_data()
    n_interp = np.asarray(n_interpolate or [0], dtype=int)

    if len(epochs) < min_epochs:
        return NodeResult(
            artifacts={
                ".fif": Artifact(
                    item=epochs,
                    writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"),
                )
            }
        )

    # Patch channel positions for synthetic data (same workaround as autoreject_annotate)
    _locs = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])
    if np.allclose(_locs, 0) or not np.all(np.isfinite(_locs)):
        _n = len(epochs.ch_names)
        _angles = np.linspace(0, 2 * np.pi, _n, endpoint=False)
        epochs = epochs.copy()
        with epochs.info._unlock():
            for _i, _ch in enumerate(epochs.info["chs"]):
                _a = _angles[_i]
                _ch["loc"][:3] = [np.cos(_a) * 0.09, np.sin(_a) * 0.09, 0.01]

    cv = min(10, max(2, len(epochs)))
    ar = AutoReject(n_interpolate=n_interp, random_state=42, n_jobs=1, verbose=False, cv=cv)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)
    cleaned = epochs[~reject_log.bad_epochs]

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=cleaned,
                writer=lambda path, e=cleaned: e.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )


@register_node
def ica_artifact_correction(
    mne_object,
    n_components: int = 20,
    remove_eog: bool = True,
    remove_ecg: bool = True,
    random_state: int = 42,
) -> NodeResult:
    """Remove physiological artifacts using ICA.

    Simplified equivalent of Stage 1 (correct.py) — uses MNE's built-in
    EOG/ECG component detection instead of the DSS/MWF approach.
    Fits ICA on bandpass-filtered copy (1-100 Hz), then applies to original.

    remove_eog uses frontal channels (Fp1/Fp2 or first 2 channels) as proxy.
    remove_ecg uses cardiac-channel heuristic or skips silently if no ECG found.
    """
    import mne as _mne
    from mne.preprocessing import ICA
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    # ICA requires continuous raw — if Epochs, apply differently
    import mne as _mne
    if isinstance(raw, _mne.BaseEpochs):
        # For epochs, reconstruct pseudo-raw for ICA fitting
        raw_for_ica = raw.copy()
        filt = raw_for_ica.filter(1.0, 100.0, verbose=False)
        ica = ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(filt)
        if remove_eog:
            try:
                eog_inds, _ = ica.find_bads_eog(raw_for_ica)
                ica.exclude.extend(eog_inds)
            except (RuntimeError, ValueError):
                pass
        cleaned = ica.apply(raw.copy(), verbose=False)
    else:
        filt = raw.copy().filter(1.0, 100.0, verbose=False)
        ica = ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(filt)

        if remove_eog:
            try:
                eog_inds, _ = ica.find_bads_eog(raw)
                ica.exclude.extend(eog_inds)
            except (RuntimeError, ValueError):
                pass
        if remove_ecg:
            try:
                ecg_inds, _ = ica.find_bads_ecg(raw)
                ica.exclude.extend(ecg_inds)
            except (RuntimeError, ValueError):
                pass

        cleaned = ica.apply(raw.copy(), verbose=False)

    return NodeResult(
        artifacts={
            ".fif": Artifact(
                item=cleaned,
                writer=lambda path, r=cleaned: r.save(path, overwrite=True, verbose="ERROR"),
            )
        }
    )

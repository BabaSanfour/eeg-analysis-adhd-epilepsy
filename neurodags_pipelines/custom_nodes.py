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
# Utility — extract sfreq scalar from xarray epoch DataArray attrs
# ---------------------------------------------------------------------------

@register_node
def extract_sfreq_from_xarray(data_like) -> NodeResult:
    """Return sfreq as a scalar NodeResult from any MNE/xarray input.

    Handles: file path string (.fif), MNE Raw/Epochs, xarray DataArray/Dataset.
    Used to pass sf dynamically to antropy_spectral_entropy instead of
    hardcoding 256.0.
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

def _group_consecutive_indices(indices):
    """Group consecutive integers into inclusive (start, end) pairs."""
    if len(indices) == 0:
        return []
    groups = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = idx
            prev = idx
    groups.append((start, prev))
    return groups


@register_node
def inflate_bad_annotations(
    mne_object,
    default_duration: float = 3.0,
    major_duration: float = 5.0,
) -> NodeResult:
    """Expand point-like manual BAD_ annotations to fixed durations by label type.

    Port of inflate_bad_annotations from base.py.
    Rare/disruptive labels (yawn, cough, blink, etc.) → major_duration (5 s).
    All other BAD_ labels → default_duration (3 s), or keep existing if longer.
    Non-BAD_ annotations are kept unchanged.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)

    raw = mne_object.copy().load_data()

    major_slugs = [
        "yawn", "cough", "yawning_coughing",
        "emotion_behavior", "oral_activity",
        "sensor_artefact", "sensor_action",
        "eye_movement", "blink",
        "jaw_face_tension",
        "sleep", "sleepy", "wakefulness",
    ]

    new_onsets, new_durations, new_descs = [], [], []
    for annot in raw.annotations:
        desc = str(annot["description"])
        onset = float(annot["onset"])
        duration = float(annot["duration"])
        if not desc.lower().startswith("bad"):
            new_onsets.append(onset)
            new_durations.append(duration)
            new_descs.append(desc)
            continue
        desc_lower = desc.lower()
        if any(slug in desc_lower for slug in major_slugs):
            new_durations.append(major_duration)
        else:
            new_durations.append(max(duration, default_duration))
        new_onsets.append(onset)
        new_descs.append(desc)

    raw.set_annotations(_mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descs,
        orig_time=raw.annotations.orig_time,
    ))

    return NodeResult(artifacts={
        ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
    })


@register_node
def autoreject_annotate_blockwise(
    mne_object,
    annotation_prefix: str = "BLOCK_",
    segment_duration: float = 1.0,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
    ar_max_chunk_minutes: float = 30.0,
    n_jobs: int = 1,
) -> NodeResult:
    """Condition-grouped AutoReject on Raw — port of annotate_artifacts_blockwise from base.py.

    Finds all unique BLOCK_* conditions, builds 1 s events within each condition's
    windows, runs one AR instance per condition group (chunked if > ar_max_chunk_minutes).
    Adds BAD_epoch_{condition} (whole-epoch) and BAD_{condition} (per-channel span)
    annotations to Raw.  Returns annotated Raw.

    Use epoch_fixed_length or extract_condition_epochs downstream to get Epochs.
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
        raise ImportError("autoreject required for autoreject_annotate_blockwise") from exc

    raw = mne_object.copy().load_data()
    n_interp = np.asarray(n_interpolate or [0], dtype=int)
    sfreq = raw.info["sfreq"]
    step = int(segment_duration * sfreq)
    tmax = max(segment_duration - 1.0 / sfreq, 0.0)
    n_per_chunk = max(1, int((ar_max_chunk_minutes * 60.0) / segment_duration))

    # Collect all BLOCK_* condition windows
    condition_windows: dict[str, list[tuple[float, float]]] = {}
    for annot in raw.annotations:
        desc = str(annot["description"])
        if desc.startswith("Comment/"):
            desc = desc[len("Comment/"):]
        if desc.startswith(annotation_prefix):
            cond = desc[len(annotation_prefix):]
            condition_windows.setdefault(cond, []).append(
                (float(annot["onset"]), float(annot["onset"]) + float(annot["duration"]))
            )

    if not condition_windows:
        return NodeResult(artifacts={
            ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR"))
        })

    bad_channels_prov = list(raw.info["bads"])  # captured after RANSAC step
    all_new_annots: list[tuple[float, float, str, tuple]] = []
    condition_plots: dict[str, Any] = {}
    condition_stats: dict[str, dict] = {}

    for cond_name, windows in condition_windows.items():
        event_rows: list[list[int]] = []
        for onset, offset in windows:
            start_samp = int(raw.time_as_index(onset)[0]) + raw.first_samp
            end_samp = int(raw.time_as_index(offset)[0]) + raw.first_samp
            t = start_samp
            while t + step <= end_samp:
                event_rows.append([t, 0, 1])
                t += step

        if len(event_rows) < min_epochs:
            continue

        events = np.array(event_rows, dtype=int)
        cond_epochs = _mne.Epochs(
            raw, events, event_id={"seg": 1}, tmin=0.0, tmax=tmax,
            baseline=None, preload=True, verbose="ERROR", reject_by_annotation=False,
        )
        if len(cond_epochs) < min_epochs:
            continue

        # Patch channel positions for synthetic data
        _locs = np.array([ch["loc"][:3] for ch in cond_epochs.info["chs"]])
        if np.allclose(_locs, 0) or not np.all(np.isfinite(_locs)):
            _n = len(cond_epochs.ch_names)
            _angles = np.linspace(0, 2 * np.pi, _n, endpoint=False)
            cond_epochs = cond_epochs.copy()
            with cond_epochs.info._unlock():
                for _i, _ch in enumerate(cond_epochs.info["chs"]):
                    _a = _angles[_i]
                    _ch["loc"][:3] = [np.cos(_a) * 0.09, np.sin(_a) * 0.09, 0.01]

        # Chunk if condition is long
        n_total = len(cond_epochs)
        if n_total <= n_per_chunk:
            chunks = [(cond_epochs, "")]
        else:
            n_chunks = int(np.ceil(n_total / n_per_chunk))
            chunks = []
            for ci in range(n_chunks):
                s = ci * n_per_chunk
                e = min((ci + 1) * n_per_chunk, n_total)
                chunk = cond_epochs[s:e]
                if len(chunk) >= 1:
                    chunks.append((chunk, f"_chunk{ci + 1}"))

        chunk_labels: list[np.ndarray] = []
        chunk_bad_epochs: list[np.ndarray] = []
        chunk_ch_names: list[str] | None = None
        cond_n_epochs = 0
        cond_n_bad = 0

        for epochs_chunk, _ in chunks:
            cv = min(10, len(epochs_chunk))
            ar = AutoReject(n_interpolate=n_interp, random_state=42, n_jobs=n_jobs, verbose=False, cv=cv)
            ar.fit(epochs_chunk)
            reject_log = ar.get_reject_log(epochs_chunk)

            chunk_labels.append(np.asarray(reject_log.labels))
            chunk_bad_epochs.append(np.asarray(reject_log.bad_epochs))
            if chunk_ch_names is None:
                chunk_ch_names = reject_log.ch_names

            cond_n_epochs += len(epochs_chunk)
            cond_n_bad += int(np.sum(reject_log.bad_epochs))

            # Whole-epoch bad annotations
            for ep_idx, is_bad in enumerate(reject_log.bad_epochs):
                if not is_bad:
                    continue
                onset = float(epochs_chunk.events[ep_idx, 0] - raw.first_samp) / sfreq
                all_new_annots.append((max(0.0, onset), segment_duration, f"BAD_epoch_{cond_name}", ()))

            # Per-channel span annotations (consecutive bad-channel runs)
            labels = np.asarray(reject_log.labels)
            if labels.ndim == 2 and labels.shape[0] == len(epochs_chunk):
                for ch_idx, ch_name in enumerate(epochs_chunk.ch_names):
                    bad_idx = np.flatnonzero(labels[:, ch_idx] != 0)
                    for first_idx, last_idx in _group_consecutive_indices(bad_idx):
                        start_s = float(epochs_chunk.events[first_idx, 0] - raw.first_samp) / sfreq
                        end_s = float(epochs_chunk.events[last_idx, 0] - raw.first_samp) / sfreq + segment_duration
                        all_new_annots.append((max(0.0, start_s), max(end_s - start_s, segment_duration), f"BAD_{cond_name}", (ch_name,)))

        if cond_n_epochs > 0:
            condition_stats[cond_name] = {
                "n_epochs": cond_n_epochs,
                "n_bad_epochs": cond_n_bad,
                "clean_fraction": round((cond_n_epochs - cond_n_bad) / cond_n_epochs, 4),
            }

        # Build combined reject-log plot for this condition
        if chunk_labels and chunk_ch_names is not None:
            try:
                from autoreject import RejectLog as _RejectLog
                combined_log = _RejectLog(
                    bad_epochs=np.concatenate(chunk_bad_epochs),
                    labels=np.concatenate(chunk_labels, axis=0),
                    ch_names=chunk_ch_names,
                )
                fig = combined_log.plot(orientation="horizontal", show=False)
                fig.set_size_inches(16, 10)
                fig.suptitle(f"AutoReject — {cond_name}", y=1.01)
                condition_plots[cond_name] = fig
            except Exception:
                pass  # plots are optional; don't break the pipeline

    if all_new_annots:
        all_new_annots.sort(key=lambda x: x[0])
        raw.set_annotations(raw.annotations + _mne.Annotations(
            onset=[a[0] for a in all_new_annots],
            duration=[a[1] for a in all_new_annots],
            description=[a[2] for a in all_new_annots],
            ch_names=[a[3] for a in all_new_annots],
        ))

    total_epochs = sum(s["n_epochs"] for s in condition_stats.values())
    total_bad = sum(s["n_bad_epochs"] for s in condition_stats.values())
    provenance = {
        "bad_channels": bad_channels_prov,
        "conditions": condition_stats,
        "overall_clean_fraction": round((total_epochs - total_bad) / total_epochs, 4) if total_epochs else None,
    }

    def _fig_writer(path: str, fig: Any) -> None:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass

    def _json_writer(path: str, data: dict) -> None:
        import json
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    artifacts: dict[str, Artifact] = {
        ".fif": Artifact(item=raw, writer=lambda path, r=raw: r.save(path, overwrite=True, verbose="ERROR")),
        "_prov.json": Artifact(item=provenance, writer=lambda path, d=provenance: _json_writer(path, d)),
    }
    for cond_name, fig in condition_plots.items():
        artifacts[f"_ar_plot_{cond_name}.png"] = Artifact(
            item=fig,
            writer=lambda path, f=fig: _fig_writer(path, f),
        )

    return NodeResult(artifacts=artifacts)


@register_node
def epoch_fixed_length(
    mne_object,
    duration: float = 2.0,
    overlap: float = 0.0,
    reject_by_annotation: str | None = None,
) -> NodeResult:
    """Create fixed-length Epochs from Raw.

    Thin wrapper around mne.make_fixed_length_epochs. Used to extract
    CleanedPrep epochs from the annotated CleanedPrepRaw.
    """
    import mne as _mne
    from neurodags.loaders import load_meeg

    if isinstance(mne_object, NodeResult):
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = _mne.io.read_raw_fif(str(mne_object), preload=True, verbose="ERROR")

    raw = mne_object.copy().load_data()
    epochs = _mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose="ERROR",
        reject_by_annotation=reject_by_annotation or False,
    )

    return NodeResult(artifacts={
        ".fif": Artifact(item=epochs, writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"))
    })

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

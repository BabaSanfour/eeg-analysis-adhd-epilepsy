"""AutoReject nodes: epoch-based artifact rejection and fixed-length epoching."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


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


def _patch_channel_positions(epochs):
    """Patch zero/invalid channel positions for synthetic data (AutoReject requirement)."""
    locs = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])
    if np.allclose(locs, 0) or not np.all(np.isfinite(locs)):
        n = len(epochs.ch_names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        epochs = epochs.copy()
        with epochs.info._unlock():
            for i, ch in enumerate(epochs.info["chs"]):
                a = angles[i]
                ch["loc"][:3] = [np.cos(a) * 0.09, np.sin(a) * 0.09, 0.01]
    return epochs


@register_node
def epoch_fixed_length(
    mne_object,
    duration: float = 2.0,
    overlap: float = 0.0,
    reject_by_annotation: str | None = None,
) -> NodeResult:
    """Create fixed-length Epochs from Raw."""
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
def autoreject_annotate(
    mne_object,
    segment_duration: float = 1.0,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
) -> NodeResult:
    """Run AutoReject on fixed-length segments and add BAD_ annotations.

    Operates on the whole recording (not condition-aware).
    Outputs Epochs with bad segments omitted.
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

    seg_epochs = _mne.make_fixed_length_epochs(raw, duration=segment_duration, preload=True, verbose="ERROR")
    if len(seg_epochs) < min_epochs:
        epochs = _mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=epoch_overlap, preload=True, verbose="ERROR")
        return NodeResult(
            artifacts={
                ".fif": Artifact(item=epochs, writer=lambda path, e=epochs: e.save(path, overwrite=True, verbose="ERROR"))
            }
        )

    seg_epochs = _patch_channel_positions(seg_epochs)

    ar = AutoReject(n_interpolate=n_interp, random_state=42, n_jobs=1, verbose=False, cv=cv)
    ar.fit(seg_epochs)
    reject_log = ar.get_reject_log(seg_epochs)

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

    When condition_name is set, AR runs only on 1s segments within
    BLOCK_{condition_name} windows. When None, runs on the whole recording.
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

    seg_epochs = _patch_channel_positions(seg_epochs)

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
def autoreject_annotate_blockwise(
    mne_object,
    annotation_prefix: str = "BLOCK_",
    segment_duration: float = 1.0,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
    ar_max_chunk_minutes: float = 30.0,
    n_jobs: int = 1,
) -> NodeResult:
    """Condition-grouped AutoReject on Raw.

    Finds all unique BLOCK_* conditions, runs one AR instance per condition
    (chunked if > ar_max_chunk_minutes). Adds BAD_epoch_{condition} (whole-epoch)
    and BAD_{condition} (per-channel span) annotations. Returns annotated Raw.
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

    bad_channels_prov = list(raw.info["bads"])
    all_new_annots: list[tuple[float, float, str, tuple]] = []
    condition_plots: dict[str, Any] = {}
    condition_stats: dict[str, dict] = {}
    global_bad_epochs = 0
    global_bad_channel_spans = 0

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

        cond_epochs = _patch_channel_positions(cond_epochs)

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
        cond_bad_spans = 0

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

            for ep_idx, is_bad in enumerate(reject_log.bad_epochs):
                if not is_bad:
                    continue
                onset = float(epochs_chunk.events[ep_idx, 0] - raw.first_samp) / sfreq
                all_new_annots.append((max(0.0, onset), segment_duration, f"BAD_epoch_{cond_name}", ()))

            labels = np.asarray(reject_log.labels)
            if labels.ndim == 2 and labels.shape[0] == len(epochs_chunk):
                for ch_idx, ch_name in enumerate(epochs_chunk.ch_names):
                    bad_idx = np.flatnonzero(labels[:, ch_idx] != 0)
                    for first_idx, last_idx in _group_consecutive_indices(bad_idx):
                        start_s = float(epochs_chunk.events[first_idx, 0] - raw.first_samp) / sfreq
                        end_s = float(epochs_chunk.events[last_idx, 0] - raw.first_samp) / sfreq + segment_duration
                        all_new_annots.append((max(0.0, start_s), max(end_s - start_s, segment_duration), f"BAD_{cond_name}", (ch_name,)))
                        cond_bad_spans += 1

        global_bad_epochs += cond_n_bad
        global_bad_channel_spans += cond_bad_spans

        if cond_n_epochs > 0:
            condition_stats[cond_name] = {
                "n_windows": len(windows),
                "n_epochs": cond_n_epochs,
                "n_bad_epochs": cond_n_bad,
                "n_bad_channel_spans": cond_bad_spans,
                "chunks_processed": len(chunks),
                "clean_fraction": round((cond_n_epochs - cond_n_bad) / cond_n_epochs, 4),
            }

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
                pass

    if all_new_annots:
        all_new_annots.sort(key=lambda x: x[0])
        raw.set_annotations(raw.annotations + _mne.Annotations(
            onset=[a[0] for a in all_new_annots],
            duration=[a[1] for a in all_new_annots],
            description=[a[2] for a in all_new_annots],
            ch_names=[a[3] for a in all_new_annots],
        ))

    total_samples = raw.n_times
    mask_manual = np.zeros(total_samples, dtype=bool)
    mask_autoreject = np.zeros(total_samples, dtype=bool)
    mask_all_bad = np.zeros(total_samples, dtype=bool)
    for annot in raw.annotations:
        desc = str(annot["description"])
        if not desc.startswith("BAD_"):
            continue
        if annot.get("ch_names"):
            continue
        i0 = max(0, int(raw.time_as_index(annot["onset"])[0]))
        i1 = min(total_samples, int(raw.time_as_index(annot["onset"] + annot["duration"])[0]))
        if i1 > i0:
            mask_all_bad[i0:i1] = True
            if desc.startswith("BAD_epoch_"):
                mask_autoreject[i0:i1] = True
            elif not desc.startswith(("BAD_ACQ_SKIP", "BAD_boundary")):
                mask_manual[i0:i1] = True

    clean_samples = int(total_samples - mask_all_bad.sum())
    total_epochs = sum(s["n_epochs"] for s in condition_stats.values())
    total_bad = sum(s["n_bad_epochs"] for s in condition_stats.values())
    provenance = {
        "bad_channels": bad_channels_prov,
        "config": {
            "annotation_prefix": annotation_prefix,
            "segment_duration": segment_duration,
            "n_interpolate": n_interp.tolist(),
            "min_epochs": min_epochs,
            "ar_max_chunk_minutes": ar_max_chunk_minutes,
            "n_jobs": n_jobs,
        },
        "artifact_stats": {
            "bad_epochs": global_bad_epochs,
            "bad_channel_spans": global_bad_channel_spans,
            "artifacts_count": len(all_new_annots),
            "by_block": [
                {"condition": cond, **stats}
                for cond, stats in condition_stats.items()
            ],
        },
        "integrity_stats": {
            "clean_duration_s": round(clean_samples / sfreq, 3),
            "clean_fraction": round(clean_samples / total_samples, 4) if total_samples else None,
            "manual_bad_fraction": round(float(mask_manual.sum()) / total_samples, 4) if total_samples else None,
            "autoreject_bad_fraction": round(float(mask_autoreject.sum()) / total_samples, 4) if total_samples else None,
        },
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
def autoreject_clean_epochs(
    mne_object,
    n_interpolate: list[int] | None = None,
    min_epochs: int = 5,
) -> NodeResult:
    """Run AutoReject on Epochs and return cleaned Epochs (bad epochs dropped).

    Use after extract_condition_epochs so AR thresholds are estimated per condition.
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

    epochs = _patch_channel_positions(epochs)

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

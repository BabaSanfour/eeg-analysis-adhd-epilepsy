#!/usr/bin/env python3
"""Extract reusable recording-level EEG foundation-model embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.decoding import (
    SignalMetadata,
    config_hash,
    get_foundation_model_spec,
)
from coco_pipe.decoding.foundation_models import (
    FoundationEmbeddingExtractor,
    check_capability,
    normalize_inclusive_endpoint,
)
from coco_pipe.io import (
    read_json,
    read_table,
    save_embedding_derivative,
)
from coco_pipe.utils import stable_hash
from mne_bids import BIDSPath

from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.analysis.utils.decoding import (
    foundation_provenance,
    require_conditions,
    require_models,
)
from eeg_adhd_epilepsy.analysis.utils.subject_resolution import (
    resolve_cohort_subjects,
    resolve_metadata_row,
)
from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    _sanitize_bids_token,
    add_recording_id,
    get_derivative_root,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], derivative_root: Path, *, shard_token: str = "full") -> Path:
    """Extract embeddings for the configured (sliced) subjects into one task shard.

    Writes artifacts + self-describing sidecars under the BIDS derivative tree, plus
    a per-subject ``_failures/<shard_token>.csv`` for the rows that leave no artifact
    (window-mismatch / unsupported / errored).
    """
    bids_root = Path(config["bids_root"]).expanduser()
    metadata_df = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )
    subject_col = config.get("subject_col", "study_id")
    subjects = resolve_cohort_subjects(metadata_df, subject_col, config.get("subjects"))
    conditions = require_conditions(config)
    models = require_models(config)
    cfg_hash = config_hash(config)

    records: list[dict[str, Any]] = []
    container_cache: dict[tuple[Any, ...], Any] = {}

    for condition in conditions:
        for model_cfg in models:
            model_key = str(model_cfg["model_key"])
            segment_duration = float(model_cfg["segment_duration"])
            overlap = float(model_cfg["overlap"])
            use_derivatives = bool(model_cfg["use_derivatives"])
            window_source = str(model_cfg["window_source"])
            spec = get_foundation_model_spec(model_key)
            provenance = foundation_provenance(model_cfg, spec, config_hash=cfg_hash)

            LOGGER.info("Processing %s for %s (segment_duration: %gs, source: %s)", model_key, condition, segment_duration, window_source)
            load_key = (condition, segment_duration, overlap, use_derivatives, window_source)
            raw = container_cache.get(load_key)
            if raw is None:
                raw = build_container(
                    bids_root=bids_root,
                    use_derivatives=use_derivatives,
                    subjects=subjects,
                    task=config["task"],
                    segment_duration=segment_duration,
                    overlap=overlap,
                    metadata_df=metadata_df,
                    subject_col=subject_col,
                    desc=config.get("desc", "base"),
                    condition=condition,
                    window_source=window_source,
                )
                container_cache[load_key] = raw

            container, window_reason = normalize_inclusive_endpoint(
                raw,
                segment_duration=segment_duration,
                expected_sfreq=float(spec.pretrained_sfreq),
                model_key=model_key,
                on_mismatch=str(model_cfg.get("window_mismatch_policy", "raise")),
            )
            if container is None:
                records.append(
                    {
                        **provenance,
                        "condition": condition,
                        "status": "skipped",
                        "reason": window_reason,
                    }
                )
                continue
            container = add_recording_id(container, subject_col)
            frame = container.observation_frame()
            sfreq = float(container.meta["sfreq"])
            channels = [str(value) for value in container.coords["channel"]]
            pooling = model_cfg.get("pooling", "mean")
            extractor: FoundationEmbeddingExtractor | None = None

            for recording_id in pd.unique(frame["recording_id"].astype(str)):
                indices = np.flatnonzero(frame["recording_id"].astype(str) == recording_id)
                recording = container.isel(obs=indices)
                row = frame.iloc[indices].reset_index(drop=True).iloc[0]
                # Per-recording manifest row: shared provenance + recording identity.
                base_metadata = {
                    **provenance,
                    "subject": row[subject_col],
                    "session": row["session"],
                    "run": row["run"],
                    "task": config["task"],
                    "condition": condition,
                    "recording_id": recording_id,
                    "input_data_type": "preprocessed_epoched_eeg",
                    "preprocessing_description": config.get("desc", "base"),
                    "preprocessing_provenance": {
                        "sfreq": container.meta.get("sfreq"),
                        "filter": container.meta.get("filter"),
                        "reference": container.meta.get("reference"),
                    },
                    "window_mismatch_policy": str(
                        model_cfg.get("window_mismatch_policy", "raise")
                    ),
                }
                artifact = BIDSPath(
                    subject=_sanitize_bids_token(row[subject_col], "subject"),
                    session=_sanitize_bids_token(row["session"], "session"),
                    task=_sanitize_bids_token(config["task"], "task"),
                    run=_sanitize_bids_token(row["run"], "run"),
                    description=_sanitize_bids_token(
                        f"{model_key}{condition}{pooling.title()}", "desc"
                    ),
                    suffix="embedding",
                    extension=".npz",
                    datatype="eeg",
                    root=derivative_root,
                    check=False,
                ).fpath

                # Resume: an existing artifact is reused only if its provenance matches.
                if artifact.exists() and not config.get("overwrite", False):
                    sidecar = artifact.with_suffix(".json")
                    existing_hash = (
                        read_json(sidecar).get("config_hash") if sidecar.exists() else None
                    )
                    if existing_hash != cfg_hash:
                        records.append(
                            {
                                **base_metadata,
                                "artifact_path": str(artifact),
                                "status": "failed",
                                "reason": (
                                    f"config_hash mismatch: existing={existing_hash!r}, "
                                    f"requested={cfg_hash!r}; use overwrite to replace."
                                ),
                            }
                        )
                        continue
                    records.append(
                        {
                            **base_metadata,
                            "artifact_path": str(artifact),
                            "status": "success",
                            "reason": "resumed",
                        }
                    )
                    LOGGER.info("Skipping %s for %s: artifact already exists (resumed)", recording_id, model_key)
                    continue

                capability = check_capability(
                    model_key,
                    train_mode="frozen",
                    sfreq=sfreq,
                    ch_names=channels,
                    n_times=int(recording.X.shape[-1]),
                    backend=model_cfg.get("backend", "auto"),
                    backend_kwargs=model_cfg.get("backend_kwargs", {}),
                )
                if capability.status != "available":
                    LOGGER.info("Skipping %s for %s: unsupported configuration (%s)", recording_id, model_key, capability.reason)
                    records.append(
                        {
                            **base_metadata,
                            **capability.to_dict(),
                            "artifact_path": str(artifact),
                            "status": "skipped",
                        }
                    )
                    continue

                try:
                    if extractor is None:
                        extractor = FoundationEmbeddingExtractor(
                            model_key,
                            backend=model_cfg.get("backend", "auto"),
                            device=model_cfg.get("device", config.get("device", "auto")),
                            pooling=pooling,
                            recording_pooling=model_cfg.get("recording_pooling", "mean"),
                            normalize_embeddings=bool(
                                model_cfg.get("normalize_embeddings", True)
                            ),
                            resample=bool(model_cfg.get("resample", True)),
                            backend_kwargs=model_cfg.get("backend_kwargs", {}),
                        )
                    n_times = recording.X.shape[-1]
                    starts = np.arange(len(recording.X), dtype=int) * n_times
                    result = extractor.extract(
                        recording.X,
                        signal_metadata=SignalMetadata(sfreq=sfreq, ch_names=channels),
                        window_start=starts,
                        window_stop=starts + n_times,
                        metadata=base_metadata,
                    )
                    save_embedding_derivative(
                        result, artifact, overwrite=bool(config.get("overwrite", False))
                    )
                    (artifact.parent / "_SUCCESS").write_text("", encoding="utf-8")
                    LOGGER.info("Successfully extracted %s for %s (shape: %s)", model_key, recording_id, result.X.shape)
                    records.append(
                        {**result.metadata, "artifact_path": str(artifact), "status": "success"}
                    )
                except (ValueError, TypeError, RuntimeError) as exc:
                    if "OutOfMemory" in type(exc).__name__ or "CUDA out of memory" in str(exc):
                        LOGGER.critical("GPU Out of Memory for model %s. Crashing job.", model_key)
                        raise
                        
                    LOGGER.exception(
                        "Embedding extraction failed for %s/%s", recording_id, model_key
                    )
                    records.append(
                        {
                            **base_metadata,
                            "artifact_path": str(artifact),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    failures = [record for record in records if record.get("status") != "success"]
    failures_path = derivative_root / "_failures" / f"{shard_token}.csv"
    if failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()
    return derivative_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract BIDS-compatible EEG foundation-model embeddings."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--bids_root", required=True, help="Path to BIDS dataset")
    parser.add_argument("--metadata", default=None, help="Path to metadata CSV")
    parser.add_argument("--derivative_root", type=str, default=None, help="Explicit path to write output derivatives")
    parser.add_argument(
        "--metadata_row",
        type=int,
        default=None,
        help=(
            "One-based metadata-CSV row to process a single subject "
            "(SLURM_ARRAY_TASK_ID)."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit subjects to process, overriding config['subjects'].",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_yaml_config(args.config)
    config["bids_root"] = args.bids_root
    if args.metadata:
        config["metadata"] = args.metadata

    if args.derivative_root:
        derivative_root = Path(args.derivative_root).expanduser()
    else:
        derivative_root = get_derivative_root(
            Path(config["bids_root"]).expanduser(), DerivativeStage.FOUNDATION_EMBEDDINGS
        )

    if args.metadata_row is not None and args.subjects:
        raise ValueError("--metadata_row and --subjects are mutually exclusive.")
    if args.metadata_row is not None and args.metadata_row < 1:
        raise ValueError("--metadata_row is one-based and must be >= 1.")

    if args.metadata_row is not None:
        if not config.get("metadata"):
            raise ValueError("--metadata_row requires config['metadata'].")
        meta_df = read_table(Path(config["metadata"]).expanduser(), sep=None)
        subject = resolve_metadata_row(
            meta_df, args.metadata_row, config.get("subject_col", "study_id")
        )
        if subject is None:
            return
        config["subjects"] = [subject]
        run(config, derivative_root, shard_token=f"row-{args.metadata_row:04d}")
    elif args.subjects:
        config["subjects"] = args.subjects
        token = stable_hash(sorted(str(value) for value in args.subjects), length=12)
        run(config, derivative_root, shard_token=f"subjects-{token}")
    else:
        run(config, derivative_root, shard_token="full")


if __name__ == "__main__":
    main()

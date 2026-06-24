#!/usr/bin/env python3
"""Extract reusable recording-level EEG foundation-model embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from coco_pipe.decoding import SignalMetadata
from coco_pipe.decoding.foundation_models import (
    FoundationEmbeddingExtractor,
    check_capability,
    normalize_inclusive_endpoint,
)
from coco_pipe.io import (
    read_json,
    read_table,
    save_embedding_derivative,
    write_embedding_dataset_description,
    write_embedding_manifest,
)
from coco_pipe.report import make_foundation_embedding_report

from eeg_adhd_epilepsy.analysis.utils.decoding import (
    config_hash,
    load_yaml_config,
    redact_sensitive,
    require_conditions,
    write_run_status,
)
from eeg_adhd_epilepsy.analysis.utils.foundation import (
    FoundationInputPlan,
    default_foundation_models,
    resolve_foundation_input_plan,
)
from eeg_adhd_epilepsy.io.bids import (
    add_recording_id,
    bids_session_label,
    bids_subject_label,
)
from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.io.report_paths import default_reports_root

LOGGER = logging.getLogger(__name__)


def _bids_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value or ""))
    return token or "unknown"


def _artifact_path(
    root: Path,
    row: pd.Series,
    *,
    task: str,
    condition: str,
    model_key: str,
    pooling: str,
) -> Path:
    subject = _bids_token(row.get("subject") or row.get("study_id"))
    session = _bids_token(row["session"])
    run = _bids_token(row.get("run") or "01")
    desc = _bids_token(f"{model_key}{condition}{pooling.title()}")
    filename = (
        f"{bids_subject_label(subject)}_{bids_session_label(session)}_"
        f"task-{_bids_token(task)}_run-{run}_desc-{desc}_embedding.npz"
    )
    return root / bids_subject_label(subject) / bids_session_label(session) / "eeg" / filename


def _record_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column in frame.columns:
        values = frame[column].dropna().unique()
        if len(values) == 1:
            value = values[0]
            metadata[str(column)] = value.item() if isinstance(value, np.generic) else value
    return metadata


def _preprocessing_provenance(
    container_meta: dict[str, Any],
    config: dict[str, Any],
    plan: FoundationInputPlan,
) -> dict[str, Any]:
    """Keep stable, scientifically relevant preprocessing provenance."""
    allowed = {
        "sfreq",
        "source",
        "source_path",
        "derivative_path",
        "preprocessing",
        "filter",
        "reference",
        "bad_channels",
        "units",
        "inclusive_endpoint_removed",
        "original_n_times",
        "normalized_n_times",
    }
    return {
        **{str(key): value for key, value in container_meta.items() if str(key) in allowed},
        "desc": config["desc"],
        "use_derivatives": plan.use_derivatives,
        "segment_duration": plan.segment_duration,
        "overlap": plan.overlap,
        "foundation_input_plan": plan.to_provenance(),
    }


def run(config: dict[str, Any]) -> Path:
    bids_root = Path(config["bids_root"]).expanduser()
    derivative_root = Path(
        config.get(
            "derivative_root",
            bids_root / "derivatives" / "eeg_foundation_embeddings",
        )
    ).expanduser()
    metadata_df = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )
    subject_col = config.get("subject_col", "study_id")
    subjects = config.get("subjects")
    if subjects is None and metadata_df is not None and subject_col in metadata_df:
        subjects = [
            f"{int(value):04d}"
            for value in pd.to_numeric(metadata_df[subject_col], errors="coerce").dropna().unique()
        ]
    conditions = require_conditions(config)
    session_col = str(config["session_col"])
    models = config.get("models", default_foundation_models())
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    extractor_cache: dict[tuple[Any, ...], FoundationEmbeddingExtractor] = {}
    cfg_hash = config_hash(config)

    container_cache: dict[tuple[Any, ...], Any] = {}
    for condition in conditions:
        for model_cfg in models:
            model_key = str(model_cfg["model_key"])
            plan = resolve_foundation_input_plan(config, model_cfg)
            if plan.skip_reason is not None:
                record = {
                    "condition": condition,
                    "model_key": model_key,
                    "status": "skipped",
                    "reason": plan.skip_reason,
                    **plan.to_provenance(),
                }
                records.append(record)
                failures.append(record)
                continue
            load_key = (
                condition,
                plan.use_derivatives,
                plan.segment_duration,
                plan.overlap,
                plan.window_source,
            )
            container = container_cache.get(load_key)
            if container is None:
                container = build_container(
                    bids_root=bids_root,
                    use_derivatives=plan.use_derivatives,
                    subjects=subjects,
                    task=config.get("task", "clinical"),
                    segment_duration=plan.segment_duration,
                    overlap=plan.overlap,
                    metadata_df=metadata_df,
                    subject_col=subject_col,
                    desc=config.get("desc", "base"),
                    condition=condition,
                    window_source=plan.window_source,
                )
                container, window_reason = normalize_inclusive_endpoint(
                    container,
                    segment_duration=plan.segment_duration,
                    expected_sfreq=plan.expected_sfreq,
                    model_key=plan.model_key,
                    on_mismatch=plan.window_mismatch_policy,
                )
                if container is None:
                    record = {
                        "condition": condition,
                        "model_key": model_key,
                        "status": "skipped",
                        "reason": window_reason,
                        **plan.to_provenance(),
                    }
                    records.append(record)
                    failures.append(record)
                    continue
                container = add_recording_id(container, subject_col)
                container_cache[load_key] = container
            frame = container.observation_frame()
            if session_col not in frame:
                raise ValueError(
                    f"Configured session_col={session_col!r} is absent from EEG metadata."
                )
            if session_col != "session":
                frame["session"] = frame[session_col]
            sfreq = float(container.meta.get("sfreq", config.get("sfreq", 200.0)))
            channels = [str(value) for value in container.coords["channel"]]
            for recording_id in pd.unique(frame["recording_id"].astype(str)):
                indices = np.flatnonzero(frame["recording_id"].astype(str) == recording_id)
                recording = container.isel(obs=indices)
                recording_frame = frame.iloc[indices].reset_index(drop=True)
                row = recording_frame.iloc[0]
                base_metadata = {
                    **_record_metadata(recording_frame),
                    "condition": condition,
                    "recording_id": recording_id,
                    "input_data_type": "preprocessed_epoched_eeg",
                    "preprocessing_description": config.get("desc", "base"),
                    "preprocessing_provenance": _preprocessing_provenance(
                        container.meta,
                        config,
                        plan,
                    ),
                    "config_hash": cfg_hash,
                    "window_source": plan.window_source,
                    "window_mismatch_policy": plan.window_mismatch_policy,
                }
                pooling = model_cfg.get("pooling", "mean")
                artifact = _artifact_path(
                    derivative_root,
                    row,
                    task=config.get("task", "clinical"),
                    condition=condition,
                    model_key=model_key,
                    pooling=pooling,
                )
                if artifact.exists() and not config.get("overwrite", False):
                    sidecar = artifact.with_suffix(".json")
                    existing_hash = None
                    if sidecar.exists():
                        existing_hash = read_json(sidecar).get("config_hash")
                    if existing_hash != cfg_hash:
                        raise RuntimeError(
                            f"Config hash mismatch for {artifact}: "
                            f"existing={existing_hash}, requested={cfg_hash}. "
                            "Use overwrite to replace it."
                        )
                    records.append(
                        {
                            **base_metadata,
                            "model_key": model_key,
                            "artifact_path": str(artifact),
                            "status": "success",
                            "reason": "resumed",
                        }
                    )
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
                    record = {
                        **base_metadata,
                        **capability.to_dict(),
                        "artifact_path": str(artifact),
                        "status": "skipped",
                    }
                    records.append(record)
                    failures.append(record)
                    continue
                try:
                    cache_key = (
                        model_key,
                        model_cfg.get("backend", "auto"),
                        model_cfg.get("device", config.get("device", "auto")),
                        pooling,
                        model_cfg.get("recording_pooling", "mean"),
                        bool(model_cfg.get("normalize_embeddings", True)),
                        bool(model_cfg.get("resample", True)),
                        tuple(channels),
                        sfreq,
                        json.dumps(
                            model_cfg.get("backend_kwargs", {}),
                            sort_keys=True,
                            default=str,
                        ),
                    )
                    extractor = extractor_cache.get(cache_key)
                    if extractor is None:
                        extractor = FoundationEmbeddingExtractor(
                            model_key,
                            backend=model_cfg.get("backend", "auto"),
                            device=model_cfg.get("device", config.get("device", "auto")),
                            pooling=pooling,
                            recording_pooling=model_cfg.get("recording_pooling", "mean"),
                            normalize_embeddings=bool(model_cfg.get("normalize_embeddings", True)),
                            resample=bool(model_cfg.get("resample", True)),
                            backend_kwargs=model_cfg.get("backend_kwargs", {}),
                        )
                        extractor_cache[cache_key] = extractor
                    n_times = recording.X.shape[-1]
                    starts = np.arange(len(recording.X), dtype=int) * n_times
                    result = extractor.extract(
                        recording.X,
                        signal_metadata=SignalMetadata(
                            sfreq=sfreq,
                            ch_names=channels,
                        ),
                        window_start=starts,
                        window_stop=starts + n_times,
                        metadata=base_metadata,
                    )
                    save_embedding_derivative(
                        result,
                        artifact,
                        overwrite=bool(config.get("overwrite", False)),
                    )
                    success_marker = artifact.parent / "_SUCCESS"
                    success_marker.write_text("", encoding="utf-8")
                    records.append(
                        {
                            **result.metadata,
                            "artifact_path": str(artifact),
                            "status": "success",
                        }
                    )
                except Exception as exc:
                    LOGGER.exception(
                        "Embedding extraction failed for %s/%s", recording_id, model_key
                    )
                    record = {
                        **base_metadata,
                        "model_key": model_key,
                        "artifact_path": str(artifact),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    records.append(record)
                    failures.append(record)

    derivative_root.mkdir(parents=True, exist_ok=True)
    (derivative_root / "config_used.yaml").write_text(
        yaml.safe_dump(redact_sensitive(config), sort_keys=False),
        encoding="utf-8",
    )
    write_embedding_manifest(derivative_root, records)
    write_embedding_dataset_description(
        derivative_root,
        name=config.get("dataset_name", "EEG foundation model embeddings"),
        bids_version=config.get("bids_version", "1.11.1"),
        generated_by=[
            {
                "Name": "coco-pipe",
                "Description": "EEG foundation-model embedding extraction",
            }
        ],
        source_datasets=[{"URL": str(bids_root)}],
    )
    pd.DataFrame(failures).to_csv(derivative_root / "failures.csv", index=False)
    successful = sum(record.get("status") == "success" for record in records)
    status = "SUCCESS" if successful and not failures else "PARTIAL" if successful else "FAILED"
    write_run_status(derivative_root, status)

    reports_root = Path(config.get("reports_root", default_reports_root(bids_root))).expanduser()
    report_dir = (
        reports_root
        / "summary"
        / "foundation_embeddings"
        / str(config.get("run_label", config.get("dataset_name", "default")))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    make_foundation_embedding_report(
        records,
        config=redact_sensitive(config),
        asset_urls=config.get("report_asset_urls", "inline"),
        output_path=str(report_dir / "dataset_summary.html"),
    )
    return derivative_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract BIDS-compatible EEG foundation-model embeddings."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(load_yaml_config(args.config))


if __name__ == "__main__":
    main()

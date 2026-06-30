#!/usr/bin/env python3
"""Aggregate foundation-embedding shards into combined tables, a manifest, and a report.

Mirrors ``merge_descriptors``: the array tasks in ``extract_foundation_embeddings``
write per-recording ``*_embedding.npz`` + self-describing sidecars (plus a per-subject
failures CSV under ``_failures/``). This step scans them and materializes one table
per (model, condition) at each :data:`~coco_pipe.io.AGGREGATION_LEVELS` granularity:

- ``combined/<model>_<condition>_epoch_embeddings.{parquet,csv}``     — 1 row / epoch
- ``combined/<model>_<condition>_recording_embeddings.{parquet,csv}`` — 1 row / recording
- ``combined/<model>_<condition>_subject_embeddings.{parquet,csv}``   — 1 row / subject

``epoch`` and ``recording`` come straight from the saved per-window / pooled arrays;
``subject`` is mean-pooled from a subject's epochs here at merge (no re-extraction).
Plus the run manifest, failures table, dataset description, run status, and HTML report.
Successful units are recovered by tree-scanning the sidecars; failed/skipped units come
from the per-subject failures CSVs (they leave no artifact to scan).
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.decoding import redact_sensitive, write_run_status
from coco_pipe.io import (
    combined_embedding_table_path,
    embedding_sidecar_path,
    load_embedding_derivatives,
    read_json,
    write_embedding_dataset_description,
    write_embedding_manifest,
)
from coco_pipe.report import make_foundation_embedding_report

from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    summary_report_dir,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)

_ID_COLUMNS = (
    "subject",
    "session",
    "run",
    "condition",
    "recording_id",
    "model_key",
    "window_index",
)


def _scan_artifacts(derivative_root: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    """Group ``*_embedding.npz`` paths by model and rebuild the success records."""
    by_model: dict[str, list[Path]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for npz_path in sorted(derivative_root.rglob("*_embedding.npz")):
        sidecar = embedding_sidecar_path(npz_path)
        if not sidecar.exists():
            continue
        metadata = read_json(sidecar)
        by_model[str(metadata.get("model_key", ""))].append(npz_path)
        records.append({**metadata, "artifact_path": str(npz_path), "status": "success"})
    return by_model, records


def _failure_records(derivative_root: Path) -> list[dict[str, Any]]:
    """Concatenate every task's per-subject failures CSV (the non-success rows)."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted((derivative_root / "_failures").glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _embedding_frame(container: Any) -> pd.DataFrame:
    """Flatten a loaded embedding container to id columns + ``embedding_*`` features."""
    metadata = container.observation_frame().reset_index(drop=True)
    features = pd.DataFrame(
        np.asarray(container.X),
        columns=[str(name) for name in container.coords["feature"]],
    )
    id_columns = [column for column in _ID_COLUMNS if column in metadata.columns]
    return pd.concat([metadata[id_columns], features], axis=1)


def _subject_frame(epoch_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean-pool a subject's epoch embeddings into one row per (condition, subject)."""
    if not {"subject", "condition"}.issubset(epoch_frame.columns):
        return pd.DataFrame()
    feature_cols = [c for c in epoch_frame.columns if str(c).startswith("embedding_")]
    keys = ["condition", "subject"]
    subject_df = epoch_frame.groupby(keys, as_index=False)[feature_cols].mean()
    if "model_key" in epoch_frame.columns:
        model_keys = epoch_frame.groupby(keys, as_index=False)["model_key"].first()
        subject_df = subject_df.merge(model_keys, on=keys)
    return subject_df


def _write_condition_tables(
    derivative_root: Path, model_key: str, representation: str, frame: pd.DataFrame
) -> None:
    """Split *frame* by condition and write one combined table per condition.

    Paths come from ``combined_embedding_table_path`` — the same helper the
    dim-reduction loader (:func:`load_combined_embedding_table`) reads back, so the
    filename convention has a single source of truth shared by writer and reader.
    """
    if frame.empty or "condition" not in frame.columns:
        return
    for condition, condition_frame in frame.groupby("condition"):
        parquet_path = combined_embedding_table_path(
            derivative_root, model_key, str(condition), representation
        )
        condition_frame.to_parquet(parquet_path, index=False)
        condition_frame.to_csv(parquet_path.with_suffix(".csv"), index=False)


def _write_combined_tables(derivative_root: Path, by_model: dict[str, list[Path]]) -> None:
    """Materialize per (model, condition) epoch/recording/subject feature tables.

    ``epoch`` and ``recording`` are the saved per-window and pooled arrays;
    ``subject`` is mean-pooled from the epoch rows here (no re-extraction needed).
    """
    (derivative_root / "combined").mkdir(parents=True, exist_ok=True)
    for model_key, paths in by_model.items():
        epoch_frame = _embedding_frame(
            load_embedding_derivatives(paths, representation="epoch", model_key=model_key)
        )
        recording_frame = _embedding_frame(
            load_embedding_derivatives(paths, representation="recording", model_key=model_key)
        )
        _write_condition_tables(derivative_root, model_key, "epoch", epoch_frame)
        _write_condition_tables(derivative_root, model_key, "recording", recording_frame)
        _write_condition_tables(derivative_root, model_key, "subject", _subject_frame(epoch_frame))


def run(config: dict[str, Any]) -> Path:
    bids_root = Path(config["bids_root"]).expanduser()
    if config.get("derivative_root"):
        derivative_root = Path(config["derivative_root"]).expanduser()
    else:
        derivative_root = get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)

    if config.get("reports_root"):
        reports_root = Path(config["reports_root"]).expanduser()
    else:
        reports_root = Path(default_reports_root(bids_root)).expanduser()

    by_model, success_records = _scan_artifacts(derivative_root)
    failures = _failure_records(derivative_root)
    records = success_records + failures

    derivative_root.mkdir(parents=True, exist_ok=True)
    write_embedding_manifest(derivative_root, records)
    write_embedding_dataset_description(
        derivative_root,
        name=str(config["dataset_name"]),
        bids_version=config.get("bids_version", "1.11.1"),
        generated_by=[
            {"Name": "coco-pipe", "Description": "EEG foundation-model embedding extraction"}
        ],
        source_datasets=[{"URL": str(bids_root)}],
    )
    pd.DataFrame(failures).to_csv(derivative_root / "failures.csv", index=False)
    _write_combined_tables(derivative_root, by_model)

    status = (
        "SUCCESS"
        if success_records and not failures
        else "PARTIAL"
        if success_records
        else "FAILED"
    )
    write_run_status(derivative_root, status)

    report_dir = summary_report_dir(reports_root, ReportStage.FOUNDATION_EMBEDDINGS) / str(
        config["dataset_name"]
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
        description="Aggregate EEG foundation-embedding shards into combined tables and a report."
    )
    parser.add_argument("--bids_root", required=True, help="Path to BIDS dataset")
    parser.add_argument(
        "--derivative_root",
        type=str,
        default=None,
        help="Explicit path to the extraction derivatives",
    )
    parser.add_argument(
        "--reports_root", type=str, default=None, help="Explicit path to write output reports"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    bids_root = Path(args.bids_root).expanduser()
    if args.derivative_root:
        derivative_root = Path(args.derivative_root).expanduser()
    else:
        derivative_root = get_derivative_root(bids_root, DerivativeStage.FOUNDATION_EMBEDDINGS)

    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config_used.yaml not found under {derivative_root}; run "
            "extract_foundation_embeddings first."
        )
    config = load_yaml_config(config_path)
    config["bids_root"] = args.bids_root
    config["derivative_root"] = str(derivative_root)
    if args.reports_root:
        config["reports_root"] = args.reports_root
    run(config)


if __name__ == "__main__":
    main()

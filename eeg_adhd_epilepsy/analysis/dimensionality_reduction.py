#!/usr/bin/env python3
"""Checkpointed dimensionality-reduction analysis for EEG data."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding
from coco_pipe.io.structures import DataContainer

from eeg_adhd_epilepsy.io.analysis import concat_containers, load_container
from eeg_adhd_epilepsy.io.bids import get_reports_root, get_stage_summary_dir, validate_bids_coverage
from eeg_adhd_epilepsy.io.table import load
from eeg_adhd_epilepsy.reports.dim_reduction import (
    SEPARATION_METRIC_KEY,
    generate_dataset_report,
    load_fit_artifact,
    load_fit_runs,
)
from eeg_adhd_epilepsy.utils.config import DEFAULT_ANALYSIS_CONDITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap"]
EXTENDED_REDUCERS = ["PCA", "UMAP", "PHATE", "Isomap", "Pacmap", "Trimap", "LLE", "TSNE"]
DEFAULT_CONDITIONS = list(DEFAULT_ANALYSIS_CONDITIONS)
DEFAULT_N_COMPONENTS_SWEEP = [2, 3, 5, 10, 20, 50, 75, 100]
FIT_METRIC_COLUMNS = [
    "trustworthiness",
    "continuity",
    "lcmc",
    "shepard_correlation",
    "mrre_intrusion",
    "mrre_extrusion",
    "mrre_total",
]
EVAL_METRIC_COLUMNS = [SEPARATION_METRIC_KEY]
POOLED_CONDITION = "pooled_all"
FIT_RUN_KEY_FIELDS = ("fit_id",)
EVAL_RUN_KEY_FIELDS = ("fit_id", "eval_name", "target_col", "group_col")
DEFAULT_EVAL_GROUP_COL = "patient_group_id"


def save_fit_artifact(
    path: Path,
    embedding: np.ndarray,
    ids: np.ndarray,
    fit_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embedding.npy", np.asarray(embedding))
    np.save(path / "ids.npy", np.asarray(ids, dtype=object))
    (path / "fit.json").write_text(json.dumps(fit_payload, indent=2), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    np.savez_compressed(path / "diagnostics.npz", payload=np.asarray([diagnostics], dtype=object))
    (path / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def save_eval_artifact(path: Path, eval_payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    (path / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def update_runs(path: Path, record: dict[str, Any], key_fields: Sequence[str]) -> None:
    if path.exists():
        runs = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(runs, list):
            raise ValueError(f"Expected list payload in {path}.")
    else:
        runs = []

    key = tuple(record.get(field) for field in key_fields)
    for idx, existing in enumerate(runs):
        if tuple(existing.get(field) for field in key_fields) == key:
            runs[idx] = dict(record)
            break
    else:
        runs.append(dict(record))

    sort_fields = list(key_fields) + [
        "scope",
        "condition",
        "analysis_mode",
        "family",
        "unit_name",
        "reducer",
        "n_components",
    ]
    runs.sort(key=lambda item: tuple(str(item.get(field, "")) for field in sort_fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def _build_result_record(
    payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metric_columns: Sequence[str],
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in dict(payload).items()
        if key not in {"metrics", "records", "metadata", "artifacts"}
    }
    record["artifact_path"] = str(artifact_path.relative_to(output_root))
    for metric_name in metric_columns:
        value = None if metrics_payload is None else metrics_payload.get(metric_name)
        record[metric_name] = np.nan if value is None else float(value)
    if error is not None:
        record["status"] = "failed"
        record["error"] = error
    else:
        record["status"] = "success"
    return record


def _build_fit_record(
    fit_payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return _build_result_record(
        fit_payload, artifact_path, output_root, FIT_METRIC_COLUMNS, metrics_payload, error
    )


def _build_eval_record(
    eval_payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return _build_result_record(
        eval_payload, artifact_path, output_root, EVAL_METRIC_COLUMNS, metrics_payload, error
    )


def iter_analysis_units(args, container: DataContainer) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    def _add_unit(u_type, u_name, u_key, u_family, u_container):
        u_container.meta = {
            **dict(u_container.meta),
            "unit_type": u_type,
            "unit_name": u_name,
            "unit_key": u_key,
            "family": u_family,
        }
        units.append(
            {
                "unit_type": u_type,
                "unit_name": u_name,
                "unit_key": u_key,
                "family": u_family,
                "container": u_container,
            }
        )

    if args.analysis_mode == "flat":
        _add_unit("global", "all", "all", None, container)
        return units

    if args.analysis_mode == "sensor":
        if args.input_mode == "raw":
            for idx, channel_name in enumerate(np.asarray(container.coords["channel"], dtype=object)):
                _add_unit("sensor", str(channel_name), str(channel_name), None, container.isel(channel=idx).flatten(preserve="obs"))
            return units

        feature_families = np.asarray(container.coords["feature_family"], dtype=object)
        allowed_families = set(args.descriptor_families or feature_families.tolist())
        feature_mask = np.isin(feature_families.astype(str), list(allowed_families))
        if not feature_mask.any():
            raise RuntimeError("No descriptor features matched the requested families.")
        feature_indices = np.flatnonzero(feature_mask).tolist()
        for idx, sensor_name in enumerate(np.asarray(container.coords["sensor"], dtype=object)):
            _add_unit("sensor", str(sensor_name), str(sensor_name), None, container.isel(sensor=idx, feature=feature_indices).flatten(preserve="obs"))
        return units

    if args.input_mode != "descriptors":
        raise ValueError(f"analysis_mode='{args.analysis_mode}' is only supported for descriptor inputs.")

    sensor_names = np.asarray(container.coords["sensor"], dtype=object).astype(str)
    feature_families = np.asarray(container.coords["feature_family"], dtype=object).astype(str)
    wanted_families = list(dict.fromkeys(args.descriptor_families or feature_families.tolist()))

    if args.analysis_mode == "family":
        for family in wanted_families:
            feature_indices = np.flatnonzero(feature_families == family).tolist()
            if not feature_indices:
                continue
            _add_unit("family", family, family, family, container.isel(feature=feature_indices).flatten(preserve="obs"))
        return units

    if args.analysis_mode == "sensor_within_family":
        for family in wanted_families:
            feature_indices = np.flatnonzero(feature_families == family).tolist()
            if not feature_indices:
                continue
            for idx, sensor_name in enumerate(sensor_names):
                _add_unit("sensor", str(sensor_name), f"{family}_{sensor_name}", family, container.isel(sensor=idx, feature=feature_indices).flatten(preserve="obs"))
        return units

    raise ValueError(f"Unsupported analysis_mode '{args.analysis_mode}'.")


def parse_eval_specs(raw_specs: Any, subject_col: str) -> list[dict[str, Any]]:
    if raw_specs is None:
        return []
    raw_specs = raw_specs.get("evals") if isinstance(raw_specs, dict) else raw_specs
    if not isinstance(raw_specs, list):
        raise ValueError("Expected eval specs to be a list or a mapping with an 'evals' key.")

    specs: list[dict[str, Any]] = []
    for idx, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Eval spec #{idx} must be a dictionary.")
        specs.append(
            {
                "name": str(raw_spec["name"]),
                "target_col": str(raw_spec["target_col"]),
                "group_col": str(raw_spec.get("group_col", DEFAULT_EVAL_GROUP_COL)),
                "filters": [
                    {
                        "column": str(item["column"]),
                        "values": [str(value) for value in item["values"]],
                    }
                    for item in raw_spec.get("filters", [])
                ],
                "label_map": {
                    str(key): str(value)
                    for key, value in (raw_spec.get("label_map") or {}).items()
                },
            }
        )
    return specs


def _prepare_eval_inputs(
    container: DataContainer,
    fit_ids: np.ndarray,
    eval_spec: dict[str, Any],
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    if container.ids is None:
        raise ValueError("Dim-reduction fit/eval expects container.ids to be present.")

    container_ids = np.asarray(container.ids, dtype=object).astype(str)
    frame = pd.DataFrame({"obs_id": container_ids})
    n_obs = len(container_ids)
    for key, values in container.coords.items():
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == n_obs and key != "feature":
            frame[key] = arr
    if container.y is not None and "y" not in frame.columns:
        frame["y"] = np.asarray(container.y)

    aligned_keys = []
    counts: dict[str, int] = {}
    for obs_id in np.asarray(fit_ids, dtype=object).astype(str):
        occurrence = counts.get(obs_id, 0)
        aligned_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1
    counts = {}
    frame_keys = []
    for obs_id in frame["obs_id"].astype(str):
        occurrence = counts.get(obs_id, 0)
        frame_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1
    frame["_obs_key"] = frame_keys

    aligned_frame = frame.drop_duplicates("_obs_key", keep="first").set_index("_obs_key")
    missing = [key for key in aligned_keys if key not in aligned_frame.index]
    if missing:
        raise RuntimeError("Saved fit ids could not be aligned to the current container.")
    aligned_frame = aligned_frame.loc[aligned_keys].reset_index(drop=True)

    for filter_spec in eval_spec["filters"]:
        column = filter_spec["column"]
        if column not in aligned_frame.columns:
            raise ValueError(f"Eval filter column '{column}' is not available.")
        values = {str(value) for value in filter_spec["values"]}
        aligned_frame = aligned_frame[aligned_frame[column].astype(str).isin(values)].copy()

    if eval_spec["target_col"] not in aligned_frame.columns:
        raise ValueError(f"Eval target column '{eval_spec['target_col']}' is not available.")
    if eval_spec["group_col"] not in aligned_frame.columns:
        raise ValueError(f"Eval group column '{eval_spec['group_col']}' is not available.")

    labels = aligned_frame[eval_spec["target_col"]].astype(str)
    if eval_spec["label_map"]:
        labels = labels.map(lambda value: eval_spec["label_map"].get(value, value))
    labels = labels.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    groups = aligned_frame[eval_spec["group_col"]].astype(str)
    valid_mask = labels.notna() & groups.notna()
    if not valid_mask.any():
        raise RuntimeError(f"Eval '{eval_spec['name']}' produced no valid samples.")

    selected_frame = aligned_frame.loc[valid_mask].copy()
    return (
        selected_frame.index,
        selected_frame["obs_id"].astype(str).to_numpy(),
        labels.loc[valid_mask].astype(str).to_numpy(),
        groups.loc[valid_mask].astype(str).to_numpy(),
    )


def run_fit(
    fit_payload: dict[str, Any],
    container: DataContainer,
    out_path: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    success_marker = out_path / "_SUCCESS"
    if success_marker.exists() and not overwrite:
        artifact = load_fit_artifact(out_path)
        return _build_fit_record(
            fit_payload=artifact["fit"],
            artifact_path=out_path,
            output_root=output_root,
            metrics_payload=artifact["metrics"],
        )

    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    X = np.asarray(container.X)
    if X.ndim != 2:
        raise ValueError("run_fit expects a 2D matrix.")
    if container.ids is None:
        raise ValueError("Dim-reduction fits expect container.ids to be present.")
    ids = np.asarray(container.ids, dtype=object).astype(str)

    reducer = DimReduction(method=fit_payload["reducer"], n_components=fit_payload["n_components"])
    embedding = reducer.fit_transform(X)
    score_payload = reducer.score(embedding, X=X)
    score_metrics = dict(reducer.get_metrics())
    metrics_payload = {
        metric_name: (
            None
            if np.isnan(score_metrics.get(metric_name, np.nan))
            else float(score_metrics.get(metric_name))
        )
        for metric_name in FIT_METRIC_COLUMNS
    }

    summary = reducer.get_summary()
    diagnostics = dict(summary.get("diagnostics") or {})
    diagnostics["score_payload"] = score_payload
    diagnostics["summary"] = summary
    try:
        components = reducer.get_components()
    except Exception:
        components = None
    if components is not None:
        diagnostics["components"] = components
    explained_variance = getattr(reducer.reducer, "explained_variance_ratio_", None)
    if explained_variance is not None:
        diagnostics["explained_variance_ratio"] = np.asarray(explained_variance)

    save_fit_artifact(out_path, embedding, ids, fit_payload, metrics_payload, diagnostics)
    return _build_fit_record(
        fit_payload=fit_payload,
        artifact_path=out_path,
        output_root=output_root,
        metrics_payload=metrics_payload,
    )


def run_eval(
    fit_payload: dict[str, Any],
    fit_artifact: dict[str, Any],
    container: DataContainer,
    eval_spec: dict[str, Any],
    out_path: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    success_marker = out_path / "_SUCCESS"
    if success_marker.exists() and not overwrite:
        eval_payload = json.loads((out_path / "eval.json").read_text(encoding="utf-8"))
        return _build_eval_record(
            eval_payload=eval_payload,
            artifact_path=out_path,
            output_root=output_root,
            metrics_payload=eval_payload.get("metrics"),
        )

    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    selected_index, selected_ids, labels, groups = _prepare_eval_inputs(
        container=container,
        fit_ids=np.asarray(fit_artifact["ids"], dtype=object).astype(str),
        eval_spec=eval_spec,
    )
    eval_id = hashlib.sha256(
        json.dumps(
            {
                "fit_id": fit_payload["fit_id"],
                "eval_name": eval_spec["name"],
                "target_col": eval_spec["target_col"],
                "group_col": eval_spec["group_col"],
                "filters": eval_spec["filters"],
                "label_map": eval_spec["label_map"],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    embedding = np.asarray(fit_artifact["embedding"])[selected_index.to_numpy()]
    if embedding.ndim != 2:
        raise ValueError("run_eval expects a 2D embedding artifact.")

    score_payload = evaluate_embedding(
        embedding,
        method_name=fit_payload["reducer"],
        metrics=EVAL_METRIC_COLUMNS,
        labels=labels,
        groups=groups,
    )
    metrics_payload = dict(score_payload["metrics"])
    eval_payload = {
        "eval_id": eval_id,
        "fit_id": fit_payload["fit_id"],
        "scope": fit_payload["scope"],
        "condition": fit_payload["condition"],
        "analysis_mode": fit_payload["analysis_mode"],
        "unit_type": fit_payload["unit_type"],
        "unit_name": fit_payload["unit_name"],
        "unit_key": fit_payload["unit_key"],
        "family": fit_payload.get("family"),
        "eval_name": eval_spec["name"],
        "input_mode": fit_payload["input_mode"],
        "representation": fit_payload["representation"],
        "reducer": fit_payload["reducer"],
        "n_components": int(fit_payload["n_components"]),
        "target_col": eval_spec["target_col"],
        "group_col": eval_spec["group_col"],
        "filters": list(eval_spec["filters"]),
        "label_map": dict(eval_spec["label_map"]),
        "descriptor_families": list(fit_payload.get("descriptor_families", [])),
        "descriptor_max_abs_value": fit_payload.get("descriptor_max_abs_value"),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(selected_ids)),
        "metrics": metrics_payload,
        "records": score_payload.get("records", []),
        "metadata": score_payload.get("metadata", {}),
        "artifacts": score_payload.get("artifacts", {}),
    }
    save_eval_artifact(out_path, eval_payload)
    return _build_eval_record(
        eval_payload=eval_payload,
        artifact_path=out_path,
        output_root=output_root,
        metrics_payload=metrics_payload,
    )


def _build_fit_task(
    args,
    scope: str,
    condition: str,
    unit_spec: dict[str, Any],
    reducer_name: str,
    n_components: int,
    output_root: Path,
) -> dict[str, Any]:
    container = unit_spec["container"]
    if container.ids is None:
        raise ValueError("Dim-reduction fits expect container.ids to be present.")
    ids = np.asarray(container.ids, dtype=object).astype(str)

    filter_specs = [
        {"column": str(column), "values": [str(value) for value in values]}
        for column, values in zip(args.filter_col, args.filter_val)
        if values
    ]
    input_signature = {
        "input_mode": args.input_mode,
        "representation": args.representation,
        "analysis_mode": args.analysis_mode,
        "descriptor_families": list(getattr(args, "descriptor_families", []) or []),
        "filters": filter_specs,
        "balance_target": args.balance_target,
        "balance_strategy": args.balance_strategy if args.balance_target else None,
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "family": unit_spec.get("family"),
    }
    if args.input_mode == "raw":
        input_signature.update(
            {
                "bids_root": str(Path(args.bids_root).expanduser()),
                "use_derivatives": bool(args.use_derivatives),
                "task": getattr(args, "task", "clinical"),
                "segment_duration": float(args.segment_duration),
                "overlap": float(args.overlap),
                "desc": args.desc,
            }
        )
    elif args.input_mode == "descriptors":
        input_signature.update(
            {
                "descriptor_table_path": str(Path(args.descriptor_table_path).expanduser()),
                "descriptor_feature_columns_path": str(
                    Path(args.descriptor_feature_columns_path).expanduser()
                ),
                "descriptor_max_abs_value": getattr(args, "descriptor_max_abs_value", None),
            }
        )
    else:
        input_signature.update(
            {
                "embeddings_root": str(Path(args.embeddings_root).expanduser()),
                "segments_root": str(Path(args.segments_root or args.bids_root).expanduser()),
                "embedding_model": args.embedding_model,
                "embedding_desc": args.embedding_desc,
                "embedding_min_overlap_fraction": float(args.embedding_min_overlap_fraction),
                "reve_segment_duration": float(args.reve_segment_duration),
                "cbramod_sampling_rate": float(args.cbramod_sampling_rate),
            }
        )

    sample_ids_sha256 = hashlib.sha256("\0".join(ids.tolist()).encode("utf-8")).hexdigest()[:16]
    fit_id = hashlib.sha256(
        json.dumps(
            {
                "scope": scope,
                "condition": condition,
                "analysis_mode": args.analysis_mode,
                "unit_type": unit_spec["unit_type"],
                "unit_name": unit_spec["unit_name"],
                "family": unit_spec.get("family"),
                "input_signature": input_signature,
                "reducer": reducer_name,
                "n_components": int(n_components),
                "sample_ids_sha256": sample_ids_sha256,
                "n_samples": int(len(ids)),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    fit_payload = {
        "fit_id": fit_id,
        "scope": scope,
        "condition": condition,
        "analysis_mode": args.analysis_mode,
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "unit_key": unit_spec["unit_key"],
        "family": unit_spec.get("family"),
        "input_mode": args.input_mode,
        "representation": args.representation,
        "descriptor_families": list(getattr(args, "descriptor_families", []) or []),
        "descriptor_max_abs_value": getattr(args, "descriptor_max_abs_value", None)
        if args.input_mode == "descriptors"
        else None,
        "reducer": reducer_name,
        "n_components": int(n_components),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(container.X.shape[0]),
        "n_subjects": int(
            pd.Index(np.asarray(container.coords.get(args.subject_col, ids), dtype=object).astype(str)).nunique()
        ),
        "loaded_obs": int(container.meta.get("loaded_obs", container.X.shape[0])),
        "samples_used": int(container.meta.get("samples_used", container.X.shape[0])),
        "input_signature": input_signature,
    }
    artifact_path = (
        output_root
        / "sub-all"
        / "ses-all"
        / "eeg"
        / "fits"
        / scope
        / condition
        / args.input_mode
        / args.analysis_mode
        / unit_spec["unit_type"]
        / unit_spec["unit_key"]
        / reducer_name
        / f"n{n_components}"
        / fit_id
    )
    return {
        "fit_payload": fit_payload,
        "container": container,
        "artifact_path": artifact_path,
        "output_root": output_root,
        "overwrite": bool(args.overwrite),
    }


def _execute_fit_task(task: dict[str, Any]) -> dict[str, Any]:
    fit_payload = task["fit_payload"]
    container = task["container"]
    artifact_path = task["artifact_path"]
    output_root = task["output_root"]
    overwrite = task["overwrite"]
    logger.info(
        "Fitting %s/%s/%s/%s/n%d",
        fit_payload["condition"],
        fit_payload["analysis_mode"],
        fit_payload["unit_name"],
        fit_payload["reducer"],
        fit_payload["n_components"],
    )
    try:
        return run_fit(
            fit_payload=fit_payload,
            container=container,
            out_path=artifact_path,
            output_root=output_root,
            overwrite=overwrite,
        )
    except Exception as err:
        logger.exception(
            "Fit failed for %s/%s/%s/n%d",
            fit_payload["condition"],
            fit_payload["unit_name"],
            fit_payload["reducer"],
            fit_payload["n_components"],
        )
        return _build_fit_record(
            fit_payload={**fit_payload, "status": "failed"},
            artifact_path=artifact_path,
            output_root=output_root,
            error=str(err),
        )


def _build_eval_task(
    fit_record: dict[str, Any],
    eval_spec: dict[str, Any],
    container: DataContainer,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    fit_path = output_root / fit_record["artifact_path"]
    fit_artifact = load_fit_artifact(fit_path)
    selected_ids: list[str] = []
    try:
        _, selected_ids_array, _, _ = _prepare_eval_inputs(
            container=container,
            fit_ids=np.asarray(fit_artifact["ids"], dtype=object).astype(str),
            eval_spec=eval_spec,
        )
        selected_ids = selected_ids_array.tolist()
    except Exception:
        selected_ids = []
    eval_id = hashlib.sha256(
        json.dumps(
            {
                "fit_id": fit_record["fit_id"],
                "eval_name": eval_spec["name"],
                "target_col": eval_spec["target_col"],
                "group_col": eval_spec["group_col"],
                "filters": eval_spec["filters"],
                "label_map": eval_spec["label_map"],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    eval_payload = {
        "eval_id": eval_id,
        "fit_id": fit_record["fit_id"],
        "scope": fit_record["scope"],
        "condition": fit_record["condition"],
        "analysis_mode": fit_record["analysis_mode"],
        "unit_type": fit_record["unit_type"],
        "unit_name": fit_record["unit_name"],
        "unit_key": fit_record["unit_key"],
        "family": fit_record.get("family"),
        "eval_name": eval_spec["name"],
        "input_mode": fit_record["input_mode"],
        "representation": fit_record["representation"],
        "reducer": fit_record["reducer"],
        "n_components": int(fit_record["n_components"]),
        "target_col": eval_spec["target_col"],
        "group_col": eval_spec["group_col"],
        "filters": list(eval_spec["filters"]),
        "label_map": dict(eval_spec["label_map"]),
        "descriptor_families": list(fit_record.get("descriptor_families", [])),
        "descriptor_max_abs_value": fit_record.get("descriptor_max_abs_value"),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(selected_ids)),
    }
    artifact_path = (
        output_root
        / "sub-all"
        / "ses-all"
        / "eeg"
        / "evals"
        / fit_record["fit_id"]
        / eval_spec["name"]
        / eval_id
    )
    return {
        "fit_record": fit_record,
        "fit_artifact": fit_artifact,
        "eval_spec": eval_spec,
        "container": container,
        "artifact_path": artifact_path,
        "output_root": output_root,
        "overwrite": bool(overwrite),
        "eval_payload": eval_payload,
    }


def _execute_eval_task(task: dict[str, Any]) -> dict[str, Any]:
    fit_record = task["fit_record"]
    fit_artifact = task["fit_artifact"]
    eval_spec = task["eval_spec"]
    container = task["container"]
    artifact_path = task["artifact_path"]
    output_root = task["output_root"]
    overwrite = task["overwrite"]
    eval_payload = task["eval_payload"]
    logger.info(
        "Evaluating %s/%s/%s/%s/n%d [%s]",
        fit_record["condition"],
        fit_record["analysis_mode"],
        fit_record["unit_name"],
        fit_record["reducer"],
        fit_record["n_components"],
        eval_spec["name"],
    )
    try:
        return run_eval(
            fit_payload=fit_artifact["fit"],
            fit_artifact=fit_artifact,
            container=container,
            eval_spec=eval_spec,
            out_path=artifact_path,
            output_root=output_root,
            overwrite=overwrite,
        )
    except Exception as err:
        logger.exception(
            "Eval failed for %s/%s/%s/%s/n%d [%s]",
            fit_record["condition"],
            fit_record["analysis_mode"],
            fit_record["unit_name"],
            fit_record["reducer"],
            fit_record["n_components"],
            eval_spec["name"],
        )
        return _build_eval_record(
            eval_payload={**eval_payload, "status": "failed"},
            artifact_path=artifact_path,
            output_root=output_root,
            error=str(err),
        )


def _resolve_subjects(args, bids_root: Path, meta_df: pd.DataFrame) -> list[str]:
    if args.subjects:
        subjects: list[str] = []
        for subject in args.subjects:
            value = str(subject).strip()
            if value.startswith("sub-"):
                value = value[4:]
            if value.isdigit():
                value = f"{int(value):04d}"
            subjects.append(value)
        return subjects
    if args.input_mode != "raw":
        return sorted(pd.Index(meta_df[args.subject_col]).astype(str).unique().tolist())

    coverage_root = bids_root / "derivatives" / "preproc" if args.use_derivatives else bids_root
    coverage_desc = args.desc if args.use_derivatives else ""
    coverage_suffix = "epo" if args.use_derivatives else None
    coverage = validate_bids_coverage(
        meta_df,
        coverage_root,
        desc=coverage_desc,
        suffix=coverage_suffix,
        subject_col=args.subject_col,
    )
    subjects = [str(subject) for subject in coverage["present_subjects"]]
    logger.info(
        "Resolved %d available subjects from %s.",
        len(subjects),
        "derivatives" if args.use_derivatives else "BIDS",
    )
    return subjects


def _build_auto_pooled_eval_spec(args) -> Optional[dict[str, Any]]:
    if not args.run_pooled or len(args.conditions) < 2:
        return None
    return {
        "name": "condition_separation",
        "target_col": "condition",
        "group_col": DEFAULT_EVAL_GROUP_COL,
        "filters": [],
        "label_map": {},
    }


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    return n_jobs


def _run_task_batch(
    tasks: Sequence[dict[str, Any]],
    worker_fn,
    max_workers: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if max_workers == 1:
        return [worker_fn(task) for task in tasks]
    return joblib.Parallel(n_jobs=min(max_workers, len(tasks)))(
        joblib.delayed(worker_fn)(task) for task in tasks
    )


def _write_run_status(
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    *,
    fatal_error: str | None = None,
    report_path: Path | None = None,
) -> None:
    fit_runs = json.loads(fit_runs_path.read_text(encoding="utf-8")) if fit_runs_path.exists() else []
    eval_runs = json.loads(eval_runs_path.read_text(encoding="utf-8")) if eval_runs_path.exists() else []

    fit_success = sum(record.get("status") == "success" for record in fit_runs)
    fit_failed = sum(record.get("status") == "failed" for record in fit_runs)
    eval_success = sum(record.get("status") == "success" for record in eval_runs)
    eval_failed = sum(record.get("status") == "failed" for record in eval_runs)
    any_success = (fit_success + eval_success) > 0
    any_failed = (fit_failed + eval_failed) > 0 or fatal_error is not None

    if any_failed and any_success:
        run_status = "partial"
    elif any_failed:
        run_status = "failed"
    elif any_success:
        run_status = "success"
    else:
        run_status = "failed" if fatal_error is not None else "partial"

    summary_payload = {
        "status": run_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_total": len(fit_runs),
        "fit_success": fit_success,
        "fit_failed": fit_failed,
        "eval_total": len(eval_runs),
        "eval_success": eval_success,
        "eval_failed": eval_failed,
        "report_path": str(report_path) if report_path is not None else None,
        "report_exists": bool(report_path is not None and report_path.exists()),
        "fatal_error": fatal_error,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    for marker_name in ("_RUN_SUCCESS", "_RUN_PARTIAL", "_RUN_FAILED"):
        marker_path = output_root / marker_name
        if marker_path.exists():
            marker_path.unlink()

    marker_name = {
        "success": "_RUN_SUCCESS",
        "partial": "_RUN_PARTIAL",
        "failed": "_RUN_FAILED",
    }[run_status]
    (output_root / marker_name).write_text("ok\n", encoding="utf-8")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    bootstrap_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Run checkpointed EEG dimensionality reduction.")
    parser.add_argument("--config", default=None, help="Path to dim-reduction YAML config.")

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--bids_root", default="/Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS")
    dataset_group.add_argument("--metadata", default=None, help="Path to canonical metadata CSV.")
    dataset_group.add_argument("--dataset_name", default=None, help="Name for this dim-reduction run namespace.")
    dataset_group.add_argument("--subject_col", default="study_id", help="Subject identifier column.")
    dataset_group.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process.")
    dataset_group.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=DEFAULT_CONDITIONS)

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--input_mode", choices=["raw", "descriptors", "embeddings"], default="raw")
    input_group.add_argument("--task", default="clinical")
    input_group.add_argument("--segment_duration", type=float, default=60.0)
    input_group.add_argument("--overlap", type=float, default=0.0)
    input_group.add_argument("--use_derivatives", action="store_true")
    input_group.add_argument("--desc", default="base")
    input_group.add_argument(
        "--representation",
        choices=[
            "epoch_native",
            "epoch_flat",
            "epoch_time_as_sample",
            "epoch_scalar_mean",
            "subject_native",
            "subject_flat",
            "subject_time_as_sample",
            "subject_scalar_mean",
        ],
        default="epoch_flat",
    )
    input_group.add_argument(
        "--analysis_mode",
        choices=["flat", "sensor", "family", "sensor_within_family"],
        default="flat",
    )
    input_group.add_argument("--descriptor_table_path", default=None)
    input_group.add_argument("--descriptor_feature_columns_path", default=None)
    input_group.add_argument("--descriptor_families", nargs="+", default=None)
    input_group.add_argument(
        "--descriptor_max_abs_value",
        type=float,
        default=1e12,
        help=(
            "For descriptor inputs, drop rows whose selected finite descriptor "
            "features exceed this absolute value threshold."
        ),
    )
    input_group.add_argument("--embeddings_root", default=None)
    input_group.add_argument("--segments_root", default=None)
    input_group.add_argument("--embedding_model", choices=["reve", "cbramod"], default="cbramod")
    input_group.add_argument("--embedding_desc", default="base")
    input_group.add_argument("--embedding_min_overlap_fraction", type=float, default=0.8)
    input_group.add_argument("--reve_segment_duration", type=float, default=10.0)
    input_group.add_argument("--cbramod_sampling_rate", type=float, default=200.0)
    input_group.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    reduction_group = parser.add_argument_group("Reduction")
    reduction_group.add_argument("--reducers", nargs="+", default=["default"])
    reduction_group.add_argument("--n_components_sweep", nargs="+", type=int, default=DEFAULT_N_COMPONENTS_SWEEP)
    reduction_group.add_argument("--run_pooled", action="store_true")
    reduction_group.add_argument("--overwrite", action="store_true")
    reduction_group.add_argument("--reports-only", action="store_true")
    reduction_group.add_argument("--eval_config", default=None)
    reduction_group.add_argument("--n_jobs", type=int, default=1)
    reduction_group.add_argument(
        "--output_group",
        default=None,
        help="Optional nested output group under derivatives/reports, e.g. 'medicated_adhd_vs_controls/lis'.",
    )

    filter_group = parser.add_argument_group("Filtering")
    filter_group.add_argument("--filter_col", action="append", default=[])
    filter_group.add_argument("--filter_val", action="append", nargs="+", default=[])
    filter_group.add_argument("--balance_target", default=None)
    filter_group.add_argument("--balance_strategy", choices=["undersample", "oversample", "auto"], default="undersample")

    report_group = parser.add_argument_group("Report")
    report_group.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
    report_group.add_argument("--save_static_figures", action="store_true")
    report_group.add_argument("--compress_viz_with_pca", action="store_true")
    report_group.add_argument(
        "--selection_metric",
        default=SEPARATION_METRIC_KEY,
        help="Metric used to select the best run in report summaries.",
    )
    report_group.add_argument(
        "--selection_eval_name",
        default=None,
        help="Eval name whose separation metric should drive report selection, e.g. 'med_adhd_vs_ctrl'.",
    )

    config_eval_specs = None
    if bootstrap_args.config:
        config_path = Path(bootstrap_args.config).expanduser()
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_config, dict):
            raise ValueError(f"Expected mapping payload in {config_path}.")
        config_eval_specs = raw_config.pop("evals", None)
        parser.set_defaults(**raw_config)
    args = parser.parse_args()

    if len(args.filter_col) != len(args.filter_val):
        raise ValueError("--filter_col and --filter_val must be provided in matching pairs.")
    if args.input_mode == "descriptors":
        if not args.descriptor_table_path or not args.descriptor_feature_columns_path:
            raise ValueError(
                "--descriptor_table_path and --descriptor_feature_columns_path are required "
                "when --input_mode descriptors."
            )
        if args.descriptor_max_abs_value is not None and args.descriptor_max_abs_value <= 0:
            raise ValueError("--descriptor_max_abs_value must be positive when provided.")
        args.representation = Path(args.descriptor_table_path).stem
    if args.input_mode == "embeddings" and not args.embeddings_root:
        raise ValueError("--embeddings_root is required when --input_mode embeddings.")
    if args.input_mode == "embeddings" and args.analysis_mode != "flat":
        raise ValueError("Embeddings currently support only analysis_mode='flat'.")
    if args.analysis_mode in {"family", "sensor_within_family"} and args.input_mode != "descriptors":
        raise ValueError(
            f"analysis_mode='{args.analysis_mode}' is only supported for descriptor inputs."
        )
    if args.input_mode == "raw" and args.analysis_mode == "sensor":
        if args.representation not in {"epoch_native", "subject_native"}:
            raise ValueError(
                "Raw sensor mode requires representation 'epoch_native' or 'subject_native'."
            )
    if args.input_mode == "raw" and args.analysis_mode != "flat":
        if args.analysis_mode != "sensor":
            raise ValueError("Raw inputs currently support only analysis_mode='flat' or 'sensor'.")
    if args.input_mode == "raw" and args.analysis_mode == "flat":
        if args.representation in {"epoch_native", "subject_native"}:
            raise ValueError(
                "Native EEG representations are reserved for sensor mode. "
                "Use --analysis_mode sensor with epoch_native or subject_native."
            )
    if args.descriptor_families and args.input_mode != "descriptors":
        raise ValueError("--descriptor_families is only supported for descriptor inputs.")
    if args.descriptor_families:
        invalid_families = [
            family
            for family in args.descriptor_families
            if family not in ("band", "complexity", "param")
        ]
        if invalid_families:
            raise ValueError(
                f"Unknown descriptor families: {invalid_families}. "
                f"Valid families: {['band', 'complexity', 'param']}"
            )
    resolved_n_jobs = _resolve_n_jobs(args.n_jobs)

    requested_reducers = [value.upper() for value in args.reducers]
    if requested_reducers == ["DEFAULT"]:
        reducers = DEFAULT_REDUCERS
    elif requested_reducers == ["EXTENDED"]:
        reducers = EXTENDED_REDUCERS
    else:
        valid_reducers = set(DEFAULT_REDUCERS + EXTENDED_REDUCERS)
        invalid_reducers = [value for value in args.reducers if value not in valid_reducers]
        if invalid_reducers:
            raise ValueError(f"Unknown reducers: {invalid_reducers}. Valid reducers: {sorted(valid_reducers)}")
        reducers = list(args.reducers)

    bids_root = Path(args.bids_root).expanduser()
    meta_df = load(str(Path(args.metadata)), sep=",")
    subjects = _resolve_subjects(args, bids_root, meta_df)
    raw_eval_specs = config_eval_specs
    if raw_eval_specs is None and args.eval_config:
        raw_eval_specs = yaml.safe_load(Path(args.eval_config).expanduser().read_text(encoding="utf-8")) or []
    eval_specs = parse_eval_specs(raw_eval_specs, args.subject_col)
    auto_pooled_eval_spec = _build_auto_pooled_eval_spec(args)
    if auto_pooled_eval_spec is not None and not any(spec["name"] == auto_pooled_eval_spec["name"] for spec in eval_specs):
        eval_specs = [*eval_specs, auto_pooled_eval_spec]

    valid_selection_metrics = { *FIT_METRIC_COLUMNS, SEPARATION_METRIC_KEY }
    eval_names = {spec["name"] for spec in eval_specs}
    if args.selection_metric in eval_names and not args.selection_eval_name:
        args.selection_eval_name = args.selection_metric
        args.selection_metric = SEPARATION_METRIC_KEY
    if args.selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Unknown selection_metric '{args.selection_metric}'. "
            f"Valid metrics: {sorted(valid_selection_metrics)}"
        )
    if args.selection_eval_name and args.selection_eval_name not in eval_names:
        raise ValueError(
            f"Unknown selection_eval_name '{args.selection_eval_name}'. "
            f"Valid eval names: {sorted(eval_names)}"
        )

    output_base = bids_root / "derivatives" / "dim_reduction"
    if args.output_group:
        output_group = Path(str(args.output_group))
        if output_group.is_absolute():
            raise ValueError("--output_group must be relative, not absolute.")
        output_base = output_base / output_group
    run_variant = f"{args.analysis_mode}__{args.representation}"
    output_root = output_base / args.dataset_name / args.input_mode / run_variant
    output_root.mkdir(parents=True, exist_ok=True)
    config_snapshot = {key: value for key, value in vars(args).items() if key not in {"config", "eval_config"}}
    if eval_specs:
        config_snapshot["evals"] = eval_specs
    (output_root / "config_used.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=True), encoding="utf-8")
    fit_runs_path = output_root / "dim_reduction_fit_runs.json"
    eval_runs_path = output_root / "dim_reduction_eval_runs.json"
    logger.info("Using %d outer worker(s) for fits/evals.", resolved_n_jobs)

    base_containers_by_scope: dict[tuple[str, str], DataContainer] = {}
    unit_containers_by_key: dict[tuple[str, str, str], DataContainer] = {}

    if args.reports_only:
        if not fit_runs_path.exists():
            raise FileNotFoundError(f"--reports-only requested but {fit_runs_path} does not exist.")
    else:
        fit_tasks: list[dict[str, Any]] = []
        for condition in args.conditions:
            logger.info("Loading input for condition '%s' (%s).", condition, args.input_mode)
            try:
                base_container = load_container(args, subjects, meta_df, condition, target_col=None)
            except Exception:
                logger.exception("Failed to load condition '%s'.", condition)
                continue

            base_containers_by_scope[("condition", condition)] = base_container
            for unit_spec in iter_analysis_units(args, base_container):
                unit_containers_by_key[("condition", condition, unit_spec["unit_key"])] = unit_spec["container"]
                for reducer_name, n_components in product(reducers, args.n_components_sweep):
                    fit_tasks.append(
                        _build_fit_task(
                            args=args,
                            scope="condition",
                            condition=condition,
                            unit_spec=unit_spec,
                            reducer_name=reducer_name,
                            n_components=n_components,
                            output_root=output_root,
                        )
                    )

        if args.run_pooled:
            available_conditions = [
                condition for condition in args.conditions if ("condition", condition) in base_containers_by_scope
            ]
            if available_conditions:
                pooled_container = concat_containers(
                    [base_containers_by_scope[("condition", condition)] for condition in available_conditions]
                )
                base_containers_by_scope[("pooled", POOLED_CONDITION)] = pooled_container
                for unit_spec in iter_analysis_units(args, pooled_container):
                    unit_containers_by_key[("pooled", POOLED_CONDITION, unit_spec["unit_key"])] = unit_spec["container"]
                    for reducer_name, n_components in product(reducers, args.n_components_sweep):
                        fit_tasks.append(
                            _build_fit_task(
                                args=args,
                                scope="pooled",
                                condition=POOLED_CONDITION,
                                unit_spec=unit_spec,
                                reducer_name=reducer_name,
                                n_components=n_components,
                                output_root=output_root,
                            )
                        )
            else:
                logger.warning("Skipping pooled mode: no condition containers were available.")

        for record in _run_task_batch(fit_tasks, _execute_fit_task, resolved_n_jobs):
            update_runs(fit_runs_path, record, key_fields=FIT_RUN_KEY_FIELDS)

        if eval_specs:
            if not fit_runs_path.exists():
                raise RuntimeError(
                    "No fit runs were produced, so post-hoc evaluations cannot run. "
                    "Check the condition load errors above."
                )
            fit_runs = [
                record
                for record in load_fit_runs(fit_runs_path)
                if record.get("status") == "success"
                and record.get("input_mode") == args.input_mode
                and record.get("analysis_mode") == args.analysis_mode
                and record.get("reducer") in reducers
                and int(record.get("n_components", 0)) in args.n_components_sweep
            ]
            if not fit_runs:
                raise RuntimeError(
                    "No successful fit runs were produced, so post-hoc evaluations cannot run. "
                    "Check the fit errors above."
                )
            eval_tasks: list[dict[str, Any]] = []
            for fit_record in fit_runs:
                unit_container = unit_containers_by_key.get(
                    (fit_record["scope"], fit_record["condition"], fit_record["unit_key"])
                )
                if unit_container is None:
                    logger.warning(
                        "Skipping evals for missing unit scope %s/%s/%s.",
                        fit_record["scope"],
                        fit_record["condition"],
                        fit_record["unit_key"],
                    )
                    continue
                for eval_spec in eval_specs:
                    if eval_spec["name"] == "condition_separation" and fit_record["scope"] != "pooled":
                        continue
                    eval_tasks.append(
                        _build_eval_task(
                            fit_record=fit_record,
                            eval_spec=eval_spec,
                            container=unit_container,
                            output_root=output_root,
                            overwrite=args.overwrite,
                        )
                    )
            for record in _run_task_batch(eval_tasks, _execute_eval_task, resolved_n_jobs):
                update_runs(eval_runs_path, record, key_fields=EVAL_RUN_KEY_FIELDS)

    report_path: Path | None = None
    fatal_error: str | None = None
    try:
        if not fit_runs_path.exists():
            raise RuntimeError(
                f"No fit runs found in {fit_runs_path}. Dim reduction produced no successful fit inventory."
            )

        report = generate_dataset_report(
            args=args,
            output_root=output_root,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            reducers=reducers,
            subjects=subjects,
            meta_df=meta_df,
            containers_by_scope=base_containers_by_scope or None,
            metric_columns=FIT_METRIC_COLUMNS,
            eval_specs=eval_specs,
            pooled_condition=POOLED_CONDITION,
        )
        reports_root = get_reports_root(bids_root)
        summary_dir = get_stage_summary_dir(reports_root, "dim_reduction", create_dir=True)
        if args.output_group:
            summary_dir = summary_dir / Path(str(args.output_group))
        summary_dir = summary_dir / args.dataset_name / args.input_mode
        summary_dir.mkdir(parents=True, exist_ok=True)
        report_path = summary_dir / f"{run_variant}_dataset_summary.html"
        report.save(report_path)
        logger.info("Report saved to: %s", report_path)
    except Exception as exc:
        fatal_error = str(exc)
        raise
    finally:
        _write_run_status(
            output_root=output_root,
            fit_runs_path=fit_runs_path,
            eval_runs_path=eval_runs_path,
            fatal_error=fatal_error,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()

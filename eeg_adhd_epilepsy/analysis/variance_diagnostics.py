#!/usr/bin/env python3
"""Model-free subject and label variance diagnostics of one embedding space.

A run reads one space's window vectors from the all-recordings file in its model's
dataset (see :func:`~eeg_adhd_epilepsy.analysis.dataset.foundation_embedding_file`),
builds the configured diagnostic tasks and scores each with coco-pipe's variance
report. With ``diagnostic_transform: leace`` the vectors are first cross-fitted: the
subjects are split in five, and each part is erased by a LEACE fitted on the others
with their conditions, so no subject is scored on directions learned from it; each
fold's number of erased directions is recorded. The subject probe trains on the first
half of each recording's windows and tests on the second (and the reverse), so
neighbouring windows are not split between the two.

Rows go to ``variance_diagnostics.csv`` under the ``eeg_variance_diagnostics``
derivative, per epoch variant and space. They carry the cohort, population and exact
target-selection fingerprint, so raw/transformed deltas are only formed from paired
observations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.decoding.targets import prepare_target
from coco_pipe.diagnostics.variance import variance_decomposition_report
from coco_pipe.io import DataContainer, load_embeddings, normalize_subject_value
from coco_pipe.transforms import LeaceEraser
from coco_pipe.utils import slug, stable_hash
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_info

from eeg_adhd_epilepsy.analysis.dataset import filter_cohort_container, foundation_embedding_file
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root, read_participants
from eeg_adhd_epilepsy.utils.config import resolve_cli_config

LOGGER = logging.getLogger(__name__)

_ASSESSMENT_KEY_COLUMNS = (
    "transform",
    "cohort_name",
    "population",
    "scope",
    "eval_name",
    "target_col",
)


@dataclass(frozen=True)
class DiagnosticTask:
    """One exact diagnostic assessment shared by raw and transformed matrices."""

    population: str
    scope: str
    eval_name: str
    target_col: str
    indices: np.ndarray
    subjects: np.ndarray
    labels: np.ndarray
    observation_ids: np.ndarray
    blocks: np.ndarray
    selection_fingerprint: str
    target_encoding: str


def _selection_fingerprint(
    task: DiagnosticTask,
) -> str:
    paired_rows = sorted(
        (str(obs_id), str(label), str(subject))
        for obs_id, label, subject in zip(
            task.observation_ids,
            task.labels,
            task.subjects,
            strict=True,
        )
    )
    return stable_hash(
        {
            "population": task.population,
            "scope": task.scope,
            "eval_name": task.eval_name,
            "target_col": task.target_col,
            "target_encoding": task.target_encoding,
            "observations": paired_rows,
        },
        length=20,
    )


def recording_halves(container: DataContainer) -> np.ndarray:
    """Whether each window is in the second half of its recording and condition (by start)."""
    rows = container.obs_table()
    key = rows["recording_id"].astype(str) + "|" + rows["condition"].astype(str)
    rank = rows.groupby(key)["start"].rank(method="first")
    return (rank > key.map(key.value_counts()) / 2).to_numpy()


def build_diagnostic_tasks(
    container: DataContainer,
    config: Mapping[str, Any],
) -> list[DiagnosticTask]:
    """Resolve exact diagnostic tasks from one metadata-enriched container."""
    subjects = np.asarray(
        [normalize_subject_value(value) for value in container.coords["subject"]],
        dtype=object,
    )
    halves = recording_halves(container)
    indexed = container.replace(
        coords={**container.coords, "_diagnostic_row": np.arange(container.X.shape[0])},
    )
    clinical = filter_cohort_container(
        indexed,
        group_filters=config.get("group_filters") or [],
        filter_col=config.get("filter_col") or [],
        filter_val=config.get("filter_val") or [],
    )

    available_conditions = list(dict.fromkeys(np.asarray(clinical.coords["condition"]).astype(str)))
    configured_conditions = [str(value) for value in config.get("conditions", [])]
    conditions = [
        condition
        for condition in (configured_conditions or available_conditions)
        if condition in available_conditions
    ]
    clinical_scopes = [
        (condition, clinical.select(condition=[condition])) for condition in conditions
    ]
    if not clinical_scopes:
        clinical_scopes = [("pooled", clinical)]
    elif len(conditions) > 1 and bool(config.get("run_pooled", True)):
        clinical_scopes.append(("pooled", clinical.select(condition=conditions)))

    scopes_by_population = {
        "transform_training_population": [("transform_fit_all", indexed)],
        "clinical_task_subset": clinical_scopes,
    }
    tasks: list[DiagnosticTask] = []
    for eval_spec in config["evals"]:
        target_col = str(eval_spec["target_col"])
        eval_name = str(eval_spec.get("name", target_col))
        for population in config["diagnostic_populations"]:
            for scope, scoped_container in scopes_by_population[population]:
                if eval_name == "condition_separation" and scope not in {
                    "pooled",
                    "transform_fit_all",
                }:
                    continue
                diagnostic_container = scoped_container
                if eval_spec.get("class_order"):
                    diagnostic_container = scoped_container.select(
                        **{target_col: list(eval_spec["class_order"])}
                    )
                _, y, _, selected_frame = prepare_target(
                    diagnostic_container,
                    eval_spec,
                    group_col=str(
                        eval_spec.get("group_col", config.get("group_col", "patient_group_id"))
                    ),
                )
                selected_indices = selected_frame["_diagnostic_row"].to_numpy(dtype=int)
                task = DiagnosticTask(
                    population=population,
                    scope=scope,
                    eval_name=eval_name,
                    target_col=target_col,
                    indices=selected_indices,
                    subjects=subjects[selected_indices],
                    labels=np.asarray(y),
                    observation_ids=selected_frame["sample_id"].astype(str).to_numpy(),
                    blocks=halves[selected_indices],
                    selection_fingerprint="",
                    target_encoding=json.dumps(
                        selected_frame.attrs.get("label_encoding", {}), sort_keys=True
                    ),
                )
                tasks.append(replace(task, selection_fingerprint=_selection_fingerprint(task)))
    return tasks


def _task_result_rows(
    report: pd.DataFrame,
    task: DiagnosticTask,
    config: Mapping[str, Any],
    *,
    transform: str,
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            **(extra or {}),
            "transform": transform,
            "cohort_name": str(config["dataset_name"]),
            "population": task.population,
            "scope": task.scope,
            "eval_name": task.eval_name,
            "target_col": task.target_col,
            "selection_fingerprint": task.selection_fingerprint,
            "target_encoding": task.target_encoding,
            **row,
        }
        for row in report.to_dict("records")
    ]


def _parallel_task_rows(
    tasks: list[DiagnosticTask],
    score_one: Any,
    n_jobs: int,
) -> list[dict[str, Any]]:
    """Run ``score_one(task)`` -> rows across tasks, in task order, with n_jobs>1
    parallelism over independent (embarrassingly parallel) diagnostic tasks.

    threadpoolctl's dl_iterate_phdr-based library scan (triggered by every
    sklearn/numpy call that manages BLAS/OpenMP thread counts) does its
    first-use imports lazily. Dozens of freshly spawned threading-backend
    workers hitting that for the first time simultaneously pile up on
    Python's per-module import lock and can stall for a very long time.
    Paying that import cost once here, from the main thread, before any
    worker starts, means workers hit an already-cached import instead of
    racing on it.
    """
    if n_jobs <= 1 or len(tasks) <= 1:
        return [row for task in tasks for row in score_one(task)]
    threadpool_info()
    with parallel_backend("threading"):
        results = Parallel(n_jobs=min(n_jobs, len(tasks)))(
            delayed(score_one)(task) for task in tasks
        )
    return [row for task_rows in results for row in task_rows]


def score_variance_diagnostics(
    features: np.ndarray,
    tasks: list[DiagnosticTask],
    config: Mapping[str, Any],
    *,
    transform: str,
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute every configured diagnostic for one embedding variant."""
    features = np.asarray(features)
    n_jobs = int(config.get("n_jobs", 1) or 1)

    def score_one(task: DiagnosticTask) -> list[dict[str, Any]]:
        diagnostic_container = DataContainer(
            X=features[task.indices],
            dims=("obs", "feature"),
            coords={
                "diagnostic_subject": task.subjects,
                "diagnostic_label": task.labels,
            },
            ids=task.observation_ids,
        )
        report = variance_decomposition_report(
            diagnostic_container,
            subject="diagnostic_subject",
            label="diagnostic_label",
            n_null_permutations=int(config["n_null_permutations"]),
            rng=np.random.default_rng(int(config["random_state"])),
            probe_blocks=task.blocks,
            probe_n_splits=2,
        )
        return _task_result_rows(report, task, config, transform=transform, extra=extra)

    return _parallel_task_rows(tasks, score_one, n_jobs)


def cross_fitted_leace(
    container: DataContainer, params: Mapping[str, Any], *, n_splits: int = 5
) -> tuple[np.ndarray, list[int]]:
    """The window vectors with subject identity erased, each subject by a LEACE fitted
    on the subjects of the other folds with their conditions; and each fold's rank."""
    x = np.asarray(container.X, dtype=np.float64)
    subjects = np.asarray(container.coords["subject"]).astype(str)
    conditions = np.asarray(container.coords["condition"]).astype(str)
    erased, ranks = np.empty_like(x), []
    for train, test in GroupKFold(n_splits=n_splits).split(x, groups=subjects):
        eraser = LeaceEraser(**params).fit(
            x[train], groups=subjects[train], condition=conditions[train]
        )
        erased[test] = eraser.transform(x[test])
        ranks.append(eraser.rank_)
    return erased, ranks


def skipped_variance_diagnostics(
    tasks: list[DiagnosticTask],
    config: Mapping[str, Any],
    *,
    transform: str,
    reason: str,
    n_features: int,
) -> list[dict[str, Any]]:
    """Return report-visible status rows for a transform that was not materialized."""
    return [
        {
            "transform": transform,
            "cohort_name": str(config["dataset_name"]),
            "population": task.population,
            "scope": task.scope,
            "eval_name": task.eval_name,
            "target_col": task.target_col,
            "selection_fingerprint": task.selection_fingerprint,
            "target_encoding": task.target_encoding,
            "metric": "total_sample_variance",
            "value": None,
            "n_observations": len(task.indices),
            "n_features": int(n_features),
            "n_constant_features": int(n_features),
            "n_subjects": len(np.unique(task.subjects)),
            "n_labels": len(np.unique(task.labels)),
            "design": "not_applicable",
            "status": "skipped",
            "reason": reason,
        }
        for task in tasks
    ]


def write_variance_diagnostics(
    rows: list[dict[str, Any]],
    diagnostics_root: Path,
) -> Path:
    """Replace the diagnostic assessment cells present in ``rows``."""
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_root / "variance_diagnostics.csv"
    incoming = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        incoming_assessments = pd.MultiIndex.from_frame(incoming.loc[:, _ASSESSMENT_KEY_COLUMNS])
        existing_assessments = pd.MultiIndex.from_frame(existing.loc[:, _ASSESSMENT_KEY_COLUMNS])
        existing = existing[~existing_assessments.isin(incoming_assessments)]
        incoming = pd.concat(
            [frame.dropna(axis=1, how="all") for frame in (existing, incoming)],
            ignore_index=True,
        )

    temporary_path = output_path.with_suffix(".tmp")
    incoming.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    return output_path


def run(config: dict[str, Any]) -> Path:
    """Score one embedding space, raw or with subject identity erased."""
    space = str(config["embedding_model_key"])
    desc = str(config.get("desc", "base"))
    transform = str(config["diagnostic_transform"])
    if transform not in ("none", "leace"):
        raise ValueError(f"diagnostic_transform must be 'none' or 'leace', got {transform!r}.")
    container = load_embeddings(
        foundation_embedding_file(
            Path(config["embedding_derivative_root"]).expanduser(),
            space,
            desc,
            task=str(config.get("task", "clinical")),
        ),
        subject_metadata_df=read_participants(Path(config["metadata"]).expanduser()),
        subject_key="subject",
    )
    tasks = build_diagnostic_tasks(container, config)
    features, extra = np.asarray(container.X), None
    if transform == "leace":
        params = (config.get("transform_params") or {}).get("leace", {})
        features, ranks = cross_fitted_leace(container, params)
        extra = {"leace_rank": f"{min(ranks)}-{max(ranks)}"}
    rows = score_variance_diagnostics(features, tasks, config, transform=transform, extra=extra)
    output_path = write_variance_diagnostics(
        rows,
        get_derivative_root(
            Path(config["bids_root"]).expanduser(),
            DerivativeStage.VARIANCE_DIAGNOSTICS,
        )
        / desc
        / slug(space),
    )
    LOGGER.info("Wrote %d variance-diagnostic row(s) to %s.", len(rows), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_config", required=True)
    parser.add_argument("--analysis_config", required=True)
    parser.add_argument("--bids_root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--embedding_derivative_root",
        required=True,
        help="Folder of the coco-pipe-<model> datasets with the all-recordings files.",
    )
    parser.add_argument(
        "--embedding_model_key",
        required=True,
        help="The space: a model key, with _pool-attention for REVE's attention vectors.",
    )
    parser.add_argument("--desc", default="base", help="Epoch variant: base or baserejected.")
    parser.add_argument(
        "--diagnostic_transform",
        required=True,
        choices=["none", "leace"],
        help="none, or leace cross-fitted over subjects.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run(
        resolve_cli_config(
            cohort_config=args.cohort_config,
            analysis_config=args.analysis_config,
            bids_root=args.bids_root,
            metadata=args.metadata,
            embedding_derivative_root=args.embedding_derivative_root,
            embedding_model_key=args.embedding_model_key,
            desc=args.desc,
            diagnostic_transform=args.diagnostic_transform,
        )
    )


if __name__ == "__main__":
    main()

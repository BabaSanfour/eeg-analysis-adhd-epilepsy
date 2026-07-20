#!/usr/bin/env python3
"""Model-free subject/label variance diagnostics over embedding derivatives.

These diagnostics are independent of subject alignment: they characterise the
subject/label variance structure of *any* embedding set. The compute here is
shared two ways:

* the alignment producer builds one set of diagnostic tasks and scores the
  before/after variance of every materialised variant in one pass, and
* :func:`run` / :func:`main` provide a standalone entry point that scores a raw
  (or already-aligned) embedding root without materialising anything.

Both paths write a single ``variance_diagnostics.csv`` per base model under the
``eeg_variance_diagnostics`` derivative stage. Rows carry the cohort,
population, and exact target-selection fingerprint used for the assessment, so
raw/transformed deltas are only formed from genuinely paired observations.
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
from coco_pipe.io import (
    DataContainer,
    load_embedding_derivatives,
    normalize_subject_value,
    read_table,
)
from coco_pipe.utils import slug, stable_hash

from eeg_adhd_epilepsy.analysis.dataset import (
    attach_subject_metadata,
    filter_cohort_container,
)
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root
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


def build_diagnostic_tasks(
    container: DataContainer,
    config: Mapping[str, Any],
) -> list[DiagnosticTask]:
    """Resolve exact diagnostic tasks from one metadata-enriched container."""
    subjects = np.asarray(
        [normalize_subject_value(value) for value in container.coords["subject"]],
        dtype=object,
    )
    indexed = replace(
        container,
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
                    selection_fingerprint="",
                    target_encoding=json.dumps(
                        selected_frame.attrs.get("label_encoding", {}), sort_keys=True
                    ),
                )
                tasks.append(replace(task, selection_fingerprint=_selection_fingerprint(task)))
    return tasks


def score_variance_diagnostics(
    features: np.ndarray,
    tasks: list[DiagnosticTask],
    config: Mapping[str, Any],
    *,
    transform: str,
) -> list[dict[str, Any]]:
    """Compute every configured diagnostic for one embedding variant."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        diagnostic_container = DataContainer(
            X=np.asarray(features)[task.indices],
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
        )
        rows.extend(
            {
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
        )
    return rows


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
        incoming = pd.concat([existing, incoming], ignore_index=True)

    temporary_path = output_path.with_suffix(".tmp")
    incoming.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    return output_path


def run(config: dict[str, Any]) -> Path:
    """Score one explicitly configured embedding variant."""
    embedding_root = Path(config["embedding_derivative_root"]).expanduser()
    model_key = str(config["embedding_model_key"])
    transform = str(config["diagnostic_transform"])

    container = load_embedding_derivatives(
        embedding_root,
        representation="epoch",
        model_key=model_key,
    )
    metadata = read_table(Path(config["metadata"]).expanduser(), sep=None)
    container = attach_subject_metadata(
        container,
        metadata,
        str(config["subject_col"]),
    )

    tasks = build_diagnostic_tasks(container, config)
    rows = score_variance_diagnostics(
        np.asarray(container.X),
        tasks,
        config,
        transform=transform,
    )

    output_path = write_variance_diagnostics(
        rows,
        get_derivative_root(
            Path(config["bids_root"]).expanduser(),
            DerivativeStage.VARIANCE_DIAGNOSTICS,
        )
        / slug(model_key),
    )
    LOGGER.info("Wrote %d variance-diagnostic row(s) to %s.", len(rows), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_config", required=True)
    parser.add_argument("--analysis_config", required=True)
    parser.add_argument("--bids_root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding_derivative_root", required=True)
    parser.add_argument("--embedding_model_key", required=True)
    parser.add_argument(
        "--diagnostic_transform",
        required=True,
        help="Exact transform label to write for this embedding variant.",
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
            diagnostic_transform=args.diagnostic_transform,
        )
    )


if __name__ == "__main__":
    main()

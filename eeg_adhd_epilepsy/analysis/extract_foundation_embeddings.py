#!/usr/bin/env python3
"""Foundation-model embeddings of every recording, one BIDS dataset per model.

One process runs one model with some of its weight sets (the pretrained weights, and
random weights drawn from a seed) over the subjects: each weight set is loaded once,
and each recording is read once for all of them. A recording gives one file per view
and weight set in ``<derivatives_root>/coco-pipe-<model>/``, written and labelled by
coco-pipe's ``write_recording_embeddings``: ``<variant>`` for the pretrained weights,
``<variant>Random<seed>`` for random ones, and REVE's attention read-out as
``<label>Attention``. Each label gets a manifest of the recordings that were written,
already done, skipped or failed.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coco_pipe.decoding import get_foundation_model_spec
from coco_pipe.decoding.foundation_models import (
    FoundationEmbeddingExtractor,
    normalize_inclusive_endpoint,
)
from coco_pipe.io import (
    DataContainer,
    NoDataError,
    embedding_label,
    embeddings_up_to_date,
    write_embedding_dataset,
    write_embedding_manifest,
    write_recording_embeddings,
)
from mne_bids import BIDSPath

from eeg_adhd_epilepsy.analysis.dataset import build_container
from eeg_adhd_epilepsy.analysis.utils.subject_resolution import resolve_cohort_subjects
from eeg_adhd_epilepsy.io.bids import DerivativeStage, get_derivative_root, read_participants
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)

# What each epoch variant of the preproc derivative holds. Every job gives
# descriptions.tsv all of them, so it comes out the same whichever job writes it last.
VARIANTS = {
    "base": "Epochs of the preprocessed recordings",
    "baserejected": (
        "Epochs of the preprocessed recordings, without those overlapping a BAD_ annotation"
    ),
}


def parse_weights(tokens: Sequence[str], seeds: Sequence[int]) -> list[int | None]:
    """The weight sets named on the command line: ``pretrained``, seeds, or ``all``."""
    if list(tokens) == ["all"]:
        return [None, *seeds]
    weights: list[int | None] = []
    for token in tokens:
        seed = None if token == "pretrained" else int(token)
        if seed is not None and seed not in seeds:
            raise ValueError(f"Seed {seed} is not in the model's random_seeds {list(seeds)}.")
        if seed in weights:
            raise ValueError(f"Weight set {token!r} is named twice.")
        weights.append(seed)
    return weights


def load_recordings(
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    subjects: list[str],
    *,
    n_jobs: int,
) -> DataContainer | None:
    """Every requested condition of ``subjects``, cut at the model's window length.

    None when no recording of theirs has one. A recording that cannot be read
    raises: it is not a recording without epochs (audit C4).
    """
    segment_duration = float(model_cfg["segment_duration"])
    try:
        container = build_container(
            bids_root=Path(config["bids_root"]),
            use_derivatives=True,
            subjects=subjects,
            task=config["task"],
            segment_duration=segment_duration,
            desc=config["desc"],
            condition=list(config["conditions"]),
            window_source=model_cfg["window_source"],
            units="uV",
            n_jobs=n_jobs,
        )
    except NoDataError:
        return None
    failures = container.meta.get("load_failures") or []
    if failures:
        raise RuntimeError(f"{len(failures)} recording(s) could not be loaded: {failures}")
    container, _ = normalize_inclusive_endpoint(
        container,
        segment_duration=segment_duration,
        expected_sfreq=float(container.meta["sfreq"]),
        on_mismatch="error",
    )
    return container


def embed_recording(
    recording: DataContainer,
    extractor: FoundationEmbeddingExtractor,
    *,
    root: Path,
    label: str,
    written: dict[tuple[str, str], BIDSPath],
    token_dtype: str,
) -> dict[str, Any]:
    """Write one recording's files for one weight set, unless they are done.

    ``label`` is the epoch variant. ``written`` holds the files of the first recording
    this process writes with the weight set, which it fills in: a recording whose
    files hold its windows with their settings is done (audit E1). That first
    recording is embedded again, but its files are kept when up to date, so a
    resubmission rewrites nothing.
    """
    rows = recording.obs_table()
    recording_id = str(rows["recording_id"].iloc[0])
    record = {
        "recording_id": recording_id,
        "subject": str(rows["subject"].iloc[0]),
        "conditions": sorted(set(rows["condition"].astype(str))),
        "n_windows": len(rows),
    }
    if written and embeddings_up_to_date(recording, written):
        return {**record, "status": "done"}
    try:
        paths = write_recording_embeddings(
            extractor.extract_recording(recording),
            root,
            label=label,
            source=rows["source"].iloc[0],
            source_dataset="preproc",
            bad_channels=rows["bad_channels"].iloc[0],
            token_dtype=token_dtype,
            skip_up_to_date=True,
        )
    except (ValueError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise
        seed = None if extractor.pretrained else extractor.seed
        LOGGER.exception("%s failed for %s", embedding_label(label, seed=seed), recording_id)
        return {**record, "status": "failed", "reason": f"{type(error).__name__}: {error}"}
    if not written:
        written.update(paths)
    return {**record, "status": "written"}


def run(
    config: dict[str, Any],
    derivatives_root: Path,
    *,
    model_key: str,
    weights: list[int | None],
    subjects: list[str],
    n_jobs: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Embed every recording of ``subjects`` with each weight set of one model.

    Returns each weight set's records by its label, which are also written as the
    manifest of each label it writes.
    """
    model_cfg = config["models"][model_key]
    desc = str(config["desc"])
    device = str(config["device"])
    root = Path(derivatives_root) / f"coco-pipe-{model_key}"
    preproc_root = get_derivative_root(Path(config["bids_root"]), DerivativeStage.PREPROC)
    spec = get_foundation_model_spec(model_key)
    # The preproc link resolves from the scratch copy too: an absolute URI, not
    # ../preproc, which on scratch has no dataset_description.json.
    write_embedding_dataset(
        root,
        VARIANTS,
        name=f"{spec.display_name or model_key} embeddings",
        dataset_links={"preproc": preproc_root.absolute().as_uri()},
    )
    poolings = tuple(model_cfg.get("extra_poolings", ()))
    # An extractor refuses a CUDA device without a GPU, before any data is read (audit D1).
    extractors = {
        seed: FoundationEmbeddingExtractor(
            model_key,
            device=device,
            pretrained=seed is None,
            seed=seed,
            extra_poolings=poolings,
            store_tokens=True,
            batch_size=model_cfg.get("batch_size"),
        )
        for seed in weights
    }
    records: dict[int | None, list[dict[str, Any]]] = {seed: [] for seed in extractors}
    written: dict[int | None, dict[tuple[str, str], BIDSPath]] = {seed: {} for seed in extractors}
    chunk = int(config["chunk_subjects"])
    for start in range(0, len(subjects), chunk):
        batch = subjects[start : start + chunk]
        LOGGER.info(
            "%s: subjects %d-%d of %d", model_key, start + 1, start + len(batch), len(subjects)
        )
        container = load_recordings(config, model_cfg, batch, n_jobs=n_jobs)
        rows = container.obs_table() if container is not None else None
        found = set(rows["subject"].astype(str)) if rows is not None else set()
        skips = [
            {"subject": subject, "status": "skipped", "reason": "no recording with a condition"}
            for subject in batch
            if subject not in found
        ]
        if container is not None:
            skips += [
                {"recording_id": recording_id, "status": "skipped", "reason": "no condition"}
                for recording_id in container.meta.get("skipped_no_matching_events", [])
            ]
        for seed_records in records.values():
            seed_records.extend(skips)
        if container is None:
            continue
        recording_ids = rows["recording_id"].astype(str)
        for recording_id in recording_ids.unique():
            recording = container.isel(obs=np.flatnonzero(recording_ids == recording_id))
            for seed, extractor in extractors.items():
                records[seed].append(
                    embed_recording(
                        recording,
                        extractor,
                        root=root,
                        label=desc,
                        written=written[seed],
                        token_dtype=str(config["token_dtype"]),
                    )
                )
    for seed, seed_records in records.items():
        for pooling in ("mean", *poolings):
            write_embedding_manifest(
                root, embedding_label(desc, seed=seed, pooling=pooling), seed_records
            )
    return {
        embedding_label(desc, seed=seed): seed_records for seed, seed_records in records.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed every recording with one foundation model, one file per view."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--bids_root", required=True, help="The raw BIDS dataset.")
    parser.add_argument(
        "--desc",
        required=True,
        choices=sorted(VARIANTS),
        help="Epoch variant of the preproc derivative (its desc-<variant> epochs).",
    )
    parser.add_argument("--metadata", required=True, help="The BIDS participants.tsv.")
    parser.add_argument(
        "--derivatives_root",
        required=True,
        help="Folder of the coco-pipe-<model> datasets (on scratch).",
    )
    parser.add_argument("--model", required=True, help="Model key, e.g. reve.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["all"],
        help="Weight sets: pretrained and/or seeds of the model's random_seeds, or all.",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None, help="Only these subjects (default: all)."
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Threads reading recordings.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_yaml_config(args.config)
    config["bids_root"] = args.bids_root
    config["desc"] = args.desc
    model_cfg = config["models"][args.model]
    subjects = resolve_cohort_subjects(
        read_participants(Path(args.metadata)), "subject", args.subjects
    )
    if not subjects:
        raise ValueError("No subject to embed.")
    records = run(
        config,
        Path(args.derivatives_root),
        model_key=args.model,
        weights=parse_weights(args.weights, model_cfg["random_seeds"]),
        subjects=subjects,
        n_jobs=args.n_jobs,
    )
    failed = {
        label: sum(record["status"] == "failed" for record in label_records)
        for label, label_records in records.items()
    }
    if any(failed.values()):
        raise SystemExit(f"Recordings failed (see the manifests): {failed}")


if __name__ == "__main__":
    main()

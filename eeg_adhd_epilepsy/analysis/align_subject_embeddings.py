#!/usr/bin/env python3
"""Subject-aligned embeddings of one model, one file holding every recording per label.

A run aligns the embeddings of one model and epoch variant, one weight set at a time
(the label of the extracted files, e.g. ``base`` or ``baseRandom3``). It reads that
label's per-recording files from the extraction's dataset (on scratch) and writes,
with coco-pipe's ``write_transformed_embeddings``, to the model's dataset on the
project, ``<derivatives_root>/coco-pipe-<model>/``:

- the window vectors aligned by each ``alignment.window`` transform (``EaMean``,
  ``EaCoral``, ``Leace``);
- the tokens aligned by each ``alignment.token`` transform: ``EaMean`` and
  ``EaCoral``, also per channel (``EaMeanCh``, ``EaCoralCh``) with ``per_channel``
  when the tokens have a channel axis, and ``Ra``'s tangent vectors of the token
  covariances (``covembeddings``).

Every transform fits each subject on all its windows. A degenerate LEACE projector
writes no file, and its manifest says why. Each aligned label gets a manifest of the
recordings it holds, and each file a sidecar.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from coco_pipe.io import (
    DataContainer,
    embedding_label,
    find_embeddings,
    load_embeddings,
    write_embedding_manifest,
    write_transformed_embeddings,
)
from coco_pipe.transforms import RiemannAlign, make_subject_transform
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone

from eeg_adhd_epilepsy.analysis.extract_foundation_embeddings import (
    VARIANTS,
    parse_weights,
    write_dataset,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def load_label(root: Path, label: str, view: str) -> DataContainer:
    """Every window of one label and view, from its per-recording files."""
    paths = find_embeddings(root, label, view)
    if not paths:
        raise FileNotFoundError(f"No {view} file of {label} in {root}.")
    container = load_embeddings(paths)
    LOGGER.info("%s %s: %d windows, shape %s.", label, view, len(container.X), container.X.shape)
    return container


def _subject_tangent(aligner: RiemannAlign, tokens: np.ndarray, subject: str) -> np.ndarray:
    """One subject's tangent vectors, as float32."""
    groups = np.full(len(tokens), subject)
    return clone(aligner).fit_transform(tokens, groups=groups).astype(np.float32)


def riemann_align(
    tokens: np.ndarray, subjects: np.ndarray, aligner: RiemannAlign, *, n_jobs: int = 1
) -> np.ndarray:
    """``aligner``'s tangent vectors of ``(window, ..., feature)`` tokens.

    Each subject has its own reference, so fitting one subject at a time gives what
    fitting every window does, while only some subjects' covariances are held. The
    subjects go to ``n_jobs`` processes, each with one BLAS thread: a subject's
    covariances are too small for threaded linear algebra to pay. The limit is
    explicit because joblib otherwise keeps the caller's ``OMP_NUM_THREADS``, so each
    process would start as many threads as the job has cores. Their vectors come back
    in order, so memory holds the result and a few subjects.
    """
    flat = tokens.reshape(len(tokens), -1, tokens.shape[-1])
    rows = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_subject_tangent)(aligner, flat[index], subjects[index[0]]) for index in rows
        )
        vectors = None
        for index, part in zip(rows, parts, strict=True):
            if vectors is None:
                vectors = np.empty((len(flat), part.shape[1]), dtype=np.float32)
            vectors[index] = part
    return vectors


def write_aligned(
    source: DataContainer,
    aligned: np.ndarray,
    root: Path,
    aligner: Any,
    records: dict[str, list[dict[str, Any]]],
    *,
    token_dtype: str,
    view: str | None = None,
) -> None:
    """Write one aligned file and put the records of its manifest in ``records``."""
    label = embedding_label(source.meta["description"], transform=aligner.label)
    if getattr(aligner, "degenerate_", False):
        reason = (
            f"degenerate projector: rank {aligner.rank_} of {source.X.shape[-1]} "
            f"features, {aligner.n_subjects_} subjects"
        )
        LOGGER.warning("%s not written: %s.", label, reason)
        for stale in root.glob(f"*_desc-{label}_*"):
            stale.unlink()
        records[label] = [{"status": "skipped", "reason": reason}]
        return
    LOGGER.info("%s fitted; writing it.", label)
    path = write_transformed_embeddings(
        source, aligned, root, transform=aligner, view=view, token_dtype=token_dtype
    )
    LOGGER.info("Wrote %s.", path.basename)
    recordings = sorted(set(np.asarray(source.coords["recording_id"]).astype(str)))
    records[label] = [{"recording_id": recording, "status": "written"} for recording in recordings]


def run(
    config: dict[str, Any],
    source_root: Path,
    derivatives_root: Path,
    *,
    model_key: str,
    seed: int | None,
    n_jobs: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Write every aligned file of one weight set; return each aligned label's records.

    ``n_jobs`` processes compute RA's tangent vectors, one subject each at a time.
    """
    label = embedding_label(str(config["desc"]), seed=seed)
    source = Path(source_root) / f"coco-pipe-{model_key}"
    root = Path(derivatives_root) / f"coco-pipe-{model_key}"
    settings = config["alignment"]
    token_dtype = str(config["token_dtype"])
    records: dict[str, list[dict[str, Any]]] = {}

    windows = load_label(source, label, "window")
    subjects = np.asarray(windows.coords["subject"]).astype(str)
    for name in settings["window"]:
        aligner = make_subject_transform(name)
        LOGGER.info("%s: fitting %s on the window vectors.", label, aligner.label)
        aligned = aligner.fit_transform(windows.X, groups=subjects)
        write_aligned(windows, aligned, root, aligner, records, token_dtype=token_dtype)
    del windows

    tokens = load_label(source, label, "token")
    subjects = np.asarray(tokens.coords["subject"]).astype(str)
    channel = tokens.dims.index("channel") if "channel" in tokens.dims else None
    for name in settings["token"]:
        if name == "ra":
            aligner = RiemannAlign()
            LOGGER.info("%s: fitting Ra on the tokens.", label)
            vectors = riemann_align(tokens.X, subjects, aligner, n_jobs=n_jobs)
            write_aligned(
                tokens, vectors, root, aligner, records, token_dtype=token_dtype, view="cov"
            )
            continue
        aligners = [make_subject_transform(name)]
        if settings.get("per_channel") and channel is not None:
            aligners.append(make_subject_transform(name, channel_axis=channel))
        for aligner in aligners:
            LOGGER.info("%s: fitting %s on the tokens.", label, aligner.label)
            aligned = aligner.fit_transform(tokens.X, groups=subjects)
            write_aligned(tokens, aligned, root, aligner, records, token_dtype=token_dtype)
            del aligned

    for written, label_records in records.items():
        write_embedding_manifest(root, written, label_records)
    write_dataset(root, model_key, Path(config["bids_root"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align one model's embeddings across subjects, one file per label."
    )
    parser.add_argument("--config", required=True, help="The embedding extraction config.")
    parser.add_argument("--bids_root", required=True, help="The raw BIDS dataset.")
    parser.add_argument(
        "--desc",
        required=True,
        choices=sorted(VARIANTS),
        help="Epoch variant of the preproc derivative (its desc-<variant> epochs).",
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help="Folder of the coco-pipe-<model> datasets with the per-recording files (scratch).",
    )
    parser.add_argument(
        "--derivatives_root",
        required=True,
        help="Folder of the coco-pipe-<model> datasets to write to (project).",
    )
    parser.add_argument("--model", required=True, help="Model key, e.g. reve.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["all"],
        help="Weight sets: pretrained and/or seeds of the model's random_seeds, or all.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Processes computing RA, one subject each."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_yaml_config(args.config)
    config["bids_root"] = args.bids_root
    config["desc"] = args.desc
    for seed in parse_weights(args.weights, config["models"][args.model]["random_seeds"]):
        run(
            config,
            Path(args.source_root),
            Path(args.derivatives_root),
            model_key=args.model,
            seed=seed,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()

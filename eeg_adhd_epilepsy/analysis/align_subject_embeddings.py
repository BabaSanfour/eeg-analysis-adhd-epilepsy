#!/usr/bin/env python3
"""LEACE-erased window vectors of one model, one file holding every recording per label.

A run erases subject identity from the window vectors of one model and epoch variant,
one weight set at a time (the label of the extracted files, e.g. ``base`` or
``baseRandom3``). It reads that label's per-recording window files from the extraction's
dataset (on scratch) and writes, with coco-pipe's ``write_transformed_embeddings``, one
``Leace`` file to the model's dataset on the project, ``<derivatives_root>/coco-pipe-
<model>/``, with a manifest of the recordings it holds and a sidecar.

LEACE (``alignment.leace`` settings) erases the directions in which subjects differ
most, fitted on every window with each window's condition. The file serves the
descriptive plots; decoding fits LEACE in its folds. When LEACE would erase almost every
direction it writes no file, and its manifest says why.
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
from coco_pipe.transforms import LeaceEraser

from eeg_adhd_epilepsy.analysis.extract_foundation_embeddings import (
    VARIANTS,
    parse_weights,
    write_dataset,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def load_windows(root: Path, label: str) -> DataContainer:
    """Every window vector of one label, from its per-recording files."""
    paths = find_embeddings(root, label, "window")
    if not paths:
        raise FileNotFoundError(f"No window file of {label} in {root}.")
    container = load_embeddings(paths)
    LOGGER.info("%s: %d windows, shape %s.", label, len(container.X), container.X.shape)
    return container


def run(
    config: dict[str, Any],
    source_root: Path,
    derivatives_root: Path,
    *,
    model_key: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Write the LEACE file of one weight set; return the records of its manifest."""
    label = embedding_label(str(config["desc"]), seed=seed)
    source = Path(source_root) / f"coco-pipe-{model_key}"
    root = Path(derivatives_root) / f"coco-pipe-{model_key}"
    windows = load_windows(source, label)
    eraser = LeaceEraser(**config["alignment"]["leace"])
    erased_label = embedding_label(label, transform=eraser.label)
    try:
        erased = eraser.fit_transform(
            windows.X,
            groups=np.asarray(windows.coords["subject"]).astype(str),
            condition=np.asarray(windows.coords["condition"]).astype(str),
        )
    except ValueError as error:  # it would erase almost every direction
        LOGGER.warning("%s not written: %s", erased_label, error)
        for stale in root.glob(f"*_desc-{erased_label}_*"):
            stale.unlink()
        records = [{"status": "skipped", "reason": str(error)}]
    else:
        LOGGER.info("%s erases %d directions; writing it.", erased_label, eraser.rank_)
        path = write_transformed_embeddings(
            windows,
            erased,
            root,
            transform=eraser,
            fitted_on=f"every window of {label}, EO and EC together",
        )
        LOGGER.info("Wrote %s.", path.basename)
        recordings = sorted(set(np.asarray(windows.coords["recording_id"]).astype(str)))
        records = [{"recording_id": recording, "status": "written"} for recording in recordings]
    write_embedding_manifest(root, erased_label, records)
    write_dataset(root, model_key, Path(config["bids_root"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erase subject identity from one model's window vectors, one file per label."
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
        )


if __name__ == "__main__":
    main()

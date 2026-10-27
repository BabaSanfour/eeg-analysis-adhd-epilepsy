#!/usr/bin/env python3
"""All-recordings embedding files of one model, from its per-recording files.

A run takes one model's per-recording files on scratch, for each epoch variant and
weight set (the extraction's labels, e.g. ``base``, ``baseRandom3`` and REVE's
``baseAttention``), and writes one file per label and view at the root of the model's
dataset on the project, with coco-pipe's ``consolidate_embeddings``. Each label is
first checked against the extraction's manifest: every recording written or done has
its file in every view, and a failed recording stops the run. The manifest goes to the
project with the files. The dataset files are written once, at the end, so they
describe every file there, the subject-erased ones included.

On a parallel file system the time goes to listing and opening a hundred thousand small
files. The dataset is listed once, not once per label and view, and with ``stage_dir`` (a
node's local disk) each label's files are first copied there by ``n_jobs`` threads, merged
from there, and the copy deleted before the next label.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coco_pipe.io import (
    EMBEDDING_VIEWS,
    consolidate_embeddings,
    embedding_label,
    read_json,
    write_embedding_manifest,
)
from mne_bids import BIDSPath, find_matching_paths

from eeg_adhd_epilepsy.analysis.extract_foundation_embeddings import (
    VARIANTS,
    parse_weights,
    write_dataset,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config

LOGGER = logging.getLogger(__name__)


def extraction_labels(
    model_cfg: Mapping[str, Any], desc: str, weights: Sequence[int | None]
) -> list[str]:
    """The labels the extraction wrote for one epoch variant and these weight sets."""
    poolings = ("mean", *model_cfg.get("extra_poolings", ()))
    return [
        embedding_label(desc, seed=seed, pooling=pooling)
        for seed in weights
        for pooling in poolings
    ]


def index_files(source: Path) -> dict[tuple[str, str], list[BIDSPath]]:
    """Every per-recording embedding file of a dataset, by label and view, sorted by path.

    One walk of the dataset, as coco-pipe's ``find_embeddings`` makes for one label and
    view; the files at the dataset root, which hold every recording, are not listed.
    """
    paths = find_matching_paths(
        source,
        suffixes=[f"{view}embeddings" for view in EMBEDDING_VIEWS],
        extensions=".nc",
        ignore_nosub=True,
        check=False,
    )
    index: dict[tuple[str, str], list[BIDSPath]] = defaultdict(list)
    for path in paths:
        index[(path.description, path.suffix.removesuffix("embeddings"))].append(path)
    return {key: sorted(found, key=lambda path: str(path.fpath)) for key, found in index.items()}


def _recording_id(path: BIDSPath) -> str:
    return path.basename.split("_desc-", 1)[0]


def check_coverage(
    label: str, records: Sequence[Mapping[str, Any]], files: Mapping[str, Sequence[BIDSPath]]
) -> None:
    """Refuse a label whose files do not match its manifest's recordings."""
    failed = [record["recording_id"] for record in records if record.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{label}: {len(failed)} recording(s) failed, e.g. {failed[0]}.")
    expected = {
        record["recording_id"] for record in records if record.get("status") in ("written", "done")
    }
    for view, view_files in files.items():
        found = {_recording_id(path) for path in view_files}
        if found != expected:
            missing, extra = sorted(expected - found), sorted(found - expected)
            raise RuntimeError(
                f"{label} {view}: {len(missing)} recording(s) of the manifest have no file "
                f"({missing[:3]}), {len(extra)} file(s) are not in it ({extra[:3]})."
            )


def stage_files(
    files: Mapping[str, Sequence[BIDSPath]], source: Path, stage: Path, *, n_jobs: int = 1
) -> dict[str, list[Path]]:
    """Copy the files to ``stage`` with ``n_jobs`` threads, keeping their place under ``source``."""
    pairs = [
        (path.fpath, stage / path.fpath.relative_to(source))
        for view_files in files.values()
        for path in view_files
    ]
    for _, copy in pairs:
        copy.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        list(pool.map(lambda pair: shutil.copyfile(*pair), pairs))
    return {
        view: [stage / path.fpath.relative_to(source) for path in view_files]
        for view, view_files in files.items()
    }


def run(
    config: dict[str, Any],
    source_root: Path,
    derivatives_root: Path,
    *,
    model_key: str,
    descs: Sequence[str],
    weights: Sequence[int | None],
    stage_dir: Path | None = None,
    n_jobs: int = 1,
) -> dict[str, list[str]]:
    """Consolidate every label of one model; return the files written per label."""
    model_cfg = config["models"][model_key]
    source = Path(source_root) / f"coco-pipe-{model_key}"
    root = Path(derivatives_root) / f"coco-pipe-{model_key}"
    written: dict[str, list[str]] = {}
    start = time.monotonic()
    index = index_files(source)
    LOGGER.info(
        "Listed %d files of %s in %.0f s.",
        sum(map(len, index.values())),
        source,
        time.monotonic() - start,
    )
    for desc in descs:
        for label in extraction_labels(model_cfg, desc, weights):
            records = read_json(source / f"desc-{label}_manifest.json")["records"]
            files = {
                view: index[(label, view)] for view in EMBEDDING_VIEWS if (label, view) in index
            }
            if not files:
                raise FileNotFoundError(f"No file of {label} in {source}.")
            check_coverage(label, records, files)
            if stage_dir is not None:
                stage = Path(stage_dir) / label
                files = stage_files(files, source, stage, n_jobs=n_jobs)
            written[label] = []
            for view, view_files in files.items():
                path = consolidate_embeddings(view_files, root)
                LOGGER.info("%s %s: %d recordings, %s", label, view, len(view_files), path.basename)
                written[label].append(path.basename)
            if stage_dir is not None:
                shutil.rmtree(stage)
            write_embedding_manifest(root, label, records)
    write_dataset(root, model_key, Path(config["bids_root"]))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one model's all-recordings embedding files, one per label and view."
    )
    parser.add_argument("--config", required=True, help="The embedding extraction config.")
    parser.add_argument("--bids_root", required=True, help="The raw BIDS dataset.")
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
        "--desc",
        nargs="+",
        default=sorted(VARIANTS),
        choices=sorted(VARIANTS),
        help="Epoch variants of the preproc derivative (default: all).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["all"],
        help="Weight sets: pretrained and/or seeds of the model's random_seeds, or all.",
    )
    parser.add_argument(
        "--stage_dir",
        default=None,
        help="A local disk to copy each label's files to before merging (e.g. $SLURM_TMPDIR).",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Threads copying files.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_yaml_config(args.config)
    config["bids_root"] = args.bids_root
    run(
        config,
        Path(args.source_root),
        Path(args.derivatives_root),
        model_key=args.model,
        descs=args.desc,
        weights=parse_weights(args.weights, config["models"][args.model]["random_seeds"]),
        stage_dir=Path(args.stage_dir) if args.stage_dir else None,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

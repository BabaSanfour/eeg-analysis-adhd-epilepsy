#!/usr/bin/env python3
"""One-time rename of descriptor artifacts to the honest aggregation-level names.

The per-recording descriptor tables were historically saved as ``*_subject_features``
(``build_descriptor_tables`` groups by ``recording_id``). The pipeline now names that
level ``*_recording_features`` and reserves ``*_subject_features`` for a true
subject-pooled table produced at merge. This script renames existing on-disk
artifacts so the new merge can read them — **no re-extraction needed**.

It renames, recursively under the descriptor derivative root, every file whose name
contains ``_subject_features``::

    <stem>_subject_features.parquet                -> <stem>_recording_features.parquet
    <stem>_subject_features.csv                    -> <stem>_recording_features.csv
    <stem>_subject_features_feature_columns.json   -> <stem>_recording_features_feature_columns.json

This catches both the per-shard tables and any pre-existing combined tables (the
latter are regenerated anyway when you re-run the merge). The script is a no-op when
already migrated; dry-run by default — pass ``--apply`` to perform the renames.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_LEGACY_TOKEN = "_subject_features"
_HONEST_TOKEN = "_recording_features"


def plan_renames(descriptor_root: Path) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for every legacy-named descriptor artifact."""
    renames: list[tuple[Path, Path]] = []
    for src in sorted(descriptor_root.rglob(f"*{_LEGACY_TOKEN}*")):
        if not src.is_file():
            continue
        dst = src.with_name(src.name.replace(_LEGACY_TOKEN, _HONEST_TOKEN))
        renames.append((src, dst))
    return renames


def migrate(descriptor_root: Path, *, apply: bool) -> int:
    """Rename legacy descriptor artifacts; return the number of files (to be) moved."""
    renames = plan_renames(descriptor_root)
    if not renames:
        LOGGER.info("Nothing to migrate under %s (already on the new names).", descriptor_root)
        return 0
    moved = 0
    for src, dst in renames:
        if dst.exists():
            LOGGER.warning("Skip %s -> %s: destination already exists.", src.name, dst.name)
            continue
        LOGGER.info("%s %s -> %s", "RENAME" if apply else "DRY-RUN", src.name, dst.name)
        if apply:
            src.rename(dst)
        moved += 1
    if not apply:
        LOGGER.info("Dry run: %d file(s) would be renamed. Re-run with --apply.", moved)
    else:
        LOGGER.info("Renamed %d file(s) under %s.", moved, descriptor_root)
    return moved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "descriptor_root",
        type=Path,
        help="Descriptor derivative root to migrate (e.g. .../signal_features/descriptors).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename the files (default is a dry run).",
    )
    args = parser.parse_args()
    if not args.descriptor_root.is_dir():
        raise SystemExit(f"Not a directory: {args.descriptor_root}")
    migrate(args.descriptor_root, apply=args.apply)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-key existing decoding run manifests to the scientific-only config hash.

Decoding resume identity used to hash the *full* run config (``{**config,
**context}``), while the run-variant directory hashed only the scientific subset
(``scientific_config`` — everything except worker counts, verbosity, and the
overwrite/reports toggles in ``_NON_SCIENTIFIC_HASH_KEYS``). Changing an
orchestration-only key such as ``n_jobs`` therefore left a run in the same
``*_cfg-<hash>`` directory (so resume fired) but changed the per-fit manifest
hash, so ``completed_for_config`` raised ``Config hash mismatch`` and aborted the
whole sweep.

The pipeline now hashes ``scientific_config(config)`` for the per-fit identity
too. This script realigns **already-completed** outputs to that scheme so they
resume instead of re-running: for every ``run_manifest.json`` it recomputes the
hash from the ``config_used.yaml`` stored beside it, dropping the non-scientific
keys, and rewrites ``config_hash`` in place. Nothing is recomputed scientifically
— only the manifest's identity field changes.

Dry-run by default; pass ``--apply`` to write. Idempotent: manifests already on
the new hash are left untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from coco_pipe.decoding import config_hash

from eeg_adhd_epilepsy.analysis.utils.decoding import scientific_config

LOGGER = logging.getLogger(__name__)

_REDACTED_SENTINEL = "<redacted>"


def _has_redacted(value: object) -> bool:
    """True if any redacted sentinel remains in the stored config."""
    if isinstance(value, str):
        return value == _REDACTED_SENTINEL
    if isinstance(value, dict):
        return any(_has_redacted(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_redacted(item) for item in value)
    return False


def plan_updates(decoding_root: Path) -> list[tuple[Path, str, str]]:
    """Return (manifest_path, old_hash, new_hash) for every stale manifest.

    Skips manifests already on the new hash, and those whose ``config_used.yaml``
    is missing or still carries a ``<redacted>`` sentinel (the live resume hash is
    computed from the un-redacted config, so a re-keyed hash could not match — those
    units re-run rather than risk a false resume).
    """
    updates: list[tuple[Path, str, str]] = []
    for manifest_path in sorted(decoding_root.rglob("run_manifest.json")):
        config_path = manifest_path.with_name("config_used.yaml")
        if not config_path.exists():
            LOGGER.warning("Skip %s: no config_used.yaml beside it.", manifest_path)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if _has_redacted(stored):
            LOGGER.warning(
                "Skip %s: config_used.yaml has redacted values; unit will re-run.",
                manifest_path.parent.name,
            )
            continue
        old_hash = manifest.get("config_hash")
        new_hash = config_hash(scientific_config(stored))
        if old_hash == new_hash:
            continue
        updates.append((manifest_path, old_hash, new_hash))
    return updates


def migrate(decoding_root: Path, *, apply: bool) -> int:
    """Rewrite stale manifest hashes; return the number of manifests (to be) updated."""
    updates = plan_updates(decoding_root)
    if not updates:
        LOGGER.info("Nothing to migrate under %s (manifests already realigned).", decoding_root)
        return 0
    for manifest_path, old_hash, new_hash in updates:
        LOGGER.info(
            "%s %s: %s -> %s",
            "UPDATE" if apply else "DRY-RUN",
            manifest_path.parent.name,
            old_hash,
            new_hash,
        )
        if apply:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config_hash"] = new_hash
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
    if not apply:
        LOGGER.info("Dry run: %d manifest(s) would be updated. Re-run with --apply.", len(updates))
    else:
        LOGGER.info("Updated %d manifest(s) under %s.", len(updates), decoding_root)
    return len(updates)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "decoding_root",
        type=Path,
        help="Decoding derivative root to migrate (e.g. .../derivatives/decoding).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite the manifests (default is a dry run).",
    )
    args = parser.parse_args()
    if not args.decoding_root.is_dir():
        raise SystemExit(f"Not a directory: {args.decoding_root}")
    migrate(args.decoding_root, apply=args.apply)


if __name__ == "__main__":
    main()

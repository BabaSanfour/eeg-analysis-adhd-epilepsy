#!/usr/bin/env python3
"""One-shot migration: split legacy single configs into cohort + analysis configs.

Legacy configs under ``configs/medicated_adhd_vs_controls/`` conflate *which
cohort* (subjects + clinical question) with *which analysis* (method +
hyperparameters). This script partitions each file by key, then writes:

* a **cohort** config under ``configs/cohorts/...`` (mirroring the source path,
  with any ``decoding_`` filename prefix stripped); and
* an **analysis** config under ``configs/analyses/<type>/...``, de-duplicated by
  content (the 71 dim-reduce configs share one identical method block, so they
  collapse to a single ``dim_reduction/default.yaml``).

Dataset paths (``bids_root`` / ``metadata``) are dropped — they belong on the
CLI/env, not in either config (see ``eeg_adhd_epilepsy/utils/config.py``).

Usage::

    python scripts/split_configs.py            # dry-run: print the plan
    python scripts/split_configs.py --apply     # write the new tree

The script never deletes the old tree; remove it by hand after verifying.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = REPO_ROOT / "configs" / "medicated_adhd_vs_controls"
COHORTS_ROOT = REPO_ROOT / "configs" / "cohorts"
ANALYSES_ROOT = REPO_ROOT / "configs" / "analyses"

# Keys that describe the cohort (the "who + which clinical question").
COHORT_KEYS = (
    "dataset_name",
    "output_group",
    "subject_col",
    "session_col",
    "group_col",
    "conditions",
    "run_pooled",
    "group_filters",
    "filter_col",
    "filter_val",
    "evals",
)
# Dataset-level paths live on the CLI/env, not in either config.
DROP_KEYS = ("bids_root", "metadata")

COHORT_KEY_ORDER = COHORT_KEYS  # preferred emit order


def _analysis_type(analysis: dict[str, Any]) -> str:
    if "train_modes" in analysis:
        return "foundation_decoding"
    if "reducers" in analysis:
        return "dim_reduction"
    if "models" in analysis:
        return "decoding"
    return "unknown"


def _ordered_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    ordered = {k: cohort[k] for k in COHORT_KEY_ORDER if k in cohort}
    for k, v in cohort.items():  # any unexpected cohort keys, appended
        ordered.setdefault(k, v)
    return ordered


def _cohort_target(source: Path) -> Path:
    rel = source.relative_to(LEGACY_ROOT)
    stem = rel.stem
    if stem.startswith("decoding_"):
        stem = stem[len("decoding_"):]
    return COHORTS_ROOT / "medicated_adhd_vs_controls" / rel.parent / f"{stem}.yaml"


def split_config(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cohort: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    for key, value in raw.items():
        if key in DROP_KEYS:
            continue
        if key in COHORT_KEYS:
            cohort[key] = value
        else:
            analysis[key] = value
    return _ordered_cohort(cohort), analysis


def _hash(payload: dict[str, Any]) -> str:
    blob = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write files (else dry-run).")
    args = parser.parse_args()

    sources = sorted(LEGACY_ROOT.rglob("*.yaml"))
    if not sources:
        raise SystemExit(f"No legacy configs found under {LEGACY_ROOT}")

    # First pass: split everything, group analysis blocks by (type, content hash).
    cohort_writes: list[tuple[Path, dict[str, Any]]] = []
    analysis_by_type: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_to_analysis_hash: dict[Path, tuple[str, str]] = {}

    for source in sources:
        cohort, analysis = split_config(source)
        cohort_writes.append((_cohort_target(source), cohort))
        atype = _analysis_type(analysis)
        ahash = _hash(analysis)
        analysis_by_type[atype][ahash] = analysis
        source_to_analysis_hash[source] = (atype, ahash)

    # Name analysis files: a single unique block per type -> "default"; otherwise
    # name by the distinguishing source stem.
    analysis_names: dict[tuple[str, str], Path] = {}
    for atype, blocks in analysis_by_type.items():
        if len(blocks) == 1:
            (ahash,) = blocks
            analysis_names[(atype, ahash)] = ANALYSES_ROOT / atype / "default.yaml"
        else:
            for source, (s_atype, s_ahash) in source_to_analysis_hash.items():
                if s_atype != atype:
                    continue
                stem = source.stem
                if stem.startswith("decoding_"):
                    stem = stem[len("decoding_"):]
                analysis_names[(atype, s_ahash)] = ANALYSES_ROOT / atype / f"{stem}.yaml"

    # Report.
    print(f"Sources: {len(sources)} legacy configs under {LEGACY_ROOT.relative_to(REPO_ROOT)}")
    print(f"Cohort configs to write: {len(cohort_writes)}")
    for atype, blocks in sorted(analysis_by_type.items()):
        names = sorted({analysis_names[(atype, h)].relative_to(REPO_ROOT) for h in blocks})
        print(f"Analysis '{atype}': {len(blocks)} unique -> {[str(n) for n in names]}")
    print()
    for source in sources:
        atype, ahash = source_to_analysis_hash[source]
        cohort_path = _cohort_target(source)
        print(
            f"  {source.relative_to(REPO_ROOT)}\n"
            f"      cohort   -> {cohort_path.relative_to(REPO_ROOT)}\n"
            f"      analysis -> {analysis_names[(atype, ahash)].relative_to(REPO_ROOT)}"
        )

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write the new tree.")
        return

    header = (
        "# Generated by scripts/split_configs.py from the legacy single-config tree.\n"
        "# Cohort config: subjects + clinical question. Pair it with an analysis\n"
        "# config (configs/analyses/<type>/) via --cohort_config + --analysis_config.\n"
    )
    for path, cohort in cohort_writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(cohort, sort_keys=False), encoding="utf-8")
    for (atype, ahash), path in analysis_names.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Generated by scripts/split_configs.py — analysis/method config.\n"
            f"# Type: {atype}. Pair with a cohort config via --cohort_config.\n"
            + yaml.safe_dump(analysis_by_type[atype][ahash], sort_keys=False)
        )
        path.write_text(body, encoding="utf-8")
    print(f"\nWrote {len(cohort_writes)} cohort + {len(analysis_names)} analysis configs.")


if __name__ == "__main__":
    main()

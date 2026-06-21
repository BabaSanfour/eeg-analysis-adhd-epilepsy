#!/usr/bin/env python3
"""``eeg-run`` — a thin local orchestrator over the per-stage CLIs.

The project is a chain of independent stages (raw → BIDS → preproc → epochs →
descriptors → merge → dim-reduction / decoding). Each stage has its own
console script and can be run by hand; this driver just sequences them in the
canonical order with resume-by-default and a ``--dry-run`` preview, so a new
user does not have to remember the order or the flags.

It runs the **sequential, single-machine** form of each stage (e.g.
``eeg-descriptors`` over all subjects at once). For large jobs use the numbered
SLURM array scripts in ``cluster/`` instead — this driver is for local runs and
for seeing the exact command chain.

Examples
--------
Preview the whole chain without running anything::

    eeg-run --dry-run --bids_root /data/BIDS --metadata /data/meta.csv \
        --raw_root /data/raw --cohort_config configs/cohorts/.../total.yaml \
        --dim_analysis_config configs/analyses/dim_reduction/default.yaml \
        --decode_analysis_config configs/analyses/decoding/EO.yaml

Run only preproc → epochs → descriptors → merge::

    eeg-run --from preprocess --to merge --bids_root /data/BIDS --metadata /data/meta.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

PKG = "eeg_adhd_epilepsy"


@dataclass
class Context:
    bids_root: Path | None
    metadata: Path | None
    raw_root: Path | None
    descriptors_config: Path
    cohort_config: Path | None
    analysis_config: Path | None
    dim_analysis_config: Path | None
    decode_analysis_config: Path | None
    segment_duration: float
    n_jobs: int


@dataclass
class Stage:
    name: str
    module: str
    #: builds the argv (after ``python -m <module>``) from the context.
    build: Callable[[Context], list[str]]
    #: optional resume predicate: returns True when the stage output already exists.
    done: Callable[[Context], bool] | None
    #: human-readable note about prerequisites (shown on --list).
    note: str = ""


def _descriptor_combined(ctx: Context) -> Path:
    assert ctx.bids_root is not None
    return ctx.bids_root / "derivatives" / "signal_features" / "descriptors" / "combined"


def _glob_exists(root: Path | None, pattern: str) -> bool:
    return root is not None and root.exists() and next(root.glob(pattern), None) is not None


STAGES: list[Stage] = [
    Stage(
        name="to-bids",
        module=f"{PKG}.preproc.to_bids",
        build=lambda c: [
            "--raw_root", str(c.raw_root),
            "--bids_root", str(c.bids_root),
            "--metadata_csv", str(c.metadata),
            "--n_jobs", str(c.n_jobs),
        ],
        done=lambda c: _glob_exists(c.bids_root, "sub-*/**/eeg/*_eeg.vhdr"),
        note="needs --raw_root, --bids_root, --metadata",
    ),
    Stage(
        name="preprocess",
        module=f"{PKG}.preproc.base",
        build=lambda c: ["--bids_root", str(c.bids_root), "--n_jobs", str(c.n_jobs)],
        done=lambda c: _glob_exists(
            c.bids_root / "derivatives" / "preproc" if c.bids_root else None,
            "**/*desc-base_eeg.fif",
        ),
        note="needs --bids_root",
    ),
    Stage(
        name="epochs",
        module=f"{PKG}.preproc.epochs",
        build=lambda c: [
            "--bids_root", str(c.bids_root),
            "--segment_duration", str(c.segment_duration),
            "--ignore_annotations",
        ],
        done=lambda c: _glob_exists(
            c.bids_root / "derivatives" if c.bids_root else None, "**/*desc-base_epo.fif"
        ),
        note="needs --bids_root, --segment_duration",
    ),
    Stage(
        name="descriptors",
        module=f"{PKG}.analysis.extract_descriptors",
        build=lambda c: [
            "--bids_root", str(c.bids_root),
            "--metadata", str(c.metadata),
            "--config", str(c.descriptors_config),
            "--conditions", "all",
        ],
        done=lambda c: _glob_exists(
            c.bids_root / "derivatives" / "signal_features" / "descriptors"
            if c.bids_root else None,
            "sub-*/**/_SUCCESS",
        ),
        note="runs all subjects sequentially (use cluster/05 for the array form)",
    ),
    Stage(
        name="merge",
        module=f"{PKG}.analysis.merge_descriptors",
        build=lambda c: ["--bids_root", str(c.bids_root)],
        done=lambda c: (_descriptor_combined(c) / "sensor_subject_features.parquet").exists(),
        note="needs descriptors complete",
    ),
    Stage(
        name="dim-reduce",
        module=f"{PKG}.analysis.dimensionality_reduction",
        build=lambda c: [
            "--bids_root", str(c.bids_root),
            "--metadata", str(c.metadata),
            "--cohort_config", str(c.cohort_config),
            "--analysis_config", str(c.dim_analysis_config or c.analysis_config),
            "--input_mode", "descriptors",
            "--analysis_mode", "flat",
            "--descriptor_table_path",
            str(_descriptor_combined(c) / "sensor_subject_features.parquet"),
            "--descriptor_feature_columns_path",
            str(_descriptor_combined(c) / "sensor_subject_features_feature_columns.json"),
            "--n_jobs", str(c.n_jobs),
        ],
        done=None,  # hashed run namespace; the tool manages its own resume
        note="needs --cohort_config + --analysis_config (configs/analyses/dim_reduction/)",
    ),
    Stage(
        name="classical-decode",
        module=f"{PKG}.analysis.classical_decoding",
        build=lambda c: [
            "--bids_root", str(c.bids_root),
            "--metadata", str(c.metadata),
            "--cohort_config", str(c.cohort_config),
            "--analysis_config", str(c.decode_analysis_config or c.analysis_config),
            "--n_jobs", str(c.n_jobs),
        ],
        done=None,
        note="needs --cohort_config + --analysis_config (configs/analyses/decoding/)",
    ),
]

STAGE_NAMES = [s.name for s in STAGES]


def _select(from_stage: str | None, to_stage: str | None) -> list[Stage]:
    start = STAGE_NAMES.index(from_stage) if from_stage else 0
    end = STAGE_NAMES.index(to_stage) + 1 if to_stage else len(STAGES)
    if start > end - 1:
        raise SystemExit(f"--from '{from_stage}' comes after --to '{to_stage}'.")
    return STAGES[start:end]


def _missing_inputs(stage: Stage, ctx: Context) -> list[str]:
    """Return required context fields the stage needs but that are unset."""
    needs: list[str] = []
    if stage.name == "to-bids" and ctx.raw_root is None:
        needs.append("--raw_root")
    if ctx.bids_root is None:
        needs.append("--bids_root")
    if stage.name in {"to-bids", "descriptors", "dim-reduce", "classical-decode"}:
        if ctx.metadata is None:
            needs.append("--metadata")
    if stage.name in {"dim-reduce", "classical-decode"}:
        if ctx.cohort_config is None:
            needs.append("--cohort_config")
        analysis_config = (
            ctx.dim_analysis_config if stage.name == "dim-reduce" else ctx.decode_analysis_config
        ) or ctx.analysis_config
        if analysis_config is None:
            specific_flag = (
                "--dim_analysis_config"
                if stage.name == "dim-reduce"
                else "--decode_analysis_config"
            )
            needs.append(f"{specific_flag} (or --analysis_config)")
    return needs


def _print_list() -> None:
    print("Pipeline stages (in order):")
    for stage in STAGES:
        print(f"  {stage.name:<18} {stage.note}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eeg-run", description="Sequence the EEG pipeline stages locally."
    )
    parser.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, default=None)
    parser.add_argument("--to", dest="to_stage", choices=STAGE_NAMES, default=None)
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Run every selected stage (ignore resume)."
    )
    parser.add_argument("--bids_root", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--raw_root", type=Path, default=None)
    parser.add_argument(
        "--descriptors_config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "descriptors.yaml",
    )
    parser.add_argument("--cohort_config", type=Path, default=None)
    parser.add_argument(
        "--analysis_config",
        type=Path,
        default=None,
        help="Analysis config for a single consumer stage (fallback for compatibility).",
    )
    parser.add_argument(
        "--dim_analysis_config",
        type=Path,
        default=None,
        help="Dimensionality-reduction config; required for a valid full-chain run.",
    )
    parser.add_argument(
        "--decode_analysis_config",
        type=Path,
        default=None,
        help="Classical-decoding config; required for a valid full-chain run.",
    )
    parser.add_argument("--segment_duration", type=float, default=10.0)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        _print_list()
        return 0

    ctx = Context(
        bids_root=args.bids_root,
        metadata=args.metadata,
        raw_root=args.raw_root,
        descriptors_config=args.descriptors_config,
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        dim_analysis_config=args.dim_analysis_config,
        decode_analysis_config=args.decode_analysis_config,
        segment_duration=args.segment_duration,
        n_jobs=args.n_jobs,
    )

    selected = _select(args.from_stage, args.to_stage)
    print(f"Selected stages: {', '.join(s.name for s in selected)}")
    for stage in selected:
        missing = _missing_inputs(stage, ctx)
        if missing:
            print(f"[{stage.name}] SKIP — missing required inputs: {', '.join(missing)}")
            continue
        if not args.overwrite and stage.done is not None and stage.done(ctx):
            print(f"[{stage.name}] skip — output already exists (resume; use --overwrite to force)")
            continue
        cmd = [sys.executable, "-m", stage.module, *stage.build(ctx)]
        print(f"[{stage.name}] $ {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[{stage.name}] FAILED (exit {result.returncode}); stopping.")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

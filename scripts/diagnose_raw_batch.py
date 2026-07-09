#!/usr/bin/env python3
"""Static, header-only scan of the epoched derivatives to explain the raw-EEG
PCA clusters (and the mysterious ``y`` colouring) *before* running any PCA.

Reads only epoch **headers** (``preload=False`` — no signal loaded, no PCA), so
it is cheap to run across the whole cohort. It answers three concrete questions:

1. **Where do the ``y`` codes come from?** ``y`` is ``epochs.events[:, -1]``.
   Codes are assigned per file as the *ordinal position* of each block condition
   (``preproc/epochs.py``: ``enumerate(conditions, start=1)``). We print, for the
   target condition, which integer code it receives in each recording — that
   distribution IS the ``{1..6}`` you see on the colour bar.

2. **Sampling rate / epoch length.** Distinct ``sfreq`` and per-epoch ``n_times``
   across recordings. If these take only a few discrete values, the downstream
   crop-to-shortest turns them into discrete feature-space clusters.

3. **What is constant (ruled out) vs varying (a candidate)?** Channel set/order,
   channel count, sfreq, n_times, event scheme — anything constant across all
   recordings cannot be causing a between-recording split.

Usage
-----
    python scripts/diagnose_raw_batch.py --bids_root /path/to/BIDS \
        --condition EO_baseline [--desc base] [--limit N] [--subjects 0002 0027]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mne
import pandas as pd

from eeg_adhd_epilepsy.io import bids as bids_io


def _iter_epoch_files(preproc_root: Path, desc: str, subjects: set[str] | None):
    for f in sorted(preproc_root.rglob(f"*desc-{desc}_epo.fif")):
        sid = bids_io.parse_bids_components(f).get("subject")
        if subjects is None or sid in subjects:
            yield f


def scan(
    bids_root: Path, desc: str, condition: str, limit: int | None, subjects: set[str] | None
) -> pd.DataFrame:
    preproc_root = bids_io.get_derivative_root(bids_root, bids_io.DerivativeStage.PREPROC)
    rows = []
    for i, f in enumerate(_iter_epoch_files(preproc_root, desc, subjects)):
        if limit is not None and i >= limit:
            break
        try:
            ep = mne.read_epochs(f, preload=False, verbose="ERROR")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[warn] could not read {f.name}: {exc}")
            continue
        code_for_condition = ep.event_id.get(condition)
        rows.append(
            {
                "file": f.name,
                "sfreq": round(float(ep.info["sfreq"]), 3),
                "n_times": int(ep.times.size),
                "n_ch": len(ep.ch_names),
                "ch_sig": "|".join(ep.ch_names),  # montage identity + order
                "n_conditions": len(ep.event_id),  # max ordinal available
                "conditions": ",".join(sorted(ep.event_id)),
                "y_code_for_condition": code_for_condition,
                "has_condition": code_for_condition is not None,
            }
        )
    return pd.DataFrame(rows)


def _report_constant_vs_varying(df: pd.DataFrame) -> None:
    print("\n========= CONSTANT (ruled out) vs VARYING (candidate) =====")
    for col, label in [
        ("n_ch", "channel count"),
        ("ch_sig", "channel set & order (montage)"),
        ("sfreq", "sampling rate"),
        ("n_times", "epoch length (samples)"),
        ("conditions", "condition/event scheme"),
    ]:
        vals = df[col].dropna().unique()
        verdict = (
            "CONSTANT  -> ruled out"
            if len(vals) <= 1
            else f"VARYING ({len(vals)} values) -> CANDIDATE"
        )
        preview = "" if col == "ch_sig" else f"  values={sorted(vals.tolist())[:8]}"
        print(f"  {label:32s}: {verdict}{preview}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bids_root", required=True, type=Path)
    p.add_argument("--desc", default="base")
    p.add_argument("--condition", default="EO_baseline")
    p.add_argument("--limit", type=int, default=None, help="scan only the first N files")
    p.add_argument(
        "--subjects", nargs="+", default=None, help="BIDS subject labels, e.g. 0002 0027"
    )
    args = p.parse_args()

    subjects = None
    if args.subjects:
        subjects = {bids_io.study_id_to_bids_subject(s) for s in args.subjects}

    df = scan(args.bids_root, args.desc, args.condition, args.limit, subjects)
    if df.empty:
        print("No epoch files found. Check --bids_root / --desc.")
        return

    print(f"Scanned {len(df)} recordings (desc-{args.desc}).")

    # (2) sampling rate / epoch length spread
    print("\n================ SAMPLING RATE / EPOCH LENGTH ================")
    print(df.value_counts(["sfreq", "n_times", "n_ch"]).to_string())

    # (3) constant vs varying
    _report_constant_vs_varying(df)

    # (1) where the y codes come from
    print(f"\n====== WHERE THE `y` CODES COME FROM (condition={args.condition!r}) ======")
    print(f"recordings containing {args.condition!r}: {int(df['has_condition'].sum())} / {len(df)}")
    present = df[df["has_condition"]]
    if not present.empty:
        dist = Counter(present["y_code_for_condition"].astype(int))
        print(f"integer code assigned to {args.condition!r}, by recording count:")
        for code in sorted(dist):
            print(f"    y == {code}:  {dist[code]} recordings")
        print(
            "(these are the 1..N values on the colour bar; they are just the block's "
            "ordinal slot per file, NOT a stable label)"
        )
    print("\nnumber of distinct block conditions per recording (the ordinal ceiling):")
    print(df["n_conditions"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

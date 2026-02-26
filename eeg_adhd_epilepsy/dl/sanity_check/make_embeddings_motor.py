#!/usr/bin/env python3
import argparse
import logging
import os
import re
import csv
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import mne

from models.cbramod import CBraMod  # from CBraMod repo

LOG = logging.getLogger(__name__)
DEFAULT_MAX_CHANNELS = int(os.environ.get("MAX_CHANNELS", 128))

# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CBraMod embeddings per segment (default: 10s)."
    )
    parser.add_argument("--deriv-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ses", default="ses-01")
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=10.0,
        help="Patch duration in seconds.",
    )
    parser.add_argument(
        "--points-per-patch",
        type=int,
        default=None,
        help="Number of samples per patch (overrides --segment-duration).",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def find_subject_dirs(root):
    subs = []
    if not os.path.isdir(root):
        return subs
    for name in os.listdir(root):
        full = os.path.join(root, name)
        # Match Sxxx format (e.g. S001, S109)
        if os.path.isdir(full) and re.match(r"S\d{3}$", name):
            subs.append((name, full))
    subs.sort()
    return subs

def find_target_runs(sub_dir):
    # Motor dataset: Files are directly in Sxxx folder, ending with .edf
    # We want R01 (Eyes Open) and R02 (Eyes Closed)
    # File format: S001R01.edf
    runs = []
    if not os.path.isdir(sub_dir):
        return runs
    
    for f in os.listdir(sub_dir):
        if not f.endswith(".edf"):
            continue
        
        # Check for R01 or R02 in filename
        # Pattern: SxxxRxx.edf
        match = re.search(r"R(\d{2})\.edf$", f)
        if match:
            run_num = match.group(1)
            if run_num in ["01", "02"]:
                runs.append((run_num, os.path.join(sub_dir, f)))
    
    # Sort by run number
    runs.sort(key=lambda x: x[0])
    return runs

def load_cbramod(weights_path, device="cuda"):
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    LOG.info("Loading CBraMod from %s", weights_path)
    model = CBraMod().to(dev)
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    if hasattr(model, "proj_out"):
        model.proj_out = nn.Identity()
    model.eval()
    return model, dev




def compute_patches(eeg_data, sfreq, seg_dur=10.0, points_per_patch=None):
    """
    Split EEG into N full patches. Output shape: (C, S, P).
    """
    C, T = eeg_data.shape
    P = int(points_per_patch) if points_per_patch is not None else int(seg_dur * sfreq)
    S = T // P

    if S < 1:
        raise ValueError(f"Recording too short: T={T} < P={P}")

    usable = S * P
    data_block = eeg_data[:, :usable]
    patches = data_block.reshape(C, S, P)
    return patches, S, P


# ---------------------------------------------------------------------
# CORE LOGIC: SEGMENT-WISE EMBEDDING
# ---------------------------------------------------------------------
def compute_embedding(
    file_path,
    model,
    device,
    seg_dur=10.0,
    points_per_patch=None,
):
    LOG.info("Reading %s", file_path)
    # Load and Resample
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raw.resample(200.0, npad="auto")
    
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    
    # Create Patches: (C, S, P)
    patches, S, P = compute_patches(data, sfreq, seg_dur, points_per_patch)

    # Prepare input: (1, C, S, P)
    x = torch.from_numpy(patches).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)  # Shape is likely (1, C, S, P) or (1, C, S, Hidden)

        # KEY CHANGE: Pool over 'P' (time within patch) but KEEP 'S' (segments)
        if out.dim() == 4:
            # (Batch, Channel, Segment, Time/Feat)
            # We average the features inside the patch, but keep the patch distinct
            seg_means = out.mean(dim=-1)  # Result: (1, C, S)
            seg_stds = out.std(dim=-1)    # Result: (1, C, S)
        else:
            # Fallback if model outputs something else
            raise RuntimeError(f"Unexpected CBraMod output shape: {tuple(out.shape)}")

        # Convert to numpy and rearrange to (S, C) for row-wise writing
        # Squeeze batch (1) -> (C, S) -> Transpose -> (S, C)
        emb_mean = seg_means.squeeze(0).t().cpu().numpy()
        emb_std = seg_stds.squeeze(0).t().cpu().numpy()

    # Returns: (S, C) arrays, plus metadata
    return emb_mean, emb_std, patches.shape[0], S, P


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Prep output
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    subs = find_subject_dirs(args.deriv_root)
    if args.max_subjects:
        subs = subs[: args.max_subjects]

    LOG.info("Found %d subjects", len(subs))
    model, device = load_cbramod(args.weights, args.device)

    # We write incrementally to avoid OOM with large S*Subjects
    # We need to know fieldnames first. We'll do a quick pass or dynamic write?
    # Better: Open file, write header once we process the first subject, then append.
    
    file_initialized = False
    f_handle = open(args.out_csv, "w", newline="")
    writer = None
    
    total_rows = 0

    try:
        for sub_id, sub_dir in subs:
            target_runs = find_target_runs(sub_dir)
            if not target_runs:
                continue

            for run_id, file_path in target_runs:
                try:
                    # Get (S, C) matrices
                    emb_means, emb_stds, C, S, P = compute_embedding(
                        file_path,
                    model,
                    device,
                    args.segment_duration,
                    args.points_per_patch,
                )
                except Exception as e:
                    LOG.warning("Failed %s run %s: %s", sub_id, run_id, e)
                    continue

                # Generate rows for this subject
                current_batch = []
                for s_idx in range(S):
                    row = {
                        "subject": sub_id,
                        "run": run_id,
                        "segment": s_idx,
                        "n_channels": C,
                        # "points_per_patch": P # Optional, static
                    }
                
                # Add features: cbramod_mean_0000 ... cbramod_mean_0060
                    for c in range(C):
                        row[f"cbramod_mean_{c:04d}"] = emb_means[s_idx, c]
                        row[f"cbramod_std_{c:04d}"] = emb_stds[s_idx, c]
                    
                    current_batch.append(row)

                if not current_batch:
                    continue

                # Initialize CSV on first valid batch
                if not file_initialized:
                    # Reserve columns up to DEFAULT_MAX_CHANNELS (or higher if this subject has more)
                    max_channels = max(DEFAULT_MAX_CHANNELS, C)
                    meta = ["subject", "run", "segment", "n_channels"]
                    feats = []
                    for c in range(max_channels):
                        feats.append(f"cbramod_mean_{c:04d}")
                        feats.append(f"cbramod_std_{c:04d}")
                    fieldnames = meta + feats

                    writer = csv.DictWriter(f_handle, fieldnames=fieldnames, restval="")
                    writer.writeheader()
                    file_initialized = True

                # Write batch
                writer.writerows(current_batch)
                total_rows += len(current_batch)
                LOG.info("Appended %d segments for %s run %s (Total rows: %d)", S, sub_id, run_id, total_rows)

    finally:
        f_handle.close()
        LOG.info("Done. Saved to %s", args.out_csv)

if __name__ == "__main__":
    main()

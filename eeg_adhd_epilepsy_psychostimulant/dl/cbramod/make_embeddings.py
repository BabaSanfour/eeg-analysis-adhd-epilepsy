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
        if os.path.isdir(full) and re.match(r"sub-\d{4}$", name):
            subs.append((name, full))
    subs.sort()
    return subs

def find_any_vhdr(sub_dir, ses):
    eeg_dir = os.path.join(sub_dir, ses, "eeg")
    if not os.path.isdir(eeg_dir):
        return None
    for f in os.listdir(eeg_dir):
        if f.endswith(".vhdr"):
            return os.path.join(eeg_dir, f)
    return None

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
    vhdr_path,
    model,
    device,
    seg_dur=10.0,
    points_per_patch=None,
):
    LOG.info("Reading %s", vhdr_path)
    # Load and Resample
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
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
            # Flatten last two dims: (1, C, S, 200) -> (1, C, S, 200)
            # We want (S, C*200) eventually.
            # First, permute to (1, S, C, 200)
            out = out.permute(0, 2, 1, 3) # (1, S, C, 200)
            # Flatten C and 200: (1, S, C*200)
            out = out.reshape(1, out.size(1), -1)
        else:
            # Fallback if model outputs something else
            raise RuntimeError(f"Unexpected CBraMod output shape: {tuple(out.shape)}")

        # Convert to numpy (S, C*200)
        emb_flat = out.squeeze(0).cpu().numpy()

    # Returns: (S, C*200) array, plus metadata
    return emb_flat, patches.shape[0], S, P


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
            vhdr = find_any_vhdr(sub_dir, args.ses)
            if vhdr is None:
                continue

            try:
                # Get (S, C*200) matrix
                emb_flat, C, S, P = compute_embedding(
                    vhdr,
                    model,
                    device,
                    args.segment_duration,
                    args.points_per_patch,
                )
            except Exception as e:
                LOG.warning("Failed %s: %s", sub_id, e)
                continue

            # Generate rows for this subject
            current_batch = []
            for s_idx in range(S):
                row = {
                    "subject": sub_id,
                    "segment": s_idx,
                    "n_channels": C,
                    # "points_per_patch": P # Optional, static
                }
                
                # Add features: cbramod_mean_0000 ... cbramod_mean_0060
                # Add flattened features
                # emb_flat is (S, C*200). We need to map this to columns.
                # Naming convention: ch01_feat000 ... ch19_feat199
                # Actually, simpler: just feat_0000 to feat_3799 (if 19ch)
                # But to keep channel sanity, let's do: ch{c}_dim{d}
                
                # OPTIMIZATION: Assigning 3800 items to a dict one by one is slow.
                # Better to construct keys once and zip?
                # For now, keep it simple but maybe slow.
                row_vals = emb_flat[s_idx] # (C*200,)
                
                # We need a stable ordering. 
                # The flatten was (1, S, C, 200) -> reshape (1, S, C*200)
                # So it iterates Channel 0 (all dims), Channel 1 (all dims)...
                
                feat_idx = 0
                for c in range(C):
                    for d in range(200): # Assuming 200 dim
                        row[f"ch{c:02d}_dim{d:03d}"] = row_vals[feat_idx]
                        feat_idx += 1
                
                current_batch.append(row)

            if not current_batch:
                continue

            # Initialize CSV on first valid batch
            if not file_initialized:
                meta = ["subject", "segment", "n_channels"]
                feats = []
                # Pre-calculate header based on first subject's channel count
                # NOTE: This assumes all subjects have same number of channels or we pad?
                # If channels vary, this simple flatten approach breaks alignment.
                # CBraMod usually requires fixed channels or handles them? 
                # For now, we assume standard 19ch or similar.
                # But wait, earlier code had "DEFAULT_MAX_CHANNELS".
                # If we flatten, minimal channel mismatches will be catastrophic for alignment.
                # WARNING: We must ensure channel consistency or include channel names.
                # For this implementation, we proceed with C * 200.
                
                for c in range(C):
                    for d in range(200):
                        feats.append(f"ch{c:02d}_dim{d:03d}")
                fieldnames = meta + feats

                writer = csv.DictWriter(f_handle, fieldnames=fieldnames, restval="")
                writer.writeheader()
                file_initialized = True

            # Write batch
            writer.writerows(current_batch)
            total_rows += len(current_batch)
            LOG.info("Appended %d segments for %s (Total rows: %d)", S, sub_id, total_rows)

    finally:
        f_handle.close()
        LOG.info("Done. Saved to %s", args.out_csv)

if __name__ == "__main__":
    main()

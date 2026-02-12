#!/usr/bin/env python3
import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne

# Try importing CBraMod, handle failure gracefully if not in path
try:
    from models.cbramod import CBraMod
except ImportError:
    # If running from a different root, we might need to adjust path or assume user has it installed
    logging.warning("Could not import models.cbramod. Ensure 'models' package is in PYTHONPATH.")
    pass

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CBraMod embeddings for '_desc-base_eeg.fif' files."
    )
    parser.add_argument("--deriv-root", required=True, help="Root directory containing subject folders")
    parser.add_argument("--out-file", required=True, help="Output file path (.h5)")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
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
    parser.add_argument("--max-subjects", type=int, default=None, help="Debug: limit number of subjects")
    
    # args unused but kept/modified for compatibility if needed
    parser.add_argument("--out-csv", help="Legacy argument, use --out-file") 
    parser.add_argument("--ses", help="Legacy argument, ignored for this flat structure")

    return parser.parse_args()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def find_subject_dirs(root):
    """Finds all subject directories (starting with sub-) in root."""
    subs = []
    if not os.path.exists(root):
        return subs
    
    # Sort to ensure consistent processing order
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith("sub-"):
            subs.append((name, full))
    return subs

def find_target_file(sub_dir):
    """
    Looks for *desc-base_eeg.fif in sub_dir/eeg/
    """
    eeg_dir = os.path.join(sub_dir, "eeg")
    if not os.path.isdir(eeg_dir):
        return None
        
    for f in os.listdir(eeg_dir):
        if f.endswith("desc-base_eeg.fif"):
            return os.path.join(eeg_dir, f)
            
    return None

def load_cbramod(weights_path, device="cuda"):
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    LOG.info("Loading CBraMod from %s (device=%s)", weights_path, dev)
    
    # Initialize model
    model = CBraMod().to(dev)
    
    # Load weights
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    
    # Remove projection head if present to get embeddings
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
    
    if P <= 0:
        raise ValueError(f"Invalid patch size P={P}")

    S = T // P

    if S < 1:
        # If signal is shorter than one patch, we can't do anything
        raise ValueError(f"Recording too short: T={T} < P={P}")

    usable = S * P
    data_block = eeg_data[:, :usable]
    
    # Reshape to (Channels, Segments, Points)
    patches = data_block.reshape(C, S, P)
    return patches, S, P


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------
def compute_embedding(
    file_path,
    model,
    device,
    seg_dur=10.0,
    points_per_patch=None,
):
    LOG.info("Processing %s", file_path)
    
    # Load .fif
    # preload=True is needed for resampling
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
    
    # Resample to 200Hz if needed (CBraMod requirement)
    if raw.info["sfreq"] != 200.0:
        # LOG.info("Resampling %.1f -> 200.0 Hz", raw.info["sfreq"])
        raw.resample(200.0, npad="auto")
    
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    
    # Create Patches: (C, S, P)
    patches, S, P = compute_patches(data, sfreq, seg_dur, points_per_patch)

    # Prepare input: (1, C, S, P)
    x = torch.from_numpy(patches).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x) 
        
        # Check output shape
        # Expected: (Batch, Channel, Segment, Time/Feat) -> (1, C, S, 200)
        if out.dim() == 4:
            # We want to stack everything into a flat vector per segment
            # Target shape: (S, C*200)
            
            # 1. Permute to (Batch, Segment, Channel, Time) -> (1, S, C, 200)
            out = out.permute(0, 2, 1, 3) 
            
            # 2. Flatten Channel and Time dimensions
            # (1, S, C*200)
            out = out.reshape(1, out.size(1), -1)
            
            emb_flat = out.squeeze(0).cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected CBraMod output shape: {tuple(out.shape)}")

    return emb_flat, patches.shape[0], S, P


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    
    # Resolve output path
    # output arg might be --out-file or --out-csv (legacy)
    out_path = args.out_file if args.out_file else args.out_csv
    if not out_path:
        raise ValueError("Output file path is required (--out-file)")

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Find subjects
    subs = find_subject_dirs(args.deriv_root)
    if not subs:
        LOG.error("No subject directories found in %s", args.deriv_root)
        return

    # Check for existing results to resume
    processed_subs = set()
    if os.path.exists(out_path):
        try:
            # Check if valid HDF and read processed subjects
            with pd.HDFStore(out_path, mode='r') as store:
                if 'embeddings' in store:
                    # Read only the 'subject', 'segment' columns just to identify unique subjects efficiently?
                    # Actually, 'subject' is sufficient.
                    # We can use select_column if indexed, but it might not be indexed.
                    # Reading full 'subject' column is safer.
                    # If file is huge, this might be slow, but better than crashing.
                    existing_df = pd.read_hdf(out_path, key="embeddings", columns=["subject"])
                    processed_subs = set(existing_df["subject"].unique())
                    LOG.info("Resuming: Found %d subjects already processed in %s.", len(processed_subs), out_path)
                else:
                    LOG.warning("File exists but 'embeddings' key not found. Starting fresh (appending).")
        except Exception as e:
            LOG.warning("Could not read existing file %s to resume: %s. Proceeding with caution (appending mode).", out_path, e)

    # Filter subjects
    original_count = len(subs)
    subs = [s for s in subs if s[0] not in processed_subs]
    
    if args.max_subjects:
        subs = subs[: args.max_subjects]

    if not subs and len(processed_subs) > 0:
        LOG.info("All subjects already processed! (%d total)", len(processed_subs))
        return

    LOG.info("Found %d pending subjects (out of %d total). Starting processing...", len(subs), original_count)

    LOG.info("Found %d subjects. Starting processing...", len(subs))
    
    # Load Model
    model, device = load_cbramod(args.weights, args.device)

    # Processing Loop
    total_segments = 0
    
    # We need to know column names for the DataFrame. 
    # Validating on the first subject, then enforcing consistency.
    feat_cols = None
    
    # Monitoring
    skipped_subs = []
    failed_subs = []
    
    for sub_id, sub_dir in subs:
        target_file = find_target_file(sub_dir)
        
        if not target_file:
            LOG.warning("SKIP [%s]: No '_desc-base_eeg.fif' found in %s", sub_id, sub_dir)
            skipped_subs.append(sub_id)
            continue
            
        try:
            emb_flat, C, S, P = compute_embedding(
                target_file,
                model,
                device,
                args.segment_duration,
                args.points_per_patch
            )
        except Exception as e:
            LOG.error("ERROR [%s]: Processing failed: %s", sub_id, e)
            failed_subs.append(sub_id)
            continue

        # Column Naming / Schema Validation
        current_n_feats = C * 200
        
        if feat_cols is None:
            # Initialize columns based on first valid subject
            feat_cols = []
            for c in range(C):
                for d in range(200):
                    feat_cols.append(f"ch{c:02d}_dim{d:03d}")
        elif len(feat_cols) != current_n_feats:
            LOG.error("ERROR [%s]: Channel mismatch (Expected %d, Got %d). Skipping.", 
                      sub_id, len(feat_cols), current_n_feats)
            failed_subs.append(sub_id)
            continue
            
        # Create DataFrame
        df = pd.DataFrame(emb_flat, columns=feat_cols)
        
        # Add Metadata
        df.insert(0, "n_channels", C)
        df.insert(0, "segment", np.arange(S))
        df.insert(0, "subject", sub_id)
        
        # Save to HDF5
        try:
            df.to_hdf(
                out_path, 
                key="embeddings", 
                mode="a", 
                format="table", 
                append=True, 
                min_itemsize={"subject": 32}, 
                index=False
            )
            total_segments += S
            LOG.info("SAVED [%s]: %d segments (Total: %d)", sub_id, S, total_segments)
            
        except Exception as e:
            LOG.error("ERROR [%s]: Failed to write HDF5: %s", sub_id, e)
            failed_subs.append(sub_id)

    LOG.info("All finished. Output saved to %s", out_path)
    if skipped_subs:
        LOG.info("SUMMARY: Skipped %d subjects: %s", len(skipped_subs), skipped_subs)
    if failed_subs:
        LOG.info("SUMMARY: Failed %d subjects: %s", len(failed_subs), failed_subs)

if __name__ == "__main__":
    main()

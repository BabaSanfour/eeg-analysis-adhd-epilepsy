#!/usr/bin/env python3
import argparse
import logging
import os
import re
import csv

import numpy as np
import torch
import torch.nn as nn
import mne

from models.cbramod import CBraMod  # from CBraMod repo

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CBraMod embeddings per segment with CAR preprocessing."
    )
    parser.add_argument("--deriv-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ses", default="ses-01")
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=1.0, # Recommended 1s for better resolution
        help="Patch duration in seconds.",
    )
    parser.add_argument(
        "--points-per-patch",
        type=int,
        default=200, # Recommended to match CBraMod 1s @ 200Hz
        help="Number of samples per patch (overrides --segment-duration).",
    )
    parser.add_argument(
        "--pooling-mode",
        choices=["meanstd", "attn"],
        default="meanstd",
        help="How to pool CBraMod outputs across time within a segment: "
             "'meanstd' (original) or 'attn' (softmax attention over time).",
    )
    parser.add_argument(
        "--max-segments-per-subject",
        type=int,
        default=None,
        help="Optional: randomly subsample this many segments per subject to reduce file size.",
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
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if hasattr(model, "proj_out"):
        model.proj_out = nn.Identity()
    model.eval()
    return model, dev

EXPECTED_CHANNELS = 19


def _pad_or_trim_channels(data: np.ndarray, expected: int = EXPECTED_CHANNELS):
    """Pad with zeros or trim channels to match CBraMod expected channel count."""
    C, T = data.shape
    if C == expected:
        return data
    if C > expected:
        return data[:expected, :]
    # pad
    pad = np.zeros((expected - C, T), dtype=data.dtype)
    return np.vstack([data, pad])


def compute_patches(eeg_data, sfreq, seg_dur=1.0, points_per_patch=None):
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
# CORE LOGIC: SEGMENT-WISE EMBEDDING WITH CAR
# ---------------------------------------------------------------------
def compute_embedding(
    vhdr_path,
    model,
    device,
    seg_dur=1.0,
    points_per_patch=None,
    pooling_mode="meanstd",
):
    LOG.info("Reading %s", vhdr_path)
    
    # 1. Load data with preload=True to allow preprocessing
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    
    # 2. Bandpass Filter (0.5 - 40 Hz) 
    # Removes low-frequency drift and high-frequency muscle noise
    raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
    
    # 3. Common Average Reference (CAR)
    # Centers signal across scalp, reducing global artifacts
    
    raw.set_eeg_reference('average', projection=False, verbose=False)
    
    # 4. Resample to 200.0 Hz (Requirement for CBraMod weights)
    raw.resample(200.0, npad="auto")
    
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    data = _pad_or_trim_channels(data, EXPECTED_CHANNELS)
    
    # Create Patches: (C, S, P)
    patches, S, P = compute_patches(data, sfreq, seg_dur, points_per_patch)

    # Prepare input: (1, C, S, P)
    x = torch.from_numpy(patches).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)  # Shape: (1, C, S, P)

        if out.dim() == 4:
            if pooling_mode == "meanstd":
                seg_means = out.mean(dim=-1).squeeze(0).t().cpu().numpy()
                seg_stds = out.std(dim=-1).squeeze(0).t().cpu().numpy()
            elif pooling_mode == "attn":
                # attention over time (P) per segment using channel-mean scores
                x_attn = out.squeeze(0)  # (C, S, P)
                scores = x_attn.mean(dim=0)             # (S, P)
                weights = torch.softmax(scores, dim=-1) # (S, P)
                weighted = weights.unsqueeze(0) * x_attn  # (C, S, P)
                seg_means = weighted.sum(dim=-1).permute(1, 0).cpu().numpy()  # (S, C)

                # weighted std
                mean_exp = torch.from_numpy(seg_means).to(x_attn.device).permute(1,0).unsqueeze(-1)  # (C,S,1)
                var = (weights.unsqueeze(0) * (x_attn - mean_exp) ** 2).sum(dim=-1)
                seg_stds = torch.sqrt(torch.clamp(var, min=1e-8)).permute(1, 0).cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling_mode '{pooling_mode}'")
        else:
            raise RuntimeError(f"Unexpected CBraMod output shape: {tuple(out.shape)}")

    return seg_means, seg_stds, patches.shape[0], S, P


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

    feat_prefix = "cbramod_attn" if args.pooling_mode == "attn" else "cbramod"

    file_initialized = False
    
    # Open file once and write rows subject by subject to prevent memory blowup
    with open(args.out_csv, "w", newline="") as f_handle:
        writer = None
        total_rows = 0

        for sub_id, sub_dir in subs:
            vhdr = find_any_vhdr(sub_dir, args.ses)
            if vhdr is None:
                continue

            try:
                # Get (S, C) matrices
                emb_means, emb_stds, C, S, P = compute_embedding(
                    vhdr,
                    model,
                    device,
                    args.segment_duration,
                    args.points_per_patch,
                    args.pooling_mode,
                )
            except Exception as e:
                LOG.warning("Failed %s: %s", sub_id, e)
                continue

            # Optional segment subsampling
            rng = np.random.default_rng(42)
            seg_indices = np.arange(S)
            if args.max_segments_per_subject:
                k = min(args.max_segments_per_subject, S)
                seg_indices = np.sort(rng.choice(seg_indices, size=k, replace=False))

            # Generate rows for this subject
            current_batch = []
            for s_idx in seg_indices:
                row = {
                    "subject": sub_id,
                    "segment": s_idx,
                    "n_channels": C,
                }
                
                # Add features: cbramod_mean_0000 ... cbramod_mean_00XX
                for c in range(C):
                    row[f"{feat_prefix}_mean_{c:04d}"] = float(emb_means[s_idx, c])
                    row[f"{feat_prefix}_std_{c:04d}"] = float(emb_stds[s_idx, c])
                
                current_batch.append(row)

            if not current_batch:
                continue

            # Initialize CSV Writer on first valid batch
            if not file_initialized:
                keys = list(current_batch[0].keys())
                meta = ["subject", "segment", "n_channels"]
                feats = sorted([k for k in keys if k not in meta])
                fieldnames = meta + feats
                
                writer = csv.DictWriter(f_handle, fieldnames=fieldnames)
                writer.writeheader()
                file_initialized = True

            # Write batch to disk
            writer.writerows(current_batch)
            total_rows += len(current_batch)
            LOG.info("Appended %d segments for %s (Total rows: %d)", S, sub_id, total_rows)

    LOG.info("Extraction complete. Final file saved to %s", args.out_csv)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate CBraMod embeddings from BIDS-formatted EEG and save to a flat table.

The output CSV is shaped like any other feature table in this repo:
rows = segments (e.g. 10s patches), columns = embedding dims (mean/std per channel).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from mne_bids import BIDSPath

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_cbramod(ckpt_path: Path, device: str):
    """Lazy import to avoid hard dependency when not requested."""
    try:
        from models.cbramod import CBraMod
    except ImportError as exc:
        raise ImportError(
            "CBraMod is not installed. Clone https://github.com/wjq-learning/CBraMod"
        ) from exc

    if not ckpt_path.exists():
        raise FileNotFoundError(f"CBraMod checkpoint not found: {ckpt_path}")

    model = CBraMod().to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        state = state.get("state_dict") or state.get("model_state_dict") or state
    
    model.load_state_dict(state)
    
    # Disable projection head if it exists to get backbone features
    if hasattr(model, "proj_out"):
        import torch.nn as nn
        model.proj_out = nn.Identity()
        
    model.eval()
    return model





def _iter_segments(
    raw: mne.io.BaseRaw,
    segment_duration: float,
) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield (segment_idx, data) for each contiguous window."""
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    samples_per_seg = int(segment_duration * sfreq)
    total_samples = data.shape[1]
    
    # Drop the remainder to ensure all segments are equal length
    n_segments = total_samples // samples_per_seg
    
    for seg in range(n_segments):
        start = seg * samples_per_seg
        end = start + samples_per_seg
        yield seg, data[:, start:end]


def _embed_segment(model, segment: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward a single segment through CBraMod and pool features.
    Returns: (means, stds) both shape (n_channels,)
    """
    # segment shape: (C, T) -> Input (1, C, T)
    x = torch.from_numpy(segment).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(x) # Expected shape (1, C, T_embed) or (1, C, Features)
        
        # Robust Pooling Strategy:
        # Calculate Mean and Std across the feature/time dimension
        # Preserving the Channel dimension.
        if out.dim() == 3: # (Batch, Channels, Features)
             means = out.mean(dim=-1)
             stds = out.std(dim=-1)
        elif out.dim() == 4: # (Batch, Channels, Time, Features)
             # Flatten last two dims then pool
             flat = out.flatten(start_dim=2)
             means = flat.mean(dim=-1)
             stds = flat.std(dim=-1)
        else:
             # Fallback for 2D output (Batch, Features) - unlikely for per-channel model
             means = out
             stds = torch.zeros_like(out)

    return means.cpu().numpy().flatten(), stds.cpu().numpy().flatten()


def compute_cbramod_embeddings_to_df(
    derivatives_dir: Path,
    ckpt_path: Path,
    subjects: Optional[List[str]] = None,
    segment_duration: int = 1, # Default to 1s patches
    processing: Optional[str] = None,
    picks: Optional[str] = "eeg",
    overwrite: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute CBraMod embeddings for all subjects and return a flat DataFrame.
    """
    if cache_path and cache_path.exists() and not overwrite:
        logger.info("Loading cached CBraMod embeddings from %s", cache_path)
        return pd.read_csv(cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_cbramod(ckpt_path, device)

    root = Path(derivatives_dir)
    if subjects is None:
        subjects = sorted(p.name.split("-")[-1] for p in root.glob("sub-*") if p.is_dir())
    
    results: List[Dict[str, float]] = []

    for subj in subjects:
        # Construct BIDS path (adjust task/run/session as needed)
        bids = BIDSPath(
            root=str(root),
            subject=str(subj),
            session="01",
            task="RESTING", 
            suffix="eeg",
            extension=".vhdr",
            datatype="eeg",
            processing=processing,
        )
        
        # Flexible finder if precise BIDS entities vary
        fpath = bids.fpath
        if not fpath.exists():
            # Try finding ANY .vhdr for this subject/session
            potentials = list(Path(bids.directory).glob("*.vhdr"))
            if potentials:
                fpath = potentials[0]
            else:
                logger.warning("Skipping sub-%s (no .vhdr found)", subj)
                continue

        try:
            raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose=False)
            if picks:
                raw.pick(picks)

            # Ensure 200Hz resampling to match CBraMod training
            if raw.info["sfreq"] != 200:
                raw.resample(200, npad='auto')
        except Exception as exc:
            logger.error("Could not load %s: %s", fpath, exc)
            continue

        for seg_idx, segment in _iter_segments(raw, segment_duration):
            try:
                means, stds = _embed_segment(model, segment, device)
            except Exception as exc:
                logger.error("Embedding failed for sub-%s seg-%d: %s", subj, seg_idx, exc)
                continue
            
            row = {
                "subject": f"sub-{subj}", # Standardize subject ID
                "segment": seg_idx,
                "n_channels": len(means)
            }
            
            # Save features with consistent naming
            for i, (m, s) in enumerate(zip(means, stds)):
                row[f"cbramod_mean_{i:04d}"] = float(m)
                row[f"cbramod_std_{i:04d}"] = float(s)
                
            results.append(row)
            
        logger.info("Processed sub-%s: %d segments", subj, seg_idx + 1)

    df = pd.DataFrame(results)
    
    # Sort columns for tidiness
    cols = sorted(df.columns.tolist())
    # Move meta columns to front
    for col in ["n_channels", "segment", "subject"]:
        if col in cols:
            cols.insert(0, cols.pop(cols.index(col)))
    df = df[cols]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Saved CBraMod embeddings to %s", cache_path)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CBraMod EEG embeddings.")
    parser.add_argument("--derivatives_dir", type=Path, required=True, help="BIDS derivatives root.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to CBraMod checkpoint.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Where to save embeddings CSV.")
    parser.add_argument("--subjects", nargs="*", help="Subject IDs to process.")
    parser.add_argument("--segment_duration", type=int, default=1, help="Segment length in seconds (default 1s).")
    parser.add_argument("--processing", type=str, default=None, help="BIDS processing label.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if out_csv exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = compute_cbramod_embeddings_to_df(
        derivatives_dir=args.derivatives_dir,
        ckpt_path=args.ckpt_path,
        subjects=args.subjects,
        segment_duration=args.segment_duration,
        processing=args.processing,
        overwrite=args.overwrite,
        cache_path=args.out_csv,
    )
    logger.info("Finished. Shape: %s", df.shape)


if __name__ == "__main__":
    main()

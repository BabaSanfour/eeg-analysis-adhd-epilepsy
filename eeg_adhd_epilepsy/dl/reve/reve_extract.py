
import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
import mne
from tqdm import tqdm
from reve_model import REVEFeatureExtractor
from reve_preprocessing import preprocess_signal

def parse_args():
    parser = argparse.ArgumentParser(description="Extract REVE embeddings for epilepsy classification")
    parser.add_argument("--data-root", required=True, help="Root directory containing BIDS-like subject folders")
    parser.add_argument("--output-dir", default="/home/mat/scratch/embeddings/reve/baseline", help="Directory to save per-subject embeddings (BIDS format)")
    parser.add_argument("--output-csv", default=None, help="Optional: Save all embeddings as CSV file")
    parser.add_argument("--model-size", default="base", choices=["base", "large"], help="REVE model size")
    parser.add_argument("--stage", default="baseline", choices=["baseline", "correct_ica", "correct_dss", "denoise_ar", "denoise_dss_ar"], help="Preprocessing stage to extract")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subject", default=None, help="Specific subject ID to process (e.g., '1' or 'sub-0001')")
    parser.add_argument("--no-pool", action="store_true", help="Disable pooling and return all tokens")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N subjects")
    return parser.parse_args()

def find_eeg_file(sub_dir, stage="base"):
    """Find the first .fif file matching the stage in the subject directory, searching recursively."""
    # Pattern map
    patterns = {
        "baseline": "desc-base_eeg.fif",
        "correct_ica": "desc-correctIca_eeg.fif",
        "correct_dss": "desc-correctDss_eeg.fif",
        "denoise_ar": "desc-denoiseAr_eeg.fif",
        "denoise_dss_ar": "denoiseWienerDss_eeg.fif"
    }
    suffix = patterns.get(stage, "desc-base_eeg.fif")
    
    # Look for .fif files matching the stage pattern
    for root, dirs, files in os.walk(sub_dir):
        for file in files:
            if file.endswith(suffix):
                return os.path.join(root, file)
    return None

def extract_features(args):
    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Find Subjects
    sub_dirs = []
    if os.path.exists(args.data_root):
        for name in sorted(os.listdir(args.data_root)):
            # Handle specific subject filter
            if args.subject:
                target_sub = str(args.subject).replace('sub-', '')
                if target_sub not in name:
                    continue
                    
            if name.startswith("sub-"):
                sub_dirs.append((name, os.path.join(args.data_root, name)))
    
    if not sub_dirs:
        print(f"No valid subject directories found in {args.data_root}")
        return
    
    if args.limit:
        sub_dirs = sub_dirs[:args.limit]
    
    # 3. Initialize Model
    print(f"Loading REVE model ({args.model_size}) on {args.device}...")
    model = REVEFeatureExtractor(model_size=args.model_size)
    model.to(args.device)
    model.eval()
    
    # Tracking for summary
    success_count = 0
    skipped_count = 0
    error_count = 0

    # 4. Process Subjects
    print(f"Processing {len(sub_dirs)} subjects...")
    for sub_id, sub_dir in tqdm(sub_dirs):
        
        # Check if already processed
        # Update filename based on stage
        suffix_map = {
            "baseline": "desc-baseline",
            "correct_ica": "desc-correctIca",
            "correct_dss": "desc-correctDss",
            "denoise_ar": "desc-denoiseAr",
            "denoise_dss_ar": "desc-denoiseDssAr"
        }

        desc = suffix_map.get(args.stage, "desc-baseline")
        
        # Create subject output directory
        sub_out_dir = os.path.join(args.output_dir, sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)
        suffix = "_no_pool" if getattr(args, "no_pool", False) else "_pool"
        emb_file = os.path.join(sub_out_dir, f"{sub_id}_{desc}_embed_reve_{args.model_size}{suffix}.npy")
        if os.path.exists(emb_file):
            print(f"SKIP [{sub_id}]: Already processed")
            success_count += 1
            continue
                
        eeg_path = find_eeg_file(sub_dir, stage=args.stage)
        if not eeg_path:
            print(f"SKIP [{sub_id}]: No EEG file found for stage '{args.stage}'")
            skipped_count += 1
            continue
            
        try:
            # Load EEG (FIF format)
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
            
            data = raw.get_data() # (C, T)
            ch_names = raw.ch_names
            
            # Z-score Normalization and Clipping (Target sfreq is 200Hz)
            data_tensor = preprocess_signal(data)
            
            # 10-second segmentation
            # 10s * 200Hz = 2000 samples
            window_size = 2000
            n_samples = data_tensor.shape[1]
            n_segments = n_samples // window_size
            
            if n_segments == 0:
                print(f"SKIP [{sub_id}]: Too short (<10s, {n_samples} samples)")
                skipped_count += 1
                continue
            
            # Truncate to integer number of segments
            data_tensor = data_tensor[:, :n_segments * window_size]
            
            # Reshape: (Channels, Segments, Window) -> (Segments, Channels, Window)
            # data_tensor: (C, S*W) -> (C, S, W)
            data_batch = data_tensor.view(data_tensor.shape[0], n_segments, window_size)
            data_batch = data_batch.permute(1, 0, 2) # (S, C, W)
            
            subject_embs = []
            
            # Process in batches to manage memory
            batch_size = 4
            with torch.no_grad():
                for i in range(0, n_segments, batch_size):
                    batch = data_batch[i : i + batch_size]
                    if args.device != "cpu":
                         batch = batch.to(args.device)
                    
                    # Forward Pass
                    # Model expects (Batch, Channels, Time)
                    do_pool = not getattr(args, "no_pool", False)
                    emb_batch = model(batch, channel_names=ch_names, pool=do_pool) 

                    
                    emb_batch = emb_batch.cpu().numpy()
                    subject_embs.append(emb_batch)
            
            # Concatenate all segments: (S, Dim)
            emb_array = np.concatenate(subject_embs, axis=0)
            
            # Save embeddings with BIDS naming
            meta_file = os.path.join(sub_out_dir, f"{sub_id}_{desc}_metadata_reve_{args.model_size}{suffix}.json")
            
            np.save(emb_file, emb_array)
            
            # Save metadata
            metadata = {
                "subject": sub_id,
                "n_segments": int(n_segments),
                "embedding_shape": list(emb_array.shape),
                "all_layers": True
            }
            if len(emb_array.shape) > 1:
                metadata["n_features"] = int(emb_array.shape[-1])
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"SAVED [{sub_id}]: {n_segments} segments, shape {emb_array.shape} → {emb_file}")
            success_count += 1
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR [{sub_id}]: {e}")
            error_count += 1
            continue

    # 5. Print Summary
    print("\n" + "="*60)
    print("REVE Embedding Extraction Complete")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Successfully processed: {success_count} subjects")
    print(f"Skipped: {skipped_count} subjects")
    print(f"Errors: {error_count} subjects")
    print("="*60)

if __name__ == "__main__":
    args = parse_args()
    extract_features(args)

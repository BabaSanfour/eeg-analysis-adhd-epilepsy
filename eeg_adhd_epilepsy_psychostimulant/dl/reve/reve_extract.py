
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
    parser.add_argument("--csv-path", default="/home/mat/projects/EEG_psychostimulant/data/results/dl/subjects.csv", help="Path to subjects CSV file")
    parser.add_argument("--output-dir", default="/home/mat/scratch/embeddings/reve/base", help="Directory to save per-subject embeddings (BIDS format)")
    parser.add_argument("--output-csv", default=None, help="Optional: Save all embeddings as CSV file")
    parser.add_argument("--model-size", default="base", choices=["base", "large"], help="REVE model size")
    parser.add_argument("--stage", default="base", choices=["base", "correct_ica", "correct_dss", "denoise_ar", "denoise_dss_ar"], help="Preprocessing stage to extract")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subject", default=None, help="Specific subject ID to process (e.g., '1' or 'sub-0001')")
    return parser.parse_args()

def find_eeg_file(sub_dir, stage="base"):
    """Find the first .fif file matching the stage in the subject directory, searching recursively."""
    # Pattern map
    patterns = {
        "base": "desc-base_eeg.fif",
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
    
    # 2. Load Subject Info
    df = pd.read_csv(args.csv_path, delimiter=";")
    # Ensure Epilepsy column exists
    if "Epilepsy" not in df.columns:
        raise ValueError("CSV must contain 'Epilepsy' column")
    
    # 3. Initialize Model
    print(f"Loading REVE model ({args.model_size}) on {args.device}...")
    model = REVEFeatureExtractor(model_size=args.model_size)
    model.to(args.device)
    model.eval()
    
    # Tracking for summary
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # For optional CSV output
    all_features = []
    all_labels = []
    all_subject_ids = []
    
    # Filter for specific subject if requested
    if args.subject:
        # Normalize target subject to string, removing 'sub-' prefix if present
        target_sub = str(args.subject).replace('sub-', '')
        
        # Filter dataframe
        # We assume 'Study ID' is the column, but let's be flexible
        # Convert column to string and compare
        df['Study ID_str'] = df['Study ID'].astype(str)
        df = df[df['Study ID_str'] == target_sub]
        
        if len(df) == 0:
            print(f"Subject {args.subject} (ID: {target_sub}) not found in CSV. Checking if it exists in data folder anyway...")
            # If not in CSV but we want to process it, we can mock a row or just warn and exit
            # For now, let's warn and exit to avoid processing without labels
            print("Processing stopped: Subject not in CSV.")
            return

    # 4. Process Subjects
    print(f"Processing {len(df)} subjects...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sub_id = f"sub-{str(row['Study ID']).zfill(4)}"
        label = row['Epilepsy']
        
        # Handle label cleaning locally if needed, but assuming clean int/str
        try:
            label = int(label)
        except:
            skipped_count += 1
            continue # Skip if label is not valid or '0 (potentiel)'
        
        # Check if already processed
        # Update filename based on stage
        suffix_map = {
            "base": "desc-base",
            "correct_ica": "desc-correctIca",
            "correct_dss": "desc-correctDss",
            "denoise_ar": "desc-denoiseAr",
            "denoise_dss_ar": "desc-denoiseDssAr"
        }

        desc = suffix_map.get(args.stage, "desc-base")
        
        # Create subject output directory
        sub_out_dir = os.path.join(args.output_dir, sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)
        
        emb_file = os.path.join(sub_out_dir, f"{sub_id}_{desc}_embed_reve.npy")
        if os.path.exists(emb_file):
            print("SKIP [{}]: Already processed".format(sub_id))
            success_count += 1
            continue
            
        sub_dir = os.path.join(args.data_root, sub_id)
        if not os.path.exists(sub_dir):
            # Try without zfill?
            sub_id_alt = f"sub-{row['Study ID']}"
            sub_dir_alt = os.path.join(args.data_root, sub_id_alt)
            if os.path.exists(sub_dir_alt):
                sub_dir = sub_dir_alt
            else:
                skipped_count += 1
                continue
                
        eeg_path = find_eeg_file(sub_dir, stage=args.stage)
        if not eeg_path:
            print("SKIP [{}]: No EEG file found for stage '{}'".format(sub_id, args.stage))
            skipped_count += 1
            continue
            
        try:
            # Load EEG (FIF format)
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
            
            # Preprocess
            target_sfreq = 200
            if raw.info['sfreq'] != target_sfreq:
                raw.resample(target_sfreq, npad="auto")
                
            data = raw.get_data() # (C, T)
            ch_names = raw.ch_names
            
            # Additional Preprocessing (Filter, Z-score, Clip)
            data_tensor = preprocess_signal(data, sfreq=target_sfreq)
            
            # 10-second segmentation
            # 10s * 200Hz = 2000 samples
            window_size = 2000
            n_samples = data_tensor.shape[1]
            n_segments = n_samples // window_size
            
            if n_segments == 0:
                print(f"SKIP [{{sub_id}}]: Too short (<10s, {{n_samples}} samples)")
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
            batch_size = 16
            with torch.no_grad():
                for i in range(0, n_segments, batch_size):
                    batch = data_batch[i : i + batch_size]
                    if args.device != "cpu":
                         batch = batch.to(args.device)
                    
                    # Forward Pass
                    # Model expects (Batch, Channels, Time)
                    emb_batch = model(batch, channel_names=ch_names) # (Batch, Dim) or (Batch, Channels, Dim)
                    
                    # Check and pool if needed (e.g. if model output is per-channel)
                    if emb_batch.dim() == 3:
                         # Use True Attention Pooling (model.pooler) instead of mean
                         # This handles cases where forward() skips pooling
                         emb_batch = model.pooler(emb_batch)
                    
                    emb_batch = emb_batch.cpu().numpy()
                    subject_embs.append(emb_batch)
            
            # Concatenate all segments: (S, Dim)
            emb_array = np.concatenate(subject_embs, axis=0)
            
            # Save embeddings with BIDS naming
            meta_file = os.path.join(sub_out_dir, f"{sub_id}_{desc}_metadata_reve.json")
            
            np.save(emb_file, emb_array)
            
            # Save metadata
            metadata = {
                "subject": sub_id,
                "label": int(label),
                "n_features": int(emb_array.shape[1]),
                "n_segments": int(n_segments),
                "embedding_shape": list(emb_array.shape)
            }
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("SAVED [{{}}]: {{}} segments, shape {{}} → {{}}".format(sub_id, n_segments, emb_array.shape, emb_file))
            success_count += 1
            
            # Also collect for CSV if requested
            # We add EACH segment as a row
            for seg_emb in emb_array:
                all_features.append(seg_emb)
                all_labels.append(label)
                all_subject_ids.append(sub_id)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ERROR [{{}}]: {{}}".format(sub_id, e))
            error_count += 1
            continue

    # 5. Optional CSV Export
    if args.output_csv and len(all_features) > 0:
        X = np.array(all_features)
        y = np.array(all_labels)
        ids = np.array(all_subject_ids)
        
        # Create output directory
        csv_dir = os.path.dirname(args.output_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        
        # Create DataFrame
        feat_cols = [f"feat_{i}" for i in range(X.shape[1])]
        df_out = pd.DataFrame(X, columns=feat_cols)
        df_out.insert(0, "subject_id", ids)
        
        df_out.to_csv(args.output_csv, index=False)
        print(f"\nSaved CSV: {args.output_csv} ({len(ids)} subjects, {X.shape[1]} features)")
    
    # 6. Print Summary
    print("\n" + "="*60)
    print("REVE Embedding Extraction Complete")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Successfully processed: {success_count} subjects")
    print(f"Skipped: {skipped_count} subjects")
    print(f"Errors: {error_count} subjects")
    if args.output_csv:
        print(f"CSV file: {args.output_csv}")
    print("="*60)

if __name__ == "__main__":
    args = parse_args()
    extract_features(args)

#!/usr/bin/env python3
"""
End-to-end Dimensionality Reduction Analysis for EEG Data.

Workflow:
1. Load EEG data from BIDS directory (Raw or Specific Segments).
2. Stack/Flatten data based on strategy (flat, per-channel, per-time).
3. Apply dimensionality reduction using coco-pipe (all 12 available reducers).
4. Compute quality metrics and compare methods using MethodSelector.
5. Generate comprehensive interactive HTML reports with all visualizations.

Available Reducers:
- pca, umap, tsne, phate, pacmap, trimap, mds, isomap, lle, spectral, kpca, factor

Available Visualizations:
- Embeddings (2D/3D), loss history, scree plots, Shepard diagrams
- Metric comparisons, feature importance, trajectory plots, local metrics
"""

import argparse
import logging
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from mne_bids import get_entity_vals

# Adjust path to ensure local modules are importable
sys.path.append(os.getcwd())

# Project imports
from eeg_adhd_epilepsy.utils.config import results_dir, csv_dir

# Coco-pipe imports
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation import MethodSelector, compute_coranking_matrix
from coco_pipe.dim_reduction import trustworthiness, continuity, lcmc
from coco_pipe.io.structures import DataContainer
from coco_pipe.io import load_data
from coco_pipe.report.core import Report, Section, PlotlyElement, ImageElement, MetricsTableElement
from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz.plotly_utils import plot_embedding_interactive

from eeg_adhd_epilepsy.io.bids import discover_bids_files
from eeg_adhd_epilepsy.qc.segmentation import extract_condition_segments
from eeg_adhd_epilepsy.utils.config import MAPPING_PSYCHOSTIMULANT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All available reducers in coco-pipe
ALL_REDUCERS = [
    "PCA", "UMAP", "TSNE", "PHATE", "PACMAP", "TRIMAP",
    "MDS", "ISOMAP", "LLE", "SPECTRALEMBEDDING"
]


def norm_id(s):
    """Normalize subject ID to sub-XXXX format."""
    s = str(s).lower().strip()
    if '_' in s: s = s.split('_')[0]
    if s.startswith('p'): s = s[1:]
    if s.startswith('sub-'): s = s[4:]
    s = s.lstrip('0')
    if not s: s = '0'
    try:
        return f"sub-{int(s):04d}"
    except:
        return f"sub-{s}"


def load_bids_data(
    bids_root: Path,
    subjects: List[str],
    task: str = "clinical",
    session: str = "01",
    segment_mode: Literal["raw", "condition"] = "condition",
    condition: Optional[str] = None,
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    stacking_mode: Literal["flat", "channel_as_sample", "time_as_sample"] = "flat",
    metadata_df: Optional[pd.DataFrame] = None,
    subject_col: str = "Study ID",
    target_col: str = "Group"
) -> DataContainer:
    """
    Load EEG data from BIDS and structure into a DataContainer.
    """
    logger.info(f"Loading data for {len(subjects)} subjects. Task: {task}, Mode: {segment_mode}, Stacking: {stacking_mode}")
    
    container = None

    logger.info("Using coco_pipe.io.load_data for raw mode...")
    container = load_data(
        mode="bids",
        path=bids_root,
        task=task,
        session=session,
        loading_mode="epochs",
        window_length=segment_duration,
        stride=segment_duration - overlap,
        subjects=subjects,
        datatype="eeg",
        suffix="eeg"
    )

    # -------------------------------------------------------------------------
    # Unified Processing (Stacking & Metadata)
    # -------------------------------------------------------------------------
    logger.info(f"Initial Container Shape: {container.X.shape}, Dims: {container.dims}")

    # 0. Capture Pre-Stack Metadata (BIDS info) from coords
    obs_dim_idx = container.dims.index('obs')
    n_obs_orig = container.shape[obs_dim_idx]
    
    pre_stack_meta = {}
    for k, v in container.coords.items():
         if len(v) == n_obs_orig:
             pre_stack_meta[k] = np.array(v)
    
    if 'subject' not in pre_stack_meta:
        pre_stack_meta['subject'] = np.array([i.split('_')[0] for i in container.ids])
        
    df_bids = pd.DataFrame(pre_stack_meta)

    # 1. Stacking / Reshaping using DataContainer methods
    if stacking_mode == "flat":
        container = container.flatten(preserve='obs')
    elif stacking_mode == "channel_as_sample":
        container = container.stack(dims=('obs', 'channel'), new_dim='obs')
    elif stacking_mode == "time_as_sample":
        container = container.stack(dims=('obs', 'time'), new_dim='obs')
        
    logger.info(f"Final Data Shape: {container.X.shape}")
    logger.info(f"Final Dims: {container.dims}")

    # 2. Metadata Enrichment & Re-Alignment
    new_subjects = [i.split('_')[0] for i in container.ids]
    df_current = pd.DataFrame({'subject': new_subjects})
    df_bids_unique = df_bids.drop_duplicates(subset=['subject'])
    merged = pd.merge(df_current, df_bids_unique, on='subject', how='left')
    
    if metadata_df is not None:
        logger.info(f"Merging metadata using subject key: '{subject_col}'")
        
        if subject_col not in metadata_df.columns:
            logger.warning(f"Subject column '{subject_col}' not found in metadata.")
        else:
            ext_df = metadata_df.drop_duplicates(subset=[subject_col]).copy()
            
            def normalize_id(s):
                s = str(s).lower().replace('sub-', '')
                import re
                clean = re.sub(r'[^a-z0-9]', '', s)
                try:
                    if clean.isdigit():
                         return str(int(clean))
                except:
                    pass
                return clean

            merged['match_key'] = merged['subject'].apply(normalize_id)
            ext_df['match_key'] = ext_df[subject_col].apply(normalize_id)
            
            merged = pd.merge(
                merged, ext_df, 
                left_on='match_key', right_on='match_key', 
                how='left', suffixes=('', '_ext')
            )
            if 'match_key' in merged.columns:
                del merged['match_key']

    merged = merged.loc[:, ~merged.columns.str.contains('^Unnamed')]
    for col in merged.columns:
        container.coords[col] = merged[col].values

    if target_col in container.coords:
         container.y = np.array(container.coords[target_col]).astype(str)
    
    if container.y is None and 'subject' in container.coords:
         container.y = container.coords['subject']

    return container


def load_derivatives_data(
    bids_root: Path,
    subjects: Optional[List[str]] = None,
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    stacking_mode: Literal["flat", "channel_as_sample", "time_as_sample"] = "flat",
    metadata_df: Optional[pd.DataFrame] = None,
    subject_col: str = "Study ID",
    target_col: str = "Group",
    desc: str = "base",
    condition: Optional[str] = None,
    ignore_annotations: bool = True
) -> DataContainer:
    """
    Load preprocessed EEG data (desc-base) from derivatives/preproc using exhaustive search.
    Manually constructs a DataContainer.
    """
    import mne
    from coco_pipe.io.structures import DataContainer
    from tqdm import tqdm
    
    # 1. Extensive Search for Files
    preproc_root = bids_root / "derivatives" / "preproc"
    logger.info(f"Searching for *desc-{desc}_eeg.fif in {preproc_root}...")
    
    # Use discover_bids_files if possible, but it searches for .vhdr primarily in config
    # Let's stick to Path.rglob for simplicity as we know the structure is derivatives/preproc
    # Or use the exact pattern we know works.
    files_to_load = sorted(list(preproc_root.rglob(f"*desc-{desc}_eeg.fif")))
    
    if not files_to_load:
        raise FileNotFoundError(f"No preprocessed files found in {preproc_root}")
        
    # 2. Filter Subjects
    if subjects:
        filtered_files = []
        normalized_subjects = [s if s.startswith('sub-') else f"sub-{s}" for s in subjects]
        for f in files_to_load:
             sid = f.name.split('_')[0]
             if sid in normalized_subjects:
                 filtered_files.append(f)
        files_to_load = filtered_files
        
    logger.info(f"Selected {len(files_to_load)} files for loading.")
    if not files_to_load:
         raise ValueError("No files matching subject selection found.")

    # 3. Load and Segment Data
    all_epochs = []
    all_ids = []
    
    # Metadata accumulator
    # We will accumulate all subjects and then merge with DataFrame later
    # to avoid repeated lookups.
    epoch_subjects = [] 
    
    import warnings
    
    for fpath in tqdm(files_to_load, desc="Loading Subjects"):
        # Load Raw
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose="ERROR")
        sid = fpath.name.split('_')[0]
        
        # Find segments CSV for this subject in BIDS root
        # Pattern: sub-sid/ses-??/eeg/sub-sid_ses-??_task-clinical_segments.csv
        segments_files = list(bids_root.rglob(f"{sid}_*task-clinical_segments.csv"))
        
        if not segments_files:
            logger.warning(f"No segments.csv found for {sid} in {bids_root}. Skipping.")
            continue
        
        csv_path = segments_files[0]
        try:
            df_segments = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read segments CSV {csv_path}: {e}")
            continue

        # Condition Filtering logic strictly using CSV
        if condition:
            cond_lower = condition.lower()
            target_types = []
            
            # Dynamic mapping of requested condition to CSV segment_type
            if cond_lower == "baseline_eo" or cond_lower == "eo_baseline":
                target_types = ["EO_baseline"]
            elif cond_lower == "baseline_ec" or cond_lower == "ec_baseline":
                target_types = ["EC_baseline"]
            elif cond_lower == "hv":
                target_types = [t for t in df_segments['segment_type'].unique() if str(t).startswith("HV")]
            elif cond_lower == "photo":
                target_types = [t for t in df_segments['segment_type'].unique() if str(t).startswith("PHOTO")]
            elif cond_lower == "raw_baseline":
                target_types = ["RAW_baseline"]
            else:
                # Direct match fallback
                target_types = [condition]
            
            mask = df_segments['segment_type'].isin(target_types)
            filtered = df_segments[mask]
            
            if filtered.empty:
                logger.warning(f"Condition '{condition}' (targets: {target_types}) not found in segments for {sid}. Skipping.")
                continue
            
            logger.info(f"Found {len(filtered)} segments for {sid} matching {condition}")
            events_list = []
            for _, seg in filtered.iterrows():
                t_start = float(seg['t_start'])
                t_stop = float(seg['t_stop'])
                if t_stop - t_start < segment_duration:
                    continue
                try:
                    # Make fixed length events WITHIN this block according to CSV timing
                    chunk_events = mne.make_fixed_length_events(
                        raw, id=1, start=t_start, stop=t_stop, duration=segment_duration, overlap=overlap
                    )
                    events_list.append(chunk_events)
                except: # TOFIX
                    continue        
            if not events_list:
                logger.warning(f"No events created for condition {condition} in {sid} within CSV boundaries.")
                continue
            events = np.concatenate(events_list)
            events = events[events[:, 0].argsort()]
            logger.info(f"Total events for {sid}: {len(events)}")
        else:
            # Use whole file
            events = mne.make_fixed_length_events(
                raw, id=1, start=0, stop=None, duration=segment_duration, overlap=overlap
            )
        
        if len(events) == 0:
            logger.warning(f"No events found for {sid}")
            continue
            
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=segment_duration, baseline=None,
            reject=None, verbose="ERROR", preload=True, proj=False,
            reject_by_annotation=not ignore_annotations
        )
        
        # Extract Data
        if len(epochs) == 0:
            logger.warning(f"Epochs object for {sid} is empty after creation. Events: {len(events)}")
            continue
            
        data = epochs.get_data() # (n_epochs, n_channels, n_times)
        logger.info(f"Successfully loaded {data.shape[0]} epochs for {sid}")
        
        if data.shape[0] > 0:
            all_epochs.append(data)
            
            n_ep = data.shape[0]
            # ID: subject_idx
            start_idx = len(all_ids)
            new_ids = [f"{sid}_{i}" for i in range(start_idx, start_idx+n_ep)]
            all_ids.extend(new_ids)
            epoch_subjects.extend([sid] * n_ep)
            
    if not all_epochs:
        raise RuntimeError(f"No valid epochs loaded. Check if condition '{condition}' exists in the data.")
        
    X_all = np.concatenate(all_epochs, axis=0) # (N, C, T)
    
    # 4. Construct DataContainer
    coords = {
        'subject': np.array(epoch_subjects),
        'channel': np.array(epochs.ch_names),
        'time': epochs.times
    }
    if condition:
        coords['condition'] = np.array([condition] * len(epoch_subjects))
    
    container = DataContainer(
        X=X_all,
        dims=('obs', 'channel', 'time'),
        coords=coords,
        ids=np.array(all_ids)
    )
    
    # -------------------------------------------------------------------------
    # 4. Stacking / Reshaping (Happening before metadata merge)
    # -------------------------------------------------------------------------
    if stacking_mode == "flat":
        container = container.flatten(preserve='obs')
    elif stacking_mode == "channel_as_sample":
        container = container.stack(dims=('obs', 'channel'), new_dim='obs')
    elif stacking_mode == "time_as_sample":
        container = container.stack(dims=('obs', 'time'), new_dim='obs')

    # -------------------------------------------------------------------------
    # 5. Metadata Integration (ON FINAL FLATTENED CONTAINER)
    # -------------------------------------------------------------------------
    if metadata_df is not None:
        logger.info(f"Merging external metadata from CSV onto {container.X.shape[0]} samples...")
        csv_df = metadata_df.copy()
        
        if subject_col in csv_df.columns:
            csv_df['match_id'] = csv_df[subject_col].apply(norm_id)
            
            sub_vals = container.coords.get('subject', container.ids)
            if sub_vals is not None:
                # Normalize container IDs too
                sub_series = pd.Series(sub_vals, name='subject_key').astype(str).str.strip().apply(norm_id)
                merged = pd.merge(sub_series, csv_df, left_on='subject_key', right_on='match_id', how='left')
                
                # Ensure normalized subject is back in coords
                container.coords['subject'] = merged['subject_key'].values

                # Prepare Derived Columns
                if 'Age' in merged.columns:
                    def get_age_group(age_str):
                        import re
                        match = re.search(r'\d+', str(age_str))
                        if not match: return "Unknown"
                        age_val = int(match.group())
                        if 5 <= age_val <= 8: return "5-8"
                        if 9 <= age_val <= 12: return "9-12"
                        if 13 <= age_val <= 18: return "13-18"
                        return "Other"
                    merged['Age_Group'] = merged['Age'].apply(get_age_group)
                    
                diag_cols = ['TDAH', 'TSA', 'Epilepsy']
                def combine_diagnosis(row):
                    diags = []
                    for col in diag_cols:
                        if col in row and str(row[col]).lower().startswith(('yes', 'oui', '1', 'true')):
                            if col == 'TDAH': diags.append("ADHD")
                            elif col == 'TSA': diags.append("ASD")
                            else: diags.append(col)
                    if not diags: return "Control"
                    return "+".join(diags)
                merged['Diagnosis_Combined'] = merged.apply(combine_diagnosis, axis=1)
                
                asm_cols = ['LEV', 'LTG', 'LCS', 'CLB', 'CBZ', 'VPA', 'ETH', 'TPM', 'RUF', 'BRV', 'STP', 'OXZ', 'CBM']
                def get_asm_types(row):
                    drugs = []
                    for col in asm_cols:
                        if col in row and str(row[col]).lower().startswith(('yes', 'oui', '1', 'true')):
                             drugs.append(col)
                    return "+".join(drugs) if drugs else "No_ASM"
                merged['ASM_Types'] = merged.apply(get_asm_types, axis=1)
                merged['ASM_dichotomous'] = np.where(merged['ASM_Types'] == "No_ASM", "No", "Yes")
                
                if 'psychostimulant_description' in merged.columns:
                     merged['Psychostim_Category_Mapped'] = merged['psychostimulant_description'].apply(lambda x: MAPPING_PSYCHOSTIMULANT.get(str(x), 0))

                ps_col = 'Psychostimulant (y/n)'
                if ps_col in merged.columns:
                    def get_meds_status(row):
                        has_asm = row['ASM_dichotomous'] == "Yes"
                        has_psy = str(row[ps_col]).lower().startswith(('yes', 'oui', '1', 'true'))
                        if has_asm and has_psy: return "ASM+Psychostim"
                        if has_asm: return "ASM"
                        if has_psy: return "Psychostim"
                        return "No Med"
                    merged['Meds'] = merged.apply(get_meds_status, axis=1)
                
                cols_to_add = [
                    'Age_Group', 'Diagnosis_Combined', 'ASM_Types', 
                    'ASM_dichotomous', 'Meds', 'Psychostim_Category_Mapped',
                    'TDAH', 'TSA', 'Epilepsy', 'Sex', 'Age', 'Group'
                ]
                for col in cols_to_add:
                    if col in merged.columns:
                         container.coords[col] = merged[col].values

                if target_col in container.coords:
                     container.y = np.array(container.coords[target_col]).astype(str)
                else:
                    for fb in ['Diagnosis_Combined', 'Meds', 'ASM_dichotomous', 'Age_Group', 'Sex']:
                        if fb in container.coords:
                             container.y = np.array(container.coords[fb]).astype(str)
                             logger.info(f"Target col '{target_col}' missing. Using '{fb}'.")
                             break

    if container.y is not None:
         container.y = np.array(container.y).astype(str)
    
    return container


def compute_quality_metrics(X_orig: np.ndarray, X_emb: np.ndarray, labels: Optional[np.ndarray] = None, k: int = 10) -> Dict[str, float]:
    """
    Compute comprehensive quality metrics for an embedding.
    Uses coranking matrix for manifold preservation metrics.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    metrics = {}
    
    # Compute coranking matrix (needed for manifold metrics)
    try:
        Q = compute_coranking_matrix(X_orig, X_emb)
        metrics['trustworthiness'] = trustworthiness(Q, k=k)
        metrics['continuity'] = continuity(Q, k=k)
        metrics['lcmc'] = lcmc(Q, k=k)
    except Exception as e:
        logger.warning(f"Could not compute coranking metrics: {e}")
    
    # Clustering metrics (if labels provided)
    if labels is not None:
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(labels):
                metrics['silhouette'] = silhouette_score(X_emb, labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(X_emb, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(X_emb, labels)
        except Exception as e:
            logger.warning(f"Could not compute clustering metrics: {e}")
    
    return metrics


def run_method_selector(
    reducers_dict: Dict[str, DimReduction],
    data: np.ndarray,
    target: Optional[np.ndarray] = None
) -> MethodSelector:
    """
    Run MethodSelector comparison across all reducers.
    """
    logger.info(f"Running MethodSelector comparison for {len(reducers_dict)} reducers...")
    selector = MethodSelector(
        reducers=reducers_dict,
        data=data,
        target=target
    )
    selector.run()
    return selector


def compute_all_embeddings(
    X: np.ndarray,
    reducer_names: List[str],
    n_components: int = 2,
    n_components_3d: int = 3
) -> Tuple[Dict[str, Dict], Dict[str, DimReduction]]:
    """
    Compute embeddings for all reducers in both 2D and 3D.
    """
    results = {}
    reducers_2d = {}
    
    for name in reducer_names:
        logger.info(f"Computing embeddings for {name}...")
        results[name] = {}
        
        try:
            dr_2d = DimReduction(method=name, n_components=n_components)
            X_emb_2d = dr_2d.fit_transform(X)
            results[name]['embedding_2d'] = X_emb_2d
            results[name]['reducer_2d'] = dr_2d
            reducers_2d[name] = dr_2d
        except Exception as e:
            logger.error(f"Failed to compute 2D {name}: {e}")
            results[name]['embedding_2d'] = None
            results[name]['reducer_2d'] = None
            
        try:
            dr_3d = DimReduction(method=name, n_components=n_components_3d)
            X_emb_3d = dr_3d.fit_transform(X)
            results[name]['embedding_3d'] = X_emb_3d
            results[name]['reducer_3d'] = dr_3d
        except Exception as e:
            logger.error(f"Failed to compute 3D {name}: {e}")
            results[name]['embedding_3d'] = None
            results[name]['reducer_3d'] = None
            
    return results, reducers_2d


def create_reducer_section(
    name: str,
    embedding_results: Dict,
    labels: np.ndarray,
    X_orig: np.ndarray,
    meta_dict: Optional[Dict[str, np.ndarray]] = None,
    times: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    interactive: bool = True
) -> Section:
    """
    Create a comprehensive report section for a single reducer.
    
    Parameters
    ----------
    meta_dict : dict, optional
        Dictionary of metadata columns for interactive color dropdown.
        Keys are column names, values are arrays of values per sample.
    """
    sec = Section(title=name.upper(), icon="📉")
    
    emb_2d = embedding_results.get('embedding_2d')
    emb_3d = embedding_results.get('embedding_3d')
    reducer_2d = embedding_results.get('reducer_2d')
    
    # 1. 2D Embedding - with metadata dropdown for color selection
    if emb_2d is not None:
        if interactive and meta_dict:
            # Use plot_embedding_interactive directly with meta for dropdown
            fig = plot_embedding_interactive(
                embedding=emb_2d,
                labels=labels,
                meta=meta_dict,
                title=f"{name.upper()} - 2D Embedding",
                dimensions=2
            )
        else:
            fig = viz.plot_embedding(
                X_emb=emb_2d,
                labels=labels,
                dims=(0, 1),
                title=f"{name.upper()} - 2D Embedding",
                interactive=interactive
            )
        if interactive:
            sec.add_element(PlotlyElement(fig))
        else:
            sec.add_element(ImageElement(fig))
    
    # 2. 3D Embedding - with metadata dropdown for color selection
    if emb_3d is not None:
        if interactive and meta_dict:
            fig_3d = plot_embedding_interactive(
                embedding=emb_3d,
                labels=labels,
                meta=meta_dict,
                title=f"{name.upper()} - 3D Embedding",
                dimensions=3
            )
        else:
            fig_3d = viz.plot_embedding(
                X_emb=emb_3d,
                labels=labels,
                dims=(0, 1, 2),
                title=f"{name.upper()} - 3D Embedding",
                interactive=interactive
            )
        if interactive:
            sec.add_element(PlotlyElement(fig_3d))
        else:
            sec.add_element(ImageElement(fig_3d))
    
    # 3. Loss History (for iterative methods)
    if reducer_2d is not None:
        reducer_obj = getattr(reducer_2d, 'reducer_', reducer_2d)
        if hasattr(reducer_obj, 'loss_history_') and reducer_obj.loss_history_ is not None:
            fig_loss = viz.plot_loss_history(
                loss_history=reducer_obj.loss_history_,
                title=f"{name.upper()} - Training Loss",
                interactive=interactive
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_loss))
            else:
                sec.add_element(ImageElement(fig_loss))
    
    # 4. Eigenvalue/Scree Plot (for PCA-like methods)
    if reducer_2d is not None:
        reducer_obj = getattr(reducer_2d, 'reducer_', reducer_2d)
        if hasattr(reducer_obj, 'explained_variance_ratio_'):
            fig_scree = viz.plot_eigenvalues(
                values=reducer_obj.explained_variance_ratio_,
                title=f"{name.upper()} - Explained Variance Ratio",
                ylabel="Explained Variance Ratio",
                interactive=interactive
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_scree))
            else:
                sec.add_element(ImageElement(fig_scree))
        elif hasattr(reducer_obj, 'eigenvalues_'):
            fig_scree = viz.plot_eigenvalues(
                values=reducer_obj.eigenvalues_,
                title=f"{name.upper()} - Eigenvalues",
                ylabel="Eigenvalue",
                interactive=interactive
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_scree))
            else:
                sec.add_element(ImageElement(fig_scree))
    
    # # 5. Shepard Diagram (quality diagnostic)
    # if emb_2d is not None:
    #     try:
    #         fig_shepard = viz.plot_shepard_diagram(
    #             X_orig=X_orig,
    #             X_emb=emb_2d,
    #             sample_size=min(500, X_orig.shape[0]),
    #             title=f"{name.upper()} - Shepard Diagram",
    #             interactive=interactive
    #         )
    #         if interactive:
    #             sec.add_element(PlotlyElement(fig_shepard))
    #         else:
    #             sec.add_element(ImageElement(fig_shepard))
    #     except Exception as e:
    #         logger.warning(f"Could not create Shepard diagram for {name}: {e}")
    
    # 6. Trajectory Plot (if temporal data)
    if emb_2d is not None and times is not None:
        try:
            fig_traj = viz.plot_trajectory(
                X=emb_2d,
                times=times,
                groups=groups,
                dimensions=2,
                title=f"{name.upper()} - Trajectory Over Time",
                interactive=interactive
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_traj))
            else:
                sec.add_element(ImageElement(fig_traj))
        except Exception as e:
            logger.warning(f"Could not create trajectory plot for {name}: {e}")
    
    return sec


def create_comparison_section(
    selector: Optional[MethodSelector],
    all_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = None,
    interactive: bool = True
) -> Section:
    """
    Create a comparison section with method selector results.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["trustworthiness", "continuity", "lcmc", "silhouette"]
    
    sec = Section(title="Method Comparison", icon="📊")
    
    # Build summary DataFrame from all_metrics
    if all_metrics:
        df_summary = pd.DataFrame(all_metrics).T
        sec.add_element(MetricsTableElement(df_summary))
        
        # Create bar chart for each metric
        for metric in metrics_to_plot:
            if metric in df_summary.columns:
                scores = df_summary[metric].dropna().to_dict()
                if scores:
                    try:
                        fig = viz.plot_metrics(
                            scores=scores,
                            title=f"Comparison: {metric.replace('_', ' ').title()}",
                            interactive=interactive
                        )
                        if interactive:
                            sec.add_element(PlotlyElement(fig))
                        else:
                            sec.add_element(ImageElement(fig))
                    except Exception as e:
                        logger.warning(f"Could not create comparison plot for {metric}: {e}")
    
    # If MethodSelector was run, use its plot_comparison
    if selector is not None:
        for metric in metrics_to_plot:
            try:
                fig = viz.plot_comparison(
                    comparison_manager=selector,
                    metric=metric,
                    title=f"MethodSelector: {metric.replace('_', ' ').title()}",
                    interactive=interactive
                )
                if interactive:
                    sec.add_element(PlotlyElement(fig))
                else:
                    sec.add_element(ImageElement(fig))
            except Exception as e:
                logger.warning(f"Could not create MethodSelector plot for {metric}: {e}")
    
    return sec


def create_quality_section(
    embedding_results: Dict[str, Dict],
    X_orig: np.ndarray,
    labels: np.ndarray,
    interactive: bool = True
) -> Tuple[Section, Dict[str, Dict[str, float]]]:
    """
    Create a section with quality metrics visualizations.
    """
    sec = Section(title="Quality Metrics", icon="📈")
    
    all_metrics = {}
    for name, res in embedding_results.items():
        emb_2d = res.get('embedding_2d')
        if emb_2d is not None:
            try:
                scores = compute_quality_metrics(X_orig, emb_2d, labels, k=10)
                all_metrics[name] = scores
                
                # Individual metrics plot
                if scores:
                    fig = viz.plot_metrics(
                        scores=scores,
                        title=f"{name.upper()} - Quality Metrics",
                        interactive=interactive
                    )
                    if interactive:
                        sec.add_element(PlotlyElement(fig))
                    else:
                        sec.add_element(ImageElement(fig))
            except Exception as e:
                logger.warning(f"Could not compute metrics for {name}: {e}")
    
    return sec, all_metrics


def create_feature_importance_section(
    embedding_results: Dict[str, Dict],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    interactive: bool = True
) -> Optional[Section]:
    """
    Create feature importance section for PCA-based methods.
    """
    sec = Section(title="Feature Importance", icon="🎯")
    added_content = False
    
    for name in ['PCA']:
        res = embedding_results.get(name, {})
        reducer = res.get('reducer_2d')
        
        if reducer is not None:
            reducer_obj = getattr(reducer, 'reducer_', reducer)
            if hasattr(reducer_obj, 'components_'):
                try:
                    loadings = np.abs(reducer_obj.components_).sum(axis=0)
                    
                    if feature_names is None:
                        feature_names_local = [f"Feature {i}" for i in range(len(loadings))]
                    else:
                        feature_names_local = feature_names[:len(loadings)]
                    
                    importance_scores = dict(zip(feature_names_local, loadings))
                    
                    fig = viz.plot_feature_importance(
                        scores=importance_scores,
                        title=f"{name.upper()} - Feature Importance",
                        top_n=top_n,
                        interactive=interactive
                    )
                    if interactive:
                        sec.add_element(PlotlyElement(fig))
                    else:
                        sec.add_element(ImageElement(fig))
                    added_content = True
                except Exception as e:
                    logger.warning(f"Could not create feature importance for {name}: {e}")
    
    return sec if added_content else None


def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive EEG DimReduction Pipeline")
    parser.add_argument("--bids_root", default="/Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS", help="Path to BIDS dataset")
    parser.add_argument("--task", default="clinical", help="Task name (default: clinical)")
    parser.add_argument("--session", default="01", help="Session ID (default: 01)")
    parser.add_argument("--metadata", default=None, help="Path to metadata CSV")
    parser.add_argument("--output_dir", default=results_dir, help="Output directory")
    parser.add_argument("--dataset_name", required=True, help="Name for this analysis run")
    
    parser.add_argument("--segment_mode", choices=["raw", "condition"], default="condition", help="Segmentation Mode")
    parser.add_argument("--condition", default=None, help="Specific condition")
    parser.add_argument("--segment_duration", type=float, default=60.0, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Window overlap in seconds")
    
    parser.add_argument("--stacking_mode", choices=["flat", "channel_as_sample", "time_as_sample"], default="flat", help="How to stack data matrices")
    
    parser.add_argument("--subsample", type=int, default=None, help="Number of subjects to random sample")
    parser.add_argument("--subject_col", default="Study ID", help="Column in metadata")
    parser.add_argument("--target_col", default="Group", help="Column to use as labels")
    
    parser.add_argument("--reducers", nargs="+", default=ALL_REDUCERS, help="List of reducers")
    parser.add_argument("--n_components_2d", type=int, default=2, help="Components for 2D")
    parser.add_argument("--n_components_3d", type=int, default=3, help="Components for 3D")
    parser.add_argument("--interactive", action="store_true", default=True, help="Interactive plots")
    parser.add_argument("--static", action="store_true", help="Static plots instead")
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings to disk")
    parser.add_argument("--with_trajectory", action="store_true", help="Include trajectory plots")
    parser.add_argument("--run_selector", action="store_true", help="Run MethodSelector comparison")
    parser.add_argument("--use_derivatives", action="store_true", help="Load from derivatives/preproc instead of raw BIDS")
    parser.add_argument("--desc", default="base", help="Desc to load from derivatives (default: base)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process")
    parser.add_argument("--ignore_annotations", action="store_true", default=True, help="Ignore BAD_ annotations during epoching")
    
    args = parser.parse_args()
    
    # Force logging level
    logging.getLogger().setLevel(logging.INFO)
    
    interactive = not args.static
    
    # Setup Paths
    bids_root = Path(args.bids_root)
    out_path = Path(args.output_dir) / args.dataset_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Subject Selection
    if args.subjects:
        subjects = args.subjects
        logger.info(f"Using provided subjects: {subjects}")
    elif args.use_derivatives:
        preproc_root = bids_root / "derivatives" / "preproc"
        # Discover subjects from the derivatives directory
        # Filenames look like sub-XXXX_desc-base_eeg.fif
        derivatives_files = list(preproc_root.rglob(f"*desc-{args.desc}_eeg.fif"))
        subjects = sorted(list(set([f.name.split('_')[0].replace('sub-', '') for f in derivatives_files])))
        logger.info(f"Found {len(subjects)} subjects with {args.desc} preprocessed data.")
    else:
        subjects = get_entity_vals(bids_root, 'subject')
        logger.info(f"Found {len(subjects)} subjects in BIDS root.")

    # Load Metadata early for filtering
    meta_df = None
    if args.metadata:
        try:
            meta_df = pd.read_csv(args.metadata)
            logger.info("Metadata loaded.")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    elif csv_dir:
         default_csv = Path(csv_dir) / "demo_psychostim_vs_none.csv"
         if default_csv.exists():
             meta_df = pd.read_csv(default_csv)
             logger.info(f"Loaded default metadata: {default_csv}")
    
    if meta_df is not None:
        meta_df = meta_df.loc[:, ~meta_df.columns.str.contains('^Unnamed')]
        
        # Clinical Filtering: Exclude subjects with "0 (potentiel)" in diagnostic columns
        diag_cols = ['TDAH', 'TSA', 'Epilepsy']
        existing_diag_cols = [c for c in diag_cols if c in meta_df.columns]
        if existing_diag_cols:
            exclude_mask = meta_df[existing_diag_cols].astype(str).apply(
                lambda x: x.str.lower().str.contains(r'0 \(potentiel\)')
            ).any(axis=1)
            
            excluded_df = meta_df[exclude_mask]
            if not excluded_df.empty:
                excluded_ids = excluded_df[args.subject_col].apply(norm_id).tolist()
                logger.info(f"Excluding {len(excluded_ids)} subjects due to 'Potential' status: {excluded_ids}")
                
                # Filter subjects list
                subjects = [s for s in subjects if norm_id(s) not in excluded_ids]
                logger.info(f"Remaining subjects after clinical filtering: {len(subjects)}")

    if args.subsample and args.subsample < len(subjects):
        random.seed(42)
        subjects = random.sample(subjects, args.subsample)
        logger.info(f"Subsampled to {len(subjects)} subjects.")
        
    # Load Data (Already loaded metadata above)
    dc = load_derivatives_data(
        bids_root=bids_root,
        subjects=subjects if subjects else None, 
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        stacking_mode=args.stacking_mode,
        metadata_df=meta_df,
        subject_col=args.subject_col,
        target_col=args.target_col,
        desc=args.desc,
        condition=args.condition,
        ignore_annotations=args.ignore_annotations
    )
    
    X = dc.X
    labels = dc.y
    
    logger.info(f"Data shape: {X.shape}, Labels: {labels.shape if labels is not None else 'None'}")
    
    # Compute All Embeddings (2D and 3D)
    embedding_results, reducers_2d = compute_all_embeddings(
        X=X,
        reducer_names=args.reducers,
        n_components=args.n_components_2d,
        n_components_3d=args.n_components_3d
    )
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings_dir = out_path / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        for name, res in embedding_results.items():
            if res['embedding_2d'] is not None:
                np.save(embeddings_dir / f"{name}_2d.npy", res['embedding_2d'])
            if res['embedding_3d'] is not None:
                np.save(embeddings_dir / f"{name}_3d.npy", res['embedding_3d'])
        logger.info(f"Embeddings saved to {embeddings_dir}")
    
    # Run MethodSelector (optional)
    selector = None
    if args.run_selector and reducers_2d:
        try:
            selector = run_method_selector(
                reducers_dict=reducers_2d,
                data=X,
                target=labels
            )
        except Exception as e:
            logger.warning(f"MethodSelector failed: {e}")
    
    # Prepare Trajectory Data (if applicable)
    times = None
    groups = None
    if args.with_trajectory:
        n_samples = X.shape[0]
        if 'epoch' in dc.coords:
            times = np.array(dc.coords['epoch'])
        else:
            times = np.linspace(0, 1, n_samples)
        
        if 'subject' in dc.coords:
            groups = np.array(dc.coords['subject'])
    
    # Generate Report
    logger.info("Generating Report...")
    report = Report(title=f"Comprehensive DimReduction Analysis: {args.dataset_name}")
    report.add_container(dc)
    
    # Build meta_dict for interactive color selection dropdown
    # Include all categorical metadata columns
    n_samples = X.shape[0]
    meta_dict = {}
    for col_name, col_values in dc.coords.items():
        if len(col_values) == n_samples:
            s = pd.Series(col_values)
            # Filter constants that are clearly not labels (like channel, time)
            if col_name in ['channel', 'time']: continue
            
            n_unique = s.astype(str).nunique()
            # Include if it has between 1 and 200 unique values
            if 1 <= n_unique <= 200:
                meta_dict[col_name] = col_values
    
    logger.info(f"Available labels for color coding: {list(meta_dict.keys())}")
    
    # Section 1: Quality Metrics
    quality_section, all_metrics = create_quality_section(
        embedding_results=embedding_results,
        X_orig=X,
        labels=labels,
        interactive=interactive
    )
    report.add_section(quality_section)
    
    # Section 2: Method Comparison (Overview)
    comp_section = create_comparison_section(
        selector=selector,
        all_metrics=all_metrics,
        interactive=interactive
    )
    report.add_section(comp_section)
    
    # Section 3: Individual Reducer Sections
    for name in args.reducers:
        if name in embedding_results:
            sec = create_reducer_section(
                name=name,
                embedding_results=embedding_results[name],
                labels=labels,
                X_orig=X,
                meta_dict=meta_dict,
                times=times,
                groups=groups,
                interactive=interactive
            )
            report.add_section(sec)
    
    # Section 4: Feature Importance (if applicable)
    feat_section = create_feature_importance_section(
        embedding_results=embedding_results,
        top_n=20,
        interactive=interactive
    )
    if feat_section:
        report.add_section(feat_section)
    
    # Save Report
    save_path = out_path / "report.html"
    report.save(save_path)
    logger.info(f"Report saved to: {save_path}")
    
    # Save static figures if requested
    if args.static:
        figures_dir = out_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        for name, res in embedding_results.items():
            emb_2d = res.get('embedding_2d')
            if emb_2d is not None:
                fig = viz.plot_embedding(
                    X_emb=emb_2d,
                    labels=labels,
                    dims=(0, 1),
                    title=f"{name.upper()} Embedding",
                    interactive=False
                )
                fig.savefig(figures_dir / f"{name}_embedding.png", dpi=150, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        logger.info(f"Static figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

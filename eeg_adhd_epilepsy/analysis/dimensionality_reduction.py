#!/usr/bin/env python3
"""
End-to-end Dimensionality Reduction Analysis for EEG Data.

Workflow:
1. Load EEG data from BIDS directory (Raw or Specific Segments).
2. Stack/Flatten data based on strategy (flat, time_as_sample, epoch_scalar_mean).
3. Apply dimensionality reduction using coco-pipe (selected supported reducers).
4. Generate interactive HTML reports with embeddings and diagnostics.

Available Reducers:
- pca, umap, phate, isomap

Available Visualizations:
- Embeddings (2D/3D), loss history, scree plots, Shepard diagrams
- Feature importance
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Literal

import numpy as np
import pandas as pd
from mne_bids import get_entity_vals

# Project imports
from eeg_adhd_epilepsy.utils.config import results_dir, csv_dir

# Coco-pipe imports
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.io.structures import DataContainer
from coco_pipe.io import load_data
from coco_pipe.report.core import Report, Section, PlotlyElement, ImageElement
from coco_pipe.viz import dim_reduction as viz
from eeg_adhd_epilepsy.analysis.utils import (
    add_derived_metadata_columns,
    add_embedding_plot,
    align_subject_metadata,
    apply_stacking_mode,
    build_meta_dict,
    capture_obs_metadata,
    coerce_sample_vector,
    norm_id,
    resolve_labels,
    subject_from_id,
    subject_key,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default reducers for this analysis profile
ALL_REDUCERS = [
    "PCA", "UMAP", "PHATE", "ISOMAP"
]

def load_bids_data(
    bids_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "clinical",
    session: str = "01",
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    stacking_mode: Literal["flat", "time_as_sample", "epoch_scalar_mean"] = "flat",
    metadata_df: Optional[pd.DataFrame] = None,
    subject_col: str = "Study ID",
    target_col: str = "Group"
) -> DataContainer:
    """
    Load EEG data from BIDS and structure into a DataContainer.
    """
    subjects = subjects or []
    logger.info(f"Loading data for {len(subjects)} subjects. Task: {task}, Stacking: {stacking_mode}")

    logger.info("Using coco_pipe.io.load_data for raw mode...")
    container = load_data(
        mode="bids",
        path=bids_root,
        task=task,
        session=session,
        loading_mode="epochs",
        window_length=segment_duration,
        stride=segment_duration - overlap,
        subjects=subjects if subjects else None,
        datatype="eeg",
        suffix="eeg"
    )
    logger.info(f"Initial Container Shape: {container.X.shape}, Dims: {container.dims}")
    obs_meta = capture_obs_metadata(container)
    container = apply_stacking_mode(container, stacking_mode)
    logger.info(f"Final Data Shape: {container.X.shape}")
    logger.info(f"Final Dims: {container.dims}")
    container = align_subject_metadata(container, obs_meta)

    if metadata_df is not None:
        logger.info(f"Merging metadata using subject key: '{subject_col}'")
        if subject_col not in metadata_df.columns:
            logger.warning(f"Subject column '{subject_col}' not found in metadata.")
        else:
            ext_df = metadata_df.drop_duplicates(subset=[subject_col]).copy()
            subject_values = coerce_sample_vector(container.coords.get("subject"), container.X.shape[0])
            if subject_values is None:
                subject_values = np.array([subject_from_id(i) for i in container.ids])
            current_df = pd.DataFrame({"subject": subject_values})
            current_df["match_key"] = current_df["subject"].apply(subject_key)
            ext_df["match_key"] = ext_df[subject_col].apply(subject_key)
            merged = pd.merge(
                current_df, ext_df, on="match_key", how="left", suffixes=("", "_ext")
            )
            merged = merged.loc[:, ~merged.columns.str.contains("^Unnamed")]
            for col in merged.columns:
                if col != "match_key":
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
    stacking_mode: Literal["flat", "time_as_sample", "epoch_scalar_mean"] = "flat",
    metadata_df: Optional[pd.DataFrame] = None,
    subject_col: str = "Study ID",
    target_col: str = "Group",
    desc: str = "base",
    condition: Optional[str] = None,
    ignore_annotations: bool = True,
    subject_average: bool = False
) -> DataContainer:
    """
    Load preprocessed EEG data (desc-base) from derivatives/preproc using exhaustive search.
    Manually constructs a DataContainer.
    """
    import mne
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
                except Exception as err:
                    logger.warning(f"Failed to create events for {sid} [{t_start}, {t_stop}]: {err}")
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

    # 4.5 Optional: Subject-Level Averaging
    if subject_average:
        logger.info(f"Averaging data across segments per subject. Groups sample: {coords['subject'][:5]}")
        logger.info(f"Unique subjects in coords: {pd.Series(coords['subject']).unique()}")
        container = container.aggregate(by='subject', method='mean')
        logger.info(f"Shape after averaging: {container.X.shape}")

    obs_meta = capture_obs_metadata(container)
    container = apply_stacking_mode(container, stacking_mode)
    container = align_subject_metadata(container, obs_meta)

    # -------------------------------------------------------------------------
    # 6. Metadata Integration (ON FINAL FLATTENED CONTAINER)
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
                merged = add_derived_metadata_columns(merged)

                # Ensure normalized subject is back in coords
                container.coords['subject'] = merged['subject_key'].values
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

    if container.y is None:
        subject_y = coerce_sample_vector(container.coords.get('subject'), container.X.shape[0])
        if subject_y is not None:
            container.y = subject_y.astype(str)

    if container.y is not None:
        container.y = np.array(container.y).astype(str)
    
    return container


def compute_all_embeddings(
    X: np.ndarray,
    reducer_names: List[str],
    n_components: int = 2,
    n_components_3d: int = 3
) -> Dict[str, Dict]:
    """
    Compute embeddings for all reducers in both 2D and 3D.
    """
    results = {}

    for name in reducer_names:
        logger.info(f"Computing embeddings for {name}...")
        results[name] = {}
        
        try:
            dr_2d = DimReduction(method=name, n_components=n_components)
            X_emb_2d = dr_2d.fit_transform(X)
            results[name]['embedding_2d'] = X_emb_2d
            results[name]['reducer_2d'] = dr_2d
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
            
    return results


def create_reducer_section(
    name: str,
    embedding_results: Dict,
    labels: Optional[np.ndarray],
    X_orig: np.ndarray,
    meta_dict: Optional[Dict[str, np.ndarray]] = None,
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

    emb_2d = embedding_results.get("embedding_2d")
    emb_3d = embedding_results.get("embedding_3d")
    reducer_2d = embedding_results.get("reducer_2d")

    n_2d = emb_2d.shape[0] if emb_2d is not None else 0
    labels_2d = coerce_sample_vector(labels, n_2d) if n_2d else None
    if labels is not None and n_2d and labels_2d is None:
        logger.warning(
            f"Skipping labels for {name} 2D plot due to size mismatch "
            f"(labels={len(np.asarray(labels).ravel())}, samples={n_2d})."
        )

    meta_2d = None
    if meta_dict and n_2d:
        filtered = {}
        for col_name, col_values in meta_dict.items():
            arr = coerce_sample_vector(col_values, n_2d)
            if arr is not None:
                filtered[col_name] = arr
        if filtered:
            meta_2d = filtered

    n_3d = emb_3d.shape[0] if emb_3d is not None else 0
    labels_3d = coerce_sample_vector(labels, n_3d) if n_3d else None
    meta_3d = None
    if meta_dict and n_3d:
        filtered = {}
        for col_name, col_values in meta_dict.items():
            arr = coerce_sample_vector(col_values, n_3d)
            if arr is not None:
                filtered[col_name] = arr
        if filtered:
            meta_3d = filtered

    reducer_diagnostics: Dict = {}

    if reducer_2d is not None and hasattr(reducer_2d, "get_diagnostics"):
        try:
            reducer_diagnostics = reducer_2d.get_diagnostics() or {}
        except Exception as err:
            logger.warning(f"Could not fetch diagnostics for {name}: {err}")

    add_embedding_plot(
        section=sec,
        embedding=emb_2d,
        labels=labels_2d,
        meta=meta_2d,
        title=f"{name.upper()} - 2D Embedding",
        dimensions=2,
        interactive=interactive,
    )

    add_embedding_plot(
        section=sec,
        embedding=emb_3d,
        labels=labels_3d,
        meta=meta_3d,
        title=f"{name.upper()} - 3D Embedding",
        dimensions=3,
        interactive=interactive,
    )

    # 3. Loss History (diagnostics API first)
    loss_history = reducer_diagnostics.get("loss_history_")
    if loss_history is None and reducer_2d is not None:
        reducer_obj = reducer_2d.reducer
        if hasattr(reducer_obj, "loss_history_"):
            loss_history = reducer_obj.loss_history_

    if loss_history is not None:
        loss_array = np.asarray(loss_history).ravel()
        if loss_array.size > 0:
            fig_loss = viz.plot_loss_history(
                loss_history=loss_array,
                title=f"{name.upper()} - Training Loss",
                interactive=interactive,
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_loss))
            else:
                sec.add_element(ImageElement(fig_loss))

    # 4. Scree / Eigen diagnostics
    variance_values = reducer_diagnostics.get("explained_variance_ratio_")
    if variance_values is None and reducer_2d is not None:
        reducer_obj = reducer_2d.reducer
        if hasattr(reducer_obj, "explained_variance_ratio_"):
            variance_values = reducer_obj.explained_variance_ratio_

    if variance_values is not None:
        variance_values = np.asarray(variance_values).ravel()
        if variance_values.size > 0:
            fig_scree = viz.plot_eigenvalues(
                values=variance_values,
                title=f"{name.upper()} - Explained Variance Ratio",
                ylabel="Explained Variance Ratio",
                interactive=interactive,
            )
            if interactive:
                sec.add_element(PlotlyElement(fig_scree))
            else:
                sec.add_element(ImageElement(fig_scree))
    else:
        eigenvalues = reducer_diagnostics.get("eigenvalues_")
        if eigenvalues is None:
            eigenvalues = reducer_diagnostics.get("eigs_")
        if eigenvalues is None and reducer_2d is not None:
            reducer_obj = reducer_2d.reducer
            if hasattr(reducer_obj, "eigenvalues_"):
                eigenvalues = reducer_obj.eigenvalues_
            elif hasattr(reducer_obj, "eigs_"):
                eigenvalues = reducer_obj.eigs_

        if eigenvalues is not None:
            eigenvalues = np.asarray(eigenvalues).ravel()
            if np.iscomplexobj(eigenvalues):
                eigenvalues = np.abs(eigenvalues)
            if eigenvalues.size > 0:
                fig_scree = viz.plot_eigenvalues(
                    values=eigenvalues,
                    title=f"{name.upper()} - Eigenvalues",
                    ylabel="Eigenvalue",
                    interactive=interactive,
                )
                if interactive:
                    sec.add_element(PlotlyElement(fig_scree))
                else:
                    sec.add_element(ImageElement(fig_scree))

    X_for_2d = None
    if emb_2d is not None and X_orig is not None and X_orig.shape[0] == emb_2d.shape[0]:
        X_for_2d = X_orig
    elif emb_2d is not None and X_orig is not None:
        logger.warning(
            f"Skipping some {name} diagnostics: X/embedding size mismatch "
            f"({X_orig.shape[0]} vs {emb_2d.shape[0]})."
        )

    # 5. Shepard Diagram
    if emb_2d is not None and X_for_2d is not None:
        X_shepard = X_for_2d
        emb_shepard = emb_2d
        fig_shepard = None
        if reducer_2d is not None:
            try:
                fig_shepard = reducer_2d.plot(
                    mode="shepard",
                    X=X_shepard,
                    sample_size=min(500, X_shepard.shape[0]),
                    title=f"{name.upper()} - Shepard Diagram",
                    interactive=interactive,
                )
            except Exception as err:
                logger.warning(f"Reducer Shepard plot failed for {name}: {err}")

        if fig_shepard is None:
            try:
                fig_shepard = viz.plot_shepard_diagram(
                    X_orig=X_shepard,
                    X_emb=emb_shepard,
                    sample_size=min(500, X_shepard.shape[0]),
                    title=f"{name.upper()} - Shepard Diagram",
                    interactive=interactive,
                )
            except Exception as err:
                logger.warning(f"Could not create Shepard diagram for {name}: {err}")

        if fig_shepard is not None:
            if interactive:
                sec.add_element(PlotlyElement(fig_shepard))
            else:
                sec.add_element(ImageElement(fig_shepard))

    return sec


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
            reducer_obj = reducer.reducer
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
    
    parser.add_argument("--condition", default=None, help="Condition filter (used with --use_derivatives)")
    parser.add_argument("--segment_duration", type=float, default=60.0, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Window overlap in seconds")
    
    parser.add_argument(
        "--stacking_mode",
        choices=["flat", "time_as_sample", "epoch_scalar_mean"],
        default="flat",
        help=(
            "Feature layout for reducers: "
            "'flat' => columns=(channel x time), "
            "'time_as_sample' => columns=channel, "
            "'epoch_scalar_mean' => columns=1 (mean over channel x time per epoch)"
        ),
    )
    
    parser.add_argument("--subsample", type=int, default=None, help="Number of subjects to random sample")
    parser.add_argument("--subject_col", default="Study ID", help="Column in metadata")
    parser.add_argument("--target_col", default="Group", help="Column to use as labels")
    
    parser.add_argument("--reducers", nargs="+", default=ALL_REDUCERS, help="List of reducers")
    parser.add_argument("--n_components_2d", type=int, default=2, help="Components for 2D")
    parser.add_argument("--n_components_3d", type=int, default=3, help="Components for 3D")
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable interactive plots (use --no-interactive for static report figures)",
    )
    parser.add_argument(
        "--save_static_figures",
        action="store_true",
        help="Save per-reducer static PNG embeddings in addition to the HTML report",
    )
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings to disk")
    parser.add_argument("--use_derivatives", action="store_true", help="Load from derivatives/preproc instead of raw BIDS")
    parser.add_argument("--desc", default="base", help="Desc to load from derivatives (default: base)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process")
    parser.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore BAD_ annotations during epoching (use --no-ignore-annotations to enforce them)",
    )
    parser.add_argument("--subject_average", action="store_true", help="Average all segments per subject into 1 point")
    
    args = parser.parse_args()
    
    # Force logging level
    logging.getLogger().setLevel(logging.INFO)
    
    interactive = args.interactive
    
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
    if args.use_derivatives:
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
            ignore_annotations=args.ignore_annotations,
            subject_average=args.subject_average
        )
    else:
        dc = load_bids_data(
            bids_root=bids_root,
            subjects=subjects if subjects else None,
            task=args.task,
            session=args.session,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            stacking_mode=args.stacking_mode,
            metadata_df=meta_df,
            subject_col=args.subject_col,
            target_col=args.target_col
        )
    
    X = dc.X
    labels = resolve_labels(dc, target_col=args.target_col)

    logger.info(
        f"Data shape: {X.shape}, Labels: {labels.shape if labels is not None else 'None'}"
    )
    
    # Compute All Embeddings (2D and 3D)
    embedding_results = compute_all_embeddings(
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
    
    # Generate Report
    logger.info("Generating Report...")
    report = Report(title=f"Comprehensive DimReduction Analysis: {args.dataset_name}")
    report.add_container(dc)
    
    meta_dict = build_meta_dict(dc)
    logger.info(f"Available labels for color coding: {list(meta_dict.keys())}")
    
    # Section 1: Individual Reducer Sections
    for name in args.reducers:
        if name in embedding_results:
            sec = create_reducer_section(
                name=name,
                embedding_results=embedding_results[name],
                labels=labels,
                X_orig=X,
                meta_dict=meta_dict,
                interactive=interactive
            )
            report.add_section(sec)
    
    # Section 2: Feature Importance (if applicable)
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
    if args.save_static_figures:
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

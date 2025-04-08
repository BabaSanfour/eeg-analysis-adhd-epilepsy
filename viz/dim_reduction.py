#!/usr/bin/env python3
"""
Dimensionality reduction script.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Append project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from viz.embeddings import load_embeddings, reshape_embeddings
from utils.config import results_dir


def load_and_reshape_embeddings(n_subjects: int,
                    segment_duration: int = 10,
                    z_score: bool = True,
                    z_score_axis: int = 1,
                    n_time_segments: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads embeddings along with subjects and time segments,
    and reshapes the embeddings.
    """
    embeddings, subjects, time_segments = load_embeddings(n_subjects, segment_duration, z_score, z_score_axis, n_time_segments)
    embeddings = reshape_embeddings(embeddings)
    return embeddings, subjects, time_segments


def clean_nan_data(embeddings, subjects, time_segments):
    """
    Removes rows from data arrays that contain NaN values.
    """
    # Find indices of rows containing any NaN
    nan_indices = np.unique(np.where(np.isnan(embeddings))[0])
    if nan_indices.size > 0:
        logging.info(f"NaN values found in row indices: {nan_indices}. Cleaning data...")
        embeddings = np.delete(embeddings, nan_indices, axis=0)
        subjects = np.delete(subjects, nan_indices, axis=0)
        time_segments = np.delete(time_segments, nan_indices, axis=0)
        logging.info("Data cleaned.")
    else:
        logging.info("No NaN values found, no cleaning needed.")
    return embeddings, subjects, time_segments


def save_cleaned_data(embeddings, subjects, time_segments, results_dir=results_dir, segment_duration=10, z_score=True, z_score_axis=1):
    """
    Saves the cleaned arrays as a compressed .npz file.
    """
    save_path = os.path.join(
        results_dir, f"embeddings_cleaned_dur-{segment_duration}s_zscore-{z_score}_axis-{z_score_axis}.npz"
    )
    np.savez_compressed(
        save_path,
        embeddings_array=embeddings,
        subjects_array=subjects,
        time_segments_array=time_segments,
    )
    logging.info(f"Cleaned data saved to {save_path}")


def umap_reduction(embeddings, n_components=2):
    """
    Applies UMAP dimensionality reduction to the embeddings.
    """
    import umap
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(embeddings)


def pca_reduction(embeddings, n_components=2):
    """
    Applies PCA dimensionality reduction to the embeddings.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)


def tsne_reduction(embeddings, n_components=2):
    """
    Applies t-SNE dimensionality reduction to the embeddings.
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    return tsne.fit_transform(embeddings)

def dimensionality_reduction(embeddings, method='umap', **kwargs):
    """
    Applies dimensionality reduction to the embeddings using the specified method.
    """
    reducers = {
        'umap': umap_reduction,
        'pca': pca_reduction,
        'tsne': tsne_reduction
    }
    try:
        func = reducers[method]
    except KeyError:
        raise ValueError(f"Unknown reduction method: {method}")
    return func(embeddings, **kwargs)


def save_reduced_data(embedding, subjects, time_segments, method, results_dir=results_dir, segment_duration=10, z_score=True, z_score_axis=1):
    """
    Saves the reduced data as a compressed .npz file.
    """
    save_path = os.path.join(
        results_dir, f"embeddings_{method}_reduced_dur-{segment_duration}s_zscore-{z_score}_axis-{z_score_axis}.npz"
    )
    np.savez_compressed(
        save_path,
        reduced_embeddings=embedding,
        subjects=subjects,
        time_segments=time_segments,
    )
    logging.info(f"{method.upper()} reduced data saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Dimensionality reduction script.")
    parser.add_argument("--n_subjects", type=int, default=253,
                        help="Number of subjects to process")
    parser.add_argument("--segment_duration", type=int, default=10,
                        help="Time segment used to segment EEG data and embeddings")
    parser.add_argument("--z_score", type=bool, default=True,
                        help="Whether the data was z-scored")
    parser.add_argument("--z_score_axis", type=int, default=1,
                        help="Axis to which we applied z-score normalization")
    parser.add_argument("--method", type=str, choices=["umap", "pca", "tsne"],
                        default="umap", help="Dimensionality reduction method")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components for reduction")
    args = parser.parse_args()
    n_time_segments = 20 * 60 // args.segment_duration
    embeddings, subjects, time_segments = load_and_reshape_embeddings(
        n_subjects = args.n_subjects,
        segment_duration = args.segment_duration,
        z_score = args.z_score,
        z_score_axis = args.z_score_axis,
        n_time_segments = n_time_segments
    )
    logging.info(f"Initial shapes: {embeddings.shape}, {subjects.shape}, {time_segments.shape}")

    embeddings, subjects, time_segments = clean_nan_data(embeddings, subjects, time_segments)
    logging.info(f"Cleaned shapes: {embeddings.shape}, {subjects.shape}, {time_segments.shape}")

    save_cleaned_data(embeddings, subjects, time_segments)

    reduced_embedding = dimensionality_reduction(
        embeddings,
        method=args.method,
        n_components=args.n_components,
    )
    logging.info(f"{args.method.upper()} reduced shape: {reduced_embedding.shape}")
    save_reduced_data(reduced_embedding, subjects, time_segments, method=args.method)
    embeddings, subjects, time_segments = load_and_reshape_embeddings(
        n_subjects=args.n_subjects, 
        segment_duration=args.segment_duration, 
        z_score=args.z_score, 
        z_score_axis=args.z_score_axis, 
        n_time_segments=n_time_segments
    )
    logging.info(f"Initial shapes: {embeddings.shape}, {subjects.shape}, {time_segments.shape}")

    embeddings, subjects, time_segments = clean_nan_data(embeddings, subjects, time_segments)
    logging.info(f"Cleaned shapes: {embeddings.shape}, {subjects.shape}, {time_segments.shape}")

    save_cleaned_data(embeddings, subjects, time_segments)

    reduction_methods =  ["umap", "pca", "tsne"]

    # Apply each reduction and save results
    for method, in reduction_methods:
        logging.info(f"Applying {method.upper()} dimensionality reduction...")
        reduced_embedding = dimensionality_reduction(embeddings, method=method)
        logging.info(f"{method.upper()} reduced shape: {reduced_embedding.shape}")
        save_reduced_data(reduced_embedding, subjects, time_segments, method=method)

if __name__ == "__main__":
    main()
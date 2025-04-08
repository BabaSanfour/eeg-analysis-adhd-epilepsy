#!/usr/bin/env python3
"""
utils for loading and reshaping EEG embeddings.
"""
import pickle
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Append project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
import os
import sys
sys.path.insert(0, str(BASE_DIR))
from utils.config import embeddings_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embeddings(n_subjects: int,
                    segment_duration: int = 10,
                    z_score: bool = True,
                    z_score_axis: int = 1,
                    n_time_segments: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and extract embeddings for all subjects.

    Args:
        n_subjects: Number of subjects to load.
        segment_duration: Duration of each segment in seconds.
        z_score: Whether the embeddings are z-scored.
        z_score_axis: Axis used for z-scoring.
        n_time_segments: Maximum time segments to include per subject.

    Returns:
        A tuple of numpy arrays: embeddings_array, subjects_array, time_segments_array.
    """
    embeddings = {}
    # Load embeddings per subject.
    for subject_id in tqdm(range(1, n_subjects + 1), desc="Loading subjects"):
        file_name = f'embeddings_sub-{subject_id}_dur-{segment_duration}s_zscore-{z_score}_axis-{z_score_axis}.pkl'
        embeddings_file = os.path.join(embeddings_dir, file_name)
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings[subject_id] = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading subject {subject_id}: {e}")

    embeddings_array = []
    subjects_array = []
    time_segments_array = []

    for subject, embed in tqdm(embeddings.items(), desc="Processing embeddings"):
        for time_segment_id, time_segment_embedding in embed.items():
            if time_segment_id > n_time_segments:
                break
            embeddings_array.append(time_segment_embedding['embedding'][0])
            subjects_array.append(subject)
            time_segments_array.append(time_segment_id)

    return (np.array(embeddings_array),
            np.array(subjects_array),
            np.array(time_segments_array))

def reshape_embeddings(embeddings_array: np.ndarray, sensorwise: bool = False) -> np.ndarray:
    """
    Reshape the embeddings array.

    If sensorwise is False, reshape to (num_items, -1).
    If sensorwise is True, reshape each sensor's embedding individually.

    Args:
        embeddings_array: The numpy array of embeddings.
        sensorwise: Flag for sensorwise reshaping.
    
    Returns:
        A reshaped numpy array.
    """
    if sensorwise:
        return embeddings_array.reshape(embeddings_array.shape[0], embeddings_array.shape[1], -1)
    return embeddings_array.reshape(embeddings_array.shape[0], -1)

if __name__ == '__main__':
    # Example usage:
    import argparse

    parser = argparse.ArgumentParser(description="Load and process EEG embeddings.")
    parser.add_argument("--n_subjects", type=int, default=10, help="Number of subjects")
    parser.add_argument("--segment_duration", type=int, default=10, help="Segment duration in seconds")
    parser.add_argument("--z_score", action="store_true", help="Apply z-scoring")
    parser.add_argument("--z_score_axis", type=int, default=1, help="Axis for z-scoring")
    parser.add_argument("--n_time_segments", type=int, default=100, help="Maximum number of time segments")
    parser.add_argument("--sensorwise", action="store_true", help="Reshape sensorwise")
    args = parser.parse_args()

    embeddings_array, subjects_array, time_segments_array = load_embeddings(
        n_subjects=args.n_subjects,
        segment_duration=args.segment_duration,
        z_score=args.z_score,
        z_score_axis=args.z_score_axis,
        n_time_segments=args.n_time_segments,
    )

    reshaped = reshape_embeddings(embeddings_array, sensorwise=args.sensorwise)
    logger.info(f"Embeddings loaded: {embeddings_array.shape}")
    logger.info(f"Subjects array shape: {subjects_array.shape}")
    logger.info(f"Time segments array shape: {time_segments_array.shape}")
    logger.info(f"Reshaped embeddings array shape: {reshaped.shape}")
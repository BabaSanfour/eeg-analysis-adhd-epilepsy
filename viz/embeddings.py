

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import derivatives_dir, results_dir, embeddings_dir

from tqdm import trange, tqdm
import pickle
import numpy as np

def load_embeddings(n_subjects, time_segment):
    """
    Load and extract embeddings for all subjects
    """
    embeddings = {}
    for subject in tqdm(range(1, n_subjects+1)):
        if subject == 133:
            continue
        embeddings_file = os.path.join(embeddings_dir, f'embeddings_sub-{subject}_{time_segment}.pkl')
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings[subject] = pickle.load(f)
        except FileNotFoundError:
            print(f"Embeddings file not found for subject {subject}")
    embeddings_array = []
    subjetcs_array = []
    time_segments_array = []
    for subject, embed in tqdm(embeddings.items()):
        for time_segment_id, time_segment_embedding in embed.items():
            # load only 20points for each subject:
            if time_segment_id > 100:
                break
            embeddings_array.append(time_segment_embedding['embedding'][0])
            subjetcs_array.append(subject)
            time_segments_array.append(time_segment_id)
    return np.array(embeddings_array), np.array(subjetcs_array), np.array(time_segments_array)

def reshape_embeddings(embeddings_array, sensorwise=False):
    """
    Reshape the embeddings_array in one of two ways:
    1. If sensorwise is False, reshape to (num_items, -1).
    2. If sensorwise is True, reshape each sensor's embedding individually
    """
    if sensorwise:
        return embeddings_array.reshape(embeddings_array.shape[0], embeddings_array.shape[1], -1)
    else:
        return embeddings_array.reshape(embeddings_array.shape[0], -1)
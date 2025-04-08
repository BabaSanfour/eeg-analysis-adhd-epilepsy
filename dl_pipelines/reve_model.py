import os
import time
import pickle
import argparse
import logging

import mne
import torch
from mne_bids import BIDSPath
from goofi.data import to_data
from goofi.nodes.analysis.reveeeg import ReveEEG

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import derivatives_dir, results_dir
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def segment_and_process(eeg_path: str, z_score: bool = True, z_score_axis: int = 1, segment_duration: int = 60) -> dict:
    """
    Segments and processes EEG data to generate embeddings for each segment.

    Parameters:
        eeg_path (str): Path to the EEG BrainVision file.
        z_score (bool, optional): Whether to perform z-score normalization on the data. Defaults to True.
        z_score_axis (int, optional): Axis along which to calculate z-score. Defaults to 1 (over channels)
        segment_duration (int, optional): Duration (in seconds) of each segment to process. Defaults to 60.
        data_length (int, optional): Unused parameter reserved for future use. Defaults to 20.

    Returns:
        dict: A dictionary where keys are segment indices (int) and values are the corresponding embeddings.
              Returns an empty dict if no segments are processed or if an error occurs.
    """
    try:
        raw = mne.io.read_raw_brainvision(eeg_path, preload=True)
    except Exception as e:
        logging.error(f"Error reading file {eeg_path}: {e}")
        return {}

    # Retrieve EEG data and sampling frequency
    raw_data = raw.get_data()
    sfreq = raw.info["sfreq"]

    # Normalize the data if z_score is True
    if z_score:
        # Calculate mean and std over the specified axis (0 or 1)
        channel_means = raw_data.mean(axis=z_score_axis, keepdims=True)
        channel_stds = raw_data.std(axis=z_score_axis, keepdims=True)
        # Perform z-score normalization and clip values exceeding 15 standard deviations
        raw_data = np.clip((raw_data - channel_means) / channel_stds, -15, 15)

    n_samples = raw_data.shape[1]
    n_timepoints_per_segment = int(segment_duration * sfreq)
    n_segments = n_samples // n_timepoints_per_segment

    if n_segments == 0:
        logging.warning(f"No segments extracted from file {eeg_path} with segment_duration {segment_duration} seconds.")
        return {}

    node = ReveEEG.create_standalone()
    node.params.reve.device.value = "cuda" if torch.cuda.is_available() else "cpu"
    node.setup()

    embeddings = {}
    for seg in tqdm(range(n_segments), desc="Processing segments"):
        tmin = seg * n_timepoints_per_segment
        tmax = (seg + 1) * n_timepoints_per_segment
        segment_data = raw_data[:, tmin:tmax]
        data_container = to_data(
            segment_data,
            {
                "sfreq": sfreq,
                "channels": {"dim0": raw.ch_names},
            },
        )
        try:
            result = node.process(data_container)
            embeddings[seg] = result['embedding'][0]
        except Exception as e:
            logging.error(f"Error processing segment {seg} in file {eeg_path}: {e}")

    return embeddings


def save_embeddings(embeddings: dict, subject_id: str, segment_duration: int, z_score: bool, z_score_axis):
    """
    Save the computed embeddings for a subject to a pickle file.
    """
    output_dir = os.path.join(results_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"embeddings_sub-{subject_id}_dur-{segment_duration}s_zscore-{z_score}_axis-{z_score_axis}.pkl"
    )
    
    try:
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info(f"Embeddings saved to {output_file}")
    except Exception as e:
        logging.error(f"Could not save embeddings for subject {subject_id}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for processing the EEG data.
    """
    parser = argparse.ArgumentParser(
        description="Segment and process EEG data to generate embeddings."
    )
    parser.add_argument("--n_subjects", type=int, default=15, 
                        help="Number of subjects to process (default: 15)")
    parser.add_argument("--start_subject", type=int, default=1, 
                        help="Subject ID to start processing from (default: 1)")
    parser.add_argument("--segment_duration", type=int, default=60, 
                        help="Duration (in seconds) of each segment to process (default: 60)")
    parser.add_argument("--z_score", type=bool, default=True,
                        help="Whether to perform z-score normalization on the data (default: True)")
    parser.add_argument("--z_score_axis", type=int, default=1,
                        help="Axis along which to calculate z-score (default: 1)")
    return parser.parse_args()


def process_subject(subject_id: int, segment_duration: int, z_score: bool = True, z_score_axis: int = 1) -> None:
    """
    Process EEG data for a single subject: segment the data, compute embeddings, and save them.
    """
    subject = f"sub-{subject_id}"

    bids_path = BIDSPath(
        root=derivatives_dir,
        subject=str(subject_id),
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
        processing="cleaned",
    )

    logging.info(f"Processing {subject} using file {bids_path.fpath}")
    start_time = time.time()
    
    embeddings = segment_and_process(bids_path.fpath, z_score=z_score, z_score_axis=z_score_axis, segment_duration=segment_duration)
    
    # Check if embeddings were successfully computed.
    if embeddings:
        save_embeddings(embeddings, subject, segment_duration)
        logging.info(f"Processed {subject} in {time.time() - start_time:.2f} seconds")
    else:
        logging.error(f"Failed to compute embeddings for {subject}")


def main() -> None:
    """
    Main function to parse arguments and process a batch of subjects.
    """
    args = parse_args()
    logging.info(
        f"Starting processing for {args.n_subjects} subjects beginning with subject {args.start_subject}"
    )
    for subject_id in range(args.start_subject, args.start_subject + args.n_subjects):
        process_subject(subject_id, args.segment_duration)


if __name__ == "__main__":
    main()
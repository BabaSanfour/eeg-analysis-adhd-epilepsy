#!/usr/bin/env python3
"""
Segment and process EEG data sequentially to generate embeddings.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

import mne
import torch
from mne_bids import BIDSPath
from goofi.data import to_data
from goofi.nodes.analysis.reveeeg import ReveEEG
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment and process EEG data sequentially to generate embeddings."
    )
    parser.add_argument(
        "--derivatives_dir",
        type=Path,
        required=True,
        help="Path to BIDS derivatives directory containing cleaned EEG files.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Directory in which to save embeddings.",
    )
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=15,
        help="Number of subjects to process (default: 15).",
    )
    parser.add_argument(
        "--start_subject",
        type=int,
        default=1,
        help="First subject index to process (default: 1).",
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=60,
        help="Duration of each segment in seconds (default: 60).",
    )
    parser.add_argument(
        "--z_score",
        action="store_true",
        help="Apply z-score normalization to each segment.",
    )
    parser.add_argument(
        "--z_score_axis",
        type=int,
        default=1,
        help="Axis along which to compute z-score (default: 1).",
    )
    parser.add_argument(
        "--notch",
        action="store_true",
        help="Apply notch filter to the EEG data before processing.",
    )
    return parser.parse_args()


def init_reve_node(device: str) -> ReveEEG:
    """
    Initialize the ReveEEG node once and return it.
    """
    node = ReveEEG.create_standalone()
    node.params.reve.device.value = device
    node.setup()
    return node


def segment_and_process(
    raw: mne.io.BaseRaw,
    node: ReveEEG,
    segment_duration: int,
) -> Dict[int, Any]:
    """
    Segment the raw data and run it through the ReveEEG node to get embeddings.
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    samples_per_seg = int(segment_duration * sfreq)
    total_samples = data.shape[1]
    n_segments = total_samples // samples_per_seg

    if n_segments == 0:
        logging.warning("No segments found for %s", getattr(raw, "filenames", "unknown file"))
        return {}

    embeddings: Dict[int, Any] = {}
    for seg in tqdm(range(n_segments), desc="Processing segments"):
        start = seg * samples_per_seg
        end = start + samples_per_seg
        segment = data[:, start:end]
        container = to_data(
            segment,
            {"sfreq": sfreq, "channels": {"dim0": raw.ch_names}},
        )
        try:
            res = node.process(container)
            embedding = res["embedding"][0]
            try:
                embedding = embedding.astype("float32")
            except AttributeError:
                embedding = embedding.to(torch.float32)
            embeddings[seg] = embedding
        except Exception as e:
            logging.error("Error processing segment %d: %s", seg, e)
    return embeddings


def save_embeddings(
    embeddings: Dict[int, Any],
    output_dir: Path,
    subject: str,
    segment_duration: int,
    z_score: bool,
    z_score_axis: int,
    notch: bool,
) -> None:
    """
    Save embeddings dict to a pickle file in a BIDS-compliant structure.
    """
    # Create processing string components based on provided flags.
    components = []
    components = [
        comp for cond, comp in (
            (notch, "notch"),
            (z_score, f"zscoreaxis{z_score_axis}"),
            (segment_duration > 0, f"seg{segment_duration}"),
        ) if cond
    ]
    processing_str = "embeddings" + (''.join(components) if components else "raw")
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"sub-{subject}_task-RESTING_run-01_{processing_str}.pkl"

    try:
        with open(file_path, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info("Saved embeddings to %s", file_path)
    except Exception as e:
        logging.error("Could not save embeddings for %s: %s", subject, e)


def process_subject(subject_idx: int, args: argparse.Namespace, node: ReveEEG) -> bool:
    """
    Load, segment, process, and save embeddings for one subject.
    Returns True on success, False otherwise.
    """
    subject = f"{subject_idx:04d}"
    # Generate processing string based on flags.
    components = []
    components = [
        comp for cond, comp in (
            (args.notch, "notch"),
            (args.z_score, f"zscoreaxis{args.z_score_axis}"),
        ) if cond
    ]
    processing_str = "cleaned" + (''.join(components) if components else "raw")

    bids = BIDSPath(
        root=str(args.derivatives_dir),
        subject=str(subject),
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
        processing=processing_str,
    )

    path = bids.fpath
    if not Path(path).exists():
        logging.error("File not found for %s: %s", subject, path)
        return False

    logging.info("Processing %s", subject)
    try:
        raw = mne.io.read_raw_brainvision(path, preload=True)
    except Exception as e:
        logging.error("Error reading %s: %s", path, e)
        return False

    embeddings = segment_and_process(raw, node, args.segment_duration)
    if not embeddings:
        logging.error("No embeddings generated for %s", subject)
        return False
    # create the output directory if it doesn't exist
    output_dir = args.embeddings_dir / f"sub-{subject}"
    save_embeddings(
        embeddings,
        output_dir,
        subject,
        args.segment_duration,
        args.z_score,
        args.z_score_axis,
        args.notch,
    )
    return True


def main() -> None:
    args = parse_args()
    logging.info(
        "Starting processing for %d subjects from subject %d",
        args.n_subjects,
        args.start_subject,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node = init_reve_node(device)
    for idx in range(args.start_subject, args.start_subject + args.n_subjects):
        success = process_subject(idx, args, node)
        if not success:
            logging.warning("Processing failed for subject %d", idx)


if __name__ == "__main__":
    main()

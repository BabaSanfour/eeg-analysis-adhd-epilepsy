"""Write the saved epochs of one epoch variant as one file per condition, which the raw
analyses read.

Opening every recording's FIF file takes minutes on a cluster file system; such a file
loads in seconds. Each holds its condition as the saved-epochs loader returns it, so write
it again whenever the epochs are.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from eeg_adhd_epilepsy.analysis.dataset import (
    build_container,
    consolidated_epochs_path,
    write_consolidated_epochs,
)
from eeg_adhd_epilepsy.io.bids import PREPROC_BIDSIGNORE, write_bidsignore

logger = logging.getLogger(__name__)


def run(
    bids_root: Path, desc: str, conditions: list[str], task: str = "clinical", n_jobs: int = 1
) -> list[Path]:
    """Write one file per condition of variant ``desc`` and return their paths."""
    paths = []
    for condition in conditions:
        container = build_container(
            bids_root,
            use_derivatives=True,
            task=task,
            desc=desc,
            condition=condition,
            n_jobs=n_jobs,
        )
        logger.info(f"{condition}: {container.X.shape[0]} epochs")
        path = consolidated_epochs_path(bids_root, desc, condition, task)
        # Moved into place once complete, so the analyses never read half a file.
        tmp = path.with_name(f".{path.name}.tmp")
        write_consolidated_epochs(container, tmp)
        os.replace(tmp, path)
        paths.append(path)
        del container
    write_bidsignore(paths[0].parent, PREPROC_BIDSIGNORE)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the saved epochs of one epoch variant as one file for the raw analyses."
    )
    parser.add_argument("--bids_root", required=True, help="The raw BIDS dataset.")
    parser.add_argument("--desc", required=True, help="Epoch variant: base or baserejected.")
    parser.add_argument(
        "--conditions", nargs="+", required=True, help="Conditions to write, e.g. EO_baseline."
    )
    parser.add_argument("--task", default="clinical", help="BIDS task of the recordings.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Threads reading the FIF files.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in run(Path(args.bids_root), args.desc, args.conditions, args.task, args.n_jobs):
        logger.info(f"Wrote {path} ({path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()

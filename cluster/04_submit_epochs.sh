#!/usr/bin/env bash
#SBATCH --job-name=eeg_epochs
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

SEGMENT_DURATION=${SEGMENT_DURATION:-10.0}
OVERLAP=${OVERLAP:-0.0}
# 1: drop the epochs overlapping a BAD_ annotation and write them as desc-clean;
# 0: every epoch, as desc-base.
REJECT_BY_ANNOTATION=${REJECT_BY_ANNOTATION:-0}

require_dir "$BIDS_ROOT"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads 1

cmd=(
  python -m eeg_adhd_epilepsy.preproc.epochs
  --bids_root "$BIDS_ROOT"
  --desc base
  --segment_duration "$SEGMENT_DURATION"
  --overlap "$OVERLAP"
  --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
  --reports_root "$SCRATCH_ROOT/reports"
)
if [ "$REJECT_BY_ANNOTATION" = "1" ]; then
  cmd+=(--reject_by_annotation)
fi

"${cmd[@]}"
validate_bids "$BIDS_ROOT/derivatives/preproc"

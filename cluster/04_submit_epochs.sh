#!/usr/bin/env bash
#SBATCH --job-name=eeg_epochs
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

SEGMENT_DURATION=${SEGMENT_DURATION:-10.0}
OVERLAP=${OVERLAP:-0.0}
OVERWRITE=${OVERWRITE:-0}

require_dir "$BIDS_ROOT"
require_dir "$VENV_PATH"

BASE_COUNT=$(find "$BIDS_ROOT/derivatives/preproc" -name '*desc-base_eeg.fif' -type f | wc -l)
[ "$BASE_COUNT" -gt 0 ] || { echo "No base-cleaned FIF files found under: $BIDS_ROOT/derivatives/preproc"; exit 1; }
echo "Found $BASE_COUNT base-cleaned FIF files for epoching."

dra_activate
dra_pin_threads 1

cmd=(
  python -m eeg_adhd_epilepsy.preproc.epochs
  --bids_root "$BIDS_ROOT"
  --desc base
  --segment_duration "$SEGMENT_DURATION"
  --overlap "$OVERLAP"
  --reports_root "$SCRATCH_ROOT/reports"
  --ignore_annotations
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"

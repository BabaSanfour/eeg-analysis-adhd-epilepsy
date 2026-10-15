#!/usr/bin/env bash
#SBATCH --job-name=eeg_epochs
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# One recording per core. Measured on all 1,237 recordings (2026-10-15): reading and
# writing the FIF files bound it, so CPU use stays at 30-45 %; 2 workers took 2.6 min in
# 10 GB, the fewest core-seconds (1: 5.6 min in 8 GB; 4: 1.8 min; 16: 32 s).

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
  --n_jobs "${SLURM_CPUS_PER_TASK:-2}"
  --reports_root "$SCRATCH_ROOT/reports"
)
if [ "$REJECT_BY_ANNOTATION" = "1" ]; then
  cmd+=(--reject_by_annotation)
fi

"${cmd[@]}"
validate_bids "$BIDS_ROOT/derivatives/preproc"

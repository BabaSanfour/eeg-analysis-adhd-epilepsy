#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

# Point BIDS_ROOT to your scratch BIDS because that's where the descriptors are saved!
SCRATCH_BIDS_ROOT=${SCRATCH_BIDS_ROOT:-$SCRATCH_ROOT/BIDS}

dra_activate
# Single-process merge that wants threaded BLAS.
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

python -m eeg_adhd_epilepsy.analysis.merge_descriptors \
  --bids_root "$SCRATCH_BIDS_ROOT" \
  --reports_root "$SCRATCH_ROOT/reports" \
  --skip_inconsistent
validate_bids "$SCRATCH_BIDS_ROOT/derivatives/signal_features/descriptors"

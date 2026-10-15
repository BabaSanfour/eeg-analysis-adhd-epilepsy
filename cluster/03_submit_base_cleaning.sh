#!/usr/bin/env bash
#SBATCH --job-name=base_clean
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

# One recording per core; peak memory was ~3.4 GB per worker on 2026-10-14.
THREADS=${SLURM_CPUS_PER_TASK:-32}

require_dir "$BIDS_ROOT"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads 1

python -m eeg_adhd_epilepsy.preproc.base \
  --bids_root "$BIDS_ROOT" \
  --reports_root "$SCRATCH_ROOT/reports" \
  --n_jobs "$THREADS"
validate_bids "$BIDS_ROOT/derivatives/preproc"

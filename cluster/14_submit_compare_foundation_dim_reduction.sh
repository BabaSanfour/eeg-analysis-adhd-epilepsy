#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_fm_compare
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Cross-model comparison of the per-model foundation dim-reduction runs produced
# by 13_batch_run_dim_reduction_foundation.sh. Each model × transform ×
# representation was a separate array task; this step gathers every cohort's
# leaderboards into one report ranking them on the same axes.
# CPU-only; run once after the array, e.g.:
#   sbatch --dependency=afterok:<array_jobid> "$0"

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# Optional: a single cohort dataset_name. Empty compares every cohort found.
DATASET_NAME=${DATASET_NAME:-}

require_dir "$VENV_PATH"
require_dir "$DIM_REDUCTION_ROOT"

# CPU-only aggregation; no BLAS pinning needed.
dra_activate

cmd=(
    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction
    --compare_only
    --bids_root "$BIDS_ROOT"
    --derivative_root "$DIM_REDUCTION_ROOT"
    --reports_root "$REPORTS_ROOT"
)
if [ -n "$DATASET_NAME" ]; then
    cmd+=(--dataset_name "$DATASET_NAME")
fi

"${cmd[@]}"

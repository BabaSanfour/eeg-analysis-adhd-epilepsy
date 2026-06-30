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
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Cross-model comparison of the per-model foundation dim-reduction runs produced
# by 12_batch_run_dim_reduction_foundation.sh. Each model × representation was a
# separate array task; this step gathers every cohort's per-model leaderboards
# into one report ranking model × representation × reducer on the same axes.
# CPU-only; run once after the array, e.g.:
#   sbatch --dependency=afterok:<array_jobid> "$0"

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
DIM_REDUCTION_ROOT=${DIM_REDUCTION_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/dim_reduction}
REPORTS_ROOT=${REPORTS_ROOT:-$SCRATCH_ROOT/reports}
# Optional: a single cohort dataset_name. Empty compares every cohort found.
DATASET_NAME=${DATASET_NAME:-}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }
[ -d "$DIM_REDUCTION_ROOT" ] || { echo "Dim-reduction root not found: $DIM_REDUCTION_ROOT"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1

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

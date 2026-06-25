#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
# Point BIDS_ROOT to your scratch BIDS because that's where the descriptors are saved!
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
SCRATCH_BIDS_ROOT=${SCRATCH_BIDS_ROOT:-$SCRATCH_ROOT/BIDS}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m eeg_adhd_epilepsy.analysis.merge_descriptors \
  --bids_root "$SCRATCH_BIDS_ROOT" \
  --reports_root "$SCRATCH_ROOT/reports" \
  --skip_inconsistent

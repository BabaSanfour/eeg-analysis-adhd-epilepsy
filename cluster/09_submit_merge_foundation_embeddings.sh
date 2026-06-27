#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb_merge
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Aggregate every per-task shard (written by 08_submit_foundation_embeddings.sh)
# into the dataset-level run_manifest.json, failures.csv, dataset_description.json,
# run status, and the HTML report. CPU-only; no model is loaded. Run once after the
# array, e.g.:  sbatch --dependency=afterok:<array_jobid> "$0"

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
REPORTS_ROOT=${REPORTS_ROOT:-$SCRATCH_ROOT/reports}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1

python -m eeg_adhd_epilepsy.analysis.merge_foundation_embeddings \
  --bids_root "$BIDS_ROOT" \
  --derivative_root "$DERIVATIVE_ROOT" \
  --reports_root "$REPORTS_ROOT"

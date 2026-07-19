#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb_merge
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Aggregate every per-task shard (written by 08_submit_foundation_embeddings.sh)
# into the dataset-level run_manifest.json, failures.csv, dataset_description.json,
# run status, combined raw/aligned tables, and the HTML report. CPU-only; no model
# is loaded. Submit this manually after the stage 09 alignment array completes.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}

require_dir "$VENV_PATH"

dra_activate
# Single-process merge that wants threaded BLAS.
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

python -m eeg_adhd_epilepsy.analysis.merge_foundation_embeddings \
  --bids_root "$BIDS_ROOT" \
  --derivative_root "$DERIVATIVE_ROOT" \
  --reports_root "$REPORTS_ROOT"

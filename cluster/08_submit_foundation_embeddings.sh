#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb
#SBATCH --account=def-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --array=1-1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
FOUNDATION_CONFIG=${FOUNDATION_CONFIG:?Set FOUNDATION_CONFIG to the dataset-wide embedding config}
BIDS_ROOT=${BIDS_ROOT:?Set BIDS_ROOT to the BIDS dataset}
METADATA=${METADATA:?Set METADATA to the metadata CSV}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
SUBMIT_STATE_DIR=${SUBMIT_STATE_DIR:-$PROJECT_ROOT/cluster/.foundation_array_state}
AUTO_SUBMIT_NEXT=${AUTO_SUBMIT_NEXT:-1}
FIRST_BATCH_SIZE=${FIRST_BATCH_SIZE:-1000}
SECOND_BATCH_SIZE=${SECOND_BATCH_SIZE:-218}

ROW_OFFSET=${ROW_OFFSET:-0}
METADATA_ROW=$((SLURM_ARRAY_TASK_ID + ROW_OFFSET))

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -f "$FOUNDATION_CONFIG" ] || { echo "Foundation config not found: $FOUNDATION_CONFIG"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -f "$METADATA" ] || { echo "Metadata not found: $METADATA"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

python -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings \
  --config "$FOUNDATION_CONFIG" \
  --bids_root "$BIDS_ROOT" \
  --metadata "$METADATA" \
  --derivative_root "$DERIVATIVE_ROOT" \
  --metadata_row "$METADATA_ROW"

if [[ "$AUTO_SUBMIT_NEXT" == "1" && "$ROW_OFFSET" == "0" ]]; then
  batch_state_dir="$SUBMIT_STATE_DIR/${SLURM_ARRAY_JOB_ID:-manual}"
  mkdir -p "$batch_state_dir"
  touch "$batch_state_dir/${SLURM_ARRAY_TASK_ID}.done"

  done_count=$(find "$batch_state_dir" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
  if [[ "$done_count" -ge "$FIRST_BATCH_SIZE" ]]; then
    if mkdir "$batch_state_dir/submit_second.lock" 2>/dev/null; then
      sbatch \
        --array=1-"$SECOND_BATCH_SIZE" \
        --export=ALL,ROW_OFFSET="$FIRST_BATCH_SIZE",AUTO_SUBMIT_NEXT=0 \
        "$0"
    fi
  fi
fi

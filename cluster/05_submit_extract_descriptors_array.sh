#!/usr/bin/env bash
#SBATCH --job-name=eeg_desc
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

CONFIG_PATH=${CONFIG_PATH:-$PROJECT_ROOT/configs/descriptors.yaml}
SUBMIT_STATE_DIR=${SUBMIT_STATE_DIR:-$PROJECT_ROOT/cluster/.descriptor_array_state}
AUTO_SUBMIT_NEXT=${AUTO_SUBMIT_NEXT:-1}
FIRST_BATCH_SIZE=${FIRST_BATCH_SIZE:-1000}
SECOND_BATCH_SIZE=${SECOND_BATCH_SIZE:-241}

THREADS=${SLURM_CPUS_PER_TASK:-16}
ROW_OFFSET=${ROW_OFFSET:-0}
METADATA_ROW=$((SLURM_ARRAY_TASK_ID + ROW_OFFSET))

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$CONFIG_PATH"
require_dir "$VENV_PATH"

dra_activate
# One process per array task doing threaded per-recording work -> threaded BLAS.
dra_pin_threads "$THREADS"

python -m eeg_adhd_epilepsy.analysis.extract_descriptors \
  --bids_root "$BIDS_ROOT" \
  --derivative_root "$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors" \
  --reports_root "$SCRATCH_ROOT/reports" \
  --metadata "$METADATA_PATH" \
  --config "$CONFIG_PATH" \
  --subject_col study_id \
  --metadata_row "$METADATA_ROW" \
  --conditions all

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

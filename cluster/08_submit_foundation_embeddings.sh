#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb
#SBATCH --account=def-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --array=1-1000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
METADATA=${METADATA:-$METADATA_PATH}
DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
SUBMIT_STATE_DIR=${SUBMIT_STATE_DIR:-$PROJECT_ROOT/cluster/.foundation_array_state}
EXTRACTION_SCRIPT=${EXTRACTION_SCRIPT:-$PROJECT_ROOT/cluster/08_submit_foundation_embeddings.sh}
AUTO_SUBMIT_NEXT=${AUTO_SUBMIT_NEXT:-1}
FIRST_BATCH_SIZE=${FIRST_BATCH_SIZE:-1000}
SECOND_BATCH_SIZE=${SECOND_BATCH_SIZE:-218}
OVERWRITE=${OVERWRITE:-0}

ROW_OFFSET=${ROW_OFFSET:-0}
METADATA_ROW=$((SLURM_ARRAY_TASK_ID + ROW_OFFSET))

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_file "$METADATA"
require_dir "$VENV_PATH"
require_file "$EXTRACTION_SCRIPT"

# GPU job (embedding extraction runs on-device); no BLAS pinning. Add a
# HuggingFace cache on node-local scratch on top of the standard caches.
dra_activate
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
mkdir -p "$HF_HOME"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset; REVE (gated) will be skipped." >&2
fi

cmd=(
  python -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings
  --config "$FOUNDATION_CONFIG"
  --bids_root "$BIDS_ROOT"
  --metadata "$METADATA"
  --derivative_root "$DERIVATIVE_ROOT"
  --metadata_row "$METADATA_ROW"
)
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
"${cmd[@]}"

if [[ "$AUTO_SUBMIT_NEXT" == "1" && "$ROW_OFFSET" == "0" ]]; then
  batch_state_dir="$SUBMIT_STATE_DIR/${SLURM_ARRAY_JOB_ID:-manual}"
  mkdir -p "$batch_state_dir"
  touch "$batch_state_dir/${SLURM_ARRAY_TASK_ID}.done"

  done_count=$(find "$batch_state_dir" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
  if [[ "$done_count" -ge "$FIRST_BATCH_SIZE" ]]; then
    if mkdir "$batch_state_dir/submit_second.lock" 2>/dev/null; then
      second_job_id=$(sbatch --parsable \
        --array=1-"$SECOND_BATCH_SIZE" \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,BIDS_ROOT=$BIDS_ROOT,FOUNDATION_CONFIG=$FOUNDATION_CONFIG,METADATA=$METADATA,DERIVATIVE_ROOT=$DERIVATIVE_ROOT,SUBMIT_STATE_DIR=$SUBMIT_STATE_DIR,ROW_OFFSET=$FIRST_BATCH_SIZE,AUTO_SUBMIT_NEXT=0,OVERWRITE=$OVERWRITE" \
        "$EXTRACTION_SCRIPT")
      second_job_id=${second_job_id%%;*}
      echo "Submitted second foundation-extraction array: $second_job_id"
    fi
  fi
fi

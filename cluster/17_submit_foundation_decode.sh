#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_decode
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

# Each task trains one foundation model in all configured modes. Keeping model
# jobs separate bounds runtime and GPU memory; stage 18 writes shared reports.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation.yaml}
read -r -a MODELS <<< "${MODELS:-cbramod labram reve luna biot signaljepa eegpt bendr}"

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$ANALYSIS_CONFIG"

MODEL_COUNT=${#MODELS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$MODEL_COUNT"

if (( TASK_ID < 1 || TASK_ID > MODEL_COUNT )); then
    echo "Array task $TASK_ID is outside valid task range 1-$MODEL_COUNT; nothing to do."
    exit 0
fi
model=${MODELS[$((TASK_ID - 1))]}

# FOUNDATION_N_JOBS is how many fits share the GPU at once: 1 on a 10 or 20 GB
# slice, more on a full H100 (each fit also needs one core and its own memory).
# GPU job (training runs on-device); no BLAS pinning. Add a HuggingFace cache on
# node-local scratch on top of the standard numba/MNE/matplotlib caches.
dra_activate

if [[ "$model" == "reve" && -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset; REVE (gated) will be skipped." >&2
fi

echo "Cohort:   $COHORT_CONFIG"
echo "Analysis: $ANALYSIS_CONFIG"
echo "Model:    $model"
echo "Input:    raw EEG epochs"

python -m eeg_adhd_epilepsy.analysis.foundation_decoding \
    --cohort_config "$COHORT_CONFIG" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --derivative_root "$DECODING_ROOT" \
    --metadata "$METADATA_PATH" \
    --reports_root "$REPORTS_ROOT" \
    --model_key "$model" \
    --n_jobs "${FOUNDATION_N_JOBS:-1}" \
    --no-write-shared-comparison-report

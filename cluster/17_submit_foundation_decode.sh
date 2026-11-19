#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_decode
#SBATCH --time=24:00:00
#SBATCH --array=1-4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# Each task trains one model of the analysis config (MODELS, default all of them, so
# --array=1-<number of models>) in all its modes. Keeping model
# jobs separate bounds runtime and GPU memory; stage 18 writes shared reports.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation.yaml}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$ANALYSIS_CONFIG"
dra_activate

# MODELS (names or model keys) defaults to every model of the analysis config, in order.
# One model alone: MODELS=<name> with --array=1. A submit without --array still runs the
# header's array, and the guard fails when it doesn't match the number of models.
if [[ -z ${MODELS:-} ]]; then
    MODELS=$(python -c 'import sys, yaml
print(" ".join(m.get("name", m["model_key"]) for m in yaml.safe_load(open(sys.argv[1]))["models"]))' "$ANALYSIS_CONFIG")
fi
read -r -a MODELS <<< "$MODELS"

MODEL_COUNT=${#MODELS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$MODEL_COUNT"

if (( TASK_ID < 1 || TASK_ID > MODEL_COUNT )); then
    echo "Array task $TASK_ID is outside valid task range 1-$MODEL_COUNT; nothing to do."
    exit 0
fi
model=${MODELS[$((TASK_ID - 1))]}

# POOLED_ONLY=1 fits EO and EC pooled only (a run of its own next to the per-condition one).
# TARGETS=<eval> decodes that eval in place of the analysis config's targets (a cohort
# whose clinical question has another name, e.g. ldx_vs_mph).
# FOUNDATION_N_JOBS is how many fits share the GPU at once: 1 on a 10 or 20 GB
# slice, more on a full H100 (each fit also needs one core and its own memory).
# GPU job (training runs on-device); no BLAS pinning.

echo "Cohort:   $COHORT_CONFIG"
echo "Analysis: $ANALYSIS_CONFIG"
echo "Model:    $model"
echo "Input:    raw EEG epochs"

python -m eeg_adhd_epilepsy.analysis.foundation_decoding \
    --cohort_config "$COHORT_CONFIG" "${MATCHED_ARGS[@]}" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --desc "$EPOCHS_DESC" \
    --derivative_root "$DECODING_ROOT" \
    --metadata "$METADATA_PATH" \
    --reports_root "$REPORTS_ROOT" \
    --model_key "$model" \
    --n_jobs "${FOUNDATION_N_JOBS:-1}" \
    ${POOLED_ONLY:+--pooled_only} \
    ${TARGETS:+--targets $TARGETS} \
    --no-write-shared-comparison-report

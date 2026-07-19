#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_decode
#SBATCH --account=def-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --array=1-8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

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

# GPU job (training runs on-device); no BLAS pinning. Add a HuggingFace cache on
# node-local scratch on top of the standard numba/MNE/matplotlib caches.
dra_activate
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
mkdir -p "$HF_HOME"

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
    --metadata "$METADATA_PATH" \
    --reports_root "$REPORTS_ROOT" \
    --representation epoch \
    --model_key "$model" \
    --no-write-shared-comparison-report

#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_fm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# array = cohorts x representation spaces x granularities; 72 x 9 x 3.
#SBATCH --array=1-3780
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/foundation.yaml}
# Merged per-model embeddings written by 10_submit_merge_foundation_embeddings.sh.
OVERWRITE=${OVERWRITE:-0}

# Each base model contributes its raw space plus every materialized alignment.
# Pooling variants without their own aligned derivatives are raw-only spaces.
read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod labram reve luna}"
read -r -a ALIGNMENT_TRANSFORMS <<< "${ALIGNMENT_TRANSFORMS:-none leace}"
read -r -a RAW_ONLY_MODELS <<< "${RAW_ONLY_MODELS:-reve_pool-attention}"

# Representations reduced, each as its own run: epoch (1 row per epoch) and recording
# (1 row per recording, so per child).
REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording})

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$CONFIGS_DIR"
require_file "$ANALYSIS_CONFIG"
require_dir "$EMBEDDING_PROJECT_ROOT"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

# Build explicit (base model, transform, saved model key) representation spaces.
SPACE_MODELS=()
SPACE_TRANSFORMS=()
SPACE_KEYS=()
for base_model in "${BASE_MODELS[@]}"; do
    for transform in "${ALIGNMENT_TRANSFORMS[@]}"; do
        SPACE_MODELS+=("$base_model")
        SPACE_TRANSFORMS+=("$transform")
        if [[ "$transform" == "none" ]]; then
            SPACE_KEYS+=("$base_model")
        else
            SPACE_KEYS+=("${base_model}_align-${transform}")
        fi
    done
done
for raw_model in "${RAW_ONLY_MODELS[@]}"; do
    SPACE_MODELS+=("$raw_model")
    SPACE_TRANSFORMS+=("none")
    SPACE_KEYS+=("$raw_model")
done

# One array task = one (scope, cohort, representation space, granularity).
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
CONFIG_COUNT=${#CONFIGS[@]}
SPACE_COUNT=${#SPACE_KEYS[@]}
REP_COUNT=${#REPRESENTATIONS[@]}
TOTAL_TASKS=$((${#SCOPES[@]} * CONFIG_COUNT * SPACE_COUNT * REP_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$TOTAL_TASKS"

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi

# Innermost axis is scope, then cohort, then representation space, then granularity.
task_index=$((TASK_ID - 1))
scope="${SCOPES[$((task_index % ${#SCOPES[@]}))]}"
dra_scope_args "$scope"
task_index=$((task_index / ${#SCOPES[@]}))
config_index=$((task_index % CONFIG_COUNT))
rest=$((task_index / CONFIG_COUNT))
space_index=$((rest % SPACE_COUNT))
rep_index=$((rest / SPACE_COUNT))
model="${SPACE_MODELS[$space_index]}"
transform="${SPACE_TRANSFORMS[$space_index]}"
space_key="${SPACE_KEYS[$space_index]}"
config="${CONFIGS[$config_index]}"
representation="${REPRESENTATIONS[$rep_index]}"

echo "================================================================================"
echo "FOUNDATION DIM REDUCTION ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Config:        $config"
echo "Scope:         $scope"
echo "Base model:    $model"
echo "Transform:     $transform"
echo "Saved key:     $space_key"
echo "Representation:$representation"
echo "Embeddings:    $EMBEDDING_PROJECT_ROOT"
echo "================================================================================"

cmd=(
    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction
    --bids_root "$BIDS_ROOT"
    --desc "$EPOCHS_DESC"
    --derivative_root "$DIM_REDUCTION_ROOT"
    --reports_root "$REPORTS_ROOT"
    --metadata "$METADATA_PATH"
    --cohort_config "$config" "${MATCHED_ARGS[@]}"
    --analysis_config "$ANALYSIS_CONFIG"
    --embedding_derivative_root "$EMBEDDING_PROJECT_ROOT"
    --embedding_model_key "$model"
    --alignment_transform "$transform"
    --representation "$representation" "${SCOPE_ARGS[@]}"
    --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_fm
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# array = cohorts x models x representations (the guard below recomputes and checks this).
#SBATCH --array=1-1998
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/foundation.yaml}
# Merged per-model embeddings written by 09_submit_merge_foundation_embeddings.sh.
EMBEDDING_ROOT=${EMBEDDING_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
OVERWRITE=${OVERWRITE:-0}

# Foundation-model embedding keys to reduce. One dim-reduction run per model keeps
# each model's embedding space separate (the right call for an unsupervised
# manifold). Override MODELS to add/remove keys (e.g. a reve-attention variant).
MODELS=(${MODELS:-cbramod labram reve luna biot signaljepa eegpt bendr})

# Embedding representations (canonical granularity ladder) to reduce, each as a
# SEPARATE run so they can be compared: epoch (1 row/epoch — most points for the
# manifold reducers), recording (1 row/recording), subject (1 row/subject).
# Override to narrow, e.g. REPRESENTATIONS="recording subject".
REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording subject})

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$CONFIGS_DIR"
require_file "$ANALYSIS_CONFIG"
require_dir "$EMBEDDING_ROOT"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

# Map this array task to one (cohort, model, representation) triple.
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
CONFIG_COUNT=${#CONFIGS[@]}
MODEL_COUNT=${#MODELS[@]}
REP_COUNT=${#REPRESENTATIONS[@]}
TOTAL_TASKS=$((CONFIG_COUNT * MODEL_COUNT * REP_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$TOTAL_TASKS"

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi

# Innermost axis is cohort, then model, then representation.
task_index=$((TASK_ID - 1))
config_index=$((task_index % CONFIG_COUNT))
rest=$((task_index / CONFIG_COUNT))
model_index=$((rest % MODEL_COUNT))
rep_index=$((rest / MODEL_COUNT))
model="${MODELS[$model_index]}"
config="${CONFIGS[$config_index]}"
representation="${REPRESENTATIONS[$rep_index]}"

# Epoch runs use all workers like every other representation: the co-ranking
# subsample cap in coco_pipe (DEFAULT_MAX_CORANKING_SAMPLES) bounds the dominant
# per-fit allocation, so the old epoch-specific worker cap is no longer needed.

echo "================================================================================"
echo "FOUNDATION DIM REDUCTION ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Config:        $config"
echo "Model:         $model"
echo "Representation:$representation"
echo "Embeddings:    $EMBEDDING_ROOT"
echo "================================================================================"

cmd=(
    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction
    --bids_root "$BIDS_ROOT"
    --reports_root "$REPORTS_ROOT"
    --metadata "$METADATA_PATH"
    --cohort_config "$config"
    --analysis_config "$ANALYSIS_CONFIG"
    --embedding_derivative_root "$EMBEDDING_ROOT"
    --embedding_model_key "$model"
    --representation "$representation"
    --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"

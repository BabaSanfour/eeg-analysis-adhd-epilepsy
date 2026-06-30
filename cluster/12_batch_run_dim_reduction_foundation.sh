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

# 1. Load Cluster Modules
module purge
module load gcc arrow/23.0.1 python/3.11

# 2. Path Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
REPORTS_ROOT="$SCRATCH_ROOT/reports"
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/foundation.yaml}
# Merged per-model embeddings written by 09_submit_merge_foundation_embeddings.sh.
EMBEDDING_ROOT=${EMBEDDING_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
OVERWRITE=${OVERWRITE:-0}

# Foundation-model embedding keys to reduce. One dim-reduction run per model keeps
# each model's embedding space separate (the right call for an unsupervised
# manifold). Override MODELS to add/remove keys (e.g. a reve-attention variant).
MODELS=(${MODELS:-cbramod labram reve luna biot signaljepa eegpt bendr})

# 3. Environment Setup
cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1
THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -f "$METADATA_PATH" ] || { echo "Metadata CSV not found: $METADATA_PATH"; exit 1; }
[ -d "$CONFIGS_DIR" ] || { echo "Config directory not found: $CONFIGS_DIR"; exit 1; }
[ -f "$ANALYSIS_CONFIG" ] || { echo "Analysis config not found: $ANALYSIS_CONFIG"; exit 1; }
[ -d "$EMBEDDING_ROOT" ] || { echo "Embedding derivative root not found: $EMBEDDING_ROOT"; exit 1; }

# Embedding representations (canonical granularity ladder) to reduce, each as a
# SEPARATE run so they can be compared: epoch (1 row/epoch — most points for the
# manifold reducers), recording (1 row/recording), subject (1 row/subject).
# Override to narrow, e.g. REPRESENTATIONS="recording subject".
REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording subject})

# 4. Map this array task to one (cohort, model, representation) triple
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
CONFIG_COUNT=${#CONFIGS[@]}
MODEL_COUNT=${#MODELS[@]}
REP_COUNT=${#REPRESENTATIONS[@]}
TOTAL_TASKS=$((CONFIG_COUNT * MODEL_COUNT * REP_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Guard: a stale #SBATCH --array bound silently drops the trailing tasks.
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$TOTAL_TASKS" ]; then
    echo "ERROR: array size $SLURM_ARRAY_TASK_COUNT != cohorts($CONFIG_COUNT) x models($MODEL_COUNT) x reps($REP_COUNT) = $TOTAL_TASKS." >&2
    echo "Update '#SBATCH --array=1-$TOTAL_TASKS' (or narrow CONFIGS_DIR / MODELS / REPRESENTATIONS)." >&2
    exit 1
fi

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

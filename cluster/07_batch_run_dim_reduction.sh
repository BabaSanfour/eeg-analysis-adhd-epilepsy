#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_batch
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=1-148
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
# Cohort configs (one per dataset/cohort/strata), each paired with the single
# analysis config below. Point CONFIGS_DIR at a subtree to narrow the sweep.
# NOTE: --array (line 9) must equal CONFIG_COUNT * MODE_COUNT (guarded below).
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/default.yaml}

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

# 4. Map this array task to one cohort/mode pair
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
MODES=("flat:recording_flat" "sensor:recording_native")
CONFIG_COUNT=${#CONFIGS[@]}
MODE_COUNT=${#MODES[@]}
TOTAL_TASKS=$((CONFIG_COUNT * MODE_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Guard: a stale #SBATCH --array bound silently drops the trailing tasks. Fail
# loudly if the submitted array size does not match cohorts x modes.
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$TOTAL_TASKS" ]; then
    echo "ERROR: array size $SLURM_ARRAY_TASK_COUNT != cohorts($CONFIG_COUNT) x modes($MODE_COUNT) = $TOTAL_TASKS." >&2
    echo "Update '#SBATCH --array=1-$TOTAL_TASKS' (or set CONFIGS_DIR to a subtree)." >&2
    exit 1
fi

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi

task_index=$((TASK_ID - 1))
mode_index=$((task_index / CONFIG_COUNT))
config_index=$((task_index % CONFIG_COUNT))
mode_spec="${MODES[$mode_index]}"
mode="${mode_spec%%:*}"
representation="${mode_spec##*:}"
config="${CONFIGS[$config_index]}"
input_mode="raw"
aggregation_unit="${AGGREGATION_UNIT:-recording}"

echo "================================================================================"
echo "DIM REDUCTION ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Config:         $config"
echo "Mode:           $mode"
echo "Representation: $representation"
echo "Report:         resolved by the configuration-hashed run namespace"
echo "================================================================================"

python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
    --bids_root "$BIDS_ROOT" \
    --reports_root "$REPORTS_ROOT" \
    --metadata "$METADATA_PATH" \
    --cohort_config "$config" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --input_mode "$input_mode" \
    --analysis_mode "$mode" \
    --representation "$representation" \
    --aggregation_unit "$aggregation_unit" \
    --n_jobs "$THREADS"

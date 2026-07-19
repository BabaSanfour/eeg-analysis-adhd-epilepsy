#!/usr/bin/env bash
#SBATCH --job-name=eeg_classical_decode
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# descriptor baseline + 8 foundation models x 3 representations.
#SBATCH --array=1-25
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# Task 1 decodes the descriptor baseline. Remaining tasks decode one saved
# foundation model and representation, including every transform declared by
# foundation_embeddings.yaml. A separate dependency job writes shared reports.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
CLASSICAL_ANALYSIS_CONFIG=${CLASSICAL_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/classical.yaml}
FOUNDATION_ANALYSIS_CONFIG=${FOUNDATION_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation_embeddings.yaml}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}

read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod labram reve luna biot signaljepa eegpt bendr}"
REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording subject})

# Descriptor table (dataset path -> supplied here, not in the analysis config).
# Recording-level table; override to sensor_subject_features for subject-pooled.
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH=${TABLE_PATH:-$DESC_ROOT/sensor_recording_features.parquet}
COLUMNS_PATH=${COLUMNS_PATH:-$DESC_ROOT/sensor_recording_features_feature_columns.json}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"

MODEL_COUNT=${#BASE_MODELS[@]}
REPRESENTATION_COUNT=${#REPRESENTATIONS[@]}
TOTAL_TASKS=$((1 + MODEL_COUNT * REPRESENTATION_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$TOTAL_TASKS"

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

common_args=(
    --cohort_config "$COHORT_CONFIG"
    --bids_root "$BIDS_ROOT"
    --metadata "$METADATA_PATH"
    --n_jobs "$THREADS"
    --no-write-shared-comparison-report
)

echo "================================================================================"
echo "CLASSICAL DECODING ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Cohort: $COHORT_CONFIG"

if (( TASK_ID == 1 )); then
    require_file "$CLASSICAL_ANALYSIS_CONFIG"
    require_file "$TABLE_PATH"
    require_file "$COLUMNS_PATH"
    echo "Input:    descriptors"
    echo "Analysis: $CLASSICAL_ANALYSIS_CONFIG"
    echo "Table:    $TABLE_PATH"

    python -m eeg_adhd_epilepsy.analysis.classical_decoding \
        "${common_args[@]}" \
        --analysis_config "$CLASSICAL_ANALYSIS_CONFIG" \
        --descriptor_table_path "$TABLE_PATH" \
        --descriptor_feature_columns_path "$COLUMNS_PATH"
    exit 0
fi

require_file "$FOUNDATION_ANALYSIS_CONFIG"
require_dir "$EMBEDDING_ROOT"

foundation_index=$((TASK_ID - 2))
model_index=$((foundation_index % MODEL_COUNT))
representation_index=$((foundation_index / MODEL_COUNT))
model=${BASE_MODELS[$model_index]}
representation=${REPRESENTATIONS[$representation_index]}

echo "Input:          foundation_embeddings"
echo "Analysis:       $FOUNDATION_ANALYSIS_CONFIG"
echo "Model:          $model"
echo "Representation: $representation"
echo "Transforms:     none, fold-local leace, ea_coral, ea_mean, ra"
echo "Embedding root: $EMBEDDING_ROOT"

python -m eeg_adhd_epilepsy.analysis.classical_decoding \
    "${common_args[@]}" \
    --analysis_config "$FOUNDATION_ANALYSIS_CONFIG" \
    --embedding_derivative_root "$EMBEDDING_ROOT" \
    --embedding_model_key "$model" \
    --representation "$representation"

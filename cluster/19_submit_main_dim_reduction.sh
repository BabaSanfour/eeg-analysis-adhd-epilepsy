#!/usr/bin/env bash
#SBATCH --job-name=eeg_main_dimred
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# One-cohort integration runner for stages 11–14. It covers all representation
# granularities and alignment paths with representative models; override the
# model lists to expand backend coverage without changing the test logic.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: You must specify what to run."
    echo "Usage: sbatch $0 [raw|descriptors|foundation]"
    exit 1
fi

PIPELINE_TYPE=$1

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# The exact cohort config
CONFIG="$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml"
DATASET_NAME=${DATASET_NAME:-pooled_01_all_subjects_total}

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-8}


if [ "$PIPELINE_TYPE" == "raw" ]; then
    echo "================================================================="
    echo " 1. RAW Dimensionality Reduction"
    echo "================================================================="
    for rep in epoch recording subject; do
        echo " -> Representation: $rep"
        python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
            --bids_root "$BIDS_ROOT" \
            --reports_root "$REPORTS_ROOT" \
            --metadata "$METADATA_PATH" \
            --cohort_config "$CONFIG" \
            --analysis_config "$PROJECT_ROOT/configs/analyses/dim_reduction/raw.yaml" \
            --representation "$rep" \
            --n_jobs "$THREADS"
    done

elif [ "$PIPELINE_TYPE" == "descriptors" ]; then
    echo "================================================================="
    echo " 2. DESCRIPTORS Dimensionality Reduction"
    echo "================================================================="
    DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"

    for rep in epoch recording subject; do
        echo " -> Representation: $rep"
        require_file "$DESC_ROOT/sensor_${rep}_features.parquet"
        require_file "$DESC_ROOT/sensor_${rep}_features_feature_columns.json"
        python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
            --bids_root "$BIDS_ROOT" \
            --reports_root "$REPORTS_ROOT" \
            --metadata "$METADATA_PATH" \
            --cohort_config "$CONFIG" \
            --analysis_config "$PROJECT_ROOT/configs/analyses/dim_reduction/descriptors.yaml" \
            --descriptor_table_path "$DESC_ROOT/sensor_${rep}_features.parquet" \
            --descriptor_feature_columns_path "$DESC_ROOT/sensor_${rep}_features_feature_columns.json" \
            --representation "$rep" \
            --n_jobs "$THREADS"
    done

elif [ "$PIPELINE_TYPE" == "foundation" ]; then
    echo "================================================================="
    echo " 3. FOUNDATION Dimensionality Reduction"
    echo "================================================================="
    FOUND_ROOT="$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings"
    DIM_ROOT="$BIDS_ROOT/derivatives/dim_reduction"
    read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod}"
    read -r -a ALIGNMENT_TRANSFORMS <<< "${ALIGNMENT_TRANSFORMS:-none leace ea_coral ea_mean ra}"
    read -r -a RAW_ONLY_MODELS <<< "${RAW_ONLY_MODELS:-reve_pool-attention}"
    REPS=(${REPRESENTATIONS:-epoch recording subject})
    require_dir "$FOUND_ROOT"

    for model in "${BASE_MODELS[@]}"; do
        for transform in "${ALIGNMENT_TRANSFORMS[@]}"; do
            for rep in "${REPS[@]}"; do
                echo " -> Model: $model | Transform: $transform | Representation: $rep"
                python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
                    --bids_root "$BIDS_ROOT" \
                    --reports_root "$REPORTS_ROOT" \
                    --metadata "$METADATA_PATH" \
                    --cohort_config "$CONFIG" \
                    --analysis_config "$PROJECT_ROOT/configs/analyses/dim_reduction/foundation.yaml" \
                    --embedding_derivative_root "$FOUND_ROOT" \
                    --embedding_model_key "$model" \
                    --alignment_transform "$transform" \
                    --representation "$rep" \
                    --n_jobs "$THREADS"
            done
        done
    done

    for model in "${RAW_ONLY_MODELS[@]}"; do
        for rep in "${REPS[@]}"; do
            echo " -> Raw-only model: $model | Representation: $rep"
            python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
                --bids_root "$BIDS_ROOT" \
                --reports_root "$REPORTS_ROOT" \
                --metadata "$METADATA_PATH" \
                --cohort_config "$CONFIG" \
                --analysis_config "$PROJECT_ROOT/configs/analyses/dim_reduction/foundation.yaml" \
                --embedding_derivative_root "$FOUND_ROOT" \
                --embedding_model_key "$model" \
                --representation "$rep" \
                --n_jobs "$THREADS"
        done
    done

    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
        --compare_only \
        --bids_root "$BIDS_ROOT" \
        --derivative_root "$DIM_ROOT" \
        --reports_root "$REPORTS_ROOT" \
        --dataset_name "$DATASET_NAME"

else
    echo "ERROR: Invalid pipeline type '$PIPELINE_TYPE'."
    echo "Usage: sbatch $0 [raw|descriptors|foundation]"
    exit 1
fi

echo "Dim reduction run complete!"

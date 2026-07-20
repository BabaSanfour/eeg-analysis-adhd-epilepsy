#!/usr/bin/env bash
#SBATCH --job-name=eeg_main_decode
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# One-cohort integration runner for stages 15–18. The classical branch covers
# descriptors, saved aligned embeddings, and their comparison; the foundation
# branch covers direct epoch-level training and its final comparison.
# Resources are intentionally supplied on the sbatch command line so CPU-only
# classical runs and GPU foundation runs can use different allocations.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: You must specify what to run."
    echo "Usage: sbatch $0 [descriptors|embeddings|foundation|all]"
    exit 1
fi

PIPELINE_TYPE=$1

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# The exact cohort config
CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
CLASSICAL_ANALYSIS_CONFIG=${CLASSICAL_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/classical.yaml}
SAVED_FOUNDATION_ANALYSIS_CONFIG=${SAVED_FOUNDATION_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation_embeddings.yaml}
FOUNDATION_ANALYSIS_CONFIG=${FOUNDATION_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation.yaml}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
OVERWRITE=${OVERWRITE:-0}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$CONFIG"

dra_activate
THREADS=${SLURM_CPUS_PER_TASK:-1}

if [ "$PIPELINE_TYPE" = "all" ]; then
    echo "WARN: 'all' runs descriptors, embeddings, and foundation inside one allocation." >&2
    echo "      Submit the three explicit modes separately to avoid resource waste." >&2
fi

run_descriptor_classical() {
    echo "================================================================="
    echo " 1. CLASSICAL Descriptor Decoding"
    echo "================================================================="
    require_file "$CLASSICAL_ANALYSIS_CONFIG"

    dra_pin_threads 1

    DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
    CLASSICAL_REPRESENTATIONS=(${CLASSICAL_REPRESENTATIONS:-epoch recording subject})
    table_override="${TABLE_PATH:-}"
    columns_override="${COLUMNS_PATH:-}"

    if [ -n "$table_override" ] && [ "${#CLASSICAL_REPRESENTATIONS[@]}" -gt 1 ]; then
        echo "ERROR: TABLE_PATH can only be used when CLASSICAL_REPRESENTATIONS has one value." >&2
        exit 1
    fi
    if [ -n "$columns_override" ] && [ "${#CLASSICAL_REPRESENTATIONS[@]}" -gt 1 ]; then
        echo "ERROR: COLUMNS_PATH can only be used when CLASSICAL_REPRESENTATIONS has one value." >&2
        exit 1
    fi

    for rep in "${CLASSICAL_REPRESENTATIONS[@]}"; do
        case "$rep" in
            epoch|recording|subject)
                stem="sensor_${rep}_features"
                ;;
            *)
                echo "ERROR: Unsupported classical representation '$rep'." >&2
                echo "Use one of: epoch recording subject" >&2
                exit 1
                ;;
        esac

        table_path="${table_override:-$DESC_ROOT/${stem}.parquet}"
        columns_path="${columns_override:-$DESC_ROOT/${stem}_feature_columns.json}"

        require_file "$table_path"
        require_file "$columns_path"

        echo " -> Representation: $rep"
        echo "    Table: $table_path"

        cmd=(
            python -m eeg_adhd_epilepsy.analysis.classical_decoding
            --cohort_config "$CONFIG"
            --analysis_config "$CLASSICAL_ANALYSIS_CONFIG"
            --bids_root "$BIDS_ROOT"
            --derivative_root "$DECODING_ROOT"
            --reports_root "$REPORTS_ROOT"
            --metadata "$METADATA_PATH"
            --descriptor_table_path "$table_path"
            --descriptor_feature_columns_path "$columns_path"
            --representation "$rep"
            --n_jobs "$THREADS"
            --no-write-shared-comparison-report
        )
        if [ "$OVERWRITE" = "1" ]; then
            cmd+=(--overwrite)
        fi
        "${cmd[@]}"
    done
}

run_saved_embedding_classical() {
    echo "================================================================="
    echo " 2. SAVED FOUNDATION EMBEDDING DECODING"
    echo "================================================================="
    require_file "$SAVED_FOUNDATION_ANALYSIS_CONFIG"
    require_dir "$EMBEDDING_ROOT"

    dra_pin_threads 1

    read -r -a SAVED_MODELS <<< "${SAVED_MODELS:-cbramod}"
    SAVED_REPRESENTATIONS=(${SAVED_REPRESENTATIONS:-epoch recording subject})

    for model in "${SAVED_MODELS[@]}"; do
        for rep in "${SAVED_REPRESENTATIONS[@]}"; do
            echo " -> Model: $model | Representation: $rep"
            echo "    Transforms: none, fold-local leace, ea_coral, ea_mean, ra"
            cmd=(
                python -m eeg_adhd_epilepsy.analysis.classical_decoding
                --cohort_config "$CONFIG"
                --analysis_config "$SAVED_FOUNDATION_ANALYSIS_CONFIG"
                --bids_root "$BIDS_ROOT"
                --derivative_root "$DECODING_ROOT"
                --reports_root "$REPORTS_ROOT"
                --metadata "$METADATA_PATH"
                --embedding_derivative_root "$EMBEDDING_ROOT"
                --embedding_model_key "$model"
                --representation "$rep"
                --n_jobs "$THREADS"
                --no-write-shared-comparison-report
            )
            if [ "$OVERWRITE" = "1" ]; then
                cmd+=(--overwrite)
            fi
            "${cmd[@]}"
        done
    done

    python -m eeg_adhd_epilepsy.analysis.classical_decoding \
        --compare_only \
        --cohort_config "$CONFIG" \
        --analysis_config "$SAVED_FOUNDATION_ANALYSIS_CONFIG" \
        --bids_root "$BIDS_ROOT" \
        --derivative_root "$DECODING_ROOT" \
        --reports_root "$REPORTS_ROOT" \
        --metadata "$METADATA_PATH"
}

run_foundation() {
    echo "================================================================="
    echo " 2. FOUNDATION Decoding"
    echo "================================================================="
    require_file "$FOUNDATION_ANALYSIS_CONFIG"

    export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
    mkdir -p "$HF_HOME"

    read -r -a DIRECT_MODELS <<< "${DIRECT_MODELS:-cbramod}"
    if [[ " ${DIRECT_MODELS[*]} " == *" reve "* && -z "${HF_TOKEN:-}" ]]; then
        echo "WARN: HF_TOKEN is unset; REVE (gated) will be skipped." >&2
    fi

    for model in "${DIRECT_MODELS[@]}"; do
        echo " -> Model: $model | Representation: epoch"
        # For foundation runs --n_jobs is the OUTER sweep concurrency: how many
        # decoding units (each a full CBraMod/REVE backbone) load at once. On a
        # single GPU that must stay at 1, otherwise N backbones + N CUDA contexts
        # co-reside and OOM the host / 20GB MIG slice. It is deliberately NOT tied
        # to $THREADS (CPU count) -- the allocated CPUs still serve torch/dataloader
        # threads inside the one active fit. Override via FOUNDATION_N_JOBS only if
        # the GPU can hold that many concurrent models.
        cmd=(
            python -m eeg_adhd_epilepsy.analysis.foundation_decoding
            --cohort_config "$CONFIG"
            --analysis_config "$FOUNDATION_ANALYSIS_CONFIG"
            --bids_root "$BIDS_ROOT"
            --derivative_root "$DECODING_ROOT"
            --reports_root "$REPORTS_ROOT"
            --metadata "$METADATA_PATH"
            --representation epoch
            --model_key "$model"
            --n_jobs "${FOUNDATION_N_JOBS:-1}"
            --no-write-shared-comparison-report
        )
        if [ "$OVERWRITE" = "1" ]; then
            cmd+=(--overwrite)
        fi
        "${cmd[@]}"
    done


    python -m eeg_adhd_epilepsy.analysis.foundation_decoding \
        --compare_only \
        --cohort_config "$CONFIG" \
        --analysis_config "$FOUNDATION_ANALYSIS_CONFIG" \
        --bids_root "$BIDS_ROOT" \
        --derivative_root "$DECODING_ROOT" \
        --reports_root "$REPORTS_ROOT" \
        --metadata "$METADATA_PATH"
}

case "$PIPELINE_TYPE" in
    descriptors)
        run_descriptor_classical
        ;;
    embeddings)
        run_saved_embedding_classical
        ;;
    foundation)
        run_foundation
        ;;
    all)
        run_descriptor_classical
        run_saved_embedding_classical
        run_foundation
        ;;
    *)
        echo "ERROR: Invalid pipeline type '$PIPELINE_TYPE'."
        echo "Usage: sbatch $0 [descriptors|embeddings|foundation|all]"
        exit 1
        ;;
esac

echo "Decoding run complete!"

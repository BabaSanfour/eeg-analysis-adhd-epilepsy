#!/usr/bin/env bash
#SBATCH --job-name=eeg_main_decode
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Debug convenience runner: run decoding for ONE hardcoded cohort serially.
# Resources are intentionally supplied on the sbatch command line so CPU-only
# classical runs and GPU foundation runs can use different allocations.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: You must specify what to run."
    echo "Usage: sbatch $0 [decoding|classical|foundation|all]"
    exit 1
fi

PIPELINE_TYPE=$1

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# The exact cohort config
CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
CLASSICAL_ANALYSIS_CONFIG=${CLASSICAL_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/classical.yaml}
FOUNDATION_ANALYSIS_CONFIG=${FOUNDATION_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation.yaml}
OVERWRITE=${OVERWRITE:-0}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$CONFIG"

dra_activate
THREADS=${SLURM_CPUS_PER_TASK:-1}

if [ "$PIPELINE_TYPE" = "all" ]; then
    echo "WARN: 'all' runs classical and foundation inside one allocation." >&2
    echo "      Submit 'decoding' and 'foundation' separately to avoid resource waste." >&2
fi

run_classical() {
    echo "================================================================="
    echo " 1. CLASSICAL Descriptor Decoding"
    echo "================================================================="
    require_file "$CLASSICAL_ANALYSIS_CONFIG"

    dra_pin_threads 1

    DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
    # Main experiment: epoch + recording. Override to narrow/expand if needed.
    CLASSICAL_REPRESENTATIONS=(${CLASSICAL_REPRESENTATIONS:-epoch recording})
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
            --reports_root "$REPORTS_ROOT"
            --metadata "$METADATA_PATH"
            --descriptor_table_path "$table_path"
            --descriptor_feature_columns_path "$columns_path"
            --representation "$rep"
            --n_jobs "$THREADS"
        )
        if [ "$OVERWRITE" = "1" ]; then
            cmd+=(--overwrite)
        fi
        "${cmd[@]}"
    done
}

run_foundation() {
    echo "================================================================="
    echo " 2. FOUNDATION Decoding"
    echo "================================================================="
    require_file "$FOUNDATION_ANALYSIS_CONFIG"

    export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
    mkdir -p "$HF_HOME"

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "WARN: HF_TOKEN is unset; REVE (gated) will be skipped." >&2
    fi

    # Main experiment: epoch + recording. Override to narrow/expand if needed.
    FOUNDATION_REPRESENTATIONS=(${FOUNDATION_REPRESENTATIONS:-epoch recording})

    for rep in "${FOUNDATION_REPRESENTATIONS[@]}"; do
        case "$rep" in
            epoch|recording|subject)
                ;;
            *)
                echo "ERROR: Unsupported foundation representation '$rep'." >&2
                echo "Use one of: epoch recording subject" >&2
                exit 1
                ;;
        esac

        echo " -> Representation: $rep"

        cmd=(
            python -m eeg_adhd_epilepsy.analysis.foundation_decoding
            --cohort_config "$CONFIG"
            --analysis_config "$FOUNDATION_ANALYSIS_CONFIG"
            --bids_root "$BIDS_ROOT"
            --reports_root "$REPORTS_ROOT"
            --metadata "$METADATA_PATH"
            --representation "$rep"
            --n_jobs "$THREADS"
        )
        if [ "$OVERWRITE" = "1" ]; then
            cmd+=(--overwrite)
        fi
        "${cmd[@]}"
    done
}

case "$PIPELINE_TYPE" in
    decoding|classical)
        run_classical
        ;;
    foundation)
        run_foundation
        ;;
    all)
        run_classical
        run_foundation
        ;;
    *)
        echo "ERROR: Invalid pipeline type '$PIPELINE_TYPE'."
        echo "Usage: sbatch $0 [decoding|classical|foundation|all]"
        exit 1
        ;;
esac

echo "Decoding run complete!"

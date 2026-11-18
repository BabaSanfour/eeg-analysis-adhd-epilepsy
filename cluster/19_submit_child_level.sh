#!/usr/bin/env bash
#SBATCH --job-name=eeg_child_level
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# Child-level decoding: one example per child from all of its windows (mean pooling,
# attention pooling, descriptors, early and late fusion), from the saved frozen embeddings
# and the recording descriptor table. Writes scores.csv, summary.json and the attention
# weights under $DECODING_ROOT/<cohort>[_matched]/child_level/.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod labram reve luna}"
TABLE="$DESCRIPTORS_ROOT/combined/sensor_recording_features.parquet"

require_dir "$EMBEDDING_PROJECT_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$TABLE"

dra_activate
THREADS=${SLURM_CPUS_PER_TASK:-16}
dra_pin_threads "$THREADS"

name=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset_name"])' "$COHORT_CONFIG")
suffix=""; [[ ${#MATCHED_ARGS[@]} -gt 0 ]] && suffix=_matched
out="$DECODING_ROOT/${name}${suffix}/child_level"
echo "Cohort: $COHORT_CONFIG  Output: $out"

python -m eeg_adhd_epilepsy.analysis.child_level \
    --cohort_config "$COHORT_CONFIG" "${MATCHED_ARGS[@]}" \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --desc "$EPOCHS_DESC" \
    --models "${BASE_MODELS[@]}" \
    --embedding_derivative_root "$EMBEDDING_PROJECT_ROOT" \
    --descriptor_table_path "$TABLE" \
    --n_threads "$THREADS" \
    --output_dir "$out"

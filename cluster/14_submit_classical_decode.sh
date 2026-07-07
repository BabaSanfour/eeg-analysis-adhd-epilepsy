#!/usr/bin/env bash
#SBATCH --job-name=eeg_classical_decode
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# One classical-decoding run = one cohort config x one analysis config. Submit
# several jobs (overriding COHORT_CONFIG/ANALYSIS_CONFIG) to cover a grid.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/classical.yaml}

# Descriptor table (dataset path -> supplied here, not in the analysis config).
# Recording-level table; override to sensor_subject_features for subject-pooled.
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH=${TABLE_PATH:-$DESC_ROOT/sensor_recording_features.parquet}
COLUMNS_PATH=${COLUMNS_PATH:-$DESC_ROOT/sensor_recording_features_feature_columns.json}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$ANALYSIS_CONFIG"
require_file "$TABLE_PATH"
require_file "$COLUMNS_PATH"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "Cohort:   $COHORT_CONFIG"
echo "Analysis: $ANALYSIS_CONFIG"
echo "Table:    $TABLE_PATH"

python -m eeg_adhd_epilepsy.analysis.classical_decoding \
    --cohort_config "$COHORT_CONFIG" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --descriptor_table_path "$TABLE_PATH" \
    --descriptor_feature_columns_path "$COLUMNS_PATH" \
    --n_jobs "$THREADS"

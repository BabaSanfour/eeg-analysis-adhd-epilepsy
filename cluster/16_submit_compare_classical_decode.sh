#!/usr/bin/env bash
#SBATCH --job-name=eeg_classical_compare
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Run once after the complete stage-15 array. It reads every completed descriptor
# and saved-foundation decoding result for the cohort, then writes the shared
# head-to-head and foundation-transform comparison reports.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
FOUNDATION_ANALYSIS_CONFIG=${FOUNDATION_ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation_embeddings.yaml}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$FOUNDATION_ANALYSIS_CONFIG"

dra_activate

echo "Cohort:   $COHORT_CONFIG"
echo "Analysis: $FOUNDATION_ANALYSIS_CONFIG"

python -m eeg_adhd_epilepsy.analysis.classical_decoding \
    --compare_only \
    --cohort_config "$COHORT_CONFIG" \
    --analysis_config "$FOUNDATION_ANALYSIS_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --derivative_root "$DECODING_ROOT" \
    --metadata "$METADATA_PATH" \
    --reports_root "$REPORTS_ROOT"

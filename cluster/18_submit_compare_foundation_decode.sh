#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_compare
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Run once after the complete stage-17 array. It writes the all-mode foundation
# comparison, then adds direct linear probes to the broader decoding comparison.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/decoding/foundation.yaml}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$ANALYSIS_CONFIG"

dra_activate

echo "Cohort:   $COHORT_CONFIG"
echo "Analysis: $ANALYSIS_CONFIG"

python -m eeg_adhd_epilepsy.analysis.foundation_decoding \
    --compare_only \
    --cohort_config "$COHORT_CONFIG" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --derivative_root "$DECODING_ROOT" \
    --metadata "$METADATA_PATH" \
    --reports_root "$REPORTS_ROOT"

#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# The per-subject shards are on scratch; the merged tables go to the project.
require_dir "$DESCRIPTOR_SHARDS_ROOT"

dra_activate
# One CPU: on full-size data (1,916 shards) the merge took 137 s and 3.4 GB on 1 CPU,
# 203 s on 16 threads.
dra_pin_threads 1

python -m eeg_adhd_epilepsy.analysis.merge_descriptors \
  --derivative_root "$DESCRIPTOR_SHARDS_ROOT" \
  --output_root "$DESCRIPTORS_ROOT" \
  --reports_root "$REPORTS_ROOT"
validate_bids "$DESCRIPTORS_ROOT"

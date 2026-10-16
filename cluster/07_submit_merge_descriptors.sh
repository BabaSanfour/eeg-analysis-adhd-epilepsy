#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# The per-subject shards are on scratch; the merged tables go to the project.
require_dir "$DESCRIPTOR_SHARDS_ROOT"

dra_activate
# Single-process merge that wants threaded BLAS.
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

python -m eeg_adhd_epilepsy.analysis.merge_descriptors \
  --derivative_root "$DESCRIPTOR_SHARDS_ROOT" \
  --output_root "$DESCRIPTORS_ROOT" \
  --reports_root "$REPORTS_ROOT" \
  --skip_inconsistent
validate_bids "$DESCRIPTORS_ROOT"

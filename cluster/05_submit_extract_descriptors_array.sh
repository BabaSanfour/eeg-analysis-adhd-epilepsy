#!/usr/bin/env bash
#SBATCH --job-name=eeg_desc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-1218
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

CONFIG_PATH=${CONFIG_PATH:-$PROJECT_ROOT/configs/descriptors.yaml}
THREADS=${SLURM_CPUS_PER_TASK:-16}
# One task per participants.tsv row (1,218), all in one array: fir takes indices up to
# 9,999. Stage 06 lists every row left without a shard.
METADATA_ROW=$SLURM_ARRAY_TASK_ID

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$CONFIG_PATH"
require_dir "$VENV_PATH"

dra_activate
# One process per array task doing threaded per-recording work -> threaded BLAS.
dra_pin_threads "$THREADS"

python -m eeg_adhd_epilepsy.analysis.extract_descriptors \
  --bids_root "$BIDS_ROOT" \
  --desc "$EPOCHS_DESC" \
  --derivative_root "$DESCRIPTOR_SHARDS_ROOT" \
  --reports_root "$REPORTS_ROOT" \
  --metadata "$METADATA_PATH" \
  --config "$CONFIG_PATH" \
  --metadata_row "$METADATA_ROW" \
  --conditions all

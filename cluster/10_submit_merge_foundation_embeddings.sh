#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge_embeddings
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# All-recordings embedding files of one model on the project: for both epoch variants
# and every weight set, the per-recording files on scratch (stage 08) become one file
# per label and view, each label checked against its manifest first; then the dataset
# files, once, describing every file there (stage 09's included). One file is copied
# at a time, so memory stays small. Run after stage 09; 10b then archives the
# per-recording files.
#   cluv submit fir cluster/10_submit_merge_foundation_embeddings.sh -- MODEL=reve

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
: "${MODEL:?Set MODEL=<model key> (after --).}"

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_dir "$EMBEDDING_SCRATCH_ROOT/coco-pipe-$MODEL"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

python -m eeg_adhd_epilepsy.analysis.merge_foundation_embeddings \
    --config "$FOUNDATION_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --source_root "$EMBEDDING_SCRATCH_ROOT" \
    --derivatives_root "$EMBEDDING_PROJECT_ROOT" \
    --model "$MODEL"
validate_bids "$EMBEDDING_PROJECT_ROOT/coco-pipe-$MODEL"

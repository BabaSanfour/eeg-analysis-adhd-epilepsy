#!/usr/bin/env bash
#SBATCH --job-name=eeg_align_subject
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-10
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# LEACE-erased window vectors of one model and epoch variant, one array task per weight
# set: task 0 the pretrained weights, task k the random seed k - 1 (the model's
# random_seeds, 0-9). Reads the label's all-recordings window file on the project, so
# run it after 10 (10b may have archived the per-recording files already), and writes
# the Leace file next to it. Memory and time not measured since the input became one
# file: a label's window vectors are at most ~0.4 GB (REVE, 87k x 512 in float64).
#   cluv submit fir cluster/09_submit_align_subject_embeddings.sh -- EPOCHS_DESC=base MODEL=cbramod

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
: "${MODEL:?Set MODEL=<model key> (after --).}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
weights=pretrained
if (( TASK_ID > 0 )); then
    weights=$((TASK_ID - 1))
fi

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_dir "$EMBEDDING_PROJECT_ROOT/coco-pipe-$MODEL"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

python -m eeg_adhd_epilepsy.analysis.align_subject_embeddings \
    --config "$FOUNDATION_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --desc "$EPOCHS_DESC" \
    --derivatives_root "$EMBEDDING_PROJECT_ROOT" \
    --model "$MODEL" \
    --weights "$weights"

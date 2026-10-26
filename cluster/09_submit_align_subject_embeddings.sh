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
# random_seeds, 0-9). Copies the label's per-recording window files from scratch to the
# node's $SLURM_TMPDIR and reads them there: on Lustre, opening 1,122 small HDF5 files
# costs minutes, the copy seconds. Writes one file holding every recording to the
# project. Memory and time not measured since EA and RA were dropped: a label's window
# vectors are at most ~0.4 GB (REVE, 87k x 512 in float64).
#   cluv submit fir cluster/09_submit_align_subject_embeddings.sh -- EPOCHS_DESC=base MODEL=cbramod

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
: "${MODEL:?Set MODEL=<model key> (after --).}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
weights=pretrained
label=$EPOCHS_DESC
if (( TASK_ID > 0 )); then
    weights=$((TASK_ID - 1))
    label=${EPOCHS_DESC}Random$weights
fi

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_dir "$EMBEDDING_SCRATCH_ROOT/coco-pipe-$MODEL"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

# The label's window files, and the dataset's own files, in the same layout.
source_dataset=$EMBEDDING_SCRATCH_ROOT/coco-pipe-$MODEL
staged_root=${SLURM_TMPDIR:?}/derivatives
mkdir -p "$staged_root/coco-pipe-$MODEL"
cp "$source_dataset"/*.json "$source_dataset"/*.tsv "$staged_root/coco-pipe-$MODEL/"
(cd "$source_dataset" && find sub-* -name "*_desc-${label}_windowembeddings.nc") \
    | rsync -a --files-from=- "$source_dataset/" "$staged_root/coco-pipe-$MODEL/"
echo "Staged $(find "$staged_root" -name '*.nc' | wc -l) files of $label to $staged_root"

python -m eeg_adhd_epilepsy.analysis.align_subject_embeddings \
    --config "$FOUNDATION_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --desc "$EPOCHS_DESC" \
    --source_root "$staged_root" \
    --derivatives_root "$EMBEDDING_PROJECT_ROOT" \
    --model "$MODEL" \
    --weights "$weights"

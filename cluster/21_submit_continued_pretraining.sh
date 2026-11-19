#!/usr/bin/env bash
#SBATCH --job-name=eeg_continued_pretraining
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# One outer fold per array task: MODEL (cbramod, luna or reve) goes on with its
# self-supervised pretraining on every saved window except that fold's test children,
# then its original and continued frozen embeddings are decoded at the child level.
# When all folds are done:
#   python -m eeg_adhd_epilepsy.analysis.continued_pretraining --combine --output_dir <out>
#   cluv submit trillium-gpu cluster/21_submit_continued_pretraining.sh -- MODEL=cbramod EPOCHS_DESC=base
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
: "${MODEL:?Set MODEL=cbramod, luna or reve (after --).}"
N_FOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$N_FOLDS"

require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
dra_activate
dra_pin_threads "${SLURM_CPUS_ON_NODE:-8}"

suffix=""; [[ ${#MATCHED_ARGS[@]} -gt 0 ]] && suffix=_matched
out="$RESULTS_ROOT/continued_pretraining_$EPOCHS_DESC/$(basename "$(dirname "$COHORT_CONFIG")")_$(basename "$COHORT_CONFIG" .yaml)$suffix"
echo "Model: $MODEL  Fold: $((TASK_ID - 1))  Output: $out"

python -m eeg_adhd_epilepsy.analysis.continued_pretraining \
    --cohort_config "$COHORT_CONFIG" "${MATCHED_ARGS[@]}" \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --desc "$EPOCHS_DESC" \
    --model "$MODEL" \
    --fold $((TASK_ID - 1)) \
    --n_folds "$N_FOLDS" \
    --epochs "${PRETRAIN_EPOCHS:-10}" \
    --output_dir "$out"

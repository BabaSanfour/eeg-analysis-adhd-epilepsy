#!/usr/bin/env bash
#SBATCH --job-name=eeg_align_subject
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=1-8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Materialize LEACE, EA-CORAL, EA-Mean, and token-based RA derivatives for one
# foundation model per array task.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/align_subject_embeddings.yaml}
SOURCE_EMBEDDING_ROOT=${SOURCE_EMBEDDING_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings}
OVERWRITE=${OVERWRITE:-0}
read -r -a MODELS <<< "${MODELS:-cbramod labram reve luna biot signaljepa eegpt bendr}"

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"
require_file "$ANALYSIS_CONFIG"
require_dir "$SOURCE_EMBEDDING_ROOT"
require_dir "$VENV_PATH"

MODEL_COUNT=${#MODELS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$MODEL_COUNT"
if (( TASK_ID < 1 || TASK_ID > MODEL_COUNT )); then
  echo "Array task $TASK_ID is outside valid task range 1-$MODEL_COUNT; nothing to do."
  exit 0
fi
model=${MODELS[$((TASK_ID - 1))]}

dra_activate
dra_pin_threads "${SLURM_CPUS_PER_TASK:-1}"

echo "================================================================================"
echo "SUBJECT ALIGNMENT ARRAY TASK $TASK_ID / $MODEL_COUNT"
echo "Model:       $model"
echo "Cohort:      $COHORT_CONFIG"
echo "Source root: $SOURCE_EMBEDDING_ROOT"
echo "================================================================================"

cmd=(
  python -m eeg_adhd_epilepsy.analysis.align_subject_embeddings
  --cohort_config "$COHORT_CONFIG"
  --analysis_config "$ANALYSIS_CONFIG"
  --bids_root "$BIDS_ROOT"
  --metadata "$METADATA_PATH"
  --source_embedding_root "$SOURCE_EMBEDDING_ROOT"
  --embedding_model_key "$model"
)
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
"${cmd[@]}"

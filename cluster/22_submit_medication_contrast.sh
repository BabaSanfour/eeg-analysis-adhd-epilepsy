#!/usr/bin/env bash
#SBATCH --job-name=eeg_medication
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# Medication contrasts from the saved frozen embeddings: unmedicated vs medicated ADHD,
# each against controls at equal size and covariate make-up, the normalisation axis, and
# methylphenidate vs lisdexamfetamine, and the transfer tests. One array task per family /
# stratum config of pooled ADHD vs controls (or COHORT_CONFIG alone) x matching scheme (the
# covariates the draws are matched on; MATCH alone overrides). Writes summary.json and
# draws.csv under $DECODING_ROOT/medication_contrasts/<cohort>/match-<scheme>/.
mapfile -t CONFIGS < <(find "$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled" -name "*.yaml" | sort)
# Outside an array (and without COHORT_CONFIG), the main cohort only.
if [[ -z ${COHORT_CONFIG:-} && -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    COHORT_CONFIG=$PROJECT_ROOT/configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml
fi
[[ -n ${COHORT_CONFIG:-} ]] && CONFIGS=("$COHORT_CONFIG")
# epilepsy first in each scheme: an unfillable cell drops the last covariates first
SCHEMES=(none epilepsy autism age_group sex age_group,sex epilepsy,age_group,sex
    autism,age_group,sex epilepsy,autism,age_group,sex)
[[ -n ${MATCH:-} ]] && SCHEMES=("$MATCH")
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if (( TASK_ID > ${#CONFIGS[@]} * ${#SCHEMES[@]} )); then
    echo "Task $TASK_ID past the ${#CONFIGS[@]} configs x ${#SCHEMES[@]} schemes; nothing to do."
    exit 0
fi
COHORT_CONFIG=${CONFIGS[$(( (TASK_ID - 1) / ${#SCHEMES[@]} ))]}
SCHEME=${SCHEMES[$(( (TASK_ID - 1) % ${#SCHEMES[@]} ))]}
read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod labram reve luna}"

require_dir "$EMBEDDING_PROJECT_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"

dra_activate
THREADS=${SLURM_CPUS_PER_TASK:-16}
dra_pin_threads 1

name=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset_name"])' "$COHORT_CONFIG")
out="$DECODING_ROOT/medication_contrasts/$name/match-${SCHEME//,/-}"
echo "Cohort: $COHORT_CONFIG  Output: $out"

python -m eeg_adhd_epilepsy.analysis.medication_contrast \
    --cohort_config "$COHORT_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --desc "$EPOCHS_DESC" \
    --models "${BASE_MODELS[@]}" \
    --embedding_derivative_root "$EMBEDDING_PROJECT_ROOT" \
    --n_draws "${N_DRAWS:-100}" \
    --match "$SCHEME" \
    --n_jobs "$THREADS" \
    --output_dir "$out"

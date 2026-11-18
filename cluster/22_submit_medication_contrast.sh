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
# methylphenidate vs lisdexamfetamine. Writes summary.json and draws.csv under
# $DECODING_ROOT/contrast_adhd_medication/.
COHORT_CONFIG=${COHORT_CONFIG:-$PROJECT_ROOT/configs/contrasts/adhd_medication.yaml}
read -r -a BASE_MODELS <<< "${BASE_MODELS:-cbramod labram reve luna}"

require_dir "$EMBEDDING_PROJECT_ROOT"
require_file "$METADATA_PATH"
require_file "$COHORT_CONFIG"

dra_activate
THREADS=${SLURM_CPUS_PER_TASK:-16}
dra_pin_threads 1

out="$DECODING_ROOT/contrast_adhd_medication"
echo "Cohort: $COHORT_CONFIG  Output: $out"

python -m eeg_adhd_epilepsy.analysis.medication_contrast \
    --cohort_config "$COHORT_CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --desc "$EPOCHS_DESC" \
    --models "${BASE_MODELS[@]}" \
    --embedding_derivative_root "$EMBEDDING_PROJECT_ROOT" \
    --n_draws "${N_DRAWS:-100}" \
    --n_jobs "$THREADS" \
    --output_dir "$out"

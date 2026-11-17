#!/usr/bin/env bash
#SBATCH --job-name=eeg_epochs_file
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# The saved epochs of one variant as one file per condition on the project,
# derivatives/preproc/task-clinical_desc-<variant>_cond-<condition>_epo.joblib, which the raw
# analyses (11, 17) read instead of the FIF files when it exists. Run it after every run of 04.
# Not measured yet: one condition is in memory at a time (EO and EC of every recording
# were 13.2 GB together with desc-base).

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

require_dir "$BIDS_ROOT/derivatives/preproc"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads 1

# The conditions the analyses use.
python -m eeg_adhd_epilepsy.analysis.consolidate_epochs \
  --bids_root "$BIDS_ROOT" \
  --desc "$EPOCHS_DESC" \
  --conditions $DESCRIPTOR_CONDITIONS \
  --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
validate_bids "$BIDS_ROOT/derivatives/preproc"

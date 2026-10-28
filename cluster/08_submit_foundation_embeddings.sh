#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# One job per epoch variant on one trillium H100 (24 cores): PROCESSES share the GPU,
# each running one model with some of its weight sets over every subject. A process
# is <model>:<weights>, the weights `all` or a comma-separated list of `pretrained`
# and seeds. Each weight set (label) must be in one process only. A process takes
# about as long per weight set as its model needs (per 64 subjects: REVE ~48 s,
# LaBraM and CBraMod ~30 s, LUNA ~23 s), so the default gives each of 12 processes
# about the same work: half the time of one process per model with REVE split in five.
#   cluv submit trillium-gpu cluster/08_submit_foundation_embeddings.sh -- EPOCHS_DESC=base
#   ... -- EPOCHS_DESC=base SUBJECTS="0001 0002"     (a check on some subjects)

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
default_processes="reve:pretrained,0,1 reve:2,3,4 reve:5,6,7 reve:8,9"
default_processes+=" labram:pretrained,0,1,2 labram:3,4,5,6 labram:7,8,9"
default_processes+=" cbramod:pretrained,0,1,2 cbramod:3,4,5,6 cbramod:7,8,9"
default_processes+=" luna:pretrained,0,1,2,3,4 luna:5,6,7,8,9"
PROCESSES=${PROCESSES:-$default_processes}
read -r -a processes <<< "$PROCESSES"
# Cores per process, for its BLAS, torch and loader threads.
THREADS=${THREADS:-$(( ${SLURM_CPUS_ON_NODE:-24} / ${#processes[@]} ))}

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$VENV_PATH"

dra_activate
dra_pin_threads "$THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

subject_args=()
if [[ -n "${SUBJECTS:-}" ]]; then
    read -r -a subjects <<< "$SUBJECTS"
    subject_args=(--subjects "${subjects[@]}")
fi

pids=()
for process in "${processes[@]}"; do
    model=${process%%:*}
    IFS=, read -r -a weights <<< "${process#*:}"
    python -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings \
        --config "$FOUNDATION_CONFIG" \
        --bids_root "$BIDS_ROOT" \
        --desc "$EPOCHS_DESC" \
        --metadata "$METADATA_PATH" \
        --derivatives_root "$EMBEDDING_SCRATCH_ROOT" \
        --model "$model" \
        --weights "${weights[@]}" \
        --n_jobs "$THREADS" \
        "${subject_args[@]}" 2>&1 | sed -u "s/^/[$process] /" &
    pids+=("$!")
done

# The job fails when any process does; the others still finish their recordings.
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"

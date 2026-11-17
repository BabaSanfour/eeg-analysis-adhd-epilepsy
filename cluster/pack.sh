#!/usr/bin/env bash
#SBATCH --job-name=eeg_pack
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# For clusters that give whole nodes (trillium): run TASKS array tasks of a stage script on
# one node, PACK at a time, each with an equal share of the cores (TASK_CPUS overrides the
# share, e.g. the whole node for tasks of very different lengths). When a task ends the next
# starts, so a slow task does not leave the node idle. Array task i of this job runs tasks
# (i-1)*TASKS+1 .. i*TASKS of SCRIPT; the stage scripts skip task ids past their range.
# TASKS defaults to PACK; without --array it runs tasks 1..TASKS. Each line of the log
# starts with its task id. Fails if any task fails.
#   cluv submit trillium cluster/pack.sh --array=1-5 -- \
#       SCRIPT=cluster/11_batch_run_dim_reduction.sh PACK=4 TASKS=8 EPOCHS_DESC=base ...

set -euo pipefail
# cluv passes VAR=value as arguments: export them for this script and the tasks.
for assignment in "$@"; do export "$assignment"; done
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
: "${SCRIPT:?Set SCRIPT=cluster/<stage script>.sh}" "${PACK:?Set PACK=<tasks at once>}"
TASKS=${TASKS:-$PACK}
cpus=${SLURM_CPUS_ON_NODE:-$(nproc)}
task_cpus=${TASK_CPUS:-$((cpus / PACK))}
first=$(( (${SLURM_ARRAY_TASK_ID:-1} - 1) * TASKS + 1 ))
# The stage scripts check the array size against their task count: this job covers
# its array size x TASKS tasks.
if [[ -n ${SLURM_ARRAY_TASK_COUNT:-} ]]; then
    export SLURM_ARRAY_TASK_COUNT=$((SLURM_ARRAY_TASK_COUNT * TASKS))
fi

status=0
running=0
for task in $(seq "$first" $((first + TASKS - 1))); do
    if (( running == PACK )); then
        wait -n || status=1
        running=$((running - 1))
    fi
    SLURM_ARRAY_TASK_ID=$task SLURM_CPUS_PER_TASK=$task_cpus \
        bash "$PROJECT_ROOT/$SCRIPT" > >(sed -u "s/^/[$task] /") 2>&1 &
    running=$((running + 1))
done
while (( running > 0 )); do
    wait -n || status=1
    running=$((running - 1))
done
exit $status

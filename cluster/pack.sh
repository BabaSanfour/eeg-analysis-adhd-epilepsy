#!/usr/bin/env bash
#SBATCH --job-name=eeg_pack
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# For clusters that give whole nodes (trillium): run PACK array tasks of a stage
# script at once on one node, each with an equal share of its cores. Array task i of
# this job runs tasks (i-1)*PACK+1 .. i*PACK of SCRIPT; the stage scripts skip
# task ids past their range. Without --array it runs tasks 1..PACK. Each line of
# the log starts with its task id. Fails if any task fails.
#   cluv submit trillium cluster/pack.sh --array=1-5 -- \
#       SCRIPT=cluster/11_batch_run_dim_reduction.sh PACK=4 EPOCHS_DESC=base ...

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
: "${SCRIPT:?Set SCRIPT=cluster/<stage script>.sh}" "${PACK:?Set PACK=<tasks per node>}"
cpus=${SLURM_CPUS_ON_NODE:-$(nproc)}
first=$(( (${SLURM_ARRAY_TASK_ID:-1} - 1) * PACK + 1 ))

pids=()
for task in $(seq "$first" $((first + PACK - 1))); do
    SLURM_ARRAY_TASK_ID=$task SLURM_CPUS_PER_TASK=$((cpus / PACK)) \
        bash "$PROJECT_ROOT/$SCRIPT" > >(sed -u "s/^/[$task] /") 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status

#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_dimred
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Archive every finished run folder of one epoch variant to the project, at its path
# under the scratch root: $SCRATCH_ROOT/<path>/<run> -> $DATA_ROOT/<path>/<run>.tar.zst.
# RUNS=dim_reduction (default): a run is finished once the REPORTS_ONLY job of 11-13 has
# written its _RUN_SUCCESS; submit this after it (--dependency=afterany).
# RUNS=decoding: submit after 16 (it reads the saved predictions) with RUN_DIRS set to
# one cohort's folders, e.g. RUN_DIRS='<dataset_name>/*'. Its runs end in _SUCCESS or
# _PARTIAL (a skipped unit makes a run partial), and 15 writes _PARTIAL while it runs,
# so only that cohort's folders are taken.
# Once the archive lists as many files as the folder, the folder's artifacts/ (one entry
# per fit) are deleted; its tables, config and status stay on scratch. A rerun into a
# pruned folder fits again. Jobs running at once skip each other's folders.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant
RUN_DIRS=${RUN_DIRS:-*/*}
case ${RUNS:-dim_reduction} in
    dim_reduction) root=$DIM_REDUCTION_ROOT; markers=(_RUN_SUCCESS) ;;
    decoding)
        [[ $RUN_DIRS != '*/*' ]] || { echo "RUNS=decoding needs RUN_DIRS" >&2; exit 1; }
        root=$DECODING_ROOT; markers=(_SUCCESS _PARTIAL) ;;
    *) echo "RUNS must be dim_reduction or decoding, not $RUNS" >&2; exit 1 ;;
esac
require_dir "$root"

runs=()
for name in "${markers[@]}"; do
    for marker in "$root"/$RUN_DIRS/"$name"; do
        if [[ -e $marker ]]; then runs+=("$(dirname "$marker")"); fi
    done
done

status=0
for run in ${runs[@]+"${runs[@]}"}; do
    [[ -d $run/artifacts ]] || continue
    mkdir "$run/.archiving" 2>/dev/null || continue
    archive=$DATA_ROOT/${run#"$SCRATCH_ROOT"/}.tar.zst
    tmp=$(dirname "$archive")/.$(basename "$archive").partial
    mkdir -p "$(dirname "$archive")"
    files=$(find "$run" -type f | wc -l)
    tar -C "$(dirname "$run")" --exclude=.archiving -cf - "$(basename "$run")" |
        zstd -T"${SLURM_CPUS_PER_TASK:-1}" -3 -q -o "$tmp"
    archived=$(zstd -q -dc "$tmp" | tar -tvf - | grep -c '^-')
    if (( files == archived )); then
        mv -f "$tmp" "$archive"
        rm -rf "$run/artifacts"
        echo "$archive: $files files"
    else
        rm -f "$tmp"
        echo "$run: $files files but $archived archived; kept" >&2
        status=1
    fi
    rmdir "$run/.archiving"
done
exit $status

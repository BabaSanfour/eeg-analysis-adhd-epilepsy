#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_dimred
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Archive every finished dim-reduction run folder of one epoch variant to the project, at
# its path under the scratch root: $SCRATCH_ROOT/<path>/<run> -> $DATA_ROOT/<path>/<run>.tar.zst.
# A run is finished once the REPORTS_ONLY job of 11-13 has written its _RUN_SUCCESS; submit
# this after it (--dependency=afterany). Once the archive lists as many files as the folder,
# the folder's artifacts/ (one file per fit and per eval) are deleted; its runs/ inventories,
# config and status stay on scratch for the merged tables. A rerun into a pruned folder fits
# again. Jobs running at once skip each other's folders.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant
require_dir "$DIM_REDUCTION_ROOT"

status=0
for marker in "$DIM_REDUCTION_ROOT"/*/*/_RUN_SUCCESS; do
    run=$(dirname "$marker")
    [[ -e $marker && -d $run/artifacts ]] || continue
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

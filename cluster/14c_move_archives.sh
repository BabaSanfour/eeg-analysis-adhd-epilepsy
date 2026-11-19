#!/usr/bin/env bash
# Run on a login node, not submitted: `bash cluster/14c_move_archives.sh` from the
# project root. For clusters whose compute nodes see the project read-only (trillium),
# where 14b stages its archives under $SCRATCH_ROOT/archive and keeps the artifacts.
# Moves each staged archive to the project at the same path, then deletes the
# artifacts/ of the run folder it holds:
#   $SCRATCH_ROOT/archive/<path>/<run>.tar.zst -> $DATA_ROOT/<path>/<run>.tar.zst
#   then rm -rf $SCRATCH_ROOT/<path>/<run>/artifacts
# A staged archive and its artifacts go only once rsync has succeeded and the project
# copy has the same size, so a cut-off move is finished by running this again.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
stage=$SCRATCH_ROOT/archive
[[ -d $stage ]] || { echo "Nothing staged in $stage"; exit 0; }

rsync -a --exclude='.*.partial' "$stage"/ "$DATA_ROOT"/
while IFS= read -r -d '' staged; do
    path=${staged#"$stage"/}
    [[ $(stat -c %s "$staged") == $(stat -c %s "$DATA_ROOT/$path" 2>/dev/null) ]] || continue
    rm -rf "$SCRATCH_ROOT/${path%.tar.zst}/artifacts"
    rm -f "$staged"
    echo "$DATA_ROOT/$path"
done < <(find "$stage" -name '*.tar.zst' -print0)
find "$stage" -mindepth 1 -type d -empty -delete

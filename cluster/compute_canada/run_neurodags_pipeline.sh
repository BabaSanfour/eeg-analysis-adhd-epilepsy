#!/usr/bin/env bash
# =============================================================================
# Neurodags pipeline — full run from scratch to dataframe CSV.
# Equivalent to the original:
#   submit_base_cleaning.sh   → step-0b_preproc_cleaned.yml
#   (epochs script)           → step-0c_conditions.yml
#   submit_extract_descriptors_array.sh → step-1_features.yml
#
# USAGE — interactive job on Compute Canada (Narval/Béluga/Graham/Cedar):
#
#   1. Request an interactive node:
#      salloc --time=4:00:00 --cpus-per-task=8 --mem=32G --account=def-<pi>
#
#   2. Run this script (or source it):
#      bash cluster/compute_canada/run_neurodags_pipeline.sh
#
#   Override any variable at call time, e.g.:
#      PROJECT_ROOT=/scratch/$USER/eeg bash .../run_neurodags_pipeline.sh
#
# NOTE: neurodags run is parallel at the file level. --n-jobs controls how many
# files are processed concurrently. AR (autoreject) inside each file is single-
# threaded (n_jobs: 1 in YAML) to avoid oversubscription.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — override via environment or edit here
# ---------------------------------------------------------------------------
PROJECT_ROOT=${PROJECT_ROOT:-/home/yorguin/code/eeg-analysis-adhd-epilepsy}
PIPELINES_DIR="$PROJECT_ROOT/neurodags_pipelines"
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}

# Parallelism: default to all CPUs allocated by SLURM, else 4
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}

# Output CSV paths
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/results}
CSV_ALL="$OUTPUT_DIR/features_all_wide.csv"
CSV_EO="$OUTPUT_DIR/features_eo_wide.csv"
CSV_EC="$OUTPUT_DIR/features_ec_wide.csv"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[ -d "$PROJECT_ROOT" ] || { echo "PROJECT_ROOT not found: $PROJECT_ROOT"; exit 1; }
[ -d "$VENV_PATH" ]    || { echo "venv not found: $VENV_PATH"; exit 1; }
[ -d "$PIPELINES_DIR" ] || { echo "PIPELINES_DIR not found: $PIPELINES_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load gcc arrow/23.0.1 python/3.11 2>/dev/null || true   # no-op outside cluster

source "$VENV_PATH/bin/activate"

# Prevent stale ~/.local packages shadowing the venv
export PYTHONNOUSERSITE=1

# Keep numerical libraries single-threaded; neurodags --n-jobs owns parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use node-local tmp for caches (fast NVMe on SLURM; /tmp otherwise)
TMPDIR_BASE="${SLURM_TMPDIR:-/tmp}"
export NUMBA_CACHE_DIR="$TMPDIR_BASE/numba_cache"
export MNE_HOME="$TMPDIR_BASE/mne_home"
export MPLCONFIGDIR="$TMPDIR_BASE/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

echo "============================================================"
echo " neurodags pipeline — $(date)"
echo " PROJECT_ROOT : $PROJECT_ROOT"
echo " N_JOBS       : $N_JOBS"
echo " OUTPUT_DIR   : $OUTPUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 0 (optional): generate synthetic test data
# Skip this on real data — your rawdata/ must already exist.
# ---------------------------------------------------------------------------
# python "$PIPELINES_DIR/generate_synthetic.py"

# ---------------------------------------------------------------------------
# STEP 1 — Preprocessing  [original: submit_base_cleaning.sh]
#
# For each source .vhdr:
#   inflate_bad_annotations → resample+bandpass → zapline → RANSAC →
#   CAR → autoreject_annotate_blockwise
#
# Writes per subject:
#   @CleanedPrepRaw.fif           fully annotated continuous Raw
#   @CleanedPrepRaw_prov.json     AR stats + config snapshot
#   @CleanedPrepRaw_ar_plot_*.png reject-log plots per condition
#   @CleanedPrep.fif              fixed-length 2 s epochs (all conditions)
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 1: preprocessing (step-0b) ---"
neurodags run "$PIPELINES_DIR/step-0b_preproc_cleaned.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 2 — Condition epoch extraction  [original: submit_epochs.sh]
#
# Reads @CleanedPrepRaw.fif; no re-preprocessing.
# Extracts 2 s epochs within BLOCK_EO / BLOCK_EC annotation windows,
# omitting any BAD_epoch_* spans from AR.
#
# Writes per subject:
#   @ConditionEO.fif
#   @ConditionEC.fif
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 2: condition epoch extraction (step-0c) ---"
neurodags run "$PIPELINES_DIR/step-0c_conditions.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 3 — Feature extraction, all epochs  [original: extract_descriptors.py]
#
# Reads @BasicPrep.fif (all conditions pooled) by default.
# Computes ~35 derivative families: PSD, FOOOF, band power, ratios,
# antropy (14) and neurokit2 (4/5) complexity measures.
#
# Writes dataset-level .nc files:
#   features@AbsBandPower.nc, features@SampleEntropy.nc, ...
#   (one file per family covering all subjects × runs × epochs)
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3: feature extraction, all epochs (step-1) ---"
neurodags run "$PIPELINES_DIR/step-1_features.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 3b — Feature extraction, per condition (mirrors --conditions EO EC)
#
# Uses step-1_datasets_conditions.yml which points to @ConditionEO.fif /
# @ConditionEC.fif. Writes to separate derivatives paths:
#   features_conditions_eo/features@*.nc
#   features_conditions_ec/features@*.nc
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3b: feature extraction, per condition ---"
neurodags run "$PIPELINES_DIR/step-1_features.yml" \
    -d "$PIPELINES_DIR/step-1_datasets_conditions.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 4 — Status check  [original: _SUCCESS markers + failures.csv]
#
# Prints done/missing/errored counts per derivative family.
# --list-errors prints paths of .error marker files.
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 4: pipeline status ---"
neurodags status "$PIPELINES_DIR/step-1_features.yml" --list-errors || true

# ---------------------------------------------------------------------------
# STEP 5 — Assemble flat CSV  [original: sensor_subject_features.csv]
#
# neurodags dataframe walks all .nc derivatives marked for_dataframe: True,
# merges them into one row per source file, and writes a wide CSV.
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 5: assemble dataframe ---"

# All epochs (default dataset)
neurodags dataframe "$PIPELINES_DIR/step-1_features.yml" \
    --format wide \
    --n-jobs "$N_JOBS" \
    --output "$CSV_ALL"

# EO-only epochs
neurodags dataframe "$PIPELINES_DIR/step-1_features.yml" \
    -d "$PIPELINES_DIR/step-1_datasets_conditions.yml" \
    --format wide \
    --n-jobs "$N_JOBS" \
    --output "$CSV_EO"

# EC-only epochs
# (step-1_datasets_conditions.yml defines both eo and ec datasets;
#  neurodags dataframe assembles both — split post-hoc if needed)

echo ""
echo "============================================================"
echo " Done. Outputs:"
echo "   $CSV_ALL"
echo "   $CSV_EO"
echo "============================================================"

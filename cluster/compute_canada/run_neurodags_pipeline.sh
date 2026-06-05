#!/usr/bin/env bash
# =============================================================================
# Neurodags pipeline — full run from scratch to dataframe CSV.
#
# Replaces the old separate scripts:
#   submit_base_cleaning.sh              → step-0_pipeline@preprocessing.yml
#   submit_epochs.sh                     → (no longer needed — epoching is in-memory in step-1)
#   submit_extract_descriptors_array.sh  → step-1_pipeline@extraction.yml
#
# Pipeline files (all in neurodags_pipelines/):
#   step-0_pipeline@preprocessing.yml  — per-subject preprocessing chain
#   step-0_pipeline@qc.yml             — QC records + HTML reports (+ dataset summary)
#   step-1_pipeline@extraction.yml     — feature extraction, all 8 conditions in one run
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

# Output CSV path (one file; split by condition post-hoc on `dataset` column)
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/results}
CSV_ALL="$OUTPUT_DIR/features_all_conditions.csv"

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
# For each source .vhdr file:
#   inject_block_annotations  → inflate_bad_annotations → preprocess_raw
#   → zapline_denoise → ransac_bad_channels → apply_car
#   → autoreject_annotate_blockwise → source_correction → residual_denoise
#
# Writes per subject:
#   @CleanedPrepRaw.fif          fully annotated + cleaned continuous Raw
#   @CleanedPrepRaw_prov.json    AR stats + config snapshot
#   @CleanedPrepRaw_ar_plot_*.png  reject-log plots per condition
#   @CorrectRaw.fif              ICA-corrected Raw (DSS+MWF)
#   @DenoiseRaw.fif              residual-denoised Raw
#
# Note: condition epoching is in-memory in step-1 — no per-condition .fif written here.
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 1: preprocessing ---"
neurodags run "$PIPELINES_DIR/step-0_pipeline@preprocessing.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 2 — Preprocessing QC  [no original equivalent — new]
#
# Reads derivatives from step-1; produces per-subject HTML reports and a
# dataset-level summary (PreprocDatasetQCReport aggregator node).
#
# Writes per subject:
#   @CleanedPrepRaw_raw_qc.json / _base_qc.json / _correct_qc.json / _denoise_qc.json
#   *_base_qc_report.html / *_correct_qc_report.html / *_denoise_qc_report.html
# Writes dataset-level:
#   preprocessing QC dataset summary HTML
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 2: preprocessing QC reports ---"
neurodags run "$PIPELINES_DIR/step-0_pipeline@qc.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 3 — Feature extraction  [original: submit_extract_descriptors_array.sh]
#
# Reads @CleanedPrepRaw.fif; conditions epoched in-memory (no intermediate .fif).
# All 8 conditions active in step-1_dataset.yml — one run covers all.
#
# Writes dataset-level .nc files per feature family:
#   derivatives/features_conditions/{condition}/features@AbsBandPower.nc
#   derivatives/features_conditions/{condition}/features@SampleEntropy.nc
#   ... (~35 families)
#
# Also writes per-subject descriptor QC records and HTML reports.
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3: feature extraction ---"
neurodags run "$PIPELINES_DIR/step-1_pipeline@extraction.yml" \
    --n-jobs "$N_JOBS"

# ---------------------------------------------------------------------------
# STEP 4 — Status check
#
# Prints done/missing/errored counts per derivative family.
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 4: pipeline status ---"
neurodags status "$PIPELINES_DIR/step-0_pipeline@preprocessing.yml" || true
neurodags status "$PIPELINES_DIR/step-1_pipeline@extraction.yml" --list-errors || true

# ---------------------------------------------------------------------------
# STEP 5 — Assemble flat CSV  [original: sensor_subject_features.csv]
#
# neurodags dataframe walks all .nc derivatives marked for_dataframe: True,
# merges them into one wide CSV with one row per source file.
# The `dataset` column identifies the condition (e.g., EO_baseline, EC_baseline).
# Split post-hoc with pandas: df[df["dataset"] == "EO_baseline"]
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 5: assemble dataframe ---"
neurodags dataframe "$PIPELINES_DIR/step-1_pipeline@extraction.yml" \
    --format wide \
    --n-jobs "$N_JOBS" \
    --output "$CSV_ALL"

echo ""
echo "============================================================"
echo " Done. Output:"
echo "   $CSV_ALL"
echo ""
echo " Split by condition post-hoc:"
echo "   df = pd.read_csv('$CSV_ALL')"
echo "   eo = df[df['dataset'] == 'EO_baseline']"
echo "   ec = df[df['dataset'] == 'EC_baseline']"
echo "============================================================"

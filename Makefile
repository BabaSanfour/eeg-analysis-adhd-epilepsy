# EEG ADHD/Epilepsy pipeline — convenience targets over the eeg-* CLIs.
#
# Set the paths once (env or `make VAR=... target`) and run a stage, a range,
# or the whole chain. Per-stage core targets delegate to `eeg-run` (which owns
# stage order, --dry-run, and resume); see `eeg-run --list`.
#
#   make dry-run BIDS_ROOT=/data/BIDS METADATA=/data/meta.csv RAW_ROOT=/data/raw \
#                COHORT=configs/cohorts/.../total.yaml \
#                DIM_ANALYSIS=configs/analyses/dim_reduction/default.yaml \
#                DECODE_ANALYSIS=configs/analyses/decoding/EO.yaml
#   make descriptors BIDS_ROOT=/data/BIDS METADATA=/data/meta.csv
#   make all ...        # whole chain
#
# Large jobs: use the numbered SLURM scripts in cluster/ instead.

# --- Configuration (override on the command line or via env) ------------------
BIDS_ROOT        ?=
METADATA         ?=
RAW_ROOT         ?=
COHORT           ?=
ANALYSIS         ?=
DIM_ANALYSIS     ?=
DECODE_ANALYSIS  ?=
DESCRIPTORS_CFG  ?= configs/descriptors.yaml
SEGMENT_DURATION ?= 10.0
N_JOBS           ?= 4
# Activate your venv first, or pass PYTHON=.venv/bin/python so the targets find
# the installed package without relying on PATH.
PYTHON           ?= python
EEG_RUN          = $(PYTHON) -m eeg_adhd_epilepsy.run

# Extra inputs for the non-core targets.
ADHD_CSV             ?=
DRUG_RESISTANT_CSV   ?=
METADATA_DIR         ?=
REPORTS_DIR          ?=
FOUNDATION_EMB_CFG   ?=

# Only pass flags whose variable is set, so unset paths don't become "." .
RUN_ARGS = \
	$(if $(RAW_ROOT),--raw_root $(RAW_ROOT)) \
	$(if $(BIDS_ROOT),--bids_root $(BIDS_ROOT)) \
	$(if $(METADATA),--metadata $(METADATA)) \
	$(if $(COHORT),--cohort_config $(COHORT)) \
	$(if $(ANALYSIS),--analysis_config $(ANALYSIS)) \
	$(if $(DIM_ANALYSIS),--dim_analysis_config $(DIM_ANALYSIS)) \
	$(if $(DECODE_ANALYSIS),--decode_analysis_config $(DECODE_ANALYSIS)) \
	--descriptors_config $(DESCRIPTORS_CFG) \
	--segment_duration $(SEGMENT_DURATION) \
	--n_jobs $(N_JOBS)

.PHONY: help install test dry-run all \
        bids preprocess epochs descriptors merge dim-reduce classical-decode \
        foundation-embeddings foundation-decode cohort-report metadata

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Editable install (refreshes the eeg-* console scripts)
	$(PYTHON) -m pip install -e .

test: ## Run the test suite
	$(PYTHON) -m pytest

dry-run: ## Preview the whole chain without running anything
	$(EEG_RUN) --dry-run $(RUN_ARGS)

all: ## Run the whole chain (resume-by-default)
	$(EEG_RUN) $(RUN_ARGS)

# --- Core stages (delegate to eeg-run for ordering/resume) --------------------
bids: ## raw -> BIDS + pre-base reports
	$(EEG_RUN) --from to-bids --to to-bids $(RUN_ARGS)

preprocess: ## BIDS -> cleaned desc-base derivatives + QC
	$(EEG_RUN) --from preprocess --to preprocess $(RUN_ARGS)

epochs: ## desc-base -> condition epochs
	$(EEG_RUN) --from epochs --to epochs $(RUN_ARGS)

descriptors: ## epochs -> descriptor shards (all subjects, sequential)
	$(EEG_RUN) --from descriptors --to descriptors $(RUN_ARGS)

merge: ## descriptor shards -> combined tables
	$(EEG_RUN) --from merge --to merge $(RUN_ARGS)

dim-reduce: ## combined descriptors -> dimensionality reduction
	$(EEG_RUN) --from dim-reduce --to dim-reduce $(RUN_ARGS)

classical-decode: ## combined descriptors -> classical decoding
	$(EEG_RUN) --from classical-decode --to classical-decode $(RUN_ARGS)

# --- Non-core stages (direct console scripts) ---------------------------------
foundation-embeddings: ## Extract foundation-model embeddings (dataset-wide)
	$(if $(FOUNDATION_EMB_CFG),,$(error Set FOUNDATION_EMB_CFG to the dataset-wide embedding config))
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings --config $(FOUNDATION_EMB_CFG) --bids_root $(BIDS_ROOT) $(if $(METADATA),--metadata $(METADATA))

foundation-decode: ## Foundation-model probing / fine-tuning (cohort + analysis)
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.foundation_decoding --cohort_config $(COHORT) --analysis_config $(ANALYSIS) \
	  $(if $(BIDS_ROOT),--bids_root $(BIDS_ROOT)) $(if $(METADATA),--metadata $(METADATA))

cohort-report: ## Build the cohort report from clean metadata
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.cohort --metadata_csv $(METADATA) --output_dir $(REPORTS_DIR)

metadata: ## Build canonical patient metadata tables from the source CSVs
	$(PYTHON) -m eeg_adhd_epilepsy.io.patients --adhd_csv $(ADHD_CSV) \
	  --drug_resistant_csv $(DRUG_RESISTANT_CSV) --output_dir $(METADATA_DIR)

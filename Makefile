# EEG ADHD/Epilepsy pipeline — direct convenience targets over each stage CLI.
#
# Set paths with environment variables or `make VAR=... target`. Every target
# invokes its owning Python module directly; `make all` runs the local stages in
# the order shown below. Large jobs should use the numbered SLURM scripts.
#
#   make descriptors BIDS_ROOT=/data/BIDS METADATA=/data/meta.csv
#   make dim-reduce BIDS_ROOT=/data/BIDS METADATA=/data/meta.csv \
#       COHORT=configs/cohorts/.../total.yaml \
#       DIM_ANALYSIS=configs/analyses/dim_reduction/descriptors.yaml

# --- Configuration -----------------------------------------------------------
BIDS_ROOT        ?=
METADATA         ?=
RAW_ROOT         ?=
COHORT           ?=
DIM_ANALYSIS     ?=
DECODE_ANALYSIS  ?=
ALIGN_ANALYSIS   ?= configs/analyses/align_subject_embeddings.yaml
ALIGN_DIM_ANALYSIS ?= configs/analyses/dim_reduction/foundation.yaml
SOURCE_EMBEDDING_ROOT ?=
EMBEDDING_MODEL_KEY   ?=
ALIGNMENT_TRANSFORM   ?= none
DESCRIPTORS_CFG       ?= configs/descriptors.yaml
SEGMENT_DURATION      ?= 10.0
N_JOBS                ?= 4
PYTHON                 ?= python

DESCRIPTOR_ROOT = $(BIDS_ROOT)/derivatives/signal_features/descriptors/combined
DESCRIPTOR_TABLE = $(DESCRIPTOR_ROOT)/sensor_recording_features.parquet
DESCRIPTOR_COLUMNS = $(DESCRIPTOR_ROOT)/sensor_recording_features_feature_columns.json

# Inputs for additional standalone targets.
ADHD_CSV                    ?=
DRUG_RESISTANT_CSV          ?=
METADATA_DIR                ?=
REPORTS_DIR                 ?=
FOUNDATION_EMB_CFG          ?=
FOUNDATION_DECODE_ANALYSIS  ?= configs/analyses/decoding/foundation.yaml

.PHONY: help install test all \
        bids preprocess epochs descriptors merge align-subject-embeddings \
        dim-reduce-alignments dim-reduce classical-decode \
        foundation-embeddings foundation-decode cohort-report metadata

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Editable install (refreshes the eeg-* console scripts)
	$(PYTHON) -m pip install -e .

test: ## Run the test suite
	$(PYTHON) -m pytest

all: ## Run every local pipeline stage in order
	$(MAKE) bids
	$(MAKE) preprocess
	$(MAKE) epochs
	$(MAKE) descriptors
	$(MAKE) merge
	$(MAKE) align-subject-embeddings
	$(MAKE) dim-reduce-alignments
	$(MAKE) dim-reduce
	$(MAKE) classical-decode

# --- Core stages --------------------------------------------------------------
bids: ## Raw EEG -> BIDS
	$(if $(RAW_ROOT),,$(error Set RAW_ROOT to the raw EEG directory))
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(PYTHON) -m eeg_adhd_epilepsy.preproc.to_bids \
	  --raw_root $(RAW_ROOT) --bids_root $(BIDS_ROOT) --metadata_csv $(METADATA) --n_jobs $(N_JOBS)

preprocess: ## BIDS -> cleaned desc-base derivatives + QC
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(PYTHON) -m eeg_adhd_epilepsy.preproc.base --bids_root $(BIDS_ROOT) --n_jobs $(N_JOBS)

epochs: ## desc-base -> condition epochs
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(PYTHON) -m eeg_adhd_epilepsy.preproc.epochs \
	  --bids_root $(BIDS_ROOT) --segment_duration $(SEGMENT_DURATION) --ignore_annotations

descriptors: ## Epochs -> descriptor shards for all subjects
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.extract_descriptors \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA) --config $(DESCRIPTORS_CFG) --conditions all

merge: ## Descriptor shards -> combined descriptor tables
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.merge_descriptors --bids_root $(BIDS_ROOT)

align-subject-embeddings: ## Foundation embeddings -> aligned variants
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(COHORT),,$(error Set COHORT to the cohort config))
	$(if $(SOURCE_EMBEDDING_ROOT),,$(error Set SOURCE_EMBEDDING_ROOT to the embedding derivatives))
	$(if $(EMBEDDING_MODEL_KEY),,$(error Set EMBEDDING_MODEL_KEY to the source model))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.align_subject_embeddings \
	  --cohort_config $(COHORT) --analysis_config $(ALIGN_ANALYSIS) \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA) \
	  --source_embedding_root $(SOURCE_EMBEDDING_ROOT) \
	  --embedding_model_key $(EMBEDDING_MODEL_KEY)

dim-reduce-alignments: ## Reduce one explicit raw/aligned embedding variant
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(COHORT),,$(error Set COHORT to the cohort config))
	$(if $(SOURCE_EMBEDDING_ROOT),,$(error Set SOURCE_EMBEDDING_ROOT to the embedding derivatives))
	$(if $(EMBEDDING_MODEL_KEY),,$(error Set EMBEDDING_MODEL_KEY to the base model))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
	  --cohort_config $(COHORT) --analysis_config $(ALIGN_DIM_ANALYSIS) \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA) --n_jobs $(N_JOBS) \
	  --alignment_transform $(ALIGNMENT_TRANSFORM) \
	  --embedding_derivative_root $(SOURCE_EMBEDDING_ROOT) \
	  --embedding_model_key $(EMBEDDING_MODEL_KEY)

dim-reduce: ## Combined descriptors -> dimensionality reduction
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(COHORT),,$(error Set COHORT to the cohort config))
	$(if $(DIM_ANALYSIS),,$(error Set DIM_ANALYSIS to the dimensionality-reduction config))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
	  --cohort_config $(COHORT) --analysis_config $(DIM_ANALYSIS) \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA) --n_jobs $(N_JOBS) \
	  --descriptor_table_path $(DESCRIPTOR_TABLE) \
	  --descriptor_feature_columns_path $(DESCRIPTOR_COLUMNS)

classical-decode: ## Combined descriptors -> classical decoding
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(COHORT),,$(error Set COHORT to the cohort config))
	$(if $(DECODE_ANALYSIS),,$(error Set DECODE_ANALYSIS to the classical-decoding config))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.classical_decoding \
	  --cohort_config $(COHORT) --analysis_config $(DECODE_ANALYSIS) \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA) --n_jobs $(N_JOBS) \
	  --descriptor_table_path $(DESCRIPTOR_TABLE) \
	  --descriptor_feature_columns_path $(DESCRIPTOR_COLUMNS)

# --- Additional standalone stages -------------------------------------------
foundation-embeddings: ## Extract foundation-model embeddings
	$(if $(FOUNDATION_EMB_CFG),,$(error Set FOUNDATION_EMB_CFG to the embedding config))
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings \
	  --config $(FOUNDATION_EMB_CFG) --bids_root $(BIDS_ROOT) \
	  $(if $(METADATA),--metadata $(METADATA))

foundation-decode: ## Foundation-model probing or fine-tuning
	$(if $(BIDS_ROOT),,$(error Set BIDS_ROOT to the BIDS dataset))
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(COHORT),,$(error Set COHORT to the cohort config))
	$(PYTHON) -m eeg_adhd_epilepsy.analysis.foundation_decoding \
	  --cohort_config $(COHORT) --analysis_config $(FOUNDATION_DECODE_ANALYSIS) \
	  --bids_root $(BIDS_ROOT) --metadata $(METADATA)

cohort-report: ## Build the cohort report from canonical metadata
	$(if $(METADATA),,$(error Set METADATA to the canonical metadata CSV))
	$(if $(REPORTS_DIR),,$(error Set REPORTS_DIR to the report output directory))
	$(PYTHON) -m eeg_adhd_epilepsy.metadata.cohort \
	  --metadata_csv $(METADATA) --output_dir $(REPORTS_DIR)

metadata: ## Build canonical patient metadata from the source CSVs
	$(if $(ADHD_CSV),,$(error Set ADHD_CSV to the ADHD source CSV))
	$(if $(DRUG_RESISTANT_CSV),,$(error Set DRUG_RESISTANT_CSV to the drug-resistant source CSV))
	$(if $(METADATA_DIR),,$(error Set METADATA_DIR to the metadata output directory))
	$(PYTHON) -m eeg_adhd_epilepsy.metadata.patients \
	  --adhd_csv $(ADHD_CSV) --drug_resistant_csv $(DRUG_RESISTANT_CSV) \
	  --output_dir $(METADATA_DIR)

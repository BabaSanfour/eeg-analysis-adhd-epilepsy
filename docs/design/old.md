# Foundation Models and Decoding Design

## Summary

This document defines how to extend the ADHD/Epilepsy analysis project with:

1. Reusable EEG foundation-model embedding extraction.
2. Classical machine-learning decoding over reduced dimensions, extracted EEG
   descriptors, and foundation-model embeddings.
3. Direct foundation-model decoding of EEG through linear probing, full
   fine-tuning, and LoRA.

The project must continue to use coco-pipe as its analysis framework. The
ADHD/Epilepsy repository should contain only study-specific BIDS discovery,
cohort filtering, configuration, output orchestration, and thin command-line
entry points. Model loading, signal adaptation, feature selection,
cross-validation, metrics, statistics, result serialization, and report
generation should be reusable coco-pipe capabilities.

The project must not restore or duplicate the deleted project-local ML and DL
implementations. No foundation-model architecture, training loop, scaler,
selector, or report renderer should be implemented from scratch in the
ADHD/Epilepsy repository.

## Current Integration Model

The existing project already follows a useful division of responsibility:

- The ADHD/Epilepsy project discovers study-specific BIDS files, reads cohort
  metadata, applies group filters, maps configured targets, and controls
  derivative paths and run status.
- coco-pipe supplies `DataContainer`, descriptor extraction, dimensionality
  reduction, decoding components, visualization, and report primitives.
- Descriptor derivatives use resumable per-recording outputs with `_SUCCESS`
  markers, merged tables, `dataset_description.json`, and `config_used.yaml`.
- Dimensionality-reduction results are written below
  `BIDS/derivatives/dim_reduction/`, while summary reports are written below the
  sibling `reports/summary/` directory.
- `patient_group_id` identifies the biological patient across repeated
  recordings and is the leakage-safe grouping variable for cross-validation.

The new modules should retain these patterns. Before project integration,
coco-pipe requires several upstream extensions: its frozen-backbone and neural
fine-tuning configurations must resolve to executable estimators, LUNA must be
registered through the public foundation-model registry, and embedding
extraction needs durable derivative I/O.

### coco-pipe State as of This Design (verified against `dev`)

The project currently pins coco-pipe `@viz`
(`pyproject.toml`), but `viz` predates the foundation-model work entirely (no
`coco_pipe.decoding.foundation_models`, no FM entries in `_specs.py`, no
`FrozenBackboneDecoderConfig`/`NeuralFineTuneConfig`). All of the following
status notes are against the `dev` branch (HEAD `441787c` at design time),
which is the revision this project should pin to going forward (see
Development Order, step 1).

Already implemented on `dev`:

- `coco_pipe.decoding.foundation_models` (`_base.py`, `_braindecode.py`,
  `_hugging_face.py`, `_loader.py`): a `BackendBase` ABC that *is* the
  sklearn estimator, with `load`, `transform`, `predict`, `fit` (shared
  skorch loop), `configure_peft`, and `frozen`/`full`/`lora`/`qlora` train
  modes. This supersedes the old empty `fm_hub/` package, which `dev` already
  removes.
- `coco_pipe.decoding.configs.FrozenBackboneDecoderConfig` and
  `NeuralFineTuneConfig` already exist and are part of the `ExperimentConfig`
  estimator union, along with `LoRAConfig`, `QuantizationConfig`,
  `CheckpointConfig`, `TrainerConfig`, `TrainStageConfig`.
- `FoundationModelSpec` registry entries for `reve`, `cbramod`, `labram`
  (plus `biot`, `eegpt`, `signaljepa`, none of which this design targets).
  `labram`'s spec already documents the 128-channel/`LABRAM_CHANNEL_ORDER`
  requirement and notes `InterpolatedLaBraM` (braindecode>=1.5) for arbitrary
  channel sets.
- `coco_pipe.report.decoding.make_decoding_report` (public wrapper
  `coco_pipe.report.api.from_experiment_result`) already provides a fairly
  complete `ExperimentResult` → `Report` builder (`overview`, `configuration`,
  `provenance`, `model_summary`, `cv_summary`, `performance`, `statistical`,
  `confusion_probability`, `features`, `fit_diagnostics`, `caveats`,
  `export_inventory`, `topomaps` sections) built on
  `coco_pipe.report.core.Report`/`Section` and `coco_pipe.report.elements`.
- Grouped cross-validation: `CVConfig` already supports
  `stratified_group_kfold`/`group_kfold`/`leave_one_group_out`/
  `group_shuffle_split` with a `group_key`; `get_cv_splitter`/`_CVWithGroups`
  build the real `StratifiedGroupKFold`, and
  `experiment._validate_fold_integrity` exists.
- Fold-local feature selection: `FeatureSelectionConfig(method="k_best"|"sfs",
  n_features=..., cv=...)` with grouped inner CV and an explicit
  `allow_nongroup_inner_cv` escape hatch.
- Statistics and aggregation: `coco_pipe.decoding.stats` already implements
  `binomial_accuracy_test`, `run_statistical_assessment` (binomial +
  full-pipeline permutation), `_bootstrap_scores` (1,000 resamples,
  group-aware), and `aggregate_predictions_for_inference`
  (`sample`/`subject`/`group_mean`/`group_majority`/`custom` units), wired via
  `StatisticalAssessmentConfig`/`ChanceAssessmentConfig`/
  `ConfidenceIntervalConfig`.
- `Experiment.run(X, y, groups=..., feature_names=..., sample_ids=...,
  sample_metadata=..., inferential_unit=...)` is the real entry point — it
  takes arrays plus grouping/metadata, **not** a `DataContainer`.

Confirmed gaps (this design's actual net-new coco-pipe scope):

- **No `luna` entry anywhere** in `_specs.py`, the registry, or the
  braindecode backend. Step 2 is required, not precautionary.
- **No `coco_pipe.io.embeddings`** module. Step 4 is net-new.
- `coco_pipe.io.units.iter_analysis_units` implements only `flat`, `sensor`,
  `family`, `sensor_within_family`. `feature` and `feature_within_family`
  (step 6) do not exist yet.
- No fold-local `ReducerConfig`/PCA pipeline step in
  `coco_pipe.decoding` (step 6). `coco_pipe.dim_reduction.reducers.linear`
  has a standalone PCA reducer, but it is not wired as a fold-local decoding
  pipeline step.
- No `FoundationEmbeddingExtractor` (recording windowing + `transform` +
  pooling + serialize). `BackendBase.transform` returns per-window embeddings
  and `coco_pipe.io.dataset` has a windowing helper, but no class composes
  them into a recording-level extractor.
- No automatic CV fold-count reduction: coco-pipe raises an actionable error
  when groups are too few for `n_splits` rather than auto-reducing.
- No cross-analysis-unit FDR: `temporal_correction="fdr_bh"` corrects within a
  single result, not across sweep units.
- No embedding-extraction or foundation-decoding report builders. The
  classical-decoding report builder largely exists already
  (`coco_pipe.report.decoding`); embedding and foundation-decoding builders
  are net-new (step 4 and the foundation-decoding milestone respectively).
- The foundation-model dependency extra (torch/braindecode/transformers/
  PEFT/skorch) is **not installed** in the project's `.venv` — `import
  braindecode` fails. Step 1/3 must include installing and pinning this
  extra and confirming the pinned braindecode version actually exports
  `LUNA` and `InterpolatedLaBraM`.

## Proposed Module Structure

### coco-pipe

The coco-pipe `dev` work (see "coco-pipe State as of This Design" above)
already supplies most of the decoding substrate. This design only needs the
following genuinely **net-new** coco-pipe additions:

- A public `FoundationEmbeddingExtractor` that wraps the existing
  `BackendBase.transform` primitive plus recording windowing (reusing the
  windowing helper in `coco_pipe.io.dataset`) and pooling. No such class
  exists today; `transform` returns per-window embeddings but there is no
  recording-level extract/pool/serialize wrapper.
- `coco_pipe.io.embeddings` for writing, validating, indexing, merging, and
  loading embedding derivatives. Net-new module.
- A LUNA `FoundationModelSpec` registry entry (CBraMod, LaBraM, REVE already
  exist; only LUNA is missing).
- A fold-local `ReducerConfig` in the decoding pipeline, initially supporting
  PCA. A standalone PCA reducer exists under
  `coco_pipe.dim_reduction.reducers.linear`, but it is not wired as a
  fold-local decoding pipeline step.
- `feature` and `feature_within_family` modes in
  `coco_pipe.io.units.iter_analysis_units`, alongside the existing `flat`,
  `family`, `sensor`, and `sensor_within_family` modes.
- Cross-analysis-unit Benjamini-Hochberg FDR correction (the existing
  `ChanceAssessmentConfig.temporal_correction="fdr_bh"` corrects *within* a
  single result across temporal coordinates; correcting *across* sensor/
  feature/family analysis units is a separate concern — see Lightweight
  Statistics).
- Automatic CV fold-count reduction for small grouped cohorts. Today
  coco-pipe raises an actionable error telling the caller to lower
  `cv.n_splits`; it does not auto-reduce. Whether this lands in coco-pipe or
  the project orchestration is an open decision (see Cross-Validation).
- Embedding-extraction and foundation-decoding report builders (the
  classical-decoding builder `coco_pipe.report.decoding.make_decoding_report`
  already exists).
- A documented foundation-model dependency extra containing the required
  PyTorch, Braindecode, Hugging Face, PEFT, skorch, and related dependencies.

Already present on `dev` and only to be **validated/configured**, not built:
working estimator dispatch for `FrozenBackboneDecoderConfig` and
`NeuralFineTuneConfig`; grouped CV (`CVConfig(strategy="stratified_group_kfold")`);
fold-local feature selection (`FeatureSelectionConfig(method="k_best")`);
group-level prediction aggregation and statistics
(`aggregate_predictions_for_inference`, `binomial_accuracy_test`,
`_bootstrap_scores`).

The coco-pipe API should expose behavior rather than project-specific paths.
The suggested public interfaces below reflect the **actual** `Experiment.run`
signature, which takes arrays plus `groups`/`sample_metadata` rather than a
`DataContainer` (the project must unpack its loaded container into these):

```python
extractor = FoundationEmbeddingExtractor(config)
result = extractor.extract(epochs, signal_metadata=signal_metadata)

save_embedding_derivative(result, output_path, provenance=provenance)
container = load_embedding_derivatives(paths, representation="recording")

# Experiment.run(X, y, groups=..., feature_names=..., sample_ids=...,
#                sample_metadata=..., inferential_unit="subject")
experiment = Experiment(config)
result = experiment.run(
    X, y,
    groups=patient_group_ids,
    sample_metadata=observation_metadata,
    inferential_unit="subject",
)
```

### ADHD/Epilepsy Project

Add three thin analysis modules:

- `eeg_adhd_epilepsy.analysis.extract_foundation_embeddings`
- `eeg_adhd_epilepsy.analysis.decoding`
- `eeg_adhd_epilepsy.analysis.foundation_decoding`

Extend the existing shared analysis loader to support `embeddings` and
fold-local `reduced_dimensions` inputs. Reuse current cohort YAML target
definitions, BIDS utilities, completion markers, config snapshots, and report
directory helpers.

The raw foundation-model workflow should remain separate from classical
decoding so that GPU dependencies, checkpoints, resume behavior, and training
artifacts do not complicate lightweight CPU decoding runs.

## Suggested Configurations

Provide example configuration files:

- `configs/foundation_embeddings.example.yaml`
- `configs/decoding.example.yaml`
- `configs/foundation_decoding.example.yaml`

A foundation-embedding config should specify:

- BIDS and metadata inputs.
- Source derivative and preprocessing description, defaulting to `desc-base`.
- Conditions, subjects, sessions, tasks, and runs.
- Model keys and checkpoint revisions.
- Model-specific channel and sampling-rate policies.
- Window duration, stride, overlap, and remainder handling.
- Within-window token/layer pooling and across-window recording pooling.
- Precision, device, batch size, resume, overwrite, and report options.

A classical-decoding config should specify (the CV/selection/statistics blocks
should mirror the real coco-pipe schema — `CVConfig`, `FeatureSelectionConfig`,
`StatisticalAssessmentConfig` — so the project YAML maps onto
`ExperimentConfig` without a translation layer):

- Input mode and source derivative.
- Existing cohort filters, target column, and label mapping.
- Analysis-unit modes.
- Models and fixed or narrowly tuned hyperparameters.
- Baseline and top-10 feature-selection variants
  (`FeatureSelectionConfig(method="k_best", n_features=10)`).
- Grouped CV (`CVConfig(strategy="stratified_group_kfold",
  group_key="patient_group_id")`), evaluation/inferential unit
  (`inferential_unit="subject"`), metrics, statistics
  (`StatisticalAssessmentConfig`), and report options.
- Whether transductive precomputed dimensionality-reduction inputs are allowed.

A foundation-decoding config should specify:

- Foundation model and checkpoint.
- Training modes to attempt.
- Source EEG derivative and signal adaptation.
- Windowing and prediction aggregation.
- Grouped outer CV and grouped validation.
- Epoch, learning-rate, batch-size, early-stopping, LoRA, checkpoint, and
  precision settings.
- Unsupported-mode policy, resume behavior, and reports.

The project should pin coco-pipe to a specific commit on `dev` (currently
`441787c` or later, once the foundation-model dependency extra is documented
and confirmed installable) rather than `viz`, which predates this work
entirely, or any other moving branch.

## Foundation-Model Embedding Extraction

### Supported Models

The first supported models are:

- CBraMod
- LaBraM
- REVE
- LUNA

The extractor should use model implementations and published pretrained
weights supplied by Braindecode or Hugging Face integrations. Project code
must not reimplement the architectures.

### Input and Signal Adaptation

The default input is cleaned, epoched EEG from the current `desc-base`
derivative. Each coco-pipe backend adapter is responsible for:

- Required resampling.
- Unit conversion and normalization expected by the checkpoint.
- Channel ordering and canonical channel names.
- Missing, extra, interpolated, or unsupported channels.
- Fixed-length window generation.
- Model-specific patch or token requirements.

Preprocessing must be deterministic and target-independent. It must not compute
cohort-wide statistics that expose held-out subjects.

#### Montage Mismatch Is a First-Order Risk

This is not a config detail — it is the single biggest scientific risk for the
foundation-model arm. The project's montage (per `configs/descriptors.yaml`
`pooling.channel_groups`) is a **low-density ~19-channel 10-20 array using old
nomenclature** (F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1,
O2). The target foundation models were pretrained on very different montages:

- **LaBraM**: 128 channels in `LABRAM_CHANNEL_ORDER` (use `InterpolatedLaBraM`
  for arbitrary sets). Going 19 → 128 is heavy interpolation/zero-filling, and
  whatever the adapter does materially changes the science.
- **EEGPT**: 62 channels @ 250 Hz. **BIOT**: 16 channels. **CBraMod / REVE**:
  more flexible, but REVE requires `ch_names` in `SignalMetadata` for its
  positional encoding, so channel naming must be exactly right.

Two concrete requirements follow:

1. **Channel-name normalization.** The old 10-20 labels T3/T4/T5/T6 map to the
   modern T7/T8/P7/P8 that model channel vocabularies expect. The extractor
   must apply an explicit, documented rename before any model sees the data;
   silent mismatches will degrade REVE/LaBraM positional/channel handling
   without erroring.
2. **Honest adaptation accounting.** Each embedding's sidecar must record the
   real channel mapping (original → model channels, interpolated, dropped,
   zero-filled) so that downstream comparisons are not misread as
   like-for-like. A 19→128 interpolated LaBraM embedding is a fundamentally
   different object from a native-resolution one and must be labelled as such
   in reports.

The capability table below should be read with this montage in mind: for this
specific cohort, the channel-adaptation rows are the decisive feasibility
question, not a formality.

### Extraction Granularity and Pooling

Do not collapse medication states, sessions, tasks, or conditions into one
patient vector during extraction. For each source recording, save:

- Window-level embeddings for reuse in later temporal analyses.
- One pooled recording embedding for standard decoding and dimensionality
  reduction.

The default recording pooling strategy is the mean over valid windows. The
sidecar must record both within-window pooling, such as token mean or CLS token,
and across-window pooling. Later loaders may aggregate recordings by
`patient_group_id`, but that is a downstream analysis decision.

### BIDS-Compatible Derivative Strategy

The embedding datatype is not currently a standardized BIDS derivative. Use a
BIDS-compatible custom derivative organization:

```text
BIDS/derivatives/eeg_foundation_embeddings/
├── dataset_description.json
├── config_used.yaml
├── run_manifest.json
├── failures.csv
└── sub-0001/
    └── ses-01/
        └── eeg/
            ├── sub-0001_ses-01_task-EObaseline_run-01_desc-cbramodMean_embedding.npz
            ├── sub-0001_ses-01_task-EObaseline_run-01_desc-cbramodMean_embedding.json
            └── _SUCCESS
```

Preserve source BIDS entities in filenames. Use an alphanumeric `desc` label
that identifies the model variant and pooling representation. Store exact
model versions and checkpoints in metadata rather than encoding unstable
version strings in filenames.

The compressed NPZ should contain:

- `window_embeddings`: shape `[n_windows, embedding_dim]`.
- `recording_embedding`: shape `[embedding_dim]`.
- `window_start`, `window_stop`, and `window_index`.
- Optional layer-level or token-level arrays only when enabled explicitly.

The matching JSON sidecar should record:

- Model name, display name, architecture variant, checkpoint, revision, and
  checkpoint hash when available.
- Backend, coco-pipe version/commit, and relevant library versions.
- Subject, session, task, run, condition, recording ID, and patient group ID.
- Source EEG paths and input data type.
- Preprocessing derivative, provenance hash, original and final sampling rates,
  units, and normalization.
- Original channels, model channels, ordering, mapping, missing channels,
  dropped channels, and interpolation policy.
- Window duration, stride, overlap, count, padding, and remainder handling.
- Within-window and recording pooling strategies.
- Array names, dimensions, axis meanings, embedding shape, and dtype.
- Device, numerical precision, config hash, and creation time.

`dataset_description.json` should identify the dataset as a derivative and
include `GeneratedBy` and `SourceDatasets`. A run manifest should index every
successful and failed artifact. Loaders should use this manifest when
available, while retaining filesystem discovery as a fallback.

### Dimensionality-Reduction Compatibility

`load_embedding_derivatives` should return a two-dimensional coco-pipe
`DataContainer` with:

- Rows representing recordings or explicitly requested patient aggregates.
- Columns representing embedding dimensions.
- Observation metadata containing BIDS entities, condition, recording ID,
  `patient_group_id`, model, checkpoint, and pooling.
- Feature metadata containing embedding dimension indices and optional
  layer/token provenance.

This container should be accepted directly by the existing dimensionality
reduction machinery without a project-specific parser.

## Classical Machine-Learning Decoding

### Primary vs. Exploratory Analyses

The `flat`/`family`/`sensor`/`feature`/`sensor_within_family`/
`feature_within_family` sweep produces a large number of analysis units.
Even with grouped CV and BH-FDR correction within families, this volume of
results is hypothesis-generating, not confirmatory: it should not be the sole
basis for claims about which sensors or features carry signal.

For each target, designate the `flat` analysis mode (all descriptors, all
sensors, baseline and top-10 variants) as the **primary, confirmatory**
result. The `family`/`sensor`/`feature`/`*_within_family` sweeps are
**exploratory** and should be reported and labelled as such, used to generate
candidate sensors/features for follow-up rather than as standalone findings.
Reports must visually distinguish the primary result from the exploratory
sweep (e.g., a dedicated "Primary result" section ahead of the sweep
browser), and acceptance criteria for a decoding run should require the
primary `flat` result to be present and complete even if some sweep units are
skipped.

### Input Spaces

Support three input modes:

- `descriptors`
- `foundation_embeddings`
- `reduced_dimensions`

Foundation embeddings are target-independent inputs and may be extracted once
for the full cohort. Scaling, feature selection, and any learned reduction
must still occur inside training folds.

For publication-grade reduced-dimension decoding, PCA must be a decoder
pipeline step fitted separately in each training fold. Existing coordinates
from a reducer fitted globally on all subjects are transductive and may only be
used when `allow_transductive_input: true`. Such results must be labelled
exploratory in outputs and reports.

### Descriptor Analysis Units

Expose these analysis modes:

- `flat`: all descriptor columns together.
- `family`: one descriptor family, such as band, spectral parameterization, or
  complexity.
- `sensor`: all descriptors available for one sensor.
- `feature`: one descriptor feature across all sensors.
- `sensor_within_family`: one sensor restricted to one family.
- `feature_within_family`: one named feature restricted to its family.

Feature coordinates should come from coco-pipe descriptor metadata rather than
ad hoc parsing in the project. Units with insufficient samples, classes, or
features should be marked skipped with a structured reason.

The project's descriptor-loading path (`eeg_adhd_epilepsy/io/analysis.py`,
`load_descriptor_table`) already produces containers with `sensor`,
`feature`, and `feature_family` coordinates for the `("obs", "sensor",
"feature")`-shaped case. The new `feature` and `feature_within_family` modes
in `coco_pipe.io.units.iter_analysis_units` should select by the `feature`
(and optionally `feature_family`) coordinate across all `sensor` values,
mirroring how `sensor`/`sensor_within_family` already select by `sensor` (and
`feature_family`). No project-side reshaping should be needed for these new
modes.

### Band-Power vs. Aperiodic (Spectral-Parameterization) Comparison

Band-power ratios, particularly frontal theta/beta, are the historically
dominant ADHD EEG biomarker but have replicated poorly. Recent work attributes
much of the apparent band-power group difference to shifts in the aperiodic
(1/f) component of the spectrum rather than genuine oscillatory differences.

This project is already set up to test exactly this, since
`configs/descriptors.yaml` extracts both sides of the comparison:

- The `bands` family produces absolute/relative/log/corrected band powers and
  band ratios — including the classic `theta`/`beta` ratio — across the
  delta/theta/alpha/beta/gamma bands.
- The `parametric` family runs `specparam` (FOOOF) and produces aperiodic
  (exponent/offset), `fit_quality`, and `peak_summary` outputs — i.e. the
  aperiodic 1/f component plus parameterized periodic peaks.

The `family` and `feature_within_family` decoding sweeps should therefore
treat `bands` (and specifically the theta/beta ratio feature) and `parametric`
(aperiodic exponent/offset) as named, directly comparable units under the same
CV, grouping, and statistics. This comparison should be promoted to a dedicated
"band-power vs. aperiodic" table/section in the classical-decoding report
rather than left implicit in the general family sweep, since it is a specific,
pre-registerable scientific question for this cohort — does the aperiodic
component carry the diagnostic signal that the theta/beta ratio is
conventionally credited with — not just one entry among many descriptor
families. The `complexity` family is a natural third comparison arm.

### Models

Use L1-penalized logistic regression as the default sparse and interpretable
classifier. This is the classification equivalent intended by the request for
LASSO-style sparsity; regression `Lasso` must not be used for classification.

Allow configurable comparison models already supported by coco-pipe,
particularly:

- L2 logistic regression.
- Random forest.

Use class weights rather than balancing or oversampling the complete dataset.
Any future resampling must be implemented as a fold-local pipeline step.
Exhaustive sensor/feature runs should use a small fixed model set to prevent a
combinatorial increase in runtime.

### Baseline and Top-10 Selection

Run two variants for every analysis unit:

1. A baseline with no separate selector.
2. A top-10 variant using fold-local `SelectKBest`, with
   `k = min(10, n_available_features)`.

This maps directly onto the existing `FeatureSelectionConfig` in coco-pipe:
set `method="k_best"`, `n_features=10`. No new selector needs to be built —
the project only needs to (a) clamp `n_features` to the available column count
per unit and (b) ensure the inner selection CV stays grouped
(`FeatureSelectionConfig.cv` with the grouped strategy; do not set
`allow_nongroup_inner_cv=True` for publication-grade runs).

The selector, scaler, optional PCA, and classifier must be fitted only on the
outer-training partition. Do not obtain a global top ten from baseline
coefficients and rerun the same cross-validation, because that would leak
held-out information.

Report feature-selection frequency and coefficient consistency across folds as
descriptive stability measures. A consensus top-ten list may be displayed in
the report, but it must not be presented as an independently validated rerun.

### Cross-Validation and Leakage Prevention

Default to:

- Five-fold `StratifiedGroupKFold` via the existing
  `CVConfig(strategy="stratified_group_kfold", n_splits=5, shuffle=True,
  random_state=...)`. The grouped strategies and the real splitter
  (`get_cv_splitter`, `_CVWithGroups`) already exist in coco-pipe.
- `patient_group_id` bound through `CVConfig.group_key`, with the matching
  column supplied in `sample_metadata`/`groups` on `Experiment.run`.
- Shuffling with a fixed random seed.
- Grouped inner folds for any hyperparameter tuning (`TuningConfig`) and for
  fold-local feature selection (`FeatureSelectionConfig.cv`).

Note an important gap: **automatic fold-count reduction does not currently
exist in coco-pipe.** When there are too few independent groups per class for
the requested `n_splits`, coco-pipe raises an actionable error instructing the
caller to lower `cv.n_splits` (it does not silently downshift). For small
ADHD/epilepsy cohorts this matters. The first build resolved this with option
(b): `decoding_common.safe_group_n_splits` computes a safe `n_splits` in the
project before constructing `CVConfig`. That works, but the helper is generic
and leakage-critical, so the Post-Implementation Review flags it for promotion
into coco-pipe (option (a)) as a shared CV helper. Until then, every project
sweep must route through `safe_group_n_splits`; the doc should not describe
auto-reduction as built-in coco-pipe behavior.

Repeated recordings, sessions, conditions, epochs, or windows from one patient
must never appear in both training and test partitions. Fold validation should
fail before fitting if a patient overlaps partitions
(`coco_pipe.decoding.experiment._validate_fold_integrity` already exists for
this).

The default observation table may contain one row per recording. Primary
metrics should be computed at patient level by aggregating recording
probabilities when repeated observations are present, using the existing
`aggregate_predictions_for_inference` with `inferential_unit="subject"` (or a
`custom_unit_column` bound to `patient_group_id`). Recording-level results
remain available as secondary outputs.

### Metrics and Saved Artifacts

Save:

- Accuracy.
- Balanced accuracy.
- F1.
- Precision.
- Recall.
- ROC-AUC.
- Confusion matrices.
- Fold-wise metrics.
- Out-of-fold predictions, probabilities, labels, observation IDs, and group
  IDs.
- Exact split assignments.
- Selected features and fold-wise feature scores.
- Coefficients or feature importances.
- Selection and sign stability across folds.
- Effective model parameters and serialized coco-pipe results.

Suggested result layout:

```text
BIDS/derivatives/decoding/
└── <output_group>/<dataset_name>/<input_mode>/<analysis_mode>/<unit_key>/<selection_mode>/
    ├── result.joblib
    ├── summary.csv
    ├── fold_scores.csv
    ├── predictions.parquet
    ├── splits.parquet
    ├── confusion_matrices.csv
    ├── selected_features.parquet
    ├── feature_importances.parquet
    ├── feature_stability.parquet
    ├── config_used.yaml
    ├── run_manifest.json
    └── _SUCCESS
```

Use run-level `_SUCCESS`, `_PARTIAL`, or `_FAILED` markers and a failure table
so broad sensor and feature sweeps can resume without repeating completed
units.

## Lightweight Statistics

Most of this already exists in `coco_pipe.decoding.stats` and is configured via
`StatisticalAssessmentConfig` — the design should *configure* it rather than
reimplement:

- Exact binomial test for accuracy against the configured chance level —
  `binomial_accuracy_test` / `ChanceAssessmentConfig(method="binomial",
  p0="auto")`. Already present.
- Group-level bootstrap resampling at the patient level — `_bootstrap_scores`
  (1,000 resamples, group-aware) and `aggregate_predictions_for_inference`
  with `unit_of_inference="group_mean"`/`"group_majority"` or
  `inferential_unit="subject"`. Already present.

Default statistical assessment should remain inexpensive. Two caveats from the
current implementation:

- **Bootstrap CIs are not a free-standing cheap default.** The standalone
  `ConfidenceIntervalConfig` produces *analytical* intervals
  (`method="wilson"`/`"clopper_pearson"`); the 1,000-resample bootstrap is
  currently reached through the permutation-assessment path. The design must
  decide whether the cheap default CI is analytical (Wilson/Clopper-Pearson,
  already free) or whether coco-pipe should expose group bootstrap CIs
  independently of permutation. Recommended default: analytical CI for the
  sweep, group bootstrap CI reserved for the primary `flat` result.
- **Cross-analysis-unit FDR is net-new.** The existing
  `ChanceAssessmentConfig.temporal_correction="fdr_bh"` corrects *within* one
  result across temporal coordinates. Benjamini-Hochberg correction *across*
  the sensor/feature/family sweep units is a separate, project-orchestration
  concern (one correction family per target × input mode × analysis mode ×
  model × selection variant). This is genuinely net-new and is the only
  statistics item this design adds rather than configures.

Full-pipeline permutation tests (`run_statistical_assessment` permutation path)
should remain disabled by default. They may be enabled explicitly for a small
set of confirmatory analyses (the primary `flat` result, not the sweep) and
already rerun the complete pipeline, including selection and reduction.

## Foundation-Model Analysis on EEG

### Head-to-Head Comparison and Primary Mode

The most defensible foundation-model experiment for this project is not
"does fine-tuning a foundation model work" in isolation, but a **head-to-head
comparison under identical CV folds and grouping**: `descriptors` vs.
`foundation_embeddings` vs. `reduced_dimensions`, and frozen linear probing
vs. classical decoding on hand-crafted descriptors, for the same target,
patients, and `StratifiedGroupKFold` splits. This directly tests whether
generic pretrained EEG representations add value over the project's existing
descriptor pipeline for these clinical cohorts.

Given the small sample sizes typical of ADHD/epilepsy clinical EEG cohorts
(tens to roughly a hundred subjects), full fine-tuning and LoRA are
high-variance and prone to overfitting at this scale. **Linear probing is the
primary foundation-model result**; full fine-tuning and LoRA are secondary
"does additional adaptation help at all" checks, reported alongside but not
in place of the linear-probe result. This is consistent with the
capability-targets table and Development Order (linear probing, step 8,
precedes full fine-tuning and LoRA, steps 9-10), but should be stated
explicitly here as a scientific prioritization, not only an engineering
sequencing decision. Reports should present the descriptors-vs-embeddings
head-to-head and the linear-probe-vs-fine-tune/LoRA comparison as named,
labelled comparisons rather than leaving them to be inferred from separate
run outputs.

### Training Modes

Support:

- Linear probing: freeze the pretrained backbone and train only a linear
  classifier head.
- Full fine-tuning: train all supported model parameters.
- LoRA: add PEFT adapters to supported modules while keeping the base model
  frozen.

Use cleaned EEG windows as model inputs. Window examples may be used for
optimization, but all outer CV and validation splits remain grouped by
`patient_group_id`.

### Capability Targets

| Model | Backend | Linear probe | Full fine-tuning | LoRA |
| --- | --- | --- | --- | --- |
| CBraMod | Braindecode | Required | Required | Validate PEFT targets |
| LaBraM | Braindecode | Required | After channel adaptation | After channel adaptation |
| REVE | Hugging Face | Required | Required | Required |
| LUNA | Braindecode | Required | Validate loader arguments | Validate PEFT targets |

Observed v3 validation on June 10, 2026:

| Model | Linear probe | Full fine-tuning | LoRA | Checkpoint reload |
| --- | --- | --- | --- | --- |
| CBraMod | Available | Available | Available (`all-linear`) | Verified |
| LaBraM | Available after 19→128 interpolation | Available after interpolation | Available after interpolation (`all-linear`) | Verified |
| REVE | Authentication required | Authentication required | Authentication required | Pending gated-model access |
| LUNA | Available with constructor-defined head | Available | Available (`all-linear`) | Verified |

The public checkpoints produced the registered embedding dimensions (CBraMod
200, LaBraM 200, LUNA 256). LaBraM reported an interpolation matrix shape of
`128 x 19`. REVE remains deliberately unavailable without `HF_TOKEN`; its
registered 512-dimensional checkpoint revision was confirmed against the
official Hugging Face model API.

LaBraM channel adaptation must be explicit and tested. If a checkpoint requires
a fixed montage, coco-pipe must either perform a documented mapping or reject
the input; it must not claim interpolation support while enforcing only a
channel-count check.

LUNA should be registered with its supported Braindecode checkpoint and
architecture variant. Required constructor arguments and feature-extraction
behavior must be validated before enabling trainable modes.

### Graceful Unsupported Behavior

Run a preflight capability check for every model and requested mode. Record one
of:

- `available`
- `unsupported`
- `missing_dependency`
- `missing_checkpoint`
- `incompatible_channels`
- `incompatible_sampling_rate`
- `invalid_configuration`

The default `on_unsupported` policy is `skip`. Reports and manifests must state
the exact reason. Never silently downgrade LoRA or full fine-tuning to a linear
probe.

### Training and Evaluation

Within every outer training fold:

- Create a group-aware training/validation split.
- Adapt channels and sampling rate using only deterministic model
  requirements.
- Fit the requested head, full model, or adapters.
- Apply early stopping and retain the best validation checkpoint.
- Predict held-out windows.
- Aggregate probabilities by mean to recording and then patient level.

Patient-level metrics are primary. Window-level metrics are diagnostic only and
must not be used for inferential statistics.

Model-specific epochs, learning rates, batch sizes, warmup, gradient
accumulation, LoRA rank, target modules, precision, and early-stopping settings
belong in configuration. Defaults should be conservative and independently
overridable for each model and training mode.

Persist:

- Fold checkpoints or LoRA adapter weights.
- Training and validation histories.
- Best epochs and stopping reasons.
- Window-, recording-, and patient-level predictions.
- Fold metrics and aggregate statistics.
- Model/checkpoint/config provenance.
- Capability decisions and skipped combinations.

## Reports

Generate reports after every run using the verified
`coco_pipe.report.core.Report(title=...)` → `Section(name, icon=...)` →
`section.add_element(...)`/`add_markdown(...)` →
`report.add_section(section)` → `report.save(path)` pattern, with
`InteractiveTableElement`, `MetricsTableElement`, `BadgeElement`,
`AccordionElement`, `TabsElement`, `CalloutElement`, etc. imported from
`coco_pipe.report.elements` (not re-exported via `core`). Do not hand-build
HTML/JS or use raw `mne.Report`, matching the pattern already established by
`coco_pipe.report.descriptor_qc` (manifest-aware, `_SUCCESS`-marker-aware,
per-unit + dataset-aggregate reports with `InteractiveTableElement` for
sensor/feature breakdowns).

Project-side report assembly should reuse
`eeg_adhd_epilepsy/reports/_common.py` helpers (`add_optional_table`,
`add_images`, `add_image_list`, `format_value`, `build_record_metric_table`,
`build_dataset_mean_metric_table`, `build_flag_reason_table`,
`build_subject_overview_table`) for any per-record/per-unit metric tables,
rather than re-implementing table assembly. These helpers operate on plain
DataFrames/dicts and are framework-agnostic, so they compose with both
project-built sections and any coco-pipe report builder output.

Classical-decoding reports should be built primarily by calling the existing
`coco_pipe.report.decoding.make_decoding_report(result, ...)` factory (public
wrapper: `coco_pipe.report.api.from_experiment_result`), which takes an
`ExperimentResult` and already covers `overview`, `configuration`,
`provenance`, `model_summary`, `cv_summary`, `performance`, `statistical`,
`confusion_probability`, `features`, `fit_diagnostics`, `caveats`,
`export_inventory`, and (with `info`/`coords`) `topomaps`. It also accepts a
`qc_result` (`coco_pipe.io.quality.QCResult`) rendered before analysis
sections, and `feature_metadata` for sensor-map sections — both of which the
project already produces in its descriptor-loading path. The project module
should add only a thin **dataset-level** wrapper that supplies cross-unit
framing (sensor/feature sweep `InteractiveTableElement`, the primary-vs-sweep
split, the band-power-vs-aperiodic comparison, links to per-unit reports),
following `descriptor_qc`'s aggregate-report pattern, rather than rebuilding
per-run sections from scratch.

Embedding reports should include:

- Extraction coverage and failures.
- Model and checkpoint inventory.
- Input preprocessing and source derivatives.
- Channel adaptation and sampling-rate changes.
- Window counts, pooling, and embedding shapes.
- Artifact links and config/provenance information.

Classical-decoding reports should include:

- Dataset, target, label, and group summaries.
- CV integrity checks.
- Aggregate and fold-wise metrics with confidence intervals.
- Confusion matrices and ROC curves.
- Baseline versus top-10 comparisons.
- Sensor, feature, and family summaries.
- Coefficients/importances and feature-selection stability.
- Corrected statistical results and caveats.

Foundation-decoding reports should include:

- Model/mode capability matrix.
- Skipped combinations and reasons.
- Training curves and checkpoint inventory.
- Fold and patient-level metrics.
- Confusion matrices, ROC curves, and predictions.
- Compute, precision, and runtime provenance.

Suggested report locations:

```text
reports/summary/foundation_embeddings/<run_label>/
reports/summary/decoding/<output_group>/<dataset_name>/
reports/summary/foundation_decoding/<output_group>/<dataset_name>/
```

For large feature sweeps, generate one aggregate dataset report with links to
unit-level artifacts. Detailed per-unit HTML reports should be configurable to
avoid producing thousands of pages.

## Expected Result Directories

```text
BIDS/derivatives/eeg_foundation_embeddings/
BIDS/derivatives/decoding/<group>/<dataset>/<input>/<analysis>/<unit>/<selection>/
BIDS/derivatives/foundation_decoding/<group>/<dataset>/<model>/<train_mode>/
reports/summary/foundation_embeddings/<run>/
reports/summary/decoding/<group>/<dataset>/
reports/summary/foundation_decoding/<group>/<dataset>/
```

All roots should contain config snapshots, software provenance, manifests,
failure inventories, and explicit run status. Config hashes should prevent
incompatible partial outputs from being resumed accidentally.

## Development Order

1. Pin the project to a coco-pipe `dev` commit (not `viz`), document and
   install the foundation-model dependency extra (torch, braindecode,
   transformers, PEFT, skorch), and confirm the pinned braindecode version
   exports `LUNA` and `InterpolatedLaBraM`.
2. Register LUNA in `_specs.py`/the registry following the existing
   `cbramod`/`labram`/`reve` `FoundationModelSpec` pattern. The other three
   models' specs and `FrozenBackboneDecoderConfig`/`NeuralFineTuneConfig`
   already exist, so this step is scoped to LUNA plus any estimator-dispatch
   gaps surfaced while validating it.
3. Validate model loading, channel adaptation, windowing, and frozen feature
   extraction for all four models against real checkpoints (smoke-test
   `BackendBase.load(...).transform(...)`), with explicit attention to the
   LaBraM channel-adaptation path (`InterpolatedLaBraM` vs. the documented
   128-channel/`LABRAM_CHANNEL_ORDER` requirement) and LUNA's constructor
   arguments and feature-extraction behavior.
4. Add `coco_pipe.io.embeddings` (canonical embedding derivative I/O:
   `save_embedding_derivative`, `load_embedding_derivatives`,
   `run_manifest`/`_SUCCESS` handling) and an embedding-extraction report
   builder, following the `descriptor_qc` manifest-aware pattern and reusing
   `coco_pipe.report.core`/`elements`.
5. Add the project embedding-extraction CLI
   (`eeg_adhd_epilepsy.analysis.extract_foundation_embeddings`) and verify
   embedding-to-dimensional-reduction loading via the existing
   `dimensionality_reduction.py` machinery.
6. In coco-pipe, add only the confirmed-missing decoding pieces: `feature` and
   `feature_within_family` modes in `iter_analysis_units` (using the same
   `sensor`/`feature`/`feature_family` coordinate conventions the project's
   descriptor loading already produces), a fold-local `ReducerConfig`/PCA
   pipeline step in the `Experiment`, cross-analysis-unit BH-FDR, and (if
   chosen over project-side handling) automatic CV fold-count reduction.
   Grouped CV, fold-local `k_best` selection, the binomial test, group
   bootstrap, and patient-level aggregation already exist and only need to be
   configured and validated, not built.
7. Add project classical-decoding orchestration
   (`eeg_adhd_epilepsy.analysis.decoding`) and aggregate reports, building on
   the existing `coco_pipe.report.decoding` `ExperimentResult` builder plus
   `eeg_adhd_epilepsy/reports/_common.py` helpers for dataset-level framing.
8. Implement and validate linear probing in coco-pipe (the
   `BackendBase`/`FrozenBackboneDecoderConfig` plumbing already exists; this
   step validates it end-to-end with real checkpoints and grouped CV).
9. Add full fine-tuning where capability checks pass.
10. Add LoRA after model-specific adapter targets are tested.
11. Add opt-in real-checkpoint integration tests and runtime documentation.

This order makes frozen embeddings and classical decoding available before the
more expensive and model-specific training modes. Steps 2-6 are now scoped
tightly to confirmed gaps rather than broad "repair"/"extend" language, since
most of the surrounding infrastructure (FM backend ABC, configs, decoding
report builder) already exists on `dev`.

## Tests and Sanity Checks

### coco-pipe Tests

- Embedding NPZ/JSON round trips preserve arrays, axes, IDs, and metadata.
- Every artifact can be traced to its source EEG and checkpoint.
- Mocked extraction validates model-specific shape, pooling, channel, and
  window behavior without downloading checkpoints.
- LUNA resolves through the public registry.
- Frozen-backbone and neural configs resolve to executable estimators.
- Channel and sampling-rate incompatibilities fail during preflight.
- Old 10-20 names (T3/T4/T5/T6) are renamed to model vocabulary (T7/T8/P7/P8)
  before any model sees the data, and the mapping is recorded in the sidecar.
- A 19-channel input to a fixed-montage model (e.g. LaBraM 128) either adapts
  via the documented interpolation path or rejects with `incompatible_channels`
  — never silently zero-fills while reporting success.
- Feature and feature-within-family units select the expected columns.
- Scalers, PCA, selectors, and models are fitted only on training folds.
- Group-aware validation contains no patient overlap.
- Patient-level prediction aggregation is deterministic.
- Cross-analysis-unit BH-FDR groups tests into the correct family and matches a
  reference implementation.
- Unsupported training modes produce structured skip records.
- Embedding and decoding report builders render with partial and failed runs.

### Project Tests

- Synthetic BIDS layouts resolve source EEG and derivative paths correctly.
- Existing cohort filters and label mappings are preserved.
- Repeated recordings from one `patient_group_id` never cross folds.
- Top-10 selection handles units with fewer than ten columns.
- The project computes a safe `n_splits` (or coco-pipe auto-reduces) so a small
  grouped cohort does not crash the run, and the chosen fold count is recorded.
- A small descriptor-decoding run writes metrics, predictions, status, and an
  HTML report.
- A synthetic embedding derivative loads into dimensionality reduction and
  decoding.
- A fake foundation backend exercises probing orchestration without GPU or
  network access.
- Resume, overwrite, partial failure, and config-hash mismatch behavior match
  existing project conventions.

Real checkpoint tests for all four models should be opt-in because they require
network access, substantial downloads, and potentially a GPU.

## Acceptance Criteria

The design is successfully implemented when:

- All four model embeddings can be extracted through coco-pipe and reloaded
  without model-specific project code.
- Embedding artifacts contain the requested provenance and are organized as
  BIDS-compatible derivatives.
- Dimensionality reduction accepts those embeddings as a normal input.
- Classical decoding supports all requested feature groupings and produces
  leakage-safe baseline and top-10 results, with the primary `flat` result
  present and complete for each target even when some sweep units are skipped.
- The band-power (`bands`, incl. theta/beta) vs. aperiodic (`parametric`)
  comparison is produced and reported under identical CV and statistics.
- Metrics, statistics, selections, predictions, fold data, and importances are
  persisted and summarized in reports.
- Raw EEG foundation-model runs support every capability validated for each
  model and report unsupported modes explicitly.
- No repeated patient crosses a training, validation, or test boundary.
- Every completed or partial run produces a reproducible manifest and report.

## Assumptions

- "Subject-level embedding" means a pooled vector per source recording, with
  optional downstream aggregation by biological patient.
- `patient_group_id` remains the authoritative leakage-safe patient identifier.
- The default source is cleaned `desc-base` epoched EEG rather than untouched
  raw acquisition files.
- Frozen pretrained extraction is target-independent.
- The custom `embedding` suffix is BIDS-compatible derivative organization,
  not a standardized BIDS datatype.
- Fold-local PCA is the default interpretation of reduced-dimension decoding.
  Globally fitted reductions are explicitly transductive.
- Efficient `SelectKBest` is the initial top-10 selector; model-based or mutual
  information selectors may be added to coco-pipe later without changing the
  project orchestration contract.
- Exact binomial tests, group bootstraps, and FDR correction are sufficient
  default statistics. Expensive permutations are opt-in. Cheap default CIs are
  analytical (Wilson/Clopper-Pearson); group bootstrap CIs are reserved for the
  primary `flat` result unless coco-pipe exposes them independently of the
  permutation path.
- The project montage is a low-density ~19-channel 10-20 array (old T3/T4/T5/T6
  nomenclature). Foundation-model channel adaptation, especially for
  fixed-montage models, is a substantive modeling choice recorded per artifact,
  not a transparent preprocessing step.
- `flat` (all descriptors) is the primary/confirmatory analysis; the
  sensor/feature/family sweep is exploratory and labelled as such.



---

# ARCHIVED — v2 Post-Implementation Review

_Moved here from `foundation_models_and_decoding.md` when that file was reset to the v3 plan. This records the state after the two remediation passes; see the v3 doc for remaining work._

## Post-Implementation Review and Remediation

A first end-to-end build of this design now exists across both repositories
(coco-pipe `dev`, uncommitted; project, untracked). This section records what
that build got right, what is unproven or wrong, and the concrete remediation
order. It is the authoritative follow-up checklist — address these before the
build is considered trustworthy.

### Round 3 — Status After the Second Remediation Pass

A second build pass addressed essentially the entire checklist below; the
per-priority detail further down is now largely historical. Verified against the
current working trees (project 31 passed; coco-pipe 1005 passed, 4 skipped;
real-load suite opt-in via `COCO_PIPE_RUN_REAL_FOUNDATION=1`):

Resolved:

- **P1.1 fake-backend harness** — `foundation_models/testing.py`
  (`FakeFoundationBackend`, a real `BackendBase`) + `register_backend`/
  `unregister_backend`; `tests/test_foundation_orchestration.py` tests
  clone-safety across linear_probe/full/lora and asserts **no patient overlap
  between the internal train/val groups**.
- **P1.2 project FM-CLI tests** — `tests/test_foundation_workflows.py` drives
  both CLIs with the fake backend (artifacts, manifest, resume, config-hash
  guard) and injects a non-serializable `container.meta` to regression-test
  P1.3.
- **P1.3 sidecar serialization** — `_preprocessing_provenance()` whitelist
  replaces the raw `container.meta` dump.
- **P1.4 channel-adaptation honesty** — `_braindecode.get_channel_adaptation()`
  reads the real `InterpolatedLaBraM` layer; `check_capability` requires an
  explicit `interpolate_channels` and otherwise returns `incompatible_channels`.
- **P1.5 LUNA / specs** — LUNA validated and corrected (`embedding_dim=256`,
  `sfreq=200`, pinned `checkpoint_revision`,
  `checkpoint_filename="LUNA_base.safetensors"`, "verified base variant" note);
  REVE corrected to `reve-base`/512 with `requires_auth=True`; all four specs
  pinned to exact checkpoint revisions, plus `pretrained_n_times`.
- **P2 moves** — `ExperimentResult.export()` (per-table guards),
  `preparation.py` (`prepare_target`/`observation_frame`/`safe_group_n_splits`),
  `artifacts.py` (resume/status/config-hash + new `redact_sensitive`).
  `safe_group_n_splits` is now wired into the CV splitter via
  `CVConfig.auto_reduce_n_splits`, so **auto-fold-reduction now lives in
  coco-pipe**. `decoding_common.py` is a 66-line re-export shim.
- **P3 dedup** — `_prepare.py` `prepare_backend()`/`PreparedBackend` is the
  single load+resample path (used by the extractor and both estimators);
  project `decoding_runner.py` factors the shared sweep scaffolding; double
  cohort filtering removed; stale `_base.py` comment removed.
- **P4 polish** — reports default `asset_urls="inline"`; `SignalMetadata`
  exported from `coco_pipe.decoding`; windowing enforced (`PreparedBackend.adapt`
  rejects an `n_times` mismatch; new `incompatible_window` preflight);
  `require_conditions` + required `session_col`.
- **Project review** — per-unit reports gated behind `detailed_unit_reports`;
  subject/group conflation replaced by `custom_unit_column="group_id"`; top-k
  clamped by `_SafeSelectKBest`; explicit `positive_class` in the eval spec.
- **New capability statuses** beyond the original list:
  `authentication_required` (gated checkpoints) and `incompatible_window`.

Genuinely remaining:

- **REVE is gated and pending Hugging Face auth.** `brain-bzh/reve-base`
  requires login; preflight correctly returns `authentication_required` until
  then. Operational step: `huggingface-cli login` (or pass
  `backend_kwargs.token`), then confirm REVE loads and yields its 512-dim
  embedding. This is the one model not yet exercised end-to-end on this machine.
- **Run the opt-in real-load suite once with auth set** —
  `COCO_PIPE_RUN_REAL_FOUNDATION=1` against all four checkpoints, with explicit
  attention to LaBraM's 19→128 `InterpolatedLaBraM` path and LUNA's safetensors
  load. Specs are pinned/validated, but the gated/large-download path should be
  exercised once.
- **Per-model input-window length.** Each model enforces `pretrained_n_times`
  (e.g. LaBraM 3000 samples); a mismatched epoch length is now skipped with
  `incompatible_window` rather than implicitly padded. The example configs
  should set `segment_duration`/epoching per model so windows map to the
  required sample count after resampling, and document the chosen lengths.
- **Minor:** confirm the `extract_foundation_embeddings` status derivation has no
  unreachable `FAILED` branch (cosmetic).

### Build Status Snapshot

Landed and passing (mock/synthetic only):

- coco-pipe: `io/embeddings.py`, `report/foundation.py`,
  `foundation_models/extraction.py` (extractor + `check_capability` +
  `normalize_channel_names`), `foundation_models/_estimators.py`
  (`FrozenBackboneTransformer`, `FoundationClassifier`), LUNA spec, `feature`/
  `feature_within_family` units, `ReducerConfig` + fold-local PCA wired in
  `experiment.py` (`scaler → reducer → fs → clf`, cloned per fold),
  `benjamini_hochberg`/`correct_sweep_pvalues`, case-insensitive inference-unit
  lookup. 11 new tests pass.
- project: `analysis/decoding.py`, `analysis/foundation_decoding.py`,
  `analysis/extract_foundation_embeddings.py`, `analysis/decoding_common.py`,
  `reports/decoding.py`, `foundation_embeddings`/`reduced_dimensions` input
  modes in `io/analysis.py`. Classical path (`test_decoding_workflow`) passes.

Not exercised by any test, in either repo: every trainable foundation-model
path. The project `.venv` already has `torch`, `skorch`, and `pytest`, but is
missing `braindecode`, `transformers`, and `peft`, so no real foundation
backend has ever loaded. (The separate coco-pipe venv likewise lacks
`braindecode`.) The 11 passing coco-pipe tests are mock/synthetic only.

### Priority 0 — Environment and Testing

**Status: DONE.** `.venv` now has braindecode 1.5.2 (torch held at 2.10.0),
transformers, peft, accelerate, huggingface-hub, plus the test-only
`beautifulsoup4` and `pytest-cov`. All four target models import
(`CBraMod`, `Labram`, `InterpolatedLaBraM`, `LUNA` — so the LUNA *class* is
real; its spec specifics still need a real load, Priority 1.5). Baseline:
project 25 passed; coco-pipe 990 passed, 3 skipped. Use `.venv` for all testing.

Do this first; Priorities 1–4 depend on a working environment.

The project `.venv` is the single, canonical environment for **all** testing.
coco-pipe is editable-installed into it (`import coco_pipe` resolves to the
local `packages/coco-pipe` checkout), so the same interpreter runs both the
project's `tests/` and coco-pipe's `tests/`. Do **not** use coco-pipe's own venv
for testing — it lacks `braindecode` and will silently skip every real-backend
path.

Install the missing foundation-model dependencies into `.venv`. These match
coco-pipe's `[foundation]` extra; the `braindecode>=1.4.0,<1.6` pin is what
provides `LUNA` and `InterpolatedLaBraM` (the 19→128 LaBraM adaptation path):

```bash
# from the project root — torch, skorch, scipy are already present
# braindecode must be >=1.5 (not just >=1.4) for LUNA + InterpolatedLaBraM
.venv/bin/pip install "braindecode>=1.5,<1.6" transformers huggingface-hub peft accelerate
# bitsandbytes is only needed for qlora (not in the planned train modes) — skip for now
# equivalent, via the local editable coco-pipe checkout and its extra:
# .venv/bin/pip install -e "/Users/hamzaabdelhedi/Projects/packages/coco-pipe[foundation]"
```

Verify braindecode actually exports the models this design targets (this is the
Development Order step 1/3 check, and gates the LUNA validation in Priority 1):

```bash
.venv/bin/python -c "import braindecode; from braindecode.models import Labram, LUNA; print('braindecode', braindecode.__version__)"
.venv/bin/python -c "from braindecode.models import InterpolatedLaBraM; print('InterpolatedLaBraM OK')"
```

Run both test suites with the project `.venv` interpreter. Do **not** use a bare
`pytest` — on this machine it resolves to `/opt/homebrew/bin/pytest` (Homebrew
python, no project deps) and fails to collect (e.g. `ModuleNotFoundError: bs4`).
Either `source .venv/bin/activate` first, or call the interpreter explicitly:

```bash
.venv/bin/python -m pytest tests -q                                        # project
.venv/bin/python -m pytest /Users/hamzaabdelhedi/Projects/packages/coco-pipe/tests -q   # coco-pipe
```

### Implementation Sequence (agreed)

No pipeline coding has started yet. The agreed build order is:

1. **Real-load validation pass first (one-time, networked).** Before building
   anything else, load each of the four models from real checkpoints and run a
   single headless forward pass to confirm the registry specs and adaptation
   paths match reality:
   - CBraMod, REVE, LaBraM, LUNA: `load(model_key, n_outputs=None,
     train_mode="frozen")` → `transform(X)` on a small synthetic batch at the
     model's expected channel count / sfreq; record the real embedding dim and
     compare it to the spec.
   - LaBraM specifically with the project's 19-channel montage through the
     `InterpolatedLaBraM` (`interpolate_channels=True`) path — confirm 19→128
     actually runs, and capture what it does to the channels (this is the data
     needed to fix the hardcoded-empty adaptation accounting in Priority 1.4).
   - Correct every `FoundationModelSpec` against the observed values — above all
     LUNA (`PulpBio/LUNA`, `embedding_dim=384`, `pretrained_sfreq=250` are
     unverified and the dim is variant-dependent). Promote any guessed field to
     a verified value, or mark the model unsupported if it cannot load.
   Keep this as a standalone, opt-in script/test: it needs network and large
   downloads (possible Hugging Face auth) and must not run in default CI.

2. **Build the rest of the pipeline against the fake-backend harness.** Once the
   specs are verified, all remaining work (the fake harness itself, the project
   CLI tests, the sidecar fix, the Priority 2 moves, Priority 3 dedup, Priority 4
   polish) is developed and tested against a deterministic registered fake
   backend, so default CI needs no GPU or network. Real-checkpoint tests stay
   opt-in.

This isolates the one slow, networked, possibly-flaky step (real loads) as a
one-time validation, while day-to-day development stays fast and deterministic.

### Priority 1 — Correctness and Coverage (blockers)

These must be fixed before any foundation-model result is believable.

1. **Fake-backend orchestration test (project + coco-pipe).** The build's one
   fake (`FakeFoundationModel`) only covers the frozen extractor. Add a fake
   backend registered through the loader/registry that drives
   `Experiment` down both the `frozen_backbone` (linear probe) and
   `neural_finetune` dispatch paths, plus `FrozenBackboneTransformer` and
   `FoundationClassifier` directly (clone-safety under `sklearn.clone`,
   `fit(X, y, groups=...)`, grouped validation split, checkpoint save, history).
   This satisfies the existing but unmet "fake foundation backend exercises
   probing orchestration without GPU" requirement and is the single highest-value
   gap.
2. **Project tests for the two foundation CLIs.** `extract_foundation_embeddings`
   and `foundation_decoding` have zero coverage. Add synthetic-BIDS runs (using
   the fake backend) asserting artifacts, manifests, capability records, status
   markers, resume, and config-hash mismatch behavior.
3. **Sidecar serialization bug.** `extract_foundation_embeddings` injects
   `preprocessing_provenance: container.meta` into embedding metadata;
   `io/embeddings._json_value` returns unknown objects unchanged, so a non-trivial
   `container.meta` (MNE `Info`, `QCResult`, …) makes `json.dumps` raise at save.
   Either whitelist serializable provenance keys in the project before passing
   them, or make `_json_value` coco-pipe-side lossy-but-safe (stringify unknowns).
   Add a test with a non-serializable `meta` value.
4. **Honest channel-adaptation accounting.** `extraction.py` hardcodes
   `dropped_channels: []` and `zero_filled_channels: []`, and `check_capability`
   returns `available` for any interpolation-capable model regardless of the
   channel-count gap. For the project's 19→128 LaBraM case this records a clean
   adaptation that never happened. Populate the real mapping from the backend's
   interpolation step, and add a test asserting a 19→128 case reports the true
   interpolated/zero-filled channel sets (or rejects with
   `incompatible_channels`).
5. **LUNA is provisional.** `hub_repo="PulpBio/LUNA"`, `embedding_dim=384`,
   `pretrained_sfreq=250.0`, and the paper URL are unverified, and the embedding
   dim is variant-dependent. Treat the current entry as a placeholder: validate
   against a real braindecode checkout (Development Order step 3) and correct the
   spec, or mark LUNA `missing_checkpoint`/unsupported until validated rather than
   advertising it as available.

### Priority 2 — Move Project Logic into coco-pipe

`decoding_common.py` is ~260 lines of mostly generic orchestration that the
design's own division of responsibility assigns to coco-pipe. Move, leaving the
project with study-specific glue only (BIDS/derivative paths, cohort YAML,
target/condition specs, BIDS artifact naming, `get_reports_root`):

- `export_experiment_result` → `ExperimentResult.export(dir)` in coco-pipe
  (generic result serialization; the project currently calls ~12 `result.get_*`
  accessors by hand).
- `safe_group_n_splits` → coco-pipe CV helper. This is the auto-fold-reduction
  the design flagged as a coco-pipe candidate; it was built project-side instead.
  It is leakage-critical and fully generic — promote it and have `CVConfig`/the
  splitter consume it (update the Cross-Validation section, which still frames
  this as an open project-vs-coco-pipe decision — it is now decided by default,
  project-side, and should move).
- `prepare_target`, `observation_frame` → generic decoding prep / a
  `DataContainer` accessor. Keep only the cohort-YAML field names in the project.
- Resume/status/config-hash (`completed_for_config`, `write_run_status`,
  `load_completed_result_records`, `config_hash`) → a coco-pipe resumable-run
  utility. The `_SUCCESS`/manifest/config-hash pattern already exists in
  `coco_pipe.report.descriptor_qc`; reuse it instead of reinventing.

### Priority 3 — Deduplicate

1. **coco-pipe load/resample boilerplate (×3).** The `load()` + model `n_times`
   + `normalize_channel_names` + `setdefault("sfreq"/"n_times")` + `_adapt_X`
   resample block is copy-pasted across `FoundationEmbeddingExtractor.extract`,
   `FrozenBackboneTransformer.fit`, and `FoundationClassifier.fit`. Extract one
   private helper (e.g. `prepare_backend(...) -> (backend, bound_metadata,
   adapt_fn)`) and call it from all three.
2. **Project sweep scaffolding.** `decoding.py` and `foundation_decoding.py`
   share ~150 lines of identical run scaffolding (path build, `prepare_target`,
   sample-metadata munging, `safe_group_n_splits`, resume, export, status,
   summary + head-to-head reports). Factor into one project sweep runner; better,
   express the unit×variant sweep as a reusable coco-pipe construct so the
   project supplies only config + path/cohort specifics.
3. **Double cohort filtering.** `load_container` already applies `group_filters`
   and `filter_col`/`filter_val`; `decoding_common.apply_cohort_filters` then
   re-applies the same filters on the returned container. Idempotent but
   redundant and confusing — pick one filtering surface (preferably coco-pipe's
   `load_container`) and delete the other.
4. **Stale `_base.py` comment.** `BackendBase` says "Backends ARE the sklearn
   estimator — no wrapping layer is needed," which the clone-safe
   `_estimators.py` adapters contradict. Amend to: direct use only; a clone-safe
   adapter is used inside sklearn pipelines/CV.

### Priority 4 — Improvements and Polish

- **Self-contained reports.** Every report currently references external CDN
  scripts (Plotly/Tailwind/pako) and renders as grey boxes offline. Pass
  `asset_urls="inline"` by default for archival scientific derivatives, in both
  the project reports and coco-pipe's `report/foundation.py`.
- **Export `SignalMetadata` publicly** from `coco_pipe.decoding` (or
  `coco_pipe.io`) so the project stops importing it from `_specs`.
- **Windowing fidelity.** Decide explicitly: either honor the config's window
  duration/stride/overlap/remainder in the extractor, or document that windows
  are upstream epochs and drop those knobs from the embedding config and the
  Extraction Granularity section. Today they are silently ignored.
- **Provenance completeness.** The sidecar omits several fields the design lists
  (checkpoint hash, padding/remainder, optional layer/token arrays). Either
  populate them or trim the design's sidecar list to what is actually written.
- **Dead status branch.** In `extract_foundation_embeddings`, the `FAILED`
  status is unreachable (every failure also appends to `records`); simplify the
  status derivation.
- **Require, don't default, study assumptions.** Conditions
  (`EO_baseline`/`EC_baseline`) and `session="01"` are silently defaulted in
  several places; make them explicit config inputs so a misconfigured run fails
  loudly rather than mislabeling outputs.

### Project Implementation Review (code that stays in the project)

Priorities 1–4 mix coco-pipe and project items (Priority 2 is entirely project).
This subsection reviews the orchestration that should *remain* in the project on
its own terms — in-place correctness and quality, independent of any move to
coco-pipe.

In-place correctness and quality:

- **Per-unit report explosion contradicts the design.** `decoding.py` calls
  `make_decoding_report(...)` for every unit × selection unconditionally. With
  the six descriptor analysis modes × sensors/features × two selection variants
  this produces thousands of HTML files and steadily leaks matplotlib figures
  (the ">20 figures" warning) — exactly what the Reports section says to avoid.
  Gate detailed per-unit reports behind a config flag; always emit the
  dataset-level aggregate.
- **Positive-class encoding is implicit.** `prepare_target` integer-encodes via
  `sorted(unique(labels))`. It is correct only because the configs pass a
  numeric `label_map` (`{"Control":"0","ADHD":"1"}`); without one, alphabetical
  order silently decides the positive class and flips ROC-AUC / F1 / precision.
  Make the positive/target class explicit and validated.
- **`subject` is overwritten with `patient_group_id`.** Both decoding CLIs set
  `sample_metadata["subject"] = groups` so `inferential_unit="subject"`
  aggregates at patient level. It works but conflates two distinct entities;
  prefer `custom_unit_column="patient_group_id"` so the real subject coordinate
  is preserved.
- **top-10 after float PCA can fail a fold.** In `reduced_dimensions` + `top10`
  with a float (variance-fraction) `n_components`, `top_k` is clamped to
  `X.shape[1]` rather than the realized post-PCA component count, so
  `SelectKBest` can request more features than exist → fold failure. Clamp
  against the realized reducer width.
- **`export_experiment_result` is brittle.** It calls ~12 `result.get_*`
  accessors; only the parquet write is guarded. One missing or raising accessor
  fails the whole export and the `_SUCCESS` marker. Guard each table (this is
  also why the function should move to a coco-pipe `ExperimentResult.export`).
- **Formatting churn mixed into feature diffs.** The
  `dimensionality_reduction.py` change is mostly black/line-length reflow around
  a small `foundation_embeddings` addition, which obscures the substantive
  change. Keep formatting-only commits separate from feature commits.

Project test coverage (separate from the FM-CLI gap in Priority 1):

- The one end-to-end test monkeypatches `_load_scope`, so the project's own
  `load_container` glue — the part that legitimately stays in the project — is
  never exercised. It covers only descriptors / `flat` / one model.
- No project test covers the `foundation_embeddings` or `reduced_dimensions`
  input modes, the non-`flat` sweep modes, the top-10 selection edge cases, the
  skip/failure recording, or that `correct_sweep_pvalues` actually runs and
  writes `p_value_fdr`.

### Constraint Check

The build does **not** restore or duplicate the deleted project-local `ml/`/
`dl/` implementations: no architecture, training loop, scaler, selector, or
report renderer is reimplemented in the project; every path delegates to
coco-pipe backends and report builders. The hard constraint holds. The
remediation above is about correctness, coverage, and keeping the project thin —
not about a constraint violation.

## References

- [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)
- [Braindecode CBraMod](https://braindecode.org/stable/generated/braindecode.models.CBraMod.html)
- [Braindecode LaBraM](https://braindecode.org/stable/generated/braindecode.models.Labram.html)
- [Braindecode LUNA](https://braindecode.org/stable/generated/braindecode.models.LUNA.html)

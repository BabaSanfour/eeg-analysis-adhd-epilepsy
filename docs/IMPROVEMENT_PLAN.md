# Run-flow Improvement Plan — Two-config pipeline + end-to-end ergonomics

> **Durable, resumable tracker.** This is the source of truth for the
> run-flow/ergonomics improvement effort. A fresh session should be able to read
> this file and continue cold. Update the checkboxes as each step lands.
>
> - **Branch:** `improve/two-config-run-flow`
> - **Started:** 2026-06-21
> - **coco-pipe pin:** keep `@dev` for now (no pin change this pass).
> - **Commits:** small, reviewable, `pytest` green at each step. Branch off `main`.

## Status / where we are

- [x] Phase 1 analysis + Phase 2 plan agreed with the user.
- [x] Branch created; this tracking doc written.
- [x] **Wave 1 complete** — two-config split, migration (74 cohorts + 4 analyses),
  validation, cluster glob fix. 82 tests green.
- [ ] Wave 2 — orchestrator, CLI consistency, structure. *(next: W2.3 console scripts, then W2.1 orchestrator)*
- [ ] Wave 3 — docs & provenance.

Leave a one-line "Resume note" at the bottom whenever you stop mid-step.

---

## Context (why)

The repo is a 9-stage EEG pipeline (raw → BIDS → preproc → epochs →
descriptors/embeddings → dim-reduction/decoding → reports) on `coco-pipe`. The
science is sound; the *run flow* has gaps. The central one: every analysis config
**conflates two independent concerns** — *which cohort/dataset* and *which
analysis*. Proof: across the 71 dim-reduce configs the method fields (`reducers`,
`n_components_sweep`, `selection_*`, viz flags) are byte-identical; only cohort
fields vary. This causes config duplication, a real cluster bug (decoding configs
get globbed into dim-reduce array jobs with stale `--array` bounds), and a
confusing tree.

**Target:** a clean **two-config** contract (each analysis script takes a cohort
config *and* an analysis config) + a top-level orchestrator so the real dataset
runs end-to-end without hand-chaining ~9 CLIs. Not building synthetic data; not
over-tailoring for new users.

## Stage DAG (current, for reference)

```
eeg-build-patients-metadata (io/patients.py; --adhd_csv --drug_resistant_csv --output_dir)
  → patients_metadata{,_clean}.csv  →  eeg-cohort-report (analysis/cohort.py)  [sidecar]
preproc.to_bids  (python -m; → eeg-to-bids)   raw+metadata → BIDS + *_segments.csv + pre-base reports
preproc.base     (python -m; → eeg-preprocess) BIDS → derivatives/preproc/*_desc-base_eeg.fif + base_qc
eeg-save-epochs  (preproc/epochs.py)          desc-base → task-<cond>_*_desc-base_epo.fif
eeg-descriptors  (analysis/extract_descriptors.py; SLURM array per metadata row) → descriptor shards
eeg-merge-descriptors (analysis/merge_descriptors.py) → descriptors/combined/*_features.{parquet,csv}
  ├ eeg-dim-reduce            (analysis/dimensionality_reduction.py)
  ├ eeg-foundation-embeddings (analysis/extract_foundation_embeddings.py)
  ├ eeg-classical-decode      (analysis/classical_decoding.py)   [renamed from eeg-decode]
  └ eeg-foundation-decode     (analysis/foundation_decoding.py)
```

## Decisions (locked)

- **Two-config API**: consumers `eeg-dim-reduce`, `eeg-classical-decode`,
  `eeg-foundation-decode` take `--cohort_config` + `--analysis_config`. Producers
  (`eeg-descriptors`, `eeg-foundation-embeddings`, `eeg-to-bids`, `eeg-preprocess`,
  `eeg-save-epochs`) stay single-config (dataset-wide, before any cohort split).
- **Field ownership**: `evals`/`label_map`/`group_filters` → cohort.
  `conditions` default in cohort, **overridable** by analysis config.
- **Layout**: `configs/cohorts/...` + `configs/analyses/...`.
- **Migration**: one-shot script auto-splits all 71+3 existing configs; verify; remove old tree.
- **Orchestrator**: Makefile + small Python driver (`eeg-run`); wraps CLIs, never removes them.
- **Renames**: `eeg-decode`→`eeg-classical-decode`; `io/analysis.py`→`io/containers.py`;
  add `eeg-to-bids`/`eeg-preprocess`; reconcile package-level `eeg_adhd_epilepsy/README.md`.
- **cohort.py**: light internal cleanup only (no subpackage split).
- **Deferred (DO NOT TOUCH this pass)**: experimental Part-2 preproc pipeline —
  `preproc/{correct,denoise,compare,run_all}.py`, `preproc/ARTIFACT_STRATEGIES.md`.

## Two-config contract (key ownership)

| Concern | Config | Keys (representative) |
|---|---|---|
| Dataset identity & paths | **cohort** | `bids_root`, `metadata`, `dataset_name`, `output_group`, `subject_col`, `session_col`, `group_col` |
| Population & question | **cohort** | `group_filters`, `filter_col`, `filter_val`, `run_pooled`, `evals` (+`label_map`,`positive_class`), `conditions` (default) |
| Method & hyperparameters | **analysis** | dim-reduce: `reducers`,`n_components_sweep`,`selection_metric`,`selection_eval_name`,viz flags · decoding: `models`,`cv`,`feature_selection`,`tuning`,`chance_method`,`n_permutations`,`metrics` · foundation: `models`,`train_modes`,`training_defaults`,`class_weight`,`device`,`precision`,`window_mismatch_policy` |
| Input shaping | **analysis** | `input_mode`, `descriptor_table_path`, `descriptor_feature_columns_path`, `location_statistic`, `embedding_*`, `use_derivatives`, `task`, `desc`, `segment_duration`, `qc`, `conditions` (override) |
| Run controls | **analysis** | `n_jobs`, `random_state`, `overwrite`, `report_asset_urls`, `detailed_unit_reports`, `verbose` |

**Merge**: load cohort, load analysis, deep-merge with **analysis overriding
cohort** on overlap. Result = the single dict existing `run()` functions consume,
so `run()` bodies stay essentially unchanged; only the CLI boundary changes.
**Cross-ref check**: analysis `selection_eval_name` must name an `evals` entry in
the cohort config — actionable error otherwise.

**Path refinement (locked during W1.1):** `bids_root` / `metadata` are
dataset-level and stay **out of both configs** — supplied via CLI/env (the 71
dim-reduce configs already omit them; cluster passes `BIDS_ROOT`/`METADATA_PATH`).
The 3 decoding configs that hardcode author paths get them stripped during
migration. Consumer CLIs accept `--bids_root`/`--metadata` (+ a few run-control)
overrides layered onto the merged dict before `run()`. So cohort-config required
keys = `dataset_name`, `output_group`, `evals` (no paths). Tests call `run(dict)`
directly, so CLI changes don't touch them.

---

## Wave 1 — Two-config split, migration, validation, cluster fix

- [x] **W1.1** New `eeg_adhd_epilepsy/utils/config.py` with
  `load_cohort_analysis_config(cohort_path, analysis_path)` → validated, merged
  dict (reuses `utils/yaml.py:load_yaml_config`). Also `apply_overrides()` for
  CLI/env path layering.
- [x] **W1.2** Validation in `utils/config.py`: required-key checks per role +
  eval cross-reference + method-marker check, each with an actionable message
  (`ConfigError`). Tests: `tests/utils/test_config.py` (8 passing).
- [x] **W1.3** Wired `--cohort_config` + `--analysis_config` (+ `--bids_root`/
  `--metadata`/`--overwrite`[/`--n_jobs`] overrides; `--config` kept as deprecated
  fallback) into `classical_decoding.py`, `foundation_decoding.py`,
  `dimensionality_reduction.py` via `utils/config.resolve_cli_config` /
  `load_cohort_analysis_config`. `run()` bodies untouched. 82 tests green, ruff clean.
- [x] **W1.4** Migration script `scripts/split_configs.py` (dry-run default,
  `--apply`). Split all 74 legacy configs → **74 cohort** files under
  `configs/cohorts/medicated_adhd_vs_controls/...` + **1 shared**
  `configs/analyses/dim_reduction/default.yaml` + **3** bespoke
  `configs/analyses/decoding/{EO,EO_EC_baseline_only,EO_amph_vs_mph}.yaml`
  (the 3 decoding files differ on both axes → distinct cohort *and* analysis).
  Verified each pair round-trips to the original effective config (minus dropped
  paths) and validates. Old `configs/medicated_adhd_vs_controls/` removed.
  *Note:* foundation_decoding/foundation_embeddings example configs are still
  single-file; they'll be split/documented in W3 (foundation_embeddings is a
  producer = stays single).
- [x] **W1.5** `cluster/06`, `cluster/07`: glob `configs/cohorts` (cohorts only,
  no analysis configs), pair each with `ANALYSIS_CONFIG`
  (default `dim_reduction/default.yaml`) via `--cohort_config`/`--analysis_config`;
  added a guard that errors when `SLURM_ARRAY_TASK_COUNT != cohorts*modes` so a
  stale `--array` bound fails loudly. 74 cohorts → 148 (06) / 740 (07), matching
  the existing bounds. *Nuance:* the 3 decoding-specific cohorts are in the swept
  tree; the amph cohort lacks `med_adhd_vs_ctrl`, so dim-reduce's selection eval
  makes those 2 tasks fail loudly (validator working) — narrow with `CONFIGS_DIR`
  if undesired.
- [x] **W1.6** `tests/utils/test_config.py` (8) green; full suite 82 green; ruff clean.

## Wave 2 — Orchestrator, CLI consistency, structure

- [x] **W2.1** `eeg_adhd_epilepsy/run.py` + console script `eeg-run`: sequences
  the 7 core stages (to-bids→…→classical-decode) with `--from/--to`, `--dry-run`,
  `--list`, resume-by-default (per-stage output globs), and per-stage
  missing-input SKIP. Tests: `tests/test_run_orchestrator.py` (6).
- [x] **W2.2** Top-level `Makefile`: `help`/`install`/`test`/`dry-run`/`all` +
  per-stage targets (core delegate to `eeg-run`; foundation/cohort/metadata call
  modules directly). Centralized vars; `$(if …)` so unset paths aren't passed;
  `PYTHON`/`EEG_RUN` keep it PATH-independent. 92 tests green; ruff clean.
- [x] **W2.3** `pyproject.toml [project.scripts]`: added `eeg-to-bids`,
  `eeg-preprocess`; renamed `eeg-decode` → `eeg-classical-decode`; grouped
  producers vs consumers. `pip install -e .` refreshed scripts (old `eeg-ml-run`/
  `eeg-decode` gone; previously-missing `eeg-foundation-*` now present).
  `resolve_cli_config` now raises an actionable error when `bids_root` is unset.
  Tests up to 12. *(README/cluster refs to `eeg-decode` → updated in W3.1.)*
- [x] **W2.4** Cluster scripts `08_submit_merge_descriptors.sh`,
  `09_submit_classical_decode.sh`, `10_submit_foundation_embeddings.sh`,
  `11_submit_foundation_decode.sh` (decoding ones take `COHORT_CONFIG`+
  `ANALYSIS_CONFIG`); `cluster/README.md` (numbered order, env vars, array math,
  two-config pairing, GPU/HF notes). Syntax-checked.
- [x] **W2.5** Added `io/__init__.py`, `preproc/__init__.py` (documented surfaces).
- [x] **W2.6** Renamed `io/analysis.py` → `io/containers.py`; updated all 5 import
  sites; package imports + 92 tests green.
- [ ] **W2.7** `analysis/cohort.py` light cleanup: section structure + docstring
  map; document relationship to `reports/cohort_report.py` and `io/patients.py`.
- [ ] **W2.8** Reconcile package-level `eeg_adhd_epilepsy/README.md` (one canonical README).

## Wave 3 — Docs & provenance

- [ ] **W3.1** README rewrite around the two-config model: corrected
  `eeg-build-patients-metadata` signature (`io/patients.py:445-447`); `pip install
  -e .` reinstall note (stale env still had `eeg-ml-run`); fixed Repository-Layout
  block (drop nonexistent `data/`,`results/`,`reports/`); pipeline-order sentence
  incl. decoding/foundation; a "Prerequisites" block (braindecode≥1.5, HF auth for
  REVE, optional GPU); a "point it at your data" subsection; a "Module map" section.
- [ ] **W3.2** `configs/README.md`: cohorts vs analyses trees, which CLI consumes
  which, copy-an-example-and-edit workflow.

## Verification (run before marking a wave done)

- `pytest` green after every wave (the gate).
- `eeg-classical-decode --cohort_config <c> --analysis_config <a>` and the
  dim-reduce / foundation equivalents run; a missing/misspelled key gives an
  actionable error, not a `KeyError`.
- `eeg-run --dry-run --from bids --to decode` prints correct ordered commands.
- Re-derive cluster array math post-W1.5; no task silently dropped.
- `ruff check` clean on touched files; every touched `--help` still works.
- Spot-check a migrated cohort+analysis pair reproduces the same effective config
  as the pre-split file (diff the merged dict).

---

## Resume note

_(Last stop:)_ **Wave 1 done & committed** (3 commits on
`improve/two-config-run-flow`: config helper, CLI wiring, migration+cluster).
74 cohort + 4 analysis configs live under `configs/{cohorts,analyses}/`; legacy
tree deleted; 82 tests green; ruff clean. Next action: **Wave 2** — start with
W2.3 (add `eeg-to-bids`/`eeg-preprocess`, rename `eeg-decode`→`eeg-classical-decode`
in `pyproject.toml`), then `pip install -e .`, then W2.1 orchestrator (`run.py`).

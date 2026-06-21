# Backlog / ideas

Forward-looking items (was the stray `eeg_adhd_epilepsy/README.md`; moved here so
the top-level `README.md` is the single canonical README).

## Scripts
- Add comprehensive documentation for each script.
  - *Partially addressed:* the two-config interface + `eeg-run`/`Makefile`
    orchestration and the README "Module map" now document the run flow.

## Analysis
- ~~Develop an automated analysis workflow over conditions/cohorts/medication
  driven by config files.~~ **Done** — the cohort/analysis two-config split
  (`configs/cohorts/` × `configs/analyses/`) plus `eeg-run` cover this.
- Incorporate additional features.
- Port multivariate outlier detection (Isolation Forest/LOF) to `coco-pipe.qc`.
- Implement raw vs. FOOOF band power consistency checks in the descriptor QC pipeline.
- Add global missingness and outlier heatmaps (Subjects × Features) to automated reports.
- Port KDE-based bimodality detection to `coco-pipe.qc`.
- Incorporate advanced distribution plots (QQ, Parallel Coordinates) into the report stack.

## Deferred (next phase)
- Wire in / document the experimental Part-2 artifact-correction pipeline
  (`preproc/{correct,denoise,compare,run_all}.py`, `ARTIFACT_STRATEGIES.md`).

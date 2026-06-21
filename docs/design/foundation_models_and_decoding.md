# Foundation Models and Decoding — v3 Iteration Plan

The original design (v1) and the two remediation passes with their full
Post-Implementation Review (v2) are archived in `docs/design/old.md`. This
document tracks only the work that **remains** for the v3 iteration.

## Status going in

- Both test suites are green: project **31 passed**; coco-pipe **1005 passed,
  4 skipped** (real-load suite is opt-in via `COCO_PIPE_RUN_REAL_FOUNDATION=1`).
- All v2 remediation (P1–P4) and the project-level review items are resolved.
  Now in place: the `FakeFoundationBackend` orchestration harness (clone-safety
  + no-patient-overlap-in-val tests across linear_probe/full/lora);
  verified-and-pinned `FoundationModelSpec`s for CBraMod/LaBraM/REVE/LUNA (exact
  checkpoint revisions, `pretrained_n_times`, `requires_auth`);
  `ExperimentResult.export`; coco-pipe `CVConfig.auto_reduce_n_splits`;
  `_prepare.prepare_backend` (single load+resample path); `_SafeSelectKBest`;
  graceful `authentication_required` / `incompatible_window` preflight;
  `redact_sensitive`; `_preprocessing_provenance` whitelist; gated per-unit
  reports with `asset_urls="inline"`.
- **v3 implementation outcome (June 10, 2026):** CBraMod, LaBraM, and LUNA
  passed real-checkpoint extraction plus one-epoch linear/full/LoRA training
  with checkpoint restoration. REVE remains explicitly
  `authentication_required` until gated-model access is supplied. Real
  descriptor-family and matched six-subject head-to-head runs were completed.
  The engineering changes are ready to commit; exact coco-pipe pinning follows
  the coco-pipe commit.

The v3 work is mostly *validation on real data* and *running the actual
experiments* — not new infrastructure.

## v3.1 closeout (implemented this session)

- **LaBraM is now included in cohort analysis via cleaned-continuous re_epoch.**
  New `io/bids.load_cleaned_continuous_container` re-epochs the **cleaned
  continuous `desc-base` derivative** at 15 s (reusing `load_stage_artifacts` +
  `preproc.epochs.make_epochs_from_preproc_raw`, block-aware EO/EC cropping),
  wired through `load_eeg_data(window_source="re_epoch")` and both CLIs;
  `configs/foundation_*.example.yaml` flip LaBraM from `skip` to
  `window_source: re_epoch`. This avoids the earlier trap where `re_epoch`
  loaded **raw, uncleaned** acquisition.
  - **Scientific caveat (recorded in `meta['autoreject_applied']=False`, the
    config comments, and the embedding sidecar):** 15 s LaBraM windows get the
    same continuous-stage cleaning (filter/ICA/interpolation) as the other
    models but **skip the per-epoch autoreject rejection** the 10 s `_epo`
    derivative received.
  - **Needs a real-data smoke test (only you can run):** per-subject
    `desc-base_eeg.fif` discovery + `metadata_df` join on real BIDS. The
    transform + container assembly are covered by a synthetic-`Raw` test
    (`tests/test_foundation_reepoch.py`); the file-discovery/metadata path is
    not. Single-session is assumed (`load_stage_artifacts` resolves one file
    per subject/desc/task).
- **Validation montage is now parameterizable** — `validate_real_checkpoints`/
  `validate_real_training` take `ch_names` (+ a `--channels` CLI flag), so the
  real cohort montage (old-named, ~17–19 ch) can be validated, giving the true
  `128 × N` LaBraM interpolation rather than the idealized 19-ch default.
- Suites green: project 44 passed; coco-pipe 1008 passed, 5 skipped.

## 1. Real-checkpoint validation (gated, networked) — do first

This is the gate. Until it passes, every real-data result is untrusted.

- [ ] `hf auth login` — REVE (`brain-bzh/reve-base`) is gated
      (`requires_auth=True`); without it preflight returns
      `authentication_required` and REVE is skipped.
- [x] Run the opt-in suite:
      `COCO_PIPE_RUN_REAL_FOUNDATION=1 .venv/bin/python -m pytest /Users/hamzaabdelhedi/Projects/packages/coco-pipe/tests/test_decoding_foundation_orchestration.py -q`
- [ ] Confirm per model: loads at the pinned revision, `transform` returns the
      documented embedding dim (CBraMod 200, LaBraM 200, REVE 512, LUNA 256),
      and a forward pass runs on a small synthetic batch.
- [x] LaBraM specifically: exercise the 19→128 `InterpolatedLaBraM` path and
      confirm `get_channel_adaptation()` reports the real interpolated channels
      and matrix shape (not empty lists).
- [x] Reconcile any drift between observed values and the specs; re-pin
      `checkpoint_revision` if a model has published a newer one.

## 2. Per-model input-window configuration

Models now enforce `pretrained_n_times`; a mismatched epoch length is skipped
with `incompatible_window` rather than implicitly padded/cropped.

- [x] Determine the required window length per model at its sfreq (e.g. LaBraM
      = 3000 samples) and set `segment_duration` per model in the example
      configs; document the chosen lengths.
- [x] Decide and encode the policy when the cohort's natural epoching cannot
      satisfy a model (skip that model vs. re-epoch the source) so a run fails
      loudly rather than silently dropping a model.

## 3. Trainable-mode validation per model (was Development Order 9–10)

Dispatch, clone-safety, and the grouped val split are covered by the fake
backend; real per-model training is not.

- [x] Full fine-tuning: confirm each accessible model trains end-to-end on a tiny
      real-checkpoint smoke run; checkpoints save and restore.
- [x] LoRA: validate PEFT target modules per accessible model (CBraMod and LUNA = "validate
      PEFT targets"; LaBraM after channel adaptation; REVE full/lora/qlora).
- [x] Materialize the capability matrix from a real run and update the design's
      Capability Targets table (in `old.md`) to observed reality — which
      (model, train_mode) pairs are actually `available` vs skipped, and why.

## 4. Scientific runs (the actual experiments)

Run on real cohort data once 1–3 pass. **Linear probing is the primary
foundation-model result;** full fine-tuning and LoRA are secondary checks.

- [x] Head-to-head under identical grouped CV: `descriptors` vs
      `foundation_embeddings` vs `reduced_dimensions`, and linear-probe vs
      classical-on-descriptors, per target; render as a named report section.
- [x] Descriptor plan restricted to `flat`, `sensor`, `subfamily`,
      `sensor_within_subfamily`, `feature`, and `feature_sensor`.
- [x] Baseline is always run; optional forward SFS is fold-local and never runs
      for one-column `feature_sensor` units.
- [x] Reports are grouped by condition, with `pooled` last, and show the full
      `flat` analysis before the narrower analyses.
- [ ] (Optional, from the v1 scientific suggestions) within-subject
      medication-state decoding; cross-condition (EO↔EC) generalization.

## 5. Config, report, and provenance polish

- [x] Fill the three example configs with the fields actually consumed now:
      per-model `segment_duration`, `positive_class`, `session_col`,
      `detailed_unit_reports`, `report_asset_urls: inline`,
      `allow_transductive_input`, and model `backend_kwargs`
      (incl. `interpolate_channels: true` for LaBraM).
- [x] Confirm saved reports render offline (inline assets) and that
      `redact_sensitive` strips any HF token from `config_used.yaml`.
- [x] Audit the `FAILED` branch in `extract_foundation_embeddings`. It is
      reachable when every requested extraction fails or is skipped, so it was
      retained rather than removed.

## 6. Engineering hygiene / reproducibility

- [ ] **Git baseline (overdue):** branch + commit the currently-uncommitted work
      in coco-pipe `dev` and in the project, so there is a revert point and a
      clean diff history.
- [ ] Update the project `pyproject.toml` coco-pipe pin from `@viz` to the exact
      `dev` commit that contains this work, with the `[foundation]` extra. The
      editable install currently masks the stale pin; a fresh environment would
      break.
- [x] Document runtime requirements (braindecode>=1.5, Hugging Face auth for
      REVE, optional GPU) in the project setup/README.

## 7. v3 acceptance

- [ ] All four models load at their pinned revisions and produce embeddings of
      the documented dimensions on real checkpoints.
- [x] LaBraM 19→128 adaptation is exercised and honestly recorded in the
      sidecar.
- [ ] Linear probing is validated end-to-end per model on real checkpoints under
      grouped CV; the capability matrix reflects observed reality.
- [x] The head-to-head and band-power-vs-aperiodic comparisons run on real data
      and appear as named report sections.
- [ ] The work is committed and the coco-pipe pin is updated; a fresh `.venv`
      reproduces both green suites (plus the opt-in real-load suite with auth).

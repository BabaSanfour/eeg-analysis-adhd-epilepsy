"""EEG preprocessing sub-package.

Canonical path (run in this order)
----------------------------------
- ``to_bids``  — raw EEG + metadata → BIDS, canonical annotations, pre-base QC (eeg-to-bids)
- ``base``     — BIDS → cleaned ``desc-base`` derivatives + post-clean QC (eeg-preprocess)
- ``epochs``   — ``desc-base`` → condition epochs (eeg-save-epochs)

Experimental "Part 2" artifact-correction pipeline (not wired into the canonical
run; see ``ARTIFACT_STRATEGIES.md``): ``correct``, ``denoise``, ``compare``,
``run_all``.
"""

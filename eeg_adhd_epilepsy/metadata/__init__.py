"""Clinical patient-metadata concern for the EEG ADHD/Epilepsy study.

This package owns everything about the canonical patient-metadata table — from
building it out of the raw clinical CSVs to analysing the assembled cohort.

Modules
-------
schema
    Shared schema constants for the canonical patient-metadata tables
    (column sets, medication categories, audit columns).
patients
    Builds the canonical metadata CSVs from the raw ADHD and drug-resistant
    sources (``eeg-build-patients-metadata`` entry point).
cohort
    Cohort-level analysis of the clean metadata: summary statistics,
    analysis-opportunity enumeration, and the HTML report
    (``eeg-cohort-report`` entry point).

Presentation lives elsewhere: :mod:`eeg_adhd_epilepsy.reports.cohort_report`
composes the HTML and :mod:`eeg_adhd_epilepsy.viz.patients` draws the figures.
"""

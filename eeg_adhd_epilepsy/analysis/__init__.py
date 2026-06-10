"""Analysis pipeline scripts for the EEG ADHD/Epilepsy study.

This package contains the study-level pipeline entry points that sit on top of
``coco-pipe``.  Each module is a self-contained CLI script (``python -m`` or
console-script entry point) with no public Python API.

Modules
-------
extract_descriptors
    Load BIDS epoched derivatives, run the ``coco-pipe`` descriptor pipeline,
    apply MAD epoch rejection, and write per-subject shards under
    ``<bids_root>/derivatives/signal_features/descriptors/``.
merge_descriptors
    Discover completed shards, verify feature-column consistency across all of
    them, concatenate into combined tables, and run dataset-level descriptor QC.
dimensionality_reduction
    Checkpointed dimensionality-reduction analysis: fit reducers (PCA, UMAP,
    PHATE, …) across analysis units (flat / sensor / family /
    sensor_within_family), evaluate embeddings, and generate HTML reports.
cohort
    Cohort-level patient report: summary statistics, analysis-opportunity
    enumeration, and optional recruitment-milestone projection.
"""

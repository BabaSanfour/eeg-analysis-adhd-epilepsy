
## To-Do (Scripts)
- Add comprehensive documentation for each script.

## To-Do (Analysis)
- Incorporate additional features.
- Develop an automated analysis workflow that tests different conditions (e.g., with/without specific factors like ADHD, Epilepsy, TSA), considers age groups, and analyzes different medication types. This may include generating selections via a JSON configuration file.
- Port multivariate outlier detection (Isolation Forest/LOF) to `coco-pipe.qc`.
- Implement raw vs. FOOOF band power consistency checks in the descriptor QC pipeline.
- Add global missingness and outlier heatmaps (Subjects × Features) to automated reports.
- Port KDE-based bimodality detection to `coco-pipe.qc`.
- Incorporate advanced distribution plots (QQ, Parallel Coordinates) into the report stack.

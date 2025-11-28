
## To-Do (Scripts)
- Add comprehensive documentation for each script.
- Automate ML functions and pipeline steps to support various datasets and CSV files.
- Integrate additional machine learning analyses.
- Merge pipeline components into a single class and consolidate the run_ml_pipe function into one class that returns a unified object. This object will maintain consistent keys (with empty values when results are unavailable) to simplify visualization.
- Enhance the reve_model to automate processing across different datasets and prepare outputs for both visualization and ML pipeline analyses.
- Create a visualzation class for decoding results and another for embeddings (umap, tsne, pca)

## To-Do (Analysis)
- Incorporate additional features.
- Conduct further experiments with the reve_model to diagnose and resolve issues (Test the reve_model on sleep and eyes open/closed EEG data; Fine tune the model on the data?).
- Develop an automated analysis workflow that tests different conditions (e.g., with/without specific factors like ADHD, Epilepsy, TSA), considers age groups, and analyzes different medication types. This may include generating selections via a JSON configuration file.
- Normalize the data before feeding it to REVE!
- Test finetuning strategies (REVE)!
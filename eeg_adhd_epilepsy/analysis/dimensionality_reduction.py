#!/usr/bin/env python3
"""
End-to-end Dimensionality Reduction Analysis for EEG Data.

Workflow:
1. Load EEG data from BIDS directory (Raw or Specific Segments).
2. Stack/Flatten data based on strategy (flat, time_as_sample, epoch_scalar_mean).
3. Apply dimensionality reduction using coco-pipe (selected supported reducers).
4. Generate interactive HTML reports with embeddings and diagnostics.

Available Reducers:
- pca, umap, phate, isomap

Available Visualizations:
- Embeddings (2D/3D), loss history
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Project imports
from eeg_adhd_epilepsy.utils.config import results_dir

# Coco-pipe imports
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.report.core import Report, Section, PlotlyElement, ImageElement, TableElement
from coco_pipe.viz import dim_reduction as viz
from eeg_adhd_epilepsy.analysis.utils import (
    add_embedding_plot,
    apply_representation,
    build_meta_dict,
    coerce_sample_vector,
    REPRESENTATION_CONFIG,
)
from eeg_adhd_epilepsy.io.bids import load_eeg_data, validate_bids_coverage
from eeg_adhd_epilepsy.io.csv import load as load_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default reducers for this analysis profile
ALL_REDUCERS = [
    "PCA", "UMAP", "PHATE", "ISOMAP"
]

DEFAULT_CONDITIONS = [
    "EO_baseline",
    "EC_baseline",
    "HV_EO",
    "HV_EC",
    "PostHV_EO",
    "PostHV_EC",
    "PHOTO_EO",
    "PHOTO_EC",
]


def compute_embeddings(
    X: np.ndarray,
    reducer_names: List[str],
) -> Dict[str, Dict]:
    """
    Compute embeddings for all reducers in both 2D and 3D.
    """
    results = {}

    for name in reducer_names:
        logger.info(f"Computing embeddings for {name}...")
        results[name] = {
            "embedding_2d": None,
            "reducer_2d": None,
            "embedding_3d": None,
            "reducer_3d": None,
        }
        try:
            dr_2d = DimReduction(method=name, n_components=2)
            results[name]["embedding_2d"] = dr_2d.fit_transform(X)
            results[name]["reducer_2d"] = dr_2d
        except Exception as err:
            logger.warning(f"Failed to compute 2D {name}: {err}")

        try:
            dr_3d = DimReduction(method=name, n_components=3)
            results[name]["embedding_3d"] = dr_3d.fit_transform(X)
            results[name]["reducer_3d"] = dr_3d
        except Exception as err:
            logger.warning(f"Failed to compute 3D {name}: {err}")
    return results


def compute_embedding_separation_score(
    embedding: Optional[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Estimate class separation from an embedding using grouped CV balanced accuracy."""
    if embedding is None:
        return np.nan

    y = np.asarray(labels).ravel().astype(str)
    group_ids = np.asarray(groups).ravel().astype(str)
    if embedding.shape[0] != y.shape[0] or embedding.shape[0] != group_ids.shape[0]:
        return np.nan
    if pd.Index(y).nunique() < 2:
        return np.nan

    subject_labels = pd.DataFrame({"group": group_ids, "label": y})
    if subject_labels.groupby("group")["label"].nunique(dropna=False).max() > 1:
        raise ValueError("Each Study ID must map to a single target label.")

    grouped_y = subject_labels.groupby("group")["label"].first()
    min_count = int(grouped_y.value_counts().min())
    if min_count < 2:
        return np.nan

    n_splits = min(5, min_count)
    experiment = Experiment(
        ExperimentConfig(
            task="classification",
            tag="embedding-separation",
            models={
                "logreg": LogisticRegressionConfig(
                    method="LogisticRegression",
                    max_iter=1000,
                    class_weight="balanced",
                )
            },
            cv=CVConfig(
                strategy="stratified_group_kfold",
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            ),
            metrics=["balanced_accuracy"],
            use_scaler=True,
            verbose=False,
            n_jobs=1,
        )
    )
    result = experiment.run(np.asarray(embedding), y, groups=group_ids)
    summary = result.summary()
    return float(summary.loc["logreg", "balanced_accuracy_mean"])


def build_condition_ranking_rows(
    condition: str,
    embeddings: Dict[str, Dict],
    labels: np.ndarray,
    groups: np.ndarray,
    loaded_subjects: int,
    loaded_epochs: int,
    samples_used: int,
) -> List[Dict[str, object]]:
    """Build reducer ranking rows for one condition."""
    rows: List[Dict[str, object]] = []
    for reducer_name, reducer_results in embeddings.items():
        for dimension, key in (("2D", "embedding_2d"), ("3D", "embedding_3d")):
            score = compute_embedding_separation_score(
                reducer_results.get(key),
                labels,
                groups,
            )
            rows.append(
                {
                    "condition": condition,
                    "reducer": reducer_name,
                    "dimension": dimension,
                    "cv_balanced_accuracy": score,
                    "loaded_subjects": loaded_subjects,
                    "loaded_epochs": loaded_epochs,
                    "samples_used": samples_used,
                }
            )
    return rows


def add_reducer_report_elements(
    section: Section,
    condition: str,
    name: str,
    embeddings: Dict,
    labels: Optional[np.ndarray],
    meta_dict: Optional[Dict[str, np.ndarray]] = None,
    interactive: bool = True
) -> None:
    """Add 2D/3D reducer plots and diagnostics to an existing section."""
    emb_2d = embeddings.get("embedding_2d")
    emb_3d = embeddings.get("embedding_3d")
    reducer_2d = embeddings.get("reducer_2d")

    n_2d = emb_2d.shape[0] if emb_2d is not None else 0
    labels_2d = coerce_sample_vector(labels, n_2d) if n_2d else None
    if labels is not None and n_2d and labels_2d is None:
        logger.warning(
            f"Skipping labels for {condition}/{name} 2D plot due to size mismatch "
            f"(labels={len(np.asarray(labels).ravel())}, samples={n_2d})."
        )

    meta_2d = None
    if meta_dict and n_2d:
        filtered = {}
        for col_name, col_values in meta_dict.items():
            arr = coerce_sample_vector(col_values, n_2d)
            if arr is not None:
                filtered[col_name] = arr
        if filtered:
            meta_2d = filtered

    n_3d = emb_3d.shape[0] if emb_3d is not None else 0
    labels_3d = coerce_sample_vector(labels, n_3d) if n_3d else None
    meta_3d = None
    if meta_dict and n_3d:
        filtered = {}
        for col_name, col_values in meta_dict.items():
            arr = coerce_sample_vector(col_values, n_3d)
            if arr is not None:
                filtered[col_name] = arr
        if filtered:
            meta_3d = filtered

    reducer_diagnostics: Dict = {}

    if reducer_2d is not None and hasattr(reducer_2d, "get_diagnostics"):
        try:
            reducer_diagnostics = reducer_2d.get_diagnostics() or {}
        except Exception as err:
            logger.warning(f"Could not fetch diagnostics for {condition}/{name}: {err}")

    section.add_markdown(f"### {name.upper()}")

    add_embedding_plot(
        section=section,
        embedding=emb_2d,
        labels=labels_2d,
        meta=meta_2d,
        title=f"{condition} - {name.upper()} - 2D",
        dimensions=2,
        interactive=interactive,
    )

    add_embedding_plot(
        section=section,
        embedding=emb_3d,
        labels=labels_3d,
        meta=meta_3d,
        title=f"{condition} - {name.upper()} - 3D",
        dimensions=3,
        interactive=interactive,
    )

    # 3. Loss History (diagnostics API first)
    loss_history = reducer_diagnostics.get("loss_history_")
    if loss_history is None and reducer_2d is not None:
        reducer_obj = reducer_2d.reducer
        if hasattr(reducer_obj, "loss_history_"):
            loss_history = reducer_obj.loss_history_

    if loss_history is not None:
        loss_array = np.asarray(loss_history).ravel()
        if loss_array.size > 0:
            fig_loss = viz.plot_loss_history(
                loss_history=loss_array,
                title=f"{condition} - {name.upper()} - Training Loss",
                interactive=interactive,
            )
            if interactive:
                section.add_element(PlotlyElement(fig_loss))
            else:
                section.add_element(ImageElement(fig_loss))


def create_condition_section(
    condition: str,
    embeddings: Dict[str, Dict],
    labels: np.ndarray,
    meta_dict: Dict[str, np.ndarray],
    ranking_rows: List[Dict[str, object]],
    loaded_subjects: int,
    loaded_epochs: int,
    samples_used: int,
    interactive: bool,
) -> Section:
    """Create one report section for a single condition."""
    section = Section(title=condition, icon="🧠")
    section.add_markdown(
        f"Condition-specific embeddings for **{condition}**. "
        f"Loaded subjects: **{loaded_subjects}**. "
        f"Loaded epochs: **{loaded_epochs}**. "
        f"Samples used after representation: **{samples_used}**."
    )
    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["dimension", "cv_balanced_accuracy"],
            ascending=[True, False],
        ).reset_index(drop=True)
        section.add_element(
            TableElement(
                ranking_df[["reducer", "dimension", "cv_balanced_accuracy"]].round(4),
                title="Reducer Ranking",
            )
        )

    for reducer_name, reducer_results in embeddings.items():
        add_reducer_report_elements(
            section=section,
            condition=condition,
            name=reducer_name,
            embeddings=reducer_results,
            labels=labels,
            meta_dict=meta_dict,
            interactive=interactive,
        )
    return section

def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive EEG DimReduction Pipeline")
    parser.add_argument("--bids_root", default="/Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS", help="Path to BIDS dataset")
    parser.add_argument("--task", default="clinical", help="Task name (default: clinical)")
    parser.add_argument("--session", default="01", help="Session ID (default: 01)")
    parser.add_argument("--metadata", default=None, help="Path to metadata CSV")
    parser.add_argument("--output_dir", default=results_dir, help="Output directory")
    parser.add_argument("--dataset_name", required=True, help="Name for this analysis run")
    
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
        help="Conditions to evaluate for separation ranking.",
    )
    parser.add_argument("--segment_duration", type=float, default=60.0, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Window overlap in seconds")
    
    parser.add_argument(
        "--representation",
        choices=list(REPRESENTATION_CONFIG.keys()),
        default="epoch_flat",
        help=(
            "Combined sample granularity + feature layout: "
            "'epoch_flat' => one row per epoch, columns=(channel x time); "
            "'epoch_time_as_sample' => one row per time-step, columns=channel; "
            "'epoch_scalar_mean' => one row per epoch, columns=1 scalar mean; "
            "'subject_*' variants first average epochs per subject then apply same layout."
        ),
    )
    
    parser.add_argument("--subsample", type=int, default=None, help="Number of subjects to random sample")
    parser.add_argument("--subject_col", default="study_id", help="Column in cleaned metadata")
    parser.add_argument("--target_col", default="Group", help="Column to use as labels")
    
    parser.add_argument("--reducers", nargs="+", default=ALL_REDUCERS, help="List of reducers")
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable interactive plots (use --no-interactive for static report figures)",
    )
    parser.add_argument(
        "--save_static_figures",
        action="store_true",
        help="Save per-reducer static PNG embeddings in addition to the HTML report",
    )
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings to disk")
    parser.add_argument("--use_derivatives", action="store_true", help="Load saved epoch derivatives instead of raw BIDS")
    parser.add_argument("--desc", default="base", help="Desc to load from derivatives (default: base)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to process")
    parser.add_argument(
        "--ignore_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore BAD_ annotations during epoching (use --no-ignore-annotations to enforce them)",
    )
    
    args = parser.parse_args()
    
    # Force logging level
    logging.getLogger().setLevel(logging.INFO)
    
    interactive = args.interactive
    logger.info(f"Using representation='{args.representation}'")
    
    # Setup Paths
    bids_root = Path(args.bids_root)
    out_path = Path(args.output_dir) / args.dataset_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    coverage_root = (
        bids_root / "derivatives" / "preproc"
        if args.use_derivatives
        else bids_root
    )
    coverage_desc = args.desc if args.use_derivatives else ""
    coverage_suffix = "epo" if args.use_derivatives else None

    meta_df = load_csv(str(Path(args.metadata)), sep=None) if args.metadata else None
    metadata_path = Path(args.metadata) if args.metadata else None
    coverage = validate_bids_coverage(
        meta_df,
        coverage_root,
        desc=coverage_desc,
        suffix=coverage_suffix,
        subject_col=args.subject_col,
    )
    available_subjects = coverage["present_subjects"]
    subjects = [f"{int(subject):04d}" for subject in args.subjects] if args.subjects else available_subjects
    subjects = [subject for subject in subjects if subject in set(available_subjects)]
    logger.info(
        f"Using {len(subjects)} available subjects from "
        f"{'derivatives' if args.use_derivatives else 'BIDS'}."
    )

    if meta_df is not None:
        meta_df = meta_df[
            meta_df[args.subject_col].map(lambda value: f"{int(value):04d}").isin(available_subjects)
        ].copy()
        logger.info(f"Metadata loaded from: {metadata_path}")
        subjects = [
            subject for subject in subjects
            if subject in set(meta_df[args.subject_col].map(lambda value: f"{int(value):04d}"))
        ]

    if args.subsample and args.subsample < len(subjects):
        random.seed(42)
        subjects = random.sample(subjects, args.subsample)
        logger.info(f"Subsampled to {len(subjects)} subjects.")
        
    logger.info("Generating condition-wise report...")
    report = Report(title=f"DimReduction Condition Screening: {args.dataset_name}")
    condition_summary_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    condition_sections: List[Section] = []

    for condition in args.conditions:
        logger.info(f"Loading condition '{condition}'")
        try:
            dc_loaded = load_eeg_data(
                bids_root=bids_root,
                use_derivatives=args.use_derivatives,
                subjects=subjects if subjects else None,
                task=args.task,
                session=args.session,
                segment_duration=args.segment_duration,
                overlap=args.overlap,
                metadata_df=meta_df,
                subject_col=args.subject_col,
                target_col=args.target_col,
                desc=args.desc,
                condition=condition,
            )
        except Exception as err:
            logger.warning(f"Skipping {condition}: {err}")
            condition_summary_rows.append(
                {
                    "condition": condition,
                    "status": "load_failed",
                    "loaded_subjects": 0,
                    "loaded_epochs": 0,
                    "samples_used": 0,
                    "best_2d_reducer": None,
                    "best_2d_score": np.nan,
                    "best_3d_reducer": None,
                    "best_3d_score": np.nan,
                }
            )
            continue

        loaded_subjects = pd.Series(dc_loaded.coords[args.subject_col]).astype(str).unique().tolist()
        missing_subjects = [subject for subject in subjects if subject not in set(loaded_subjects)]
        loaded_epochs = int(dc_loaded.X.shape[0])
        logger.info(f"{condition}: loaded data for {len(loaded_subjects)}/{len(subjects)} subjects.")
        if missing_subjects:
            logger.info(f"{condition}: skipped subjects during loading: {missing_subjects}")

        logger.info(f"{condition}: applying representation '{args.representation}'")
        dc = apply_representation(dc_loaded, args.representation, args.subject_col)
        labels = np.asarray(dc.y).ravel().astype(str)
        groups = np.asarray(dc.coords[args.subject_col]).ravel().astype(str)
        logger.info(f"{condition}: final data shape {dc.X.shape}")

        embeddings = compute_embeddings(X=dc.X, reducer_names=args.reducers)
        meta_dict = build_meta_dict(dc)
        logger.info(f"{condition}: color options {list(meta_dict.keys())}")

        condition_ranking_rows = build_condition_ranking_rows(
            condition=condition,
            embeddings=embeddings,
            labels=labels,
            groups=groups,
            loaded_subjects=len(loaded_subjects),
            loaded_epochs=loaded_epochs,
            samples_used=int(dc.X.shape[0]),
        )
        ranking_rows.extend(condition_ranking_rows)

        condition_df = pd.DataFrame(condition_ranking_rows)
        best_2d = condition_df[
            (condition_df["dimension"] == "2D")
            & condition_df["cv_balanced_accuracy"].notna()
        ].sort_values(
            "cv_balanced_accuracy", ascending=False, na_position="last"
        )
        best_3d = condition_df[
            (condition_df["dimension"] == "3D")
            & condition_df["cv_balanced_accuracy"].notna()
        ].sort_values(
            "cv_balanced_accuracy", ascending=False, na_position="last"
        )
        condition_summary_rows.append(
            {
                "condition": condition,
                "status": "ok",
                "loaded_subjects": len(loaded_subjects),
                "loaded_epochs": loaded_epochs,
                "samples_used": int(dc.X.shape[0]),
                "best_2d_reducer": best_2d.iloc[0]["reducer"] if not best_2d.empty else None,
                "best_2d_score": best_2d.iloc[0]["cv_balanced_accuracy"] if not best_2d.empty else np.nan,
                "best_3d_reducer": best_3d.iloc[0]["reducer"] if not best_3d.empty else None,
                "best_3d_score": best_3d.iloc[0]["cv_balanced_accuracy"] if not best_3d.empty else np.nan,
            }
        )

        condition_sections.append(
            create_condition_section(
                condition=condition,
                embeddings=embeddings,
                labels=labels,
                meta_dict=meta_dict,
                ranking_rows=condition_ranking_rows,
                loaded_subjects=len(loaded_subjects),
                loaded_epochs=loaded_epochs,
                samples_used=int(dc.X.shape[0]),
                interactive=interactive,
            )
        )

        if args.save_embeddings:
            embeddings_dir = out_path / "embeddings" / condition
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            for name, res in embeddings.items():
                if res["embedding_2d"] is not None:
                    np.save(embeddings_dir / f"{name}_2d.npy", res["embedding_2d"])
                if res["embedding_3d"] is not None:
                    np.save(embeddings_dir / f"{name}_3d.npy", res["embedding_3d"])

        if args.save_static_figures:
            figures_dir = out_path / "figures" / condition
            figures_dir.mkdir(parents=True, exist_ok=True)
            for name, res in embeddings.items():
                emb_2d = res.get("embedding_2d")
                if emb_2d is not None:
                    fig = viz.plot_embedding(
                        X_emb=emb_2d,
                        labels=labels,
                        dims=(0, 1),
                        title=f"{condition} - {name.upper()}",
                        interactive=False,
                    )
                    fig.savefig(figures_dir / f"{name}_embedding.png", dpi=150, bbox_inches="tight")
                    import matplotlib.pyplot as plt
                    plt.close(fig)

    if not condition_sections:
        raise RuntimeError("No conditions produced usable embeddings.")

    overview_df = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset_name,
                "representation": args.representation,
                "target_col": args.target_col,
                "reducers": ", ".join(args.reducers),
                "requested_subjects": len(subjects),
                "conditions_tested": ", ".join(args.conditions),
            }
        ]
    )
    overview_sec = Section("Overview", icon="📋")
    overview_sec.add_markdown(
        "This report ranks **conditions** by how well each reducer separates the target labels "
        "in 2D and 3D embeddings."
    )
    overview_sec.add_element(TableElement(overview_df, title="Run Configuration"))
    report.add_section(overview_sec)

    condition_summary_df = pd.DataFrame(condition_summary_rows)
    summary_sec = Section("Condition Summary", icon="🧾")
    summary_sec.add_markdown(
        "Per-condition data availability and best reducer scores."
    )
    summary_sec.add_element(
        TableElement(
            condition_summary_df[
                [
                    "condition",
                    "status",
                    "loaded_subjects",
                    "loaded_epochs",
                    "samples_used",
                    "best_2d_reducer",
                    "best_2d_score",
                    "best_3d_reducer",
                    "best_3d_score",
                ]
            ].round(4),
            title="Condition Summary",
        )
    )
    report.add_section(summary_sec)

    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_sec = Section("Condition Ranking", icon="🏁")
        ranking_sec.add_markdown(
            "Cross-validated balanced accuracy on the low-dimensional embeddings. "
            "Higher is better."
        )
        ranking_sec.add_element(
            TableElement(
                ranking_df.sort_values(
                    by="cv_balanced_accuracy",
                    ascending=False,
                    na_position="last",
                ).reset_index(drop=True).round(4),
                title="Reducer x Condition Ranking",
            )
        )
        report.add_section(ranking_sec)

    for section in condition_sections:
        report.add_section(section)

    # Save Report
    save_path = out_path / "report.html"
    report.save(save_path)
    logger.info(f"Report saved to: {save_path}")


if __name__ == "__main__":
    main()

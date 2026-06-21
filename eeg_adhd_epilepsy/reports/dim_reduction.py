"""Dimensionality-reduction report assembly."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction import (
    DimReduction,
    EVAL_METRIC_COLUMNS,
    SEPARATION_METRIC_KEY,
    grouped_condition_stats,
    load_fit_artifact,
    load_fit_runs,
    paired_condition_stats,
)
from coco_pipe.io import DataContainer
from coco_pipe.report import (
    ImageElement,
    InteractiveTableElement,
    PlotlyElement,
    Report,
    Section,
    TableElement,
)
from coco_pipe.report.qc import build_qc_section
from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz.interactive.dim_reduction import (
    plot_component_loadings,
    plot_embedding as plot_embedding_interactive,
    plot_radar_comparison,
)

from eeg_adhd_epilepsy.io.containers import load_container
from eeg_adhd_epilepsy.io.bids import get_reports_root
from eeg_adhd_epilepsy.utils.metadata_schema import EPILEPSY_MED_COLS
from eeg_adhd_epilepsy.viz.topo import plot_topomap_from_channel_values, plot_topomap_selector
from eeg_adhd_epilepsy.viz.utils import save_fig

logger = logging.getLogger(__name__)

_UNIT_LABELS = {
    "flat": "global",
    "sensor": "sensor",
    "family": "family",
    "subfamily": "subfamily",
    "sensor_within_family": "sensor",
    "sensor_within_subfamily": "sensor",
    "feature": "feature",
    "feature_within_family": "feature",
    "descriptor": "descriptor",
    "descriptor_sensor": "descriptor × sensor",
}


def _unit_label(analysis_mode: str) -> str:
    return _UNIT_LABELS.get(analysis_mode, "analysis unit")


def _unit_intro(analysis_mode: str) -> str:
    return _UNIT_LABELS.get(analysis_mode, analysis_mode.replace("_", " "))


_PLOT_META_EXCLUDED_COLUMNS = {
    "obs",
    "channel",
    "time",
    "subject",
    "study_id",
    "patient_id",
    "patient_group_id",
    "obs_id",
    "run",
    "recording_id",
    "eeg_date",
    *EPILEPSY_MED_COLS,
}
_PLOT_META_EXCLUDED_NORMALIZED = {
    "obs",
    "subject",
    "channel",
    "time",
    "studyid",
    "patientid",
    "patientgroupid",
    "obsid",
    "run",
    "recordingid",
    "eegdate",
    "age",
    "epochcount",
    "firsteeg",
    "psychostimulant",
}
_FIT_FAILURE_COLUMNS = [
    "scope",
    "condition",
    "analysis_mode",
    "family",
    "unit_name",
    "reducer",
    "n_components",
    "status",
    "error",
    "timestamp",
]
_EVAL_FAILURE_COLUMNS = [
    "scope",
    "condition",
    "analysis_mode",
    "family",
    "unit_name",
    "eval_name",
    "reducer",
    "n_components",
    "status",
    "error",
    "timestamp",
]


def _family_label(args: Any) -> str:
    """Human-readable descriptor-family label, or empty string for raw inputs."""
    if args.input_mode == "descriptors" and args.descriptor_families:
        return ", ".join(args.descriptor_families)
    if args.input_mode == "descriptors":
        return "all descriptor families"
    return ""


def _get_feature_names(container) -> Optional[list[str]]:
    """Return the feature-axis names from a container, or None if unavailable."""
    if container is None:
        return None
    try:
        feat = (container.coords or {}).get("feature")
        if feat is not None:
            names = [str(f) for f in np.asarray(feat)]
            return names if names else None
    except Exception:
        pass
    return None


def _build_condition_scalar_df(
    fit_eval_ranking: pd.DataFrame,
    analysis_mode: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Build a long-format scalar DataFrame suitable for paired_condition_stats.

    The ``subject`` key is the (reducer, n_components) pair for flat mode and
    the unit_name for family / sensor modes — so that the same model
    configuration is treated as one replicate across conditions.
    """
    rows = []
    for _, row in fit_eval_ranking.iterrows():
        if str(row.get("scope", "")) != "condition":
            continue
        if analysis_mode == "flat":
            subject = f"{row.get('reducer', '')}__n{int(row.get('n_components', 0))}"
        else:
            subject = str(row.get("unit_name", ""))
        condition = str(row.get("condition", ""))
        for metric in metrics:
            if metric in row.index and pd.notna(row[metric]):
                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "metric": metric,
                    "value": float(row[metric]),
                })
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["subject", "condition", "metric", "value"])
    )


def _build_condition_stats_section(
    args: Any,
    fit_eval_ranking: pd.DataFrame,
    metrics: list[str],
) -> Optional[Section]:
    """Build a paired-stats section comparing embedding quality across conditions.

    Runs ``paired_condition_stats`` across all condition pairs and, when both
    EO and EC conditions are present, ``grouped_condition_stats`` for the
    eye-state contrast.  Returns ``None`` when fewer than two conditions are
    available or no valid pairs can be formed.
    """
    if len(args.conditions) < 2:
        return None

    available_metrics = [
        m for m in metrics if m in fit_eval_ranking.columns
    ]
    if not available_metrics:
        return None

    scalar_df = _build_condition_scalar_df(fit_eval_ranking, args.analysis_mode, available_metrics)
    if scalar_df.empty:
        return None

    section = Section("Condition Statistics", icon="📐")
    section.add_markdown(
        "Paired t-tests (FDR-BH corrected) comparing embedding quality metrics across "
        "conditions.  The unit of replication is the model configuration "
        f"({'(reducer, n_components)' if args.analysis_mode == 'flat' else 'analysis unit'}) "
        "observed in each condition."
    )

    paired_df = paired_condition_stats(
        scalar_df,
        conditions=args.conditions,
        metric_col="metric",
        condition_col="condition",
        subject_col="subject",
        value_col="value",
    )
    if not paired_df.empty:
        section.add_element(
            TableElement(
                paired_df.round(4),
                title="Pairwise condition comparisons (paired t-test, FDR corrected)",
            )
        )
    else:
        section.add_markdown("*No valid condition pairs for paired t-test (insufficient matched replicates).*")

    # Eye-state grouped test: EO vs EC when both families are present
    eo_conds = [c for c in args.conditions if "EO" in c.upper()]
    ec_conds = [c for c in args.conditions if "EC" in c.upper()]
    if eo_conds and ec_conds:
        condition_sets = {"eye_state": {"EO": eo_conds, "EC": ec_conds}}
        grouped_df = grouped_condition_stats(
            scalar_df,
            condition_sets=condition_sets,
            metric_col="metric",
            condition_col="condition",
            subject_col="subject",
            value_col="value",
        )
        if not grouped_df.empty:
            section.add_element(
                TableElement(
                    grouped_df.round(4),
                    title="Eye-state grouped contrast (EO vs EC, FDR corrected)",
                )
            )

    return section


def _filter_runs(
    df: pd.DataFrame,
    args: Any,
    reducers: Sequence[str],
    pooled_condition: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df[df["input_mode"] == args.input_mode].copy()
    filtered = filtered[filtered["analysis_mode"] == args.analysis_mode].copy()
    filtered = filtered[filtered["reducer"].isin(reducers)].copy()
    filtered = filtered[filtered["n_components"].isin(args.n_components_sweep)].copy()
    wanted_conditions = set(args.conditions) | ({pooled_condition} if args.run_pooled else set())
    filtered = filtered[filtered["condition"].isin(wanted_conditions)].copy()
    run_config_hash = getattr(args, "run_config_hash", None)
    if run_config_hash and "input_signature" in filtered.columns:
        filtered = filtered[
            filtered["input_signature"].apply(
                lambda value: isinstance(value, dict)
                and value.get("run_config_hash") == run_config_hash
            )
        ].copy()
    if "representation" in filtered.columns:
        filtered = filtered[filtered["representation"] == args.representation].copy()
    if args.input_mode == "foundation_embeddings":
        for column, desired in (
            ("embedding_model_key", getattr(args, "embedding_model_key", None)),
            (
                "embedding_representation",
                getattr(args, "embedding_representation", None),
            ),
            ("embedding_aggregate_by", getattr(args, "embedding_aggregate_by", None)),
        ):
            if column in filtered.columns:
                filtered = filtered[
                    filtered[column].fillna("") == (desired or "")
                ].copy()

    desired_families = list(args.descriptor_families or [])
    if args.input_mode == "descriptors" and "descriptor_families" in filtered.columns:
        filtered = filtered[
            filtered["descriptor_families"].apply(
                lambda value: list(value or []) == desired_families
                if isinstance(value, list)
                else desired_families == []
            )
        ].copy()
    if args.input_mode == "descriptors" and "descriptor_max_abs_value" in filtered.columns:
        desired_max_abs = getattr(args, "descriptor_max_abs_value", None)
        if desired_max_abs is not None:
            filtered = filtered[
                pd.to_numeric(filtered["descriptor_max_abs_value"], errors="coerce").eq(float(desired_max_abs))
            ].copy()
    return filtered


def _plot_meta(meta: Optional[dict[str, np.ndarray]], n_samples: int) -> Optional[dict[str, np.ndarray]]:
    if not meta:
        return None
    filtered = {}
    for key, value in meta.items():
        arr = np.asarray(value).ravel()
        if arr.shape[0] == n_samples:
            filtered[key] = arr
    return filtered or None


def _selection_view(
    frame: pd.DataFrame,
    selection_metric: str,
    selection_eval_name: Optional[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if (
        selection_metric == SEPARATION_METRIC_KEY
        and selection_eval_name
        and "eval_name" in frame.columns
    ):
        selected = frame[frame["eval_name"] == selection_eval_name].copy()
        if not selected.empty:
            return selected
    return frame


def _primary_eval_spec(args: Any, eval_specs: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not eval_specs:
        return None
    selection_eval_name = getattr(args, "selection_eval_name", None)
    if selection_eval_name:
        for spec in eval_specs:
            if spec["name"] == selection_eval_name:
                return spec
    return eval_specs[0]


def _overview_container(
    args: Any,
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    containers_by_scope: Optional[dict[tuple[str, str], Any]],
    pooled_condition: str,
):
    if containers_by_scope:
        pooled = containers_by_scope.get(("pooled", pooled_condition))
        if pooled is not None:
            return pooled
        condition_containers = [
            containers_by_scope[("condition", condition)]
            for condition in args.conditions
            if ("condition", condition) in containers_by_scope
        ]
        if condition_containers:
            if len(condition_containers) == 1:
                return condition_containers[0]
            return DataContainer.concat(condition_containers)

    loaded = []
    for condition in args.conditions:
        try:
            loaded.append(load_container(args, subjects, meta_df, condition, target_col=None))
        except Exception:
            continue
    if not loaded:
        return None
    if len(loaded) == 1:
        return loaded[0]
    return DataContainer.concat(loaded)


def _subject_frame(container, subject_col: str) -> pd.DataFrame:
    n_obs = container.X.shape[0]
    data: dict[str, np.ndarray] = {}
    for key, values in container.coords.items():
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == n_obs:
            data[str(key)] = arr
    if container.ids is not None:
        data["obs_id"] = np.asarray(container.ids, dtype=object).astype(str)
    frame = pd.DataFrame(data)
    if subject_col in frame.columns:
        frame = frame.drop_duplicates(subset=[subject_col], keep="first").reset_index(drop=True)
    return frame


def _add_overview_cohort_summary(
    overview_sec: Section,
    args: Any,
    eval_specs: Sequence[dict[str, Any]],
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    containers_by_scope: Optional[dict[tuple[str, str], Any]],
    pooled_condition: str,
) -> None:
    primary_spec = _primary_eval_spec(args, eval_specs)
    if primary_spec is None:
        return
    container = _overview_container(args, subjects, meta_df, containers_by_scope, pooled_condition)
    if container is None:
        return
    frame = _subject_frame(container, args.subject_col)
    if primary_spec["target_col"] not in frame.columns:
        return

    labels = frame[primary_spec["target_col"]].astype(str)
    label_map = primary_spec.get("label_map") or {}
    if label_map:
        labels = labels.map(lambda value: label_map.get(value, value))
    frame = frame.assign(_primary_class=labels.astype(str))
    if frame["_primary_class"].nunique(dropna=True) <= 1:
        return

    overview_sec.add_markdown(
        f"Primary cohort summary for **{primary_spec['name']}** using **{frame[args.subject_col].nunique() if args.subject_col in frame.columns else len(frame)}** unique subjects."
    )

    summary_rows = []
    for class_value, class_df in frame.groupby("_primary_class", dropna=False):
        row = {
            "class": class_value,
            "n_subjects": int(len(class_df)),
            "pct_subjects": round(100.0 * len(class_df) / len(frame), 1),
        }
        if "age" in class_df.columns:
            age = pd.to_numeric(class_df["age"], errors="coerce")
            if age.notna().any():
                row["mean_age"] = round(float(age.mean()), 2)
                row["sd_age"] = round(float(age.std(ddof=0)), 2)
        summary_rows.append(row)
    overview_sec.add_element(
        TableElement(pd.DataFrame(summary_rows), title="Primary class counts")
    )

    if "sex" in frame.columns:
        sex_table = (
            frame.assign(sex=frame["sex"].astype(str))
            .groupby(["_primary_class", "sex"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={"_primary_class": "class"})
        )
        overview_sec.add_element(TableElement(sex_table, title="Sex by class"))

    if "age_group" in frame.columns:
        age_group_table = (
            frame.assign(age_group=frame["age_group"].astype(str))
            .groupby(["_primary_class", "age_group"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={"_primary_class": "class"})
        )
        overview_sec.add_element(TableElement(age_group_table, title="Age group by class"))

    clinical_columns = [
        column
        for column in ["autism", "epilepsy", "asm", "asm_resistant", "psychostimulant"]
        if column in frame.columns
    ]
    if clinical_columns:
        clinical_rows = []
        for class_value, class_df in frame.groupby("_primary_class", dropna=False):
            row = {"class": class_value}
            for column in clinical_columns:
                values = class_df[column].astype(str).str.strip().str.lower()
                present = values.isin({"1", "true", "yes", "y", "present"})
                total = int(present.notna().sum())
                if total == 0:
                    row[column] = ""
                else:
                    count = int(present.sum())
                    row[column] = f"{count} ({(100.0 * count / total):.1f}%)"
            clinical_rows.append(row)
        overview_sec.add_element(
            TableElement(pd.DataFrame(clinical_rows), title="Clinical composition by class")
        )

    if "psychostimulant_category" in frame.columns:
        medication_rows = []
        for class_value, class_df in frame.groupby("_primary_class", dropna=False):
            counts = class_df["psychostimulant_category"].fillna("None").astype(str).value_counts()
            medication_rows.append(
                {
                    "class": class_value,
                    "psychostimulant_category_counts": "; ".join(
                        f"{name}={count}" for name, count in counts.items()
                    ),
                }
            )
        overview_sec.add_element(
            TableElement(pd.DataFrame(medication_rows), title="Medication category by class")
        )


def _add_data_availability_summary(
    overview_sec: Section,
    args: Any,
    containers_by_scope: Optional[dict[tuple[str, str], Any]],
    fit_runs_df: pd.DataFrame,
    pooled_condition: str,
) -> None:
    rows = []
    for scope, condition in [("condition", condition) for condition in args.conditions] + (
        [("pooled", pooled_condition)] if getattr(args, "run_pooled", False) else []
    ):
        container = None if containers_by_scope is None else containers_by_scope.get((scope, condition))
        condition_runs = (
            fit_runs_df[
                (fit_runs_df["scope"] == scope)
                & (fit_runs_df["condition"] == condition)
                & (fit_runs_df["status"] == "success")
            ]
            if not fit_runs_df.empty
            else pd.DataFrame()
        )
        if container is None and condition_runs.empty:
            continue
        row = {
            "scope": scope,
            "condition": condition,
            "loaded_observations": "",
            "samples_used": "",
            "unique_subjects": "",
            "unique_recordings": "",
            "successful_fits": int(len(condition_runs)),
            "reducers": ", ".join(sorted(condition_runs["reducer"].dropna().astype(str).unique())) if "reducer" in condition_runs else "",
            "valid_n_components": ", ".join(map(str, sorted(condition_runs["n_components"].dropna().astype(int).unique()))) if "n_components" in condition_runs and not condition_runs.empty else "",
        }
        if container is not None:
            row["loaded_observations"] = int(container.meta.get("loaded_obs", container.X.shape[0]))
            row["samples_used"] = int(container.X.shape[0])
            if getattr(args, "subject_col", None) in container.coords:
                row["unique_subjects"] = int(pd.Index(np.asarray(container.coords[args.subject_col]).astype(str)).nunique())
            if "recording_id" in container.coords:
                row["unique_recordings"] = int(pd.Index(np.asarray(container.coords["recording_id"]).astype(str)).nunique())
        rows.append(row)
    if rows:
        overview_sec.add_element(TableElement(pd.DataFrame(rows), title="Data Availability"))


def _add_embedding_plot(
    section: Section,
    embedding: np.ndarray,
    meta: Optional[dict[str, np.ndarray]],
    title: str,
    dimensions: int,
    interactive: bool,
) -> None:
    if interactive:
        section.add_element(
            PlotlyElement(
                plot_embedding_interactive(
                    embedding=embedding,
                    metadata=meta,
                    title=title,
                    dimensions=dimensions,
                )
            )
        )
        return

    dims = (0, 1) if dimensions == 2 else (0, 1, 2)
    section.add_element(
        ImageElement(
            viz.plot_embedding(
                X_emb=embedding,
                labels=None,
                dims=dims,
                title=title,
                interactive=False,
            )
        )
    )


def _add_best_fit_plots(
    section: Section,
    title: str,
    artifact: dict[str, Any],
    meta_dict: dict[str, np.ndarray],
    interactive: bool,
    compress_viz_with_pca: bool = False,
    feature_names: Optional[list[str]] = None,
) -> None:
    embedding = np.asarray(artifact["embedding"])
    if embedding.ndim != 2:
        return

    plot_meta = _plot_meta(meta_dict, embedding.shape[0])
    if embedding.shape[1] == 2:
        _add_embedding_plot(section, embedding, plot_meta, f"{title} - native 2D", 2, interactive)
    elif embedding.shape[1] >= 3:
        _add_embedding_plot(
            section, embedding[:, :2], plot_meta, f"{title} - first 2 dims", 2, interactive
        )
        _add_embedding_plot(
            section, embedding[:, :3], plot_meta, f"{title} - first 3 dims", 3, interactive
        )

    if compress_viz_with_pca and embedding.shape[1] > 3:
        pca_2d = DimReduction(method="PCA", n_components=2).fit_transform(embedding)
        pca_meta = _plot_meta(meta_dict, pca_2d.shape[0])
        _add_embedding_plot(
            section,
            pca_2d,
            pca_meta,
            f"{title} - PCA compressed 2D",
            2,
            interactive,
        )

    # Component loadings heatmap for linear reducers (PCA, ICA, …)
    components = (artifact.get("diagnostics") or {}).get("components")
    if components is not None:
        try:
            comp_arr = np.asarray(components, dtype=float)
            # sklearn convention: (n_components, n_features) → transpose to (n_features, n_components)
            if comp_arr.ndim == 2:
                loadings = comp_arr.T
                n_comp = loadings.shape[1]
                reducer_name = (artifact.get("fit") or {}).get("reducer", "")
                section.add_element(
                    PlotlyElement(
                        plot_component_loadings(
                            loadings,
                            feature_names=feature_names,
                            n_components=min(n_comp, 10),
                            title=f"{title} - component loadings (top {min(n_comp, 10)})",
                        )
                    )
                )
        except Exception as exc:
            logger.debug("Could not render component loadings for %s: %s", title, exc)


def _container_obs_frame(container) -> pd.DataFrame:
    n_samples = container.X.shape[0]
    data: dict[str, np.ndarray] = {}
    for col_name, col_values in container.coords.items():
        arr = np.asarray(col_values)
        if arr.ndim == 1 and len(arr) == n_samples and str(col_name) != "feature":
            data[str(col_name)] = arr
    if container.y is not None and "y" not in data:
        data["y"] = np.asarray(container.y)
    if container.ids is not None:
        data["obs_id"] = np.asarray(container.ids, dtype=object).astype(str)
    return pd.DataFrame(data)


def _align_obs_frame(container, ids: Optional[np.ndarray]) -> pd.DataFrame:
    frame = _container_obs_frame(container)
    if ids is None or "obs_id" not in frame.columns:
        return frame.reset_index(drop=True)

    requested_ids = np.asarray(ids, dtype=object).astype(str)
    if len(requested_ids) == len(frame) and np.array_equal(
        requested_ids,
        frame["obs_id"].astype(str).to_numpy(),
    ):
        return frame.reset_index(drop=True)

    counts: dict[str, int] = {}
    frame_keys = []
    for obs_id in frame["obs_id"].astype(str):
        occurrence = counts.get(obs_id, 0)
        frame_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1

    counts = {}
    requested_keys = []
    for obs_id in requested_ids:
        occurrence = counts.get(obs_id, 0)
        requested_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1

    keyed = frame.assign(_obs_key=frame_keys).drop_duplicates("_obs_key", keep="first").set_index("_obs_key")
    if all(key in keyed.index for key in requested_keys):
        return keyed.loc[requested_keys].drop(columns=[], errors="ignore").reset_index(drop=True)
    if len(frame) == len(requested_ids):
        return frame.reset_index(drop=True)
    return pd.DataFrame()


def _build_meta_dict(
    container,
    ids: Optional[np.ndarray] = None,
    eval_specs: Sequence[dict[str, Any]] = (),
) -> dict[str, np.ndarray]:
    frame = _align_obs_frame(container, ids)
    if frame.empty:
        return {}

    meta: dict[str, np.ndarray] = {}
    for col_name in frame.columns:
        if col_name == "_obs_key":
            continue
        normalized = "".join(ch for ch in str(col_name).lower() if ch.isalnum())
        if (
            col_name in _PLOT_META_EXCLUDED_COLUMNS
            or normalized in _PLOT_META_EXCLUDED_NORMALIZED
            or "psychostimulant" in normalized
            or str(col_name).endswith("_bool")
            or str(col_name).endswith("_clean")
        ):
            continue
        values = frame[col_name].to_numpy()
        if 1 < pd.Index(values.astype(str)).nunique() <= 200:
            meta[col_name] = values

    for spec in eval_specs:
        target_col = spec.get("target_col")
        if target_col not in frame.columns:
            continue
        labels = frame[target_col].astype(str)
        label_map = spec.get("label_map") or {}
        if label_map:
            labels = labels.map(lambda value: label_map.get(value, value))
        labels = labels.replace({"nan": "unknown", "None": "unknown", "": "unknown"})
        if spec.get("filters"):
            filter_mask = pd.Series(True, index=frame.index)
            for filter_spec in spec["filters"]:
                column = filter_spec["column"]
                if column not in frame.columns:
                    filter_mask[:] = False
                    continue
                values = {str(value) for value in filter_spec["values"]}
                filter_mask &= frame[column].astype(str).isin(values)
            labels = labels.where(filter_mask, "unknown")
        values = labels.to_numpy(dtype=object)
        if 1 < pd.Index(values.astype(str)).nunique() <= 200:
            meta[str(spec["name"])] = values

    if "condition" in meta:
        eye_state = np.array(
            [
                "EO"
                if str(value).lower().startswith("eo")
                else "EC"
                if str(value).lower().startswith("ec")
                else str(value)
                for value in meta["condition"]
            ],
            dtype=object,
        )
        if pd.Index(eye_state).nunique() > 1:
            meta["eye_state"] = eye_state
    return meta


def _build_flat_condition_section(
    args,
    output_root: Path,
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Section:
    container = load_container(args, subjects, meta_df, condition, target_col=None)
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    fam_label = _family_label(args)
    section = Section(condition, icon="🧠")
    section.add_markdown(
        (
            f"Input mode: **{args.input_mode}**. "
            f"{f'Descriptor families: **{fam_label}**. ' if fam_label else ''}"
            f"Representation: **{args.representation}**. "
            f"Loaded observations: **{container.meta.get('loaded_obs', container.X.shape[0])}**."
        )
    )

    ranking_df = condition_runs.merge(
        eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]],
        on="fit_id",
        how="left",
    )
    section.add_element(
        InteractiveTableElement(
            ranking_df.loc[
                :,
                ["reducer", "n_components", "trustworthiness", "continuity", "eval_name", SEPARATION_METRIC_KEY],
            ].round(4),
            title="Fit ranking",
            selector_columns=["reducer", "eval_name"],
            default_sort={"column": args.selection_metric, "direction": "desc"},
            page_size=5,
        )
    )

    for reducer_name in reducers:
        reducer_runs = condition_runs[condition_runs["reducer"] == reducer_name].copy()
        if reducer_runs.empty:
            continue
        best_row = (
            _selection_view(
                reducer_runs.merge(
                eval_frame.loc[:, ["fit_id", "eval_name", SEPARATION_METRIC_KEY]],
                on="fit_id",
                how="left",
                ),
                args.selection_metric,
                getattr(args, "selection_eval_name", None),
            )
            .sort_values(
                [args.selection_metric, "trustworthiness", "continuity"],
                ascending=[False, False, False],
                na_position="last",
            )
            .iloc[0]
        )
        best_artifact = artifacts[str(best_row["fit_id"])]
        section.add_markdown(f"### {reducer_name} (best n={int(best_row['n_components'])})")
        meta_dict = _build_meta_dict(container, best_artifact["ids"], eval_specs)
        _add_best_fit_plots(
            section,
            f"{condition} - {reducer_name}",
            best_artifact,
            meta_dict,
            interactive=args.interactive,
            compress_viz_with_pca=args.compress_viz_with_pca,
            feature_names=_get_feature_names(container),
        )
        sweep_df = reducer_runs.merge(
            eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]],
            on="fit_id",
            how="left",
        )
        if not sweep_df.empty:
            fig = go.Figure()
            for metric_name in ["trustworthiness", "continuity", "shepard_correlation"]:
                if metric_name not in reducer_runs.columns:
                    continue
                metric_df = reducer_runs.dropna(subset=[metric_name]).sort_values("n_components")
                if metric_df.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=metric_df["n_components"],
                        y=metric_df[metric_name],
                        mode="lines+markers",
                        name=metric_name,
                    )
                )
            # Separation metrics (Eval level)
            if not sweep_df.empty and SEPARATION_METRIC_KEY in sweep_df.columns:
                for eval_name, eval_group in sweep_df.dropna(subset=[SEPARATION_METRIC_KEY]).groupby("eval_name"):
                    eval_group = eval_group.sort_values("n_components")
                    fig.add_trace(
                        go.Scatter(
                            x=eval_group["n_components"],
                            y=eval_group[SEPARATION_METRIC_KEY],
                            mode="lines+markers",
                            name=f"separation: {eval_name}",
                        )
                    )
            fig.update_layout(
                title=f"{condition} - {reducer_name} metrics vs n_components",
                xaxis_title="n_components",
                yaxis_title="score",
            )
            section.add_element(PlotlyElement(fig))
            section.add_element(
                InteractiveTableElement(
                    sweep_df.loc[
                        :,
                        [
                            column
                            for column in [
                                "n_components",
                                "trustworthiness",
                                "continuity",
                                "shepard_correlation",
                                "eval_name",
                                SEPARATION_METRIC_KEY,
                            ]
                            if column in sweep_df.columns
                        ],
                    ].round(4),
                    title=f"{condition} - {reducer_name} sweep summary",
                    page_size=5,
                )
            )
    return section


def _add_unit_summary(
    section: Section,
    unit_runs: pd.DataFrame,
    *,
    args: Any,
    eval_specs: Sequence[dict[str, Any]],
    meta_df: Optional[pd.DataFrame],
    artifacts: dict[str, Any],
    output_root: Path,
    subjects: Optional[Sequence[str]],
    unit_label: str,
    title_prefix: str,
    input_mode: str,
    selection_metric: str,
    selection_eval_name: Optional[str] = None,
) -> None:
    if unit_runs.empty:
        return

    unit_runs = _selection_view(unit_runs, selection_metric, selection_eval_name)
    unit_column = (
        "unit_key" if args.analysis_mode == "descriptor_sensor" else "unit_name"
    )

    group_columns = [
        column
        for column in ["family", "subfamily", "eval_name", "target_col"]
        if column in unit_runs.columns and unit_runs[column].notna().any()
    ]
    sort_columns = list(
        dict.fromkeys(
            column
            for column in [selection_metric, "trustworthiness", "continuity", SEPARATION_METRIC_KEY]
            if column in unit_runs.columns
        )
    )
    best_units = (
        unit_runs.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
            na_position="last",
        )
        .groupby([*group_columns, unit_column], dropna=False)
        .head(1)
        .copy()
    )
    display_columns = [
        column
        for column in [*group_columns, unit_column, "reducer", "n_components", *sort_columns]
        if column in best_units.columns
    ]
    section.add_element(
        InteractiveTableElement(
            best_units.loc[:, display_columns].round(4),
            title=f"{title_prefix} {unit_label} ranking",
            selector_columns=[
                column
                for column in [*group_columns, "reducer"]
                if column in best_units.columns
            ],
            default_sort=(
                {"column": sort_columns[0], "direction": "desc"}
                if sort_columns
                else None
            ),
            page_size=5,
        )
    )

    sweep_df = unit_runs.loc[
        :,
        [
            column
            for column in [*group_columns, unit_column, "reducer", "n_components", *sort_columns]
            if column in unit_runs.columns
        ],
    ].copy()
    section.add_element(
        InteractiveTableElement(
            sweep_df.round(4),
            title=f"{title_prefix} {unit_label} sweep",
            selector_columns=[
                column
                for column in [*group_columns, unit_column, "reducer"]
                if column in sweep_df.columns
            ],
            page_size=5,
        )
    )

    best_by_unit = best_units.copy()
    plot_metric = sort_columns[0] if sort_columns else "trustworthiness"
    grouped_best = (
        list(best_by_unit.groupby(group_columns, dropna=False))
        if group_columns
        else [((), best_by_unit)]
    )
    bar_fig = go.Figure()
    for idx, (group_key, group_df) in enumerate(grouped_best):
        plot_df = group_df.dropna(subset=[plot_metric]).copy()
        if plot_df.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label_parts = [
            str(value)
            for value in group_key
            if pd.notna(value) and str(value) != ""
        ]
        trace_label = " / ".join(label_parts) if label_parts else plot_metric
        bar_fig.add_trace(
            go.Bar(
                x=plot_df[unit_column].astype(str),
                y=plot_df[plot_metric],
                text=plot_df["n_components"].map(lambda value: f"n={int(value)}"),
                name=trace_label,
                visible=len(bar_fig.data) == 0,
            )
        )
    if len(bar_fig.data) > 1:
        bar_fig.update_layout(
            updatemenus=[
                {
                    "type": "dropdown",
                    "direction": "down",
                    "x": 1.0,
                    "y": 1.16,
                    "xanchor": "right",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": trace.name,
                            "method": "update",
                            "args": [
                                {"visible": [j == idx for j in range(len(bar_fig.data))]},
                                {
                                    "title": (
                                        f"{title_prefix} best {plot_metric} by "
                                        f"{unit_label} - {trace.name}"
                                    )
                                },
                            ],
                        }
                        for idx, trace in enumerate(bar_fig.data)
                    ],
                }
            ]
        )
    if bar_fig.data:
        first_bar_label = bar_fig.data[0].name
        title_suffix = (
            f" - {first_bar_label}"
            if first_bar_label and len(bar_fig.data) > 1
            else ""
        )
        bar_fig.update_layout(
            title=f"{title_prefix} best {plot_metric} by {unit_label}{title_suffix}",
            xaxis_title=unit_label,
            yaxis_title=plot_metric,
        )
        section.add_element(PlotlyElement(bar_fig))
    if unit_label == "sensor":
        topo_groups: dict[str, tuple[list[str], np.ndarray]] = {}
        for group_key, group_df in grouped_best:
            topo_df = group_df.dropna(subset=[plot_metric]).copy()
            if topo_df.empty:
                continue
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label_parts = [
                str(value)
                for value in group_key
                if pd.notna(value) and str(value) != ""
            ]
            topo_label = " / ".join(label_parts) if label_parts else plot_metric
            topo_groups[topo_label] = (
                topo_df[unit_column].astype(str).tolist(),
                topo_df[plot_metric].astype(float).to_numpy(),
            )
        if topo_groups:
            topo_plot = plot_topomap_selector(
                topo_groups,
                title=f"{title_prefix} best {plot_metric}",
                unit=plot_metric,
            )
            if topo_plot is not None:
                section.add_element(PlotlyElement(topo_plot))
                return
            topo_label, (topo_names, topo_values) = next(iter(topo_groups.items()))
            try:
                topo_fig = plot_topomap_from_channel_values(
                    channel_names=topo_names,
                    values=topo_values,
                    title=f"{title_prefix} best {plot_metric} - {topo_label}",
                    unit=plot_metric,
                )
            except Exception:
                topo_fig = None
            if topo_fig is not None:
                section.add_element(ImageElement(topo_fig, width="42%"))


def _build_nonflat_condition_section(
    args,
    output_root: Path,
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    metric_columns: Sequence[str],
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Section:
    unit_label = _unit_label(args.analysis_mode)
    fam_label = _family_label(args)
    section = Section(condition, icon="📊")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    intro = f"Primary analysis unit: **{_unit_intro(args.analysis_mode)}**."
    section.add_markdown(
        (
            f"Input mode: **{args.input_mode}**. "
            f"{f'Descriptor families: **{fam_label}**. ' if fam_label else ''}"
            f"{intro}"
        )
    )

    merged = condition_runs.merge(
        eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]],
        on="fit_id",
        how="left",
    )
    if args.analysis_mode == "family":
        _add_unit_summary(
            section,
            merged,
            args=args,
            eval_specs=eval_specs,
            meta_df=meta_df,
            artifacts=artifacts,
            output_root=output_root,
            subjects=subjects,
            unit_label="family",
            title_prefix=condition,
            input_mode=args.input_mode,
            selection_metric=args.selection_metric,
            selection_eval_name=getattr(args, "selection_eval_name", None),
        )
        family_container = load_container(args, subjects, meta_df, condition, target_col=None)
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue
            fig = go.Figure()
            comparison_metrics = [
                metric
                for metric in ["trustworthiness", "continuity", "shepard_correlation"]
                if metric in reducer_runs.columns
            ]
            if SEPARATION_METRIC_KEY in reducer_runs.columns:
                comparison_metrics.append(SEPARATION_METRIC_KEY)
            comparison_metrics = list(dict.fromkeys(comparison_metrics))
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = _selection_view(
                    family_runs,
                    args.selection_metric,
                    getattr(args, "selection_eval_name", None),
                ).sort_values(
                    [args.selection_metric, "trustworthiness", "continuity"],
                    ascending=[False, False, False],
                    na_position="last",
                )
                family_best_by_n = (
                    family_best_by_n.groupby("n_components", dropna=False).head(1).sort_values("n_components")
                )
                for metric_name in comparison_metrics:
                    metric_df = family_best_by_n.dropna(subset=[metric_name])
                    if metric_df.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=metric_df["n_components"],
                            y=metric_df[metric_name],
                            mode="lines+markers",
                            name=f"{family}: {metric_name}",
                        )
                    )
            if fig.data:
                fig.update_layout(
                    title=f"{condition} - {reducer_name} family comparison vs n_components",
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                section.add_element(PlotlyElement(fig))
            section.add_element(
                InteractiveTableElement(
                    reducer_runs.loc[
                        :,
                        list(dict.fromkeys(["family", "n_components", *comparison_metrics, "eval_name"])),
                    ].round(4),
                    title=f"{condition} - {reducer_name} family sweep",
                    selector_columns=["family", "eval_name"],
                    page_size=5,
                )
            )
        for family, family_runs in merged.groupby("family", dropna=False):
            best_row = (
                _selection_view(
                    family_runs,
                    args.selection_metric,
                    getattr(args, "selection_eval_name", None),
                ).sort_values(
                    [args.selection_metric, "trustworthiness", "continuity"],
                    ascending=[False, False, False],
                    na_position="last",
                )
                .iloc[0]
            )
            best_artifact = artifacts[str(best_row["fit_id"])]
            family_meta = _build_meta_dict(family_container, best_artifact["ids"], eval_specs)
            section.add_markdown(
                f"### {family} (best {best_row['reducer']} n={int(best_row['n_components'])})"
            )
            _add_best_fit_plots(
                section,
                f"{condition} - {family}",
                best_artifact,
                family_meta,
                interactive=args.interactive,
                compress_viz_with_pca=args.compress_viz_with_pca,
                feature_names=_get_feature_names(family_container),
            )
            section.add_element(
                InteractiveTableElement(
                    family_runs.loc[
                        :,
                        list(
                            dict.fromkeys(
                                [
                                    "reducer",
                                    "n_components",
                                    *[
                                        metric
                                        for metric in ["trustworthiness", "continuity", "shepard_correlation"]
                                        if metric in family_runs.columns
                                    ],
                                    "eval_name",
                                    *([SEPARATION_METRIC_KEY] if SEPARATION_METRIC_KEY in family_runs.columns else []),
                                ]
                            )
                        ),
                    ].round(4),
                    title=f"{condition} - {family} reducer/n sweep",
                    selector_columns=["reducer", "eval_name"],
                    page_size=5,
                )
            )
        return section
    if args.analysis_mode in {"sensor_within_family", "sensor_within_subfamily"}:
        group_columns = ["family"]
        if (
            args.analysis_mode == "sensor_within_subfamily"
            and "subfamily" in merged.columns
        ):
            group_columns.append("subfamily")
        for group_key, family_runs in merged.groupby(group_columns, dropna=False):
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_label = " / ".join(str(value) for value in group_values)
            section.add_markdown(f"### {group_label}")
            _add_unit_summary(
                section,
                family_runs,
                args=args,
                eval_specs=eval_specs,
                meta_df=meta_df,
                artifacts=artifacts,
                output_root=output_root,
                subjects=subjects,
                unit_label="sensor",
                title_prefix=f"{condition} - {group_label}",
                input_mode=args.input_mode,
                selection_metric=args.selection_metric,
                selection_eval_name=getattr(args, "selection_eval_name", None),
            )
        return section

    _add_unit_summary(
        section,
        merged,
        args=args,
        eval_specs=eval_specs,
        meta_df=meta_df,
        artifacts=artifacts,
        output_root=output_root,
        subjects=subjects,
        unit_label=unit_label,
        title_prefix=condition,
        input_mode=args.input_mode,
        selection_metric=args.selection_metric,
        selection_eval_name=getattr(args, "selection_eval_name", None),
    )
    if args.analysis_mode == "sensor":
        sensor_container = load_container(args, subjects, meta_df, condition, target_col=None)
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue
            sensor_group_columns = [
                column
                for column in ["eval_name", "target_col"]
                if column in reducer_runs.columns and reducer_runs[column].notna().any()
            ]
            sensor_groups = list(reducer_runs.groupby(sensor_group_columns, dropna=False)) if sensor_group_columns else [((), reducer_runs)]
            for group_key, group_df in sensor_groups:
                best_row = (
                    _selection_view(
                        group_df,
                        args.selection_metric,
                        getattr(args, "selection_eval_name", None),
                    ).sort_values(
                        [args.selection_metric, "trustworthiness", "continuity"],
                        ascending=[False, False, False],
                        na_position="last",
                    )
                    .iloc[0]
                )
                best_artifact = artifacts[str(best_row["fit_id"])]
                sensor_meta = _build_meta_dict(sensor_container, best_artifact["ids"], eval_specs)
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                label_parts = [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
                label_suffix = f" [{' / '.join(label_parts)}]" if label_parts else ""
                section.add_markdown(
                    f"### {reducer_name}{label_suffix} best sensor: {best_row['unit_name']} (n={int(best_row['n_components'])})"
                )
                _add_best_fit_plots(
                    section,
                    f"{condition} - {reducer_name}{label_suffix} - {best_row['unit_name']}",
                    best_artifact,
                    sensor_meta or {},
                    interactive=args.interactive,
                    compress_viz_with_pca=args.compress_viz_with_pca,
                    feature_names=_get_feature_names(sensor_container),
                )
    return section


def _build_pooled_section(
    args,
    output_root: Path,
    pooled_runs: pd.DataFrame,
    pooled_eval_runs: pd.DataFrame,
    metric_columns: Sequence[str],
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Optional[Section]:
    if pooled_runs.empty:
        return None
    fam_label = _family_label(args)
    section = Section("Pooled Multi-condition", icon="🌐")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in pooled_runs.iterrows()
    }
    section.add_markdown(
        (
            "Shared fits across all requested conditions. "
            f"{f'Descriptor families: **{fam_label}**. ' if fam_label else ''}"
            "Condition-separation scores show EO vs EC-style pooled separability when available."
        )
    )
    merged = pooled_runs.merge(
        pooled_eval_runs.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]] if not pooled_eval_runs.empty else pd.DataFrame(columns=["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]),
        on="fit_id",
        how="left",
    )
    if args.analysis_mode == "family":
        _add_unit_summary(
            section,
            merged,
            args=args,
            eval_specs=eval_specs,
            meta_df=meta_df,
            artifacts=artifacts,
            output_root=output_root,
            subjects=subjects,
            unit_label="family",
            title_prefix="Pooled",
            input_mode=args.input_mode,
            selection_metric=args.selection_metric,
            selection_eval_name=getattr(args, "selection_eval_name", None),
        )
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue
            fig = go.Figure()
            comparison_metrics = [
                metric
                for metric in ["trustworthiness", "continuity", "shepard_correlation"]
                if metric in reducer_runs.columns
            ]
            if SEPARATION_METRIC_KEY in reducer_runs.columns:
                comparison_metrics.append(SEPARATION_METRIC_KEY)
            comparison_metrics = list(dict.fromkeys(comparison_metrics))
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = (
                    _selection_view(
                        family_runs,
                        args.selection_metric,
                        getattr(args, "selection_eval_name", None),
                    ).sort_values(
                        [args.selection_metric, "trustworthiness", "continuity"],
                        ascending=[False, False, False],
                        na_position="last",
                    )
                    .groupby("n_components", dropna=False)
                    .head(1)
                    .sort_values("n_components")
                )
                for metric_name in comparison_metrics:
                    metric_df = family_best_by_n.dropna(subset=[metric_name])
                    if metric_df.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=metric_df["n_components"],
                            y=metric_df[metric_name],
                            mode="lines+markers",
                            name=f"{family}: {metric_name}",
                        )
                    )
            if fig.data:
                fig.update_layout(
                    title=f"Pooled - {reducer_name} family comparison vs n_components",
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                section.add_element(PlotlyElement(fig))
        return section
    if args.analysis_mode == "flat":
        section.add_element(
            InteractiveTableElement(
                merged.loc[
                    :,
                    [
                        col for col in
                        ["reducer", "n_components", "trustworthiness", "continuity", "eval_name", SEPARATION_METRIC_KEY]
                        if col in merged.columns
                    ],
                ].round(4),
                title="Pooled fit ranking",
                selector_columns=["reducer", "eval_name"],
                default_sort={"column": "trustworthiness", "direction": "desc"},
                page_size=5,
            )
        )
        # Best embedding visualisation per reducer (mirrors per-condition flat sections)
        pooled_container = None
        try:
            pooled_container = load_container(args, subjects, meta_df, args.conditions[0], target_col=None)
        except Exception:
            pooled_container = None
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy() if not merged.empty else pd.DataFrame()
            if reducer_runs.empty:
                continue
            best_row = (
                _selection_view(
                    reducer_runs,
                    args.selection_metric,
                    getattr(args, "selection_eval_name", None),
                )
                .sort_values(
                    [args.selection_metric, "trustworthiness", "continuity"],
                    ascending=[False, False, False],
                    na_position="last",
                )
                .iloc[0]
            )
            best_artifact = artifacts[str(best_row["fit_id"])]
            section.add_markdown(f"### {reducer_name} (best n={int(best_row['n_components'])})")
            pool_meta = (
                _build_meta_dict(pooled_container, best_artifact["ids"], eval_specs)
                if pooled_container is not None
                else {}
            )
            _add_best_fit_plots(
                section,
                f"Pooled - {reducer_name}",
                best_artifact,
                pool_meta,
                interactive=args.interactive,
                compress_viz_with_pca=args.compress_viz_with_pca,
                feature_names=_get_feature_names(pooled_container),
            )
    elif args.analysis_mode in {"sensor_within_family", "sensor_within_subfamily"}:
        group_columns = ["family"]
        if (
            args.analysis_mode == "sensor_within_subfamily"
            and "subfamily" in merged.columns
        ):
            group_columns.append("subfamily")
        for group_key, family_runs in merged.groupby(group_columns, dropna=False):
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_label = " / ".join(str(value) for value in group_values)
            section.add_markdown(f"### {group_label}")
            _add_unit_summary(
                section,
                family_runs,
                args=args,
                eval_specs=eval_specs,
                meta_df=meta_df,
                artifacts=artifacts,
                output_root=output_root,
                subjects=subjects,
                unit_label="sensor",
                title_prefix=f"Pooled - {group_label}",
                input_mode=args.input_mode,
                selection_metric=args.selection_metric,
                selection_eval_name=getattr(args, "selection_eval_name", None),
            )
    else:
        unit_label = _unit_label(args.analysis_mode)
        _add_unit_summary(
            section,
            merged,
            args=args,
            eval_specs=eval_specs,
            meta_df=meta_df,
            artifacts=artifacts,
            output_root=output_root,
            subjects=subjects,
            unit_label=unit_label,
            title_prefix="Pooled",
            input_mode=args.input_mode,
            selection_metric=args.selection_metric,
            selection_eval_name=getattr(args, "selection_eval_name", None),
        )
    return section


def generate_dataset_report(
    args,
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    reducers: Sequence[str],
    subjects: Optional[Sequence[str]],
    meta_df: Optional[pd.DataFrame],
    containers_by_scope: Optional[dict[tuple[str, str], Any]],
    metric_columns: Sequence[str],
    eval_specs: Sequence[dict[str, Any]],
    pooled_condition: str,
) -> Report:
    fit_runs_df = _filter_runs(pd.DataFrame(load_fit_runs(fit_runs_path)), args, reducers, pooled_condition)

    if eval_runs_path.exists():
        eval_runs_df = _filter_runs(pd.DataFrame(json.loads(eval_runs_path.read_text(encoding="utf-8"))), args, reducers, pooled_condition)
    else:
        eval_runs_df = pd.DataFrame()
    if not eval_runs_df.empty and eval_specs:
        wanted_eval_names = {spec["name"] for spec in eval_specs}
        if "eval_name" in eval_runs_df.columns:
            eval_runs_df = eval_runs_df[eval_runs_df["eval_name"].isin(wanted_eval_names)].copy()

    _available_eval_metrics = [col for col in EVAL_METRIC_COLUMNS if col in eval_runs_df.columns]
    if eval_runs_df.empty or not _available_eval_metrics:
        eval_frame = pd.DataFrame()
    else:
        _eval_base_cols = [
            "fit_id", "scope", "condition", "analysis_mode",
            "family", "unit_name", "eval_name", "target_col", "reducer", "n_components",
        ]
        eval_frame = eval_runs_df.loc[
            eval_runs_df["status"] == "success",
            [col for col in [*_eval_base_cols, *_available_eval_metrics] if col in eval_runs_df.columns],
        ].copy()
    fit_eval_ranking = fit_runs_df[fit_runs_df["status"] == "success"].copy().merge(
        eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]] if not eval_frame.empty else pd.DataFrame(columns=["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]),
        on="fit_id",
        how="left",
    )

    fam_label = _family_label(args)
    run_variant = getattr(args, "run_variant", f"{args.analysis_mode}__{args.representation}")
    run_label = getattr(args, "run_label", None) or args.dataset_name
    report_title = f"Dimensionality Reduction: {run_label} ({args.input_mode} / {run_variant})"
    if fam_label:
        report_title += f" [{fam_label}]"
    report = Report(title=report_title)
    if containers_by_scope:
        for (scope, condition), container in containers_by_scope.items():
            qc_result = (container.meta or {}).get("qc_result")
            if qc_result is None:
                continue
            qc_section = build_qc_section(qc_result)
            qc_section.title = f"Data Quality (QC): {scope} / {condition}"
            report.add_section(qc_section)

    overview_sec = Section("Overview", icon="📋")
    overview_sec.add_element(
        TableElement(
            pd.DataFrame(
                [
                    {
                        "dataset_name": args.dataset_name,
                        "run_label": run_label,
                        "input_mode": args.input_mode,
                        "analysis_mode": args.analysis_mode,
                        "representation": args.representation,
                        "aggregation_unit": getattr(args, "aggregation_unit", ""),
                        "run_variant": run_variant,
                        "descriptor_families": fam_label,
                        "descriptor_max_abs_value": getattr(args, "descriptor_max_abs_value", ""),
                        "reducers": ", ".join(reducers),
                        "n_components_sweep": ", ".join(map(str, args.n_components_sweep)),
                        "conditions": ", ".join(args.conditions),
                        "eval_names": ", ".join(spec["name"] for spec in eval_specs) if eval_specs else "",
                        "selection_metric": args.selection_metric,
                        "selection_eval_name": getattr(args, "selection_eval_name", None) or "",
                        "run_pooled": args.run_pooled,
                    }
                ]
            ),
            title="Run configuration",
        )
    )
    _add_data_availability_summary(
        overview_sec,
        args,
        containers_by_scope,
        fit_runs_df,
        pooled_condition,
    )

    # --- Best overall summary (headline numbers without digging into condition sections) ---
    success_runs = fit_eval_ranking[fit_eval_ranking["status"] == "success"].copy()
    if not success_runs.empty:
        sort_metric = args.selection_metric if args.selection_metric in success_runs.columns else "trustworthiness"
        best_overall = (
            success_runs.sort_values(sort_metric, ascending=False, na_position="last").iloc[0]
        )
        best_row_dict: dict[str, Any] = {
            "best_reducer": best_overall.get("reducer", ""),
            "best_n_components": int(best_overall.get("n_components", 0)) if pd.notna(best_overall.get("n_components")) else "",
            "best_condition": best_overall.get("condition", ""),
        }
        for col in ["trustworthiness", "continuity", SEPARATION_METRIC_KEY]:
            if col in best_overall.index and pd.notna(best_overall[col]):
                best_row_dict[col] = round(float(best_overall[col]), 4)
        overview_sec.add_element(
            TableElement(
                pd.DataFrame([best_row_dict]),
                title=f"Best overall run (by {sort_metric})",
            )
        )

    _add_overview_cohort_summary(
        overview_sec,
        args,
        eval_specs,
        subjects,
        meta_df,
        containers_by_scope,
        pooled_condition,
    )
    report.add_section(overview_sec)

    # --- Fit / eval failure sections ---
    fit_failures = fit_runs_df[fit_runs_df["status"] != "success"].copy()
    if not fit_failures.empty:
        failures_sec = Section("Fit Failures", icon="⚠️")
        failures_sec.add_element(
            InteractiveTableElement(
                fit_failures.loc[:, [column for column in _FIT_FAILURE_COLUMNS if column in fit_failures.columns]],
                title="Failed fits",
                selector_columns=[column for column in ["scope", "condition", "family", "unit_name", "reducer"] if column in fit_failures.columns],
                default_sort={"column": "condition", "direction": "asc"} if "condition" in fit_failures.columns else None,
                page_size=5,
            )
        )
        report.add_section(failures_sec)

    eval_failures = eval_runs_df[eval_runs_df["status"] != "success"].copy()
    if not eval_failures.empty:
        failures_sec = Section("Eval Failures", icon="⚠️")
        failures_sec.add_element(
            InteractiveTableElement(
                eval_failures.loc[:, [column for column in _EVAL_FAILURE_COLUMNS if column in eval_failures.columns]],
                title="Failed evals",
                selector_columns=[column for column in ["scope", "condition", "family", "unit_name", "reducer", "eval_name"] if column in eval_failures.columns],
                default_sort={"column": "condition", "direction": "asc"} if "condition" in eval_failures.columns else None,
                page_size=5,
            )
        )
        report.add_section(failures_sec)

    if not eval_frame.empty:
        _default_eval_sort_col = SEPARATION_METRIC_KEY if SEPARATION_METRIC_KEY in eval_frame.columns else (eval_frame.columns[-1] if len(eval_frame.columns) else "eval_name")
        eval_sec = Section("Evaluation Results", icon="🧪")
        eval_sec.add_element(
            InteractiveTableElement(
                eval_frame.round(4),
                title="Post-hoc evaluations",
                selector_columns=[column for column in ["scope", "condition", "family", "unit_name", "reducer", "eval_name"] if column in eval_frame.columns],
                default_sort={"column": _default_eval_sort_col, "direction": "desc"},
                page_size=5,
            )
        )
        report.add_section(eval_sec)

    # --- Condition Ranking section with cross-condition comparison chart ---
    condition_runs = fit_eval_ranking[
        (fit_eval_ranking["scope"] == "condition") & (fit_eval_ranking["status"] == "success")
    ].copy()
    ranking_cols = [
        column
        for column in ["condition", "family", "unit_name", "reducer", "n_components", "trustworthiness", "continuity", "eval_name", SEPARATION_METRIC_KEY]
        if column in condition_runs.columns
    ]
    ranking_sec = Section("Condition Ranking", icon="🏁")
    ranking_sec.add_element(
        InteractiveTableElement(
            condition_runs.loc[:, ranking_cols].round(4),
            title="Condition ranking",
            selector_columns=[column for column in ["condition", "family", "unit_name", "reducer", "eval_name"] if column in ranking_cols],
            default_sort={"column": "trustworthiness", "direction": "desc"},
            page_size=5,
        )
    )
    # Cross-condition comparison bar chart: best metric per (condition × reducer)
    if not condition_runs.empty and len(args.conditions) > 1:
        compare_metric = args.selection_metric if args.selection_metric in condition_runs.columns else "trustworthiness"
        cond_compare_fig = go.Figure()
        for reducer_name in reducers:
            reducer_cond_runs = condition_runs[condition_runs["reducer"] == reducer_name].copy()
            if reducer_cond_runs.empty:
                continue
            best_per_condition = (
                reducer_cond_runs
                .sort_values(compare_metric, ascending=False, na_position="last")
                .groupby("condition", dropna=False)
                .head(1)
            )
            best_per_condition = best_per_condition.set_index("condition").reindex(args.conditions).reset_index()
            cond_compare_fig.add_trace(
                go.Bar(
                    name=reducer_name,
                    x=best_per_condition["condition"],
                    y=best_per_condition[compare_metric],
                    text=best_per_condition["n_components"].map(
                        lambda v: f"n={int(v)}" if pd.notna(v) else ""
                    ),
                    textposition="outside",
                )
            )
        if cond_compare_fig.data:
            cond_compare_fig.update_layout(
                title=f"Best {compare_metric} per condition × reducer",
                xaxis_title="condition",
                yaxis_title=compare_metric,
                barmode="group",
                legend_title="Reducer",
            )
            ranking_sec.add_element(PlotlyElement(cond_compare_fig))

    # Radar chart: reducer × metric comparison (best value per reducer across all conditions)
    radar_metric_cols = [
        m for m in ["trustworthiness", "continuity", "shepard_correlation", SEPARATION_METRIC_KEY]
        if m in condition_runs.columns
    ]
    if len(reducers) > 1 and len(radar_metric_cols) >= 3:
        radar_rows: dict[str, dict[str, float]] = {}
        for reducer_name in reducers:
            reducer_cond_runs = condition_runs[condition_runs["reducer"] == reducer_name]
            if reducer_cond_runs.empty:
                continue
            row: dict[str, float] = {}
            for m in radar_metric_cols:
                vals = reducer_cond_runs[m].dropna()
                if not vals.empty:
                    row[m] = float(vals.max())
            if len(row) >= 3:
                radar_rows[reducer_name] = row
        if len(radar_rows) > 1:
            radar_df = pd.DataFrame(radar_rows).T
            ranking_sec.add_element(
                PlotlyElement(
                    plot_radar_comparison(
                        radar_df,
                        title="Reducer comparison — best metric across conditions",
                    )
                )
            )

    report.add_section(ranking_sec)

    # --- Per-condition sections ---
    if args.analysis_mode == "flat":
        for condition in args.conditions:
            condition_fit_runs = fit_runs_df[
                (fit_runs_df["scope"] == "condition")
                & (fit_runs_df["condition"] == condition)
                & (fit_runs_df["status"] == "success")
            ].copy()
            if condition_fit_runs.empty:
                continue
            report.add_section(
                _build_flat_condition_section(
                    args,
                    output_root,
                    condition,
                    condition_fit_runs,
                    eval_frame[eval_frame["condition"] == condition].copy() if not eval_frame.empty else pd.DataFrame(),
                    subjects,
                    meta_df,
                    eval_specs,
                    reducers,
                )
            )
    else:
        for condition in args.conditions:
            condition_fit_runs = fit_runs_df[
                (fit_runs_df["scope"] == "condition")
                & (fit_runs_df["condition"] == condition)
                & (fit_runs_df["status"] == "success")
            ].copy()
            if condition_fit_runs.empty:
                continue
            report.add_section(
                _build_nonflat_condition_section(
                    args,
                    output_root,
                    condition,
                    condition_fit_runs,
                    eval_frame[eval_frame["condition"] == condition].copy() if not eval_frame.empty else pd.DataFrame(),
                    metric_columns,
                    subjects,
                    meta_df,
                    eval_specs,
                    reducers,
                )
            )

    if args.run_pooled:
        pooled_runs = fit_runs_df[
            (fit_runs_df["scope"] == "pooled")
            & (fit_runs_df["condition"] == pooled_condition)
            & (fit_runs_df["status"] == "success")
        ].copy()
        pooled_eval = (
            eval_frame[
                (eval_frame["scope"] == "pooled")
                & (eval_frame["condition"] == pooled_condition)
            ].copy()
            if not eval_frame.empty
            else pd.DataFrame()
        )
        pooled_section = _build_pooled_section(
            args,
            output_root,
            pooled_runs,
            pooled_eval,
            metric_columns,
            subjects,
            meta_df,
            eval_specs,
            reducers,
        )
        if pooled_section is not None:
            report.add_section(pooled_section)

    # --- Condition statistics section (paired t-tests across conditions) ---
    stats_metrics = [
        m for m in ["trustworthiness", "continuity", SEPARATION_METRIC_KEY]
        if m in fit_eval_ranking.columns
    ]
    stats_section = _build_condition_stats_section(args, fit_eval_ranking, stats_metrics)
    if stats_section is not None:
        report.add_section(stats_section)

    return report

"""Dimensionality-reduction report assembly."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.report.core import Report, Section
from coco_pipe.report.elements import (
    ImageElement,
    InteractiveTableElement,
    PlotlyElement,
    TableElement,
)
from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz.plotly_utils import plot_embedding_interactive

from eeg_adhd_epilepsy.io.analysis import concat_containers, load_container
from eeg_adhd_epilepsy.io.bids import get_reports_root
from eeg_adhd_epilepsy.utils.metadata_schema import EPILEPSY_MED_COLS
from eeg_adhd_epilepsy.viz.topo import plot_topomap_from_channel_values, plot_topomap_selector
from eeg_adhd_epilepsy.viz.utils import save_fig

logger = logging.getLogger(__name__)

SEPARATION_METRIC_KEY = "separation_logreg_balanced_accuracy"

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


def load_fit_artifact(path: Path) -> dict[str, Any]:
    manifest_path = path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    fit_path = path / manifest.get("fit", "fit.json")
    metrics_path = path / manifest.get("metrics", "metrics.json")
    embedding_path = path / manifest.get("embedding", "embedding.npy")
    ids_path = path / manifest.get("ids", "ids.npy")
    diagnostics_path = path / manifest.get("diagnostics", "diagnostics.npz")
    if not fit_path.exists():
        fit_matches = sorted(path.glob("*_fit.json"))
        if fit_matches:
            fit_path = fit_matches[0]
    if not metrics_path.exists():
        metrics_matches = sorted(path.glob("*_metrics.json"))
        if metrics_matches:
            metrics_path = metrics_matches[0]
    if not embedding_path.exists():
        embedding_matches = sorted(path.glob("*_embedding.npy"))
        if embedding_matches:
            embedding_path = embedding_matches[0]
    if not ids_path.exists():
        ids_matches = sorted(path.glob("*_ids.npy"))
        if ids_matches:
            ids_path = ids_matches[0]
    if not diagnostics_path.exists():
        diagnostics_matches = sorted(path.glob("*_diagnostics.npz"))
        if diagnostics_matches:
            diagnostics_path = diagnostics_matches[0]

    diagnostics = {}
    if diagnostics_path.exists():
        with np.load(diagnostics_path, allow_pickle=True) as npz:
            diagnostics = dict(npz["payload"][0])
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "embedding": np.load(embedding_path, allow_pickle=True),
        "ids": np.load(ids_path, allow_pickle=True),
        "fit": fit,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "manifest": manifest,
        "path": path,
    }


def load_fit_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"No fit runs found in {path}.")
    runs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(runs, list):
        raise ValueError(f"Expected list payload in {path}.")
    return runs


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
            return concat_containers(condition_containers)

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
    return concat_containers(loaded)


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
) -> Section:
    container = load_container(args, subjects, meta_df, condition, target_col=None)
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    family_label = (
        ", ".join(args.descriptor_families)
        if args.input_mode == "descriptors" and args.descriptor_families
        else "all descriptor families"
        if args.input_mode == "descriptors"
        else ""
    )
    section = Section(condition, icon="🧠")
    section.add_markdown(
        (
            f"Input mode: **{args.input_mode}**. "
            f"{f'Descriptor families: **{family_label}**. ' if family_label else ''}"
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

    for reducer_name in args.reducers_resolved:
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

    group_columns = [
        column
        for column in ["family", "eval_name", "target_col"]
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
        unit_runs.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        .groupby([*group_columns, "unit_name"], dropna=False)
        .head(1)
        .copy()
    )
    display_columns = [
        column
        for column in [*group_columns, "unit_name", "reducer", "n_components", *sort_columns]
        if column in best_units.columns
    ]
    section.add_element(
        InteractiveTableElement(
            best_units.loc[:, display_columns].round(4),
            title=f"{title_prefix} {unit_label} ranking",
            selector_columns=[column for column in [*group_columns, "reducer"] if column in best_units.columns],
            default_sort={"column": sort_columns[0], "direction": "desc"} if sort_columns else None,
            page_size=5,
        )
    )

    sweep_df = unit_runs.loc[
        :,
        [
            column
            for column in [*group_columns, "unit_name", "reducer", "n_components", *sort_columns]
            if column in unit_runs.columns
        ],
    ].copy()
    section.add_element(
        InteractiveTableElement(
            sweep_df.round(4),
            title=f"{title_prefix} {unit_label} sweep",
            selector_columns=[column for column in [*group_columns, "unit_name", "reducer"] if column in sweep_df.columns],
            page_size=5,
        )
    )

    best_by_unit = best_units.copy()
    plot_metric = sort_columns[0] if sort_columns else "trustworthiness"
    grouped_best = list(best_by_unit.groupby(group_columns, dropna=False)) if group_columns else [((), best_by_unit)]
    bar_fig = go.Figure()
    for idx, (group_key, group_df) in enumerate(grouped_best):
        plot_df = group_df.dropna(subset=[plot_metric]).copy()
        if plot_df.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label_parts = [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
        trace_label = " / ".join(label_parts) if label_parts else plot_metric
        bar_fig.add_trace(
            go.Bar(
                x=plot_df["unit_name"].astype(str),
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
                                {"title": f"{title_prefix} best {plot_metric} by {unit_label} - {trace.name}"},
                            ],
                        }
                        for idx, trace in enumerate(bar_fig.data)
                    ],
                }
            ]
        )
    if bar_fig.data:
        first_bar_label = bar_fig.data[0].name
        bar_fig.update_layout(
            title=f"{title_prefix} best {plot_metric} by {unit_label}{f' - {first_bar_label}' if first_bar_label and len(bar_fig.data) > 1 else ''}",
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
            label_parts = [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
            topo_label = " / ".join(label_parts) if label_parts else plot_metric
            topo_groups[topo_label] = (
                topo_df["unit_name"].astype(str).tolist(),
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
) -> Section:
    unit_label = "family" if args.analysis_mode == "family" else "sensor"
    family_label = (
        ", ".join(args.descriptor_families)
        if args.input_mode == "descriptors" and args.descriptor_families
        else "all descriptor families"
        if args.input_mode == "descriptors"
        else ""
    )
    section = Section(condition, icon="📊")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    intro = f"Primary analysis unit: **{unit_label}**."
    if args.analysis_mode == "sensor_within_family":
        intro = "Primary analysis unit: **sensor within family**."
    section.add_markdown(
        (
            f"Input mode: **{args.input_mode}**. "
            f"{f'Descriptor families: **{family_label}**. ' if family_label else ''}"
            f"{intro}"
        )
    )

    merged = condition_runs.merge(
        eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]],
        on="fit_id",
        how="left",
    )
    if args.analysis_mode == "family":
        artifacts = {
            str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
            for _, row in condition_runs.iterrows()
        }
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
        for reducer_name in args.reducers_resolved:
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
    if args.analysis_mode == "sensor_within_family":
        for family, family_runs in merged.groupby("family", dropna=False):
            section.add_markdown(f"### {family}")
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
                title_prefix=f"{condition} - {family}",
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
        for reducer_name in args.reducers_resolved:
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
) -> Optional[Section]:
    if pooled_runs.empty:
        return None
    family_label = (
        ", ".join(args.descriptor_families)
        if args.input_mode == "descriptors" and args.descriptor_families
        else "all descriptor families"
        if args.input_mode == "descriptors"
        else ""
    )
    section = Section("Pooled Multi-condition", icon="🌐")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in pooled_runs.iterrows()
    }
    section.add_markdown(
        (
            "Shared fits across all requested conditions. "
            f"{f'Descriptor families: **{family_label}**. ' if family_label else ''}"
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
        for reducer_name in args.reducers_resolved:
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
                    ["reducer", "n_components", "trustworthiness", "continuity", "eval_name", SEPARATION_METRIC_KEY],
                ].round(4),
                title="Pooled fit ranking",
                selector_columns=["reducer", "eval_name"],
                default_sort={"column": "trustworthiness", "direction": "desc"},
                page_size=5,
            )
        )
    elif args.analysis_mode == "sensor_within_family":
        for family, family_runs in merged.groupby("family", dropna=False):
            section.add_markdown(f"### {family}")
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
                title_prefix=f"Pooled - {family}",
                input_mode=args.input_mode,
                selection_metric=args.selection_metric,
                selection_eval_name=getattr(args, "selection_eval_name", None),
            )
    else:
        unit_label = "family" if args.analysis_mode == "family" else "sensor"
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

    if eval_runs_df.empty or SEPARATION_METRIC_KEY not in eval_runs_df.columns:
        eval_frame = pd.DataFrame()
    else:
        eval_frame = eval_runs_df.loc[
        eval_runs_df["status"] == "success",
        [
            "fit_id",
            "scope",
            "condition",
            "analysis_mode",
            "family",
            "unit_name",
            "eval_name",
            "target_col",
            "reducer",
            "n_components",
            SEPARATION_METRIC_KEY,
        ],
        ].copy()
    fit_eval_ranking = fit_runs_df[fit_runs_df["status"] == "success"].copy().merge(
        eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]] if not eval_frame.empty else pd.DataFrame(columns=["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]),
        on="fit_id",
        how="left",
    )

    family_label = (
        ", ".join(args.descriptor_families)
        if args.input_mode == "descriptors" and args.descriptor_families
        else "all descriptor families"
        if args.input_mode == "descriptors"
        else ""
    )
    run_variant = getattr(args, "run_variant", f"{args.analysis_mode}__{args.representation}")
    run_label = getattr(args, "run_label", None) or args.dataset_name
    report_title = f"Dimensionality Reduction: {run_label} ({args.input_mode} / {run_variant})"
    if family_label:
        report_title += f" [{family_label}]"
    report = Report(title=report_title)

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
                        "descriptor_families": family_label,
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
        eval_sec = Section("Evaluation Results", icon="🧪")
        eval_sec.add_element(
            InteractiveTableElement(
                eval_frame.round(4),
                title="Post-hoc evaluations",
                selector_columns=[column for column in ["scope", "condition", "family", "unit_name", "reducer", "eval_name"] if column in eval_frame.columns],
                default_sort={"column": SEPARATION_METRIC_KEY, "direction": "desc"},
                page_size=5,
            )
        )
        report.add_section(eval_sec)

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
    report.add_section(ranking_sec)

    args.reducers_resolved = list(reducers)
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
                    eval_frame[eval_frame["condition"] == condition].copy(),
                    subjects,
                    meta_df,
                    eval_specs,
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
                    eval_frame[eval_frame["condition"] == condition].copy(),
                    metric_columns,
                    subjects,
                    meta_df,
                    eval_specs,
                )
            )

    if args.run_pooled:
        pooled_runs = fit_runs_df[
            (fit_runs_df["scope"] == "pooled")
            & (fit_runs_df["condition"] == pooled_condition)
            & (fit_runs_df["status"] == "success")
        ].copy()
        pooled_section = _build_pooled_section(
            args,
            output_root,
            pooled_runs,
            eval_frame[
                (eval_frame["scope"] == "pooled")
                & (eval_frame["condition"] == pooled_condition)
            ].copy(),
            metric_columns,
            subjects,
            meta_df,
            eval_specs,
        )
        if pooled_section is not None:
            report.add_section(pooled_section)

    return report

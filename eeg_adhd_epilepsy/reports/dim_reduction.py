"""Dimensionality-reduction report assembly."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coco_pipe.dim_reduction import (
    EVAL_METRIC_COLUMNS,
    SEPARATION_METRIC_KEY,
    load_fit_artifact,
    load_fit_runs,
)
from coco_pipe.report import (
    AccordionElement,
    CalloutElement,
    ColumnsElement,
    ContainerElement,
    ImageElement,
    InteractiveTableElement,
    PlotlyElement,
    Report,
    Section,
    StatCardElement,
    TableElement,
    TabsElement,
)
from coco_pipe.report.qc import build_qc_section
from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz.interactive.base import (
    plot_bar,
    plot_distribution_groups,
    plot_grouped_bar,
    plot_scatter,
)
from coco_pipe.viz.interactive.dim_reduction import (
    plot_component_loadings,
    plot_radar_comparison,
)
from coco_pipe.viz.interactive.dim_reduction import (
    plot_embedding as plot_embedding_interactive,
)

from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.metadata.schema import EPILEPSY_MED_COLS
from eeg_adhd_epilepsy.reports._common import (
    add_overview_cohort_summary,
    family_label,
    get_feature_names,
)
from eeg_adhd_epilepsy.viz.topo import (
    feature_names_are_sensors,
    plot_topomap_from_channel_values,
    plot_topomap_selector,
)

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

logger = logging.getLogger(__name__)

_PCA_DIAG_COLUMNS = ["participation_ratio", "cumulative_explained_variance"]


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


def _add_data_availability_summary(
    overview_sec: Section,
    args: Any,
    dataset_stats: list[dict[str, Any]] | None,
    fit_runs_df: pd.DataFrame,
    pooled_condition: str,
) -> None:
    scopes_conditions = [("condition", c) for c in args.conditions]
    if args.run_pooled:
        scopes_conditions.append(("pooled", pooled_condition))

    stats_map = {(s.get("scope"), s.get("condition")): s for s in (dataset_stats or [])}

    rows = []
    for scope, condition in scopes_conditions:
        stat_dict = stats_map.get((scope, condition), {})

        condition_runs = pd.DataFrame()
        if not fit_runs_df.empty:
            condition_runs = fit_runs_df[
                (fit_runs_df["scope"] == scope)
                & (fit_runs_df["condition"] == condition)
                & (fit_runs_df["status"] == "success")
            ]

        if not stat_dict and condition_runs.empty:
            continue

        reducers_str = ""
        n_comps_str = ""
        if not condition_runs.empty:
            if "reducer" in condition_runs.columns:
                reducers_str = ", ".join(
                    sorted(condition_runs["reducer"].dropna().astype(str).unique())
                )
            if "n_components" in condition_runs.columns:
                n_comps_str = ", ".join(
                    map(str, sorted(condition_runs["n_components"].dropna().astype(int).unique()))
                )

        rows.append(
            {
                "scope": scope,
                "condition": condition,
                "loaded_observations": stat_dict.get("loaded_observations", ""),
                "samples_used": stat_dict.get("samples_used", ""),
                "unique_subjects": stat_dict.get("unique_subjects", ""),
                "unique_recordings": stat_dict.get("unique_recordings", ""),
                "successful_fits": len(condition_runs),
                "reducers": reducers_str,
                "valid_n_components": n_comps_str,
            }
        )

    if rows:
        acc = AccordionElement("Show Data Availability", open=False)
        acc.add_element(
            InteractiveTableElement(pd.DataFrame(rows), title="Data Availability", page_size=10)
        )
        overview_sec.add_element(acc)


def _add_embedding_plot(
    embedding: np.ndarray,
    meta: dict[str, np.ndarray] | None,
    title: str,
    dimensions: int,
    interactive: bool,
):
    if interactive:
        return PlotlyElement(
            plot_embedding_interactive(
                embedding=embedding,
                metadata=meta,
                title=title,
                dimensions=dimensions,
            )
        )

    dims = (0, 1) if dimensions == 2 else (0, 1, 2)
    fig, _ = viz.plot_embedding(
        X_emb=embedding,
        metadata=meta,
        dims=dims,
        title=title,
    )
    return ImageElement(fig)


def _add_best_fit_plots(
    title: str,
    artifact: dict[str, Any],
    meta_dict: dict[str, np.ndarray],
    interactive: bool,
    feature_names: list[str] | None = None,
):
    embedding = np.asarray(artifact["embedding"])
    if embedding.ndim != 2:
        return

    plots = []

    if embedding.shape[1] == 2:
        plots.append(
            _add_embedding_plot(embedding, meta_dict, f"{title} - native 2D", 2, interactive)
        )
    elif embedding.shape[1] >= 3:
        plots.append(
            _add_embedding_plot(
                embedding[:, :3], meta_dict, f"{title} - first 3 dims", 3, interactive
            )
        )

    components = (artifact.get("diagnostics") or {}).get("components")
    if components is not None:
        loadings = np.asarray(components, dtype=float).T
        if loadings.ndim == 2:
            n_comp = min(loadings.shape[1], 10)
            topo_fig = None
            if feature_names_are_sensors(feature_names):
                topo_fig = plot_topomap_selector(
                    {f"PC{i + 1}": (feature_names, loadings[:, i]) for i in range(n_comp)},
                    title=f"{title} - component loadings (topomap, top {n_comp})",
                    unit="loading",
                )
            if topo_fig is not None:
                plots.append(PlotlyElement(topo_fig))
            else:
                plots.append(
                    PlotlyElement(
                        plot_component_loadings(
                            loadings,
                            feature_names=feature_names,
                            n_components=n_comp,
                            title=f"{title} - component loadings (top {n_comp})",
                        )
                    )
                )

    if plots:
        return ColumnsElement(plots, cols=len(plots))
    return None


def _build_meta_dict(
    container,
    ids: np.ndarray | None = None,
    eval_specs: Sequence[dict[str, Any]] = (),
) -> dict[str, np.ndarray]:
    # 1. Fetch and format metadata frame
    frame = container.observation_frame()
    frame = frame.drop(columns=["feature"], errors="ignore")
    if container.y is not None and "y" not in frame.columns:
        frame["y"] = np.asarray(container.y)
    frame = frame.rename(columns={"sample_id": "obs_id"})

    # 2. Align frame rows to match requested ids
    if ids is not None and "obs_id" in frame.columns:
        requested_ids = np.asarray(ids, dtype=object).astype(str)
        is_aligned = len(requested_ids) == len(frame) and np.array_equal(
            requested_ids, frame["obs_id"].astype(str).to_numpy()
        )
        if not is_aligned:
            keyed = frame.set_index("obs_id")
            if all(key in keyed.index for key in requested_ids):
                frame = keyed.loc[requested_ids].reset_index()
            elif len(frame) != len(requested_ids):
                frame = pd.DataFrame()

    if frame.empty:
        return {}

    meta: dict[str, np.ndarray] = {}

    # 3. Extract valid standard columns
    for col_name in frame.columns:
        col_str = str(col_name)
        normalized = "".join(ch for ch in col_str.lower() if ch.isalnum())

        is_excluded = (
            col_str in _PLOT_META_EXCLUDED_COLUMNS
            or normalized in _PLOT_META_EXCLUDED_NORMALIZED
            or "psychostimulant" in normalized
            or col_str.endswith(("_bool", "_clean"))
        )
        if is_excluded:
            continue

        if 1 < frame[col_str].nunique(dropna=False) <= 200:
            meta[col_str] = frame[col_str].to_numpy()

    # 4. Extract columns defined in evaluation specs (YAML)
    for spec in eval_specs:
        target_col = spec.get("target_col")
        if target_col not in frame.columns:
            continue

        labels = frame[target_col].astype(str)
        if label_map := spec.get("label_map"):
            labels = labels.map(lambda v: label_map.get(v, v))

        labels = labels.replace({"nan": "unknown", "None": "unknown", "": "unknown"})

        if filters := spec.get("filters"):
            mask = pd.Series(True, index=frame.index)
            for f_spec in filters:
                col = f_spec["column"]
                if col in frame.columns:
                    valid_vals = {str(v) for v in f_spec["values"]}
                    mask &= frame[col].astype(str).isin(valid_vals)
                else:
                    mask[:] = False
                    break
            labels = labels.where(mask, "unknown")

        if 1 < labels.nunique(dropna=False) <= 200:
            meta[str(spec["name"])] = labels.to_numpy(dtype=object)

    # 5. Extract eye state if condition exists
    if "condition" in meta:

        def _get_eye_state(val):
            v = str(val).lower()
            if v.startswith("eo"):
                return "EO"
            if v.startswith("ec"):
                return "EC"
            return str(val)

        eye_state = np.array([_get_eye_state(v) for v in meta["condition"]], dtype=object)
        if len(np.unique(eye_state)) > 1:
            meta["eye_state"] = eye_state

    return meta


def _build_flat_condition_section(
    args,
    output_root: Path,
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    meta_df: pd.DataFrame | None,
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Section:
    container = build_dataset(args, meta_df, condition, target_col=None)
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    fam_label = family_label(args)
    section = Section(condition, icon="🧠")

    callout_text = (
        f"Input mode: **{args.input_mode}**<br/>Representation: **{args.representation}**"
    )
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(CalloutElement(callout_text, kind="info", title="Configuration Details"))

    section.add_element(
        ColumnsElement(
            [
                StatCardElement(
                    "Observations",
                    container.meta.get("loaded_obs", container.X.shape[0]),
                    color="blue",
                ),
                StatCardElement("Successful Fits", len(condition_runs), color="green"),
            ],
            cols=4,
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
                [
                    col
                    for col in [
                        "reducer",
                        "n_components",
                        "eval_name",
                        SEPARATION_METRIC_KEY,
                        "trustworthiness",
                        "continuity",
                    ]
                    if col in ranking_df.columns
                ],
            ].round(4),
            title="Fit ranking",
            selector_columns=["reducer", "eval_name"],
            default_sort={"column": args.selection_metric, "direction": "desc"},
            page_size=5,
        )
    )

    reducer_tabs = {}
    for reducer_name in reducers:
        reducer_runs = condition_runs[condition_runs["reducer"] == reducer_name].copy()
        if reducer_runs.empty:
            continue
        best_row = (
            reducer_runs.merge(
                eval_frame.loc[:, ["fit_id", "eval_name", SEPARATION_METRIC_KEY]],
                on="fit_id",
                how="left",
            )
            .sort_values(
                [args.selection_metric],
                ascending=[False],
                na_position="last",
            )
            .iloc[0]
        )
        best_artifact = artifacts[str(best_row["fit_id"])]

        tab_elements = []
        meta_dict = _build_meta_dict(container, best_artifact["ids"], eval_specs)

        plots_elem = _add_best_fit_plots(
            f"{condition} - {reducer_name}",
            best_artifact,
            meta_dict,
            interactive=args.interactive,
            feature_names=get_feature_names(container),
        )
        if plots_elem:
            tab_elements.append(plots_elem)

        sweep_df = reducer_runs.merge(
            eval_frame.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]],
            on="fit_id",
            how="left",
        )
        if not sweep_df.empty and SEPARATION_METRIC_KEY in sweep_df.columns:
            sep_df = sweep_df.dropna(subset=[SEPARATION_METRIC_KEY]).copy()
            sep_df["series"] = "separation: " + sep_df["eval_name"].astype(str)
            fig = plot_scatter(
                sep_df,
                x="n_components",
                y=SEPARATION_METRIC_KEY,
                color="series",
                mode="lines+markers",
                title=f"{condition} - {reducer_name} separation vs n_components",
                xaxis_title="n_components",
                yaxis_title="score",
            )
            acc = AccordionElement("Show Hyperparameter Sweep Data", open=False)
            acc.add_element(PlotlyElement(fig))

            sweep_table = InteractiveTableElement(
                sweep_df.loc[
                    :,
                    [
                        col
                        for col in [
                            "n_components",
                            "eval_name",
                            SEPARATION_METRIC_KEY,
                            "trustworthiness",
                            "continuity",
                        ]
                        if col in sweep_df.columns
                    ],
                ].round(4),
                title=f"{condition} - {reducer_name} sweep summary",
                page_size=5,
            )
            acc.add_element(sweep_table)
            tab_elements.append(acc)

        if tab_elements:
            reducer_tabs[reducer_name] = (
                ColumnsElement(tab_elements, cols=1) if len(tab_elements) > 1 else tab_elements[0]
            )

    if reducer_tabs:
        section.add_element(TabsElement(reducer_tabs))

    return section


def _add_unit_summary(
    section: Section,
    unit_runs: pd.DataFrame,
    *,
    args: Any,
    unit_label: str,
    title_prefix: str,
    selection_metric: str,
) -> None:
    if unit_runs.empty:
        return

    unit_column = "unit_key" if args.analysis_mode == "descriptor_sensor" else "unit_name"

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
                column for column in [*group_columns, "reducer"] if column in best_units.columns
            ],
            default_sort=(
                {"column": sort_columns[0], "direction": "desc"} if sort_columns else None
            ),
            page_size=5,
        )
    )

    sweep_df = unit_runs.loc[
        :,
        [
            column
            for column in [
                *group_columns,
                unit_column,
                "reducer",
                "n_components",
                *sort_columns,
            ]
            if column in unit_runs.columns
        ],
    ].copy()
    acc = AccordionElement("Show Full Unit Sweep Data", open=False)
    acc.add_element(
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
    section.add_element(acc)

    best_by_unit = best_units.copy()
    plot_metric = sort_columns[0] if sort_columns else "trustworthiness"
    grouped_best = (
        list(best_by_unit.groupby(group_columns, dropna=False))
        if group_columns
        else [((), best_by_unit)]
    )
    grouped_sweep = (
        list(sweep_df.groupby(group_columns, dropna=False)) if group_columns else [((), sweep_df)]
    )

    perf_tabs = {}
    stab_tabs = {}
    topo_tabs = {}

    for idx, (group_key, group_df) in enumerate(grouped_best):
        plot_df = group_df.dropna(subset=[plot_metric]).copy()
        if plot_df.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label_parts = [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
        trace_label = " / ".join(label_parts) if label_parts else "All"

        scores_series = plot_df.set_index(unit_column)[plot_metric]
        n_comp_series = plot_df.set_index(unit_column)["n_components"]

        sep_fig = plot_bar(
            scores=scores_series,
            title=f"{title_prefix} best {plot_metric} by {unit_label}",
            xaxis_title=unit_label,
            yaxis_title=plot_metric,
        )

        n_comp_fig = plot_bar(
            scores=n_comp_series,
            title=f"{title_prefix} optimal n_components",
            xaxis_title=unit_label,
            yaxis_title="n_components",
            color="orange",
        )
        perf_tabs[trace_label] = ColumnsElement(
            [PlotlyElement(sep_fig), PlotlyElement(n_comp_fig)], cols=2
        )

        if unit_label == "sensor":
            topomaps = []
            for topo_metric in [plot_metric, "trustworthiness", "continuity"]:
                if topo_metric not in group_df.columns:
                    continue
                topo_df = group_df.dropna(subset=[topo_metric]).copy()
                if topo_df.empty:
                    continue

                topo_groups = {
                    topo_metric: (
                        topo_df[unit_column].astype(str).tolist(),
                        topo_df[topo_metric].astype(float).to_numpy(),
                    )
                }
                topo_plot = plot_topomap_selector(
                    topo_groups,
                    title=f"{title_prefix} best {topo_metric}",
                    unit=topo_metric,
                )
                if topo_plot is not None:
                    topomaps.append(PlotlyElement(topo_plot))
                else:
                    topo_label, (topo_names, topo_values) = next(iter(topo_groups.items()))
                    try:
                        topo_fig = plot_topomap_from_channel_values(
                            channel_names=topo_names,
                            values=topo_values,
                            title=f"{title_prefix} best {topo_metric} - {topo_label}",
                            unit=topo_metric,
                        )
                        if topo_fig is not None:
                            topomaps.append(ImageElement(topo_fig, width="100%"))
                    except Exception:
                        pass
            if topomaps:
                topo_tabs[trace_label] = ColumnsElement(topomaps, cols=len(topomaps))

    for idx, (group_key, group_df) in enumerate(grouped_sweep):
        plot_df = group_df.dropna(subset=[plot_metric]).copy()
        if plot_df.empty:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label_parts = [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
        trace_label = " / ".join(label_parts) if label_parts else "All"

        units = plot_df[unit_column].unique()
        groups = [plot_df[plot_df[unit_column] == u][plot_metric].values for u in units]

        box_fig = plot_distribution_groups(
            groups=groups,
            labels=units,
            kind="box",
            title=f"{title_prefix} stability by {unit_label}",
            xaxis_title=unit_label,
            yaxis_title=plot_metric,
        )
        stab_tabs[trace_label] = PlotlyElement(box_fig)

    viz_tabs = {}
    if perf_tabs:
        viz_tabs["Peak Performance"] = (
            next(iter(perf_tabs.values())) if len(perf_tabs) == 1 else TabsElement(perf_tabs)
        )
    if stab_tabs:
        viz_tabs["Hyperparameter Stability"] = (
            next(iter(stab_tabs.values())) if len(stab_tabs) == 1 else TabsElement(stab_tabs)
        )
    if topo_tabs:
        viz_tabs["Spatial Topomaps"] = (
            next(iter(topo_tabs.values())) if len(topo_tabs) == 1 else TabsElement(topo_tabs)
        )

    if viz_tabs:
        section.add_element(TabsElement(viz_tabs))


def _build_nonflat_condition_section(
    args,
    output_root: Path,
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    meta_df: pd.DataFrame | None,
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Section:
    unit_label = _UNIT_LABELS.get(args.analysis_mode, "analysis unit")
    fam_label = family_label(args)
    section = Section(condition, icon="📊")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    unit_label = _UNIT_LABELS.get(args.analysis_mode, args.analysis_mode.replace("_", " "))
    intro = f"Primary analysis unit: **{unit_label}**"
    callout_text = f"Input mode: **{args.input_mode}**<br/>{intro}"
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(CalloutElement(callout_text, kind="info", title="Configuration Details"))
    section.add_element(StatCardElement("Successful Fits", len(condition_runs), color="green"))

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
            unit_label="family",
            title_prefix=condition,
            selection_metric=args.selection_metric,
        )
        family_container = build_dataset(args, meta_df, condition, target_col=None)

        reducer_tabs = {}
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()

            comparison_metrics = []
            if SEPARATION_METRIC_KEY in reducer_runs.columns:
                comparison_metrics.append(SEPARATION_METRIC_KEY)
            for m in ["trustworthiness", "continuity"]:
                if m in reducer_runs.columns:
                    comparison_metrics.append(m)

            curve_frames = []
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = family_runs.sort_values(
                    [args.selection_metric, "trustworthiness", "continuity"],
                    ascending=[False, False, False],
                    na_position="last",
                )
                family_best_by_n = (
                    family_best_by_n.groupby("n_components", dropna=False)
                    .head(1)
                    .sort_values("n_components")
                )
                for metric_name in comparison_metrics:
                    metric_df = family_best_by_n.dropna(subset=[metric_name])
                    if metric_df.empty:
                        continue
                    curve_frames.append(
                        pd.DataFrame(
                            {
                                "n_components": metric_df["n_components"].to_numpy(),
                                "score": metric_df[metric_name].to_numpy(),
                                "series": f"{family}: {metric_name}",
                            }
                        )
                    )

            acc = AccordionElement(f"Show {reducer_name} Hyperparameter Sweep Details", open=False)
            if curve_frames:
                fig = plot_scatter(
                    pd.concat(curve_frames, ignore_index=True),
                    x="n_components",
                    y="score",
                    color="series",
                    mode="lines+markers",
                    title=f"{condition} - {reducer_name} family comparison vs n_components",
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                acc.add_element(PlotlyElement(fig))

            sweep_cols = ["family", "n_components"]
            if "eval_name" in reducer_runs.columns:
                sweep_cols.append("eval_name")
            if SEPARATION_METRIC_KEY in reducer_runs.columns:
                sweep_cols.append(SEPARATION_METRIC_KEY)
            for gm in ["trustworthiness", "continuity"]:
                if gm in reducer_runs.columns:
                    sweep_cols.append(gm)
            sweep_cols = list(dict.fromkeys(sweep_cols))

            acc.add_element(
                InteractiveTableElement(
                    reducer_runs.loc[:, sweep_cols].round(4),
                    title=f"{condition} - {reducer_name} family sweep",
                    selector_columns=["family", "eval_name"]
                    if "eval_name" in sweep_cols
                    else ["family"],
                    page_size=5,
                )
            )
            tab_section.add_element(acc)
            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

        family_tabs = {}
        for family, family_runs in merged.groupby("family", dropna=False):
            best_row = family_runs.sort_values(
                [args.selection_metric, "trustworthiness", "continuity"],
                ascending=[False, False, False],
                na_position="last",
            ).iloc[0]
            best_artifact = artifacts[str(best_row["fit_id"])]
            family_meta = _build_meta_dict(family_container, best_artifact["ids"], eval_specs)

            fam_container = ContainerElement()
            fam_container.add_element(
                CalloutElement(
                    f"Best fit for family **{family}** uses **{best_row['reducer']}** "
                    f"with n={int(best_row['n_components'])}",
                    kind="tip",
                )
            )

            plots_elem = _add_best_fit_plots(
                f"{condition} - {family}",
                best_artifact,
                family_meta,
                interactive=args.interactive,
                feature_names=get_feature_names(family_container),
            )
            if plots_elem:
                fam_container.add_element(plots_elem)

            sweep_cols = ["reducer", "n_components"]
            if "eval_name" in family_runs.columns:
                sweep_cols.append("eval_name")
            if SEPARATION_METRIC_KEY in family_runs.columns:
                sweep_cols.append(SEPARATION_METRIC_KEY)
            for gm in ["trustworthiness", "continuity"]:
                if gm in family_runs.columns:
                    sweep_cols.append(gm)
            sweep_cols = list(dict.fromkeys(sweep_cols))

            acc = AccordionElement("Show Reducer Sweep Details", open=False)
            acc.add_element(
                InteractiveTableElement(
                    family_runs.loc[:, sweep_cols].round(4),
                    title=f"{condition} - {family} reducer/n sweep",
                    selector_columns=["reducer", "eval_name"]
                    if "eval_name" in sweep_cols
                    else ["reducer"],
                    page_size=5,
                )
            )
            fam_container.add_element(acc)
            family_tabs[str(family)] = fam_container

        if family_tabs:
            section.add_element(TabsElement(family_tabs))

        return section
    if args.analysis_mode in {"sensor_within_family", "sensor_within_subfamily"}:
        group_columns = ["family"]
        if args.analysis_mode == "sensor_within_subfamily" and "subfamily" in merged.columns:
            group_columns.append("subfamily")
        for group_key, family_runs in merged.groupby(group_columns, dropna=False):
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_label = " / ".join(str(value) for value in group_values)
            section.add_markdown(f"### {group_label}")
            _add_unit_summary(
                section,
                family_runs,
                args=args,
                unit_label="sensor",
                title_prefix=f"{condition} - {group_label}",
                selection_metric=args.selection_metric,
            )
        return section

    _add_unit_summary(
        section,
        merged,
        args=args,
        unit_label=unit_label,
        title_prefix=condition,
        selection_metric=args.selection_metric,
    )
    if args.analysis_mode != "family":
        top_n = 2 if args.analysis_mode == "sensor" else 1
        sensor_container = build_dataset(args, meta_df, condition, target_col=None)

        reducer_tabs = {}
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()

            sensor_group_columns = [
                column
                for column in ["eval_name", "target_col"]
                if column in reducer_runs.columns and reducer_runs[column].notna().any()
            ]
            sensor_groups = (
                list(reducer_runs.groupby(sensor_group_columns, dropna=False))
                if sensor_group_columns
                else [((), reducer_runs)]
            )
            for group_key, group_df in sensor_groups:
                best_rows = group_df.sort_values(
                    [args.selection_metric, "trustworthiness", "continuity"],
                    ascending=[False, False, False],
                    na_position="last",
                ).head(top_n)

                for rank, (_, best_row) in enumerate(best_rows.iterrows(), 1):
                    best_artifact = artifacts[str(best_row["fit_id"])]
                    sensor_meta = _build_meta_dict(
                        sensor_container, best_artifact["ids"], eval_specs
                    )
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    label_parts = [
                        str(value) for value in group_key if pd.notna(value) and str(value) != ""
                    ]
                    label_suffix = f" [{' / '.join(label_parts)}]" if label_parts else ""

                    rank_prefix = f"#{rank} " if top_n > 1 else ""
                    tab_section.add_element(
                        CalloutElement(
                            f"**{reducer_name}{label_suffix}** {rank_prefix}best unit: "
                            f"{best_row['unit_name']} (n={int(best_row['n_components'])})",
                            kind="tip",
                        )
                    )

                    title = (
                        f"{condition} - {reducer_name}{label_suffix} - "
                        f"Rank {rank} ({best_row['unit_name']})"
                    )
                    plots_elem = _add_best_fit_plots(
                        title,
                        best_artifact,
                        sensor_meta or {},
                        interactive=args.interactive,
                        feature_names=get_feature_names(sensor_container),
                    )
                    if plots_elem:
                        tab_section.add_element(plots_elem)

            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

    return section


def _build_pooled_section(
    args,
    output_root: Path,
    pooled_runs: pd.DataFrame,
    pooled_eval_runs: pd.DataFrame,
    meta_df: pd.DataFrame | None,
    eval_specs: Sequence[dict[str, Any]],
    reducers: Sequence[str],
) -> Section | None:
    if pooled_runs.empty:
        return None
    fam_label = family_label(args)
    section = Section("Pooled Multi-condition", icon="🌐")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(output_root / row["artifact_path"])
        for _, row in pooled_runs.iterrows()
    }
    callout_text = (
        "Shared fits across all requested conditions.<br/>Condition-separation scores "
        "show EO vs EC-style pooled separability when available."
    )
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(CalloutElement(callout_text, kind="info", title="Pooled Configuration"))
    section.add_element(StatCardElement("Pooled Fits", len(pooled_runs), color="purple"))
    merged = pooled_runs.merge(
        pooled_eval_runs.loc[:, ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]]
        if not pooled_eval_runs.empty
        else pd.DataFrame(columns=["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]),
        on="fit_id",
        how="left",
    )
    pooled_container = None
    try:
        pooled_container = build_dataset(args, meta_df, args.conditions[0], target_col=None)
    except Exception:
        pass

    if args.analysis_mode == "family":
        _add_unit_summary(
            section,
            merged,
            args=args,
            unit_label="family",
            title_prefix="Pooled",
            selection_metric=args.selection_metric,
        )
        reducer_tabs = {}
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()
            comparison_metrics = [
                metric
                for metric in ["trustworthiness", "continuity", "shepard_correlation"]
                if metric in reducer_runs.columns
            ]
            if SEPARATION_METRIC_KEY in reducer_runs.columns:
                comparison_metrics.append(SEPARATION_METRIC_KEY)
            comparison_metrics = list(dict.fromkeys(comparison_metrics))
            curve_frames = []
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = (
                    family_runs.sort_values(
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
                    curve_frames.append(
                        pd.DataFrame(
                            {
                                "n_components": metric_df["n_components"].to_numpy(),
                                "score": metric_df[metric_name].to_numpy(),
                                "series": f"{family}: {metric_name}",
                            }
                        )
                    )
            if curve_frames:
                fig = plot_scatter(
                    pd.concat(curve_frames, ignore_index=True),
                    x="n_components",
                    y="score",
                    color="series",
                    mode="lines+markers",
                    title=f"Pooled - {reducer_name} family comparison vs n_components",
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                acc = AccordionElement("Show Family Comparison Sweep Curves", open=False)
                acc.add_element(PlotlyElement(fig))

                sweep_cols = ["reducer", "n_components"]
                if "eval_name" in reducer_runs.columns:
                    sweep_cols.append("eval_name")
                if SEPARATION_METRIC_KEY in reducer_runs.columns:
                    sweep_cols.append(SEPARATION_METRIC_KEY)
                for gm in ["trustworthiness", "continuity"]:
                    if gm in reducer_runs.columns:
                        sweep_cols.append(gm)
                sweep_cols = list(dict.fromkeys(sweep_cols))
                acc.add_element(
                    InteractiveTableElement(
                        reducer_runs.loc[
                            :, [c for c in sweep_cols if c in reducer_runs.columns]
                        ].round(4),
                        title=f"Pooled - {reducer_name} sweep",
                        selector_columns=["reducer", "eval_name"]
                        if "eval_name" in sweep_cols
                        else ["reducer"],
                        page_size=5,
                    )
                )
                tab_section.add_element(acc)
            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(CalloutElement("Family Sweeps by Reducer", kind="info"))
            section.add_element(TabsElement(reducer_tabs))

        family_tabs = {}
        top_n = 2
        for family, family_runs in merged.groupby("family", dropna=False):
            family_best = family_runs.sort_values(
                [args.selection_metric, "trustworthiness", "continuity"],
                ascending=[False, False, False],
                na_position="last",
            ).head(top_n)

            fam_container = ContainerElement()
            for rank, (_, best_row) in enumerate(family_best.iterrows(), 1):
                best_artifact = artifacts[str(best_row["fit_id"])]
                pool_meta = (
                    _build_meta_dict(pooled_container, best_artifact["ids"], eval_specs)
                    if pooled_container is not None
                    else {}
                )

                rank_prefix = f"#{rank} " if top_n > 1 else ""
                fam_container.add_element(
                    CalloutElement(
                        f"**{family}** {rank_prefix}best fit uses **{best_row['reducer']}** "
                        f"with n={int(best_row['n_components'])}",
                        kind="tip",
                    )
                )

                plots_elem = _add_best_fit_plots(
                    f"Pooled - {family} - Rank {rank}",
                    best_artifact,
                    pool_meta,
                    interactive=args.interactive,
                    feature_names=get_feature_names(pooled_container) if pooled_container else None,
                )
                if plots_elem:
                    fam_container.add_element(plots_elem)
            if fam_container.elements:
                family_tabs[str(family)] = fam_container

        if family_tabs:
            section.add_element(TabsElement(family_tabs))

        return section

    if args.analysis_mode == "flat":
        acc = AccordionElement("Show Hyperparameter Sweep Tables", open=False)
        sweep_cols = ["reducer", "n_components"]
        if "eval_name" in merged.columns:
            sweep_cols.append("eval_name")
        if SEPARATION_METRIC_KEY in merged.columns:
            sweep_cols.append(SEPARATION_METRIC_KEY)
        for gm in ["trustworthiness", "continuity"]:
            if gm in merged.columns:
                sweep_cols.append(gm)
        sweep_cols = list(dict.fromkeys(sweep_cols))
        acc.add_element(
            InteractiveTableElement(
                merged.loc[:, [c for c in sweep_cols if c in merged.columns]].round(4),
                title="Pooled fit ranking",
                selector_columns=["reducer", "eval_name"]
                if "eval_name" in sweep_cols
                else ["reducer"],
                default_sort={"column": args.selection_metric, "direction": "desc"},
                page_size=5,
            )
        )
        section.add_element(acc)

        reducer_tabs = {}
        for reducer_name in reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue
            best_row = reducer_runs.sort_values(
                [args.selection_metric, "trustworthiness", "continuity"],
                ascending=[False, False, False],
                na_position="last",
            ).iloc[0]
            best_artifact = artifacts[str(best_row["fit_id"])]
            pool_meta = (
                _build_meta_dict(pooled_container, best_artifact["ids"], eval_specs)
                if pooled_container is not None
                else {}
            )

            plots_elem = _add_best_fit_plots(
                f"Pooled - {reducer_name}",
                best_artifact,
                pool_meta,
                interactive=args.interactive,
                feature_names=get_feature_names(pooled_container) if pooled_container else None,
            )
            if plots_elem:
                reducer_tabs[f"{reducer_name} (n={int(best_row['n_components'])})"] = plots_elem
        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

        return section

    unit_label = _UNIT_LABELS.get(args.analysis_mode, "analysis unit")
    _add_unit_summary(
        section,
        merged,
        args=args,
        unit_label=unit_label,
        title_prefix="Pooled",
        selection_metric=args.selection_metric,
    )

    top_n = 2 if args.analysis_mode == "sensor" else 1
    reducer_tabs = {}
    for reducer_name in reducers:
        reducer_runs = merged[merged["reducer"] == reducer_name].copy()
        if reducer_runs.empty:
            continue

        tab_section = ContainerElement()
        sensor_group_columns = [
            column
            for column in ["eval_name", "target_col"]
            if column in reducer_runs.columns and reducer_runs[column].notna().any()
        ]
        sensor_groups = (
            list(reducer_runs.groupby(sensor_group_columns, dropna=False))
            if sensor_group_columns
            else [((), reducer_runs)]
        )

        for group_key, group_df in sensor_groups:
            best_rows = group_df.sort_values(
                [args.selection_metric, "trustworthiness", "continuity"],
                ascending=[False, False, False],
                na_position="last",
            ).head(top_n)

            for rank, (_, best_row) in enumerate(best_rows.iterrows(), 1):
                best_artifact = artifacts[str(best_row["fit_id"])]
                pool_meta = (
                    _build_meta_dict(pooled_container, best_artifact["ids"], eval_specs)
                    if pooled_container is not None
                    else {}
                )

                label_parts = (
                    [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
                    if not isinstance(group_key, tuple)
                    else [str(value) for value in group_key if pd.notna(value) and str(value) != ""]
                )
                label_suffix = f" [{' / '.join(label_parts)}]" if label_parts else ""

                rank_prefix = f"#{rank} " if top_n > 1 else ""
                tab_section.add_element(
                    CalloutElement(
                        f"**{reducer_name}{label_suffix}** {rank_prefix}best unit: "
                        f"{best_row.get('unit_name', 'unit')} (n={int(best_row['n_components'])})",
                        kind="tip",
                    )
                )

                title = (
                    f"Pooled - {reducer_name}{label_suffix} - "
                    f"Rank {rank} ({best_row.get('unit_name', 'unit')})"
                )
                plots_elem = _add_best_fit_plots(
                    title,
                    best_artifact,
                    pool_meta,
                    interactive=args.interactive,
                    feature_names=get_feature_names(pooled_container) if pooled_container else None,
                )
                if plots_elem:
                    tab_section.add_element(plots_elem)

        reducer_tabs[reducer_name] = tab_section

    if reducer_tabs:
        section.add_element(TabsElement(reducer_tabs))

    return section


def generate_dataset_report(
    args,
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    reducers: Sequence[str],
    meta_df: pd.DataFrame,
    containers_by_scope: dict[tuple[str, str], Any] | None,
    dataset_stats: list[dict[str, Any]] | None,
    eval_specs: Sequence[dict[str, Any]],
    pooled_condition: str,
) -> Report:
    fit_runs_df = pd.DataFrame(load_fit_runs(fit_runs_path))

    if eval_runs_path.exists():
        eval_runs_df = pd.DataFrame(json.loads(eval_runs_path.read_text(encoding="utf-8")))
    else:
        eval_runs_df = pd.DataFrame()

    if not eval_runs_df.empty and eval_specs:
        wanted_eval_names = {spec["name"] for spec in eval_specs}
        if "eval_name" in eval_runs_df.columns:
            eval_runs_df = eval_runs_df[eval_runs_df["eval_name"].isin(wanted_eval_names)]

    _available_eval_metrics = [col for col in EVAL_METRIC_COLUMNS if col in eval_runs_df.columns]
    if eval_runs_df.empty or not _available_eval_metrics:
        eval_frame = pd.DataFrame()
    else:
        _eval_base_cols = [
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
        ]
        cols_to_keep = [
            c for c in [*_eval_base_cols, *_available_eval_metrics] if c in eval_runs_df.columns
        ]
        eval_frame = eval_runs_df.loc[eval_runs_df["status"] == "success", cols_to_keep].copy()

    fit_success = fit_runs_df[fit_runs_df["status"] == "success"].copy()
    eval_merge_cols = ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]

    if not eval_frame.empty:
        eval_subset = eval_frame.loc[:, [c for c in eval_merge_cols if c in eval_frame.columns]]
        fit_eval_ranking = fit_success.merge(eval_subset, on="fit_id", how="left")
    else:
        empty_evals = pd.DataFrame(columns=eval_merge_cols)
        fit_eval_ranking = fit_success.merge(empty_evals, on="fit_id", how="left")

    fam_label = family_label(args)
    run_variant = args.run_variant or f"{args.analysis_mode}_{args.representation}"
    run_label = args.run_label or args.dataset_name
    report_title = f"Dimensionality Reduction: {run_label} ({args.input_mode} / {run_variant})"
    if fam_label:
        report_title += f" [{fam_label}]"
    report = Report(title=report_title)

    overview_sec = Section("Overview", icon="📋")
    # Configuration StatCards
    config_cards = [
        StatCardElement("Dataset", args.dataset_name, color="blue"),
        StatCardElement("Input Mode", args.input_mode, color="purple"),
        StatCardElement("Analysis Mode", args.analysis_mode, color="indigo"),
    ]
    if args.representation:
        config_cards.append(StatCardElement("Representation", args.representation, color="cyan"))

    overview_sec.add_element(ColumnsElement(config_cards, cols=len(config_cards)))

    config_html = (
        f"**Conditions**: {', '.join(args.conditions)}<br/>**Reducers**: {', '.join(reducers)}"
    )
    if fam_label:
        config_html += f"<br/>**Families**: {fam_label}"
    overview_sec.add_element(
        CalloutElement(config_html, kind="info", title="Run Configuration Details")
    )

    _add_data_availability_summary(
        overview_sec,
        args,
        dataset_stats,
        fit_runs_df,
        pooled_condition,
    )

    # --- Best overall summary per condition ---
    success_runs = fit_eval_ranking[fit_eval_ranking["status"] == "success"].copy()
    if not success_runs.empty:
        sort_metric = (
            args.selection_metric
            if args.selection_metric in success_runs.columns
            else "trustworthiness"
        )

        # Get top run per condition
        best_per_condition = (
            success_runs.sort_values(sort_metric, ascending=False, na_position="last")
            .groupby("condition", dropna=False)
            .head(1)
        )

        for _, best_overall in best_per_condition.iterrows():
            condition_name = best_overall.get("condition", "Unknown")

            best_row_dict: dict[str, Any] = {
                "best_reducer": best_overall.get("reducer", ""),
                "best_n_components": int(best_overall.get("n_components", 0))
                if pd.notna(best_overall.get("n_components"))
                else "",
            }
            if (
                "unit_name" in best_overall
                and pd.notna(best_overall["unit_name"])
                and best_overall["unit_name"]
            ):
                best_row_dict["best_unit"] = best_overall["unit_name"]

            for col in ["trustworthiness", "continuity", SEPARATION_METRIC_KEY]:
                if col in best_overall.index and pd.notna(best_overall[col]):
                    best_row_dict[col] = round(float(best_overall[col]), 4)

            overview_sec.add_element(
                CalloutElement(
                    f"Best run for condition: **{condition_name}**",
                    kind="tip",
                    title=f"Peak Performance ({sort_metric})",
                )
            )

            stat_cards = [
                StatCardElement("Reducer", best_row_dict["best_reducer"], color="blue"),
                StatCardElement("Components", best_row_dict["best_n_components"], color="purple"),
            ]
            if "best_unit" in best_row_dict:
                stat_cards.insert(
                    0, StatCardElement("Best Unit", best_row_dict["best_unit"], color="indigo")
                )

            if SEPARATION_METRIC_KEY in best_row_dict:
                stat_cards.append(
                    StatCardElement(
                        "Separation", best_row_dict[SEPARATION_METRIC_KEY], color="blue"
                    )
                )
            if "trustworthiness" in best_row_dict:
                stat_cards.append(
                    StatCardElement(
                        "Trustworthiness", best_row_dict["trustworthiness"], color="green"
                    )
                )
            if "continuity" in best_row_dict:
                stat_cards.append(
                    StatCardElement("Continuity", best_row_dict["continuity"], color="yellow")
                )

            overview_sec.add_element(ColumnsElement(stat_cards, cols=len(stat_cards)))

    add_overview_cohort_summary(
        overview_sec,
        args,
        eval_specs,
        containers_by_scope,
        pooled_condition,
    )
    report.add_section(overview_sec)

    if containers_by_scope:
        for (scope, condition), container in containers_by_scope.items():
            qc_result = (container.meta or {}).get("qc_result")
            if qc_result is None:
                continue
            qc_section = build_qc_section(qc_result)
            qc_section.title = f"Data Quality (QC): {scope} / {condition}"
            report.add_section(qc_section)

    if not eval_frame.empty:
        _default_eval_sort_col = (
            SEPARATION_METRIC_KEY
            if SEPARATION_METRIC_KEY in eval_frame.columns
            else (eval_frame.columns[-1] if len(eval_frame.columns) else "eval_name")
        )
        eval_sec = Section("Evaluation Results", icon="🧪")
        acc = AccordionElement("Show Post-hoc Evaluation Results", open=False)
        acc.add_element(
            InteractiveTableElement(
                eval_frame.round(4),
                title="Post-hoc evaluations",
                selector_columns=[
                    column
                    for column in [
                        "scope",
                        "condition",
                        "family",
                        "unit_name",
                        "reducer",
                        "eval_name",
                    ]
                    if column in eval_frame.columns
                ],
                default_sort={"column": _default_eval_sort_col, "direction": "desc"},
                page_size=5,
            )
        )
        eval_sec.add_element(acc)
        report.add_section(eval_sec)

    # --- Condition Ranking section with cross-condition comparison chart ---
    condition_runs = fit_eval_ranking[
        (fit_eval_ranking["scope"] == "condition") & (fit_eval_ranking["status"] == "success")
    ].copy()
    ranking_cols = [
        column
        for column in [
            "condition",
            "family",
            "unit_name",
            "reducer",
            "n_components",
            "trustworthiness",
            "continuity",
            "eval_name",
            SEPARATION_METRIC_KEY,
        ]
        if column in condition_runs.columns
    ]
    ranking_sec = Section("Condition Ranking", icon="🏁")
    ranking_sec.add_element(
        InteractiveTableElement(
            condition_runs.loc[:, ranking_cols].round(4),
            title="Condition ranking",
            selector_columns=[
                column
                for column in ["condition", "family", "unit_name", "reducer", "eval_name"]
                if column in ranking_cols
            ],
            default_sort={"column": "trustworthiness", "direction": "desc"},
            page_size=5,
        )
    )
    ranking_tabs = {}
    # Cross-condition comparison bar chart: best metric per (condition × reducer)
    if not condition_runs.empty and len(args.conditions) > 1:
        compare_metric = (
            args.selection_metric
            if args.selection_metric in condition_runs.columns
            else "trustworthiness"
        )
        bar_frames = []
        for reducer_name in reducers:
            reducer_cond_runs = condition_runs[condition_runs["reducer"] == reducer_name].copy()
            if reducer_cond_runs.empty:
                continue
            best_per_condition = (
                reducer_cond_runs.sort_values(compare_metric, ascending=False, na_position="last")
                .groupby("condition", dropna=False)
                .head(1)
            )
            best_per_condition = (
                best_per_condition.set_index("condition").reindex(args.conditions).reset_index()
            )
            best_per_condition["reducer"] = reducer_name
            best_per_condition["n_label"] = best_per_condition["n_components"].map(
                lambda v: f"n={int(v)}" if pd.notna(v) else ""
            )
            bar_frames.append(best_per_condition)
        if bar_frames:
            cond_compare_fig = plot_grouped_bar(
                pd.concat(bar_frames, ignore_index=True),
                x="condition",
                y=compare_metric,
                group="reducer",
                text="n_label",
                x_order=args.conditions,
                title=f"Best {compare_metric} per condition × reducer",
                xaxis_title="condition",
                yaxis_title=compare_metric,
                legend_title="Reducer",
            )
            ranking_tabs["Cross-Condition Summary"] = PlotlyElement(cond_compare_fig)

    # Radar chart: reducer × metric comparison (best value per reducer across all conditions)
    radar_metric_cols = [
        m
        for m in ["trustworthiness", "continuity", "shepard_correlation", SEPARATION_METRIC_KEY]
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
            ranking_tabs["Reducer Profile"] = PlotlyElement(
                plot_radar_comparison(
                    radar_df,
                    title="Reducer comparison — best metric across conditions",
                )
            )

    if ranking_tabs:
        ranking_sec.add_element(TabsElement(ranking_tabs))

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
                    eval_frame[eval_frame["condition"] == condition].copy()
                    if not eval_frame.empty
                    else pd.DataFrame(),
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
                    eval_frame[eval_frame["condition"] == condition].copy()
                    if not eval_frame.empty
                    else pd.DataFrame(),
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
                (eval_frame["scope"] == "pooled") & (eval_frame["condition"] == pooled_condition)
            ].copy()
            if not eval_frame.empty
            else pd.DataFrame()
        )
        pooled_section = _build_pooled_section(
            args,
            output_root,
            pooled_runs,
            pooled_eval,
            meta_df,
            eval_specs,
            reducers,
        )
        if pooled_section is not None:
            report.add_section(pooled_section)
    # --- Fit / eval failure sections ---
    fit_failures = fit_runs_df[fit_runs_df["status"] != "success"].copy()
    if not fit_failures.empty:
        failures_sec = Section("Fit Failures", icon="⚠️")
        acc = AccordionElement("Show Fit Failures", open=False)
        acc.add_element(
            InteractiveTableElement(
                fit_failures.loc[
                    :, [column for column in _FIT_FAILURE_COLUMNS if column in fit_failures.columns]
                ],
                title="Failed fits",
                selector_columns=[
                    column
                    for column in ["scope", "condition", "family", "unit_name", "reducer"]
                    if column in fit_failures.columns
                ],
                default_sort={"column": "condition", "direction": "asc"}
                if "condition" in fit_failures.columns
                else None,
                page_size=5,
            )
        )
        failures_sec.add_element(acc)
        report.add_section(failures_sec)

    eval_failures = eval_runs_df[eval_runs_df["status"] != "success"].copy()
    if not eval_failures.empty:
        failures_sec = Section("Eval Failures", icon="⚠️")
        acc = AccordionElement("Show Eval Failures", open=False)
        acc.add_element(
            InteractiveTableElement(
                eval_failures.loc[
                    :,
                    [column for column in _EVAL_FAILURE_COLUMNS if column in eval_failures.columns],
                ],
                title="Failed evals",
                selector_columns=[
                    column
                    for column in [
                        "scope",
                        "condition",
                        "family",
                        "unit_name",
                        "reducer",
                        "eval_name",
                    ]
                    if column in eval_failures.columns
                ],
                default_sort={"column": "condition", "direction": "asc"}
                if "condition" in eval_failures.columns
                else None,
                page_size=5,
            )
        )
        failures_sec.add_element(acc)
        report.add_section(failures_sec)

    return report


def collect_mode_leaderboard(
    *,
    args: Any,
    fit_runs_path: Path,
    eval_runs_path: Path,
    reducers: list[str] | None = None,
    pooled_condition: str | None = None,
) -> pd.DataFrame:
    """Best run per (scope, condition) for one analysis mode, for the roll-up.

    Returns one row per scope/condition tagged with ``analysis_mode`` (and the
    foundation ``model`` / embedding ``representation`` when relevant), ranked by
    the run's selection metric. Empty when the mode produced no successful fits.

    ``reducers`` restricts the ranking to the reducers the run actually requested
    (a stale inventory may carry others). ``pooled_condition`` standardizes the
    condition label shown for pooled-scope rows so the cross-model roll-up groups
    them consistently.
    """
    fit_runs_df = pd.DataFrame(load_fit_runs(fit_runs_path))
    if fit_runs_df.empty or "status" not in fit_runs_df.columns:
        return pd.DataFrame()
    fit_runs_df = fit_runs_df[fit_runs_df["status"] == "success"].copy()
    if reducers and "reducer" in fit_runs_df.columns:
        fit_runs_df = fit_runs_df[fit_runs_df["reducer"].isin(list(reducers))].copy()
    if fit_runs_df.empty:
        return pd.DataFrame()

    if eval_runs_path.exists():
        eval_runs_df = pd.DataFrame(json.loads(eval_runs_path.read_text(encoding="utf-8")))
    else:
        eval_runs_df = pd.DataFrame()
    if (
        not eval_runs_df.empty
        and "status" in eval_runs_df.columns
        and SEPARATION_METRIC_KEY in eval_runs_df.columns
    ):
        eval_frame = eval_runs_df.loc[
            eval_runs_df["status"] == "success",
            [
                col
                for col in ["fit_id", "eval_name", "target_col", SEPARATION_METRIC_KEY]
                if col in eval_runs_df.columns
            ],
        ].copy()

        selection_eval_name = getattr(args, "selection_eval_name", None)
        if selection_eval_name and "eval_name" in eval_frame.columns:
            eval_frame = eval_frame[eval_frame["eval_name"] == selection_eval_name].copy()
    else:
        eval_frame = pd.DataFrame(columns=["fit_id", "eval_name", SEPARATION_METRIC_KEY])

    merged = fit_runs_df.merge(
        eval_frame.loc[
            :,
            [c for c in ["fit_id", "eval_name", SEPARATION_METRIC_KEY] if c in eval_frame.columns],
        ],
        on="fit_id",
        how="left",
    )
    sort_cols = [
        col
        for col in [args.selection_metric, "trustworthiness", "continuity", SEPARATION_METRIC_KEY]
        if col in merged.columns
    ]
    ranked = (
        merged.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        if sort_cols
        else merged
    )
    best = ranked.groupby(["scope", "condition"], dropna=False).head(1).copy()

    # Vectorized column assignment instead of iterrows loop
    best["analysis_mode"] = args.analysis_mode
    best["input_mode"] = args.input_mode
    best["representation"] = args.representation if hasattr(args, "representation") else ""

    keep_cols = [
        "analysis_mode",
        "input_mode",
        "representation",
        "scope",
        "condition",
        "unit_name",
        "reducer",
        "n_components",
    ]

    if pooled_condition is not None and "scope" in best.columns:
        best.loc[best["scope"] == "pooled", "condition"] = pooled_condition

    if args.input_mode == "foundation_embeddings":
        best["model"] = args.embedding_model_key if hasattr(args, "embedding_model_key") else ""
        best["representation"] = args.representation or ""
        keep_cols.append("model")

    for metric in ["trustworthiness", "continuity", SEPARATION_METRIC_KEY, "eval_name"]:
        if metric in best.columns:
            keep_cols.append(metric)

    # Ensure base columns exist gracefully
    for col in keep_cols:
        if col not in best.columns:
            best[col] = ""

    return best[keep_cols]


def generate_rollup_report(
    *,
    args: Any,
    summaries: Sequence[dict[str, Any]],
    task_failures: Sequence[dict[str, str]] = (),
) -> Report:
    """Cross-mode leaderboard answering which representation wins for this cohort.

    Aggregates the per-mode leaderboards into one sortable table plus a
    faithful-vs-discriminative scatter (trustworthiness × separation), so the
    EEG / descriptor-mode / foundation-model comparison lands on a single axis.
    """
    run_label = (
        args.run_label if hasattr(args, "run_label") and args.run_label else args.dataset_name
    )
    report = Report(
        title=f"Dim Reduction Roll-up: {run_label} ({args.input_mode})",
        asset_urls="inline",
    )

    frames = [
        summary["leaderboard"]
        for summary in summaries
        if isinstance(summary.get("leaderboard"), pd.DataFrame) and not summary["leaderboard"].empty
    ]
    leaderboard = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    overview = Section("Roll-up Overview", icon="🏆")
    selection_note = (
        f" (eval: {args.selection_eval_name})"
        if hasattr(args, "selection_eval_name") and args.selection_eval_name
        else ""
    )

    callout_text = (
        f"Best run per analysis mode &times; condition for **{run_label}** ({args.input_mode}), "
        f"ranked by **{args.selection_metric}**{selection_note}.<br/><br/>"
        "💡 A strong representation is both geometrically faithful (high trustworthiness) "
        "and clinically separating (high separation balanced accuracy) — "
        "aim for the top-right in the scatter plot below."
    )
    overview.add_element(CalloutElement(callout_text, kind="info", title="Roll-up Strategy"))
    report.add_section(overview)

    if leaderboard.empty:
        overview.add_markdown("*No successful runs available for the leaderboard.*")
    else:
        sort_col = (
            SEPARATION_METRIC_KEY
            if SEPARATION_METRIC_KEY in leaderboard.columns
            else "trustworthiness"
            if "trustworthiness" in leaderboard.columns
            else str(leaderboard.columns[0])
        )
        display = leaderboard.copy()
        for col in display.select_dtypes(include="number").columns:
            display[col] = display[col].round(4)
        board_sec = Section("Leaderboard", icon="📊")
        board_sec.add_element(
            InteractiveTableElement(
                display,
                title="Best run per mode × condition",
                selector_columns=[
                    col
                    for col in [
                        "analysis_mode",
                        "scope",
                        "condition",
                        "reducer",
                        "model",
                        "eval_name",
                    ]
                    if col in display.columns
                ],
                default_sort={"column": sort_col, "direction": "desc"},
                page_size=15,
            )
        )
        has_scatter_axes = (
            "trustworthiness" in leaderboard.columns
            and SEPARATION_METRIC_KEY in leaderboard.columns
        )
        if has_scatter_axes:
            scatter_df = leaderboard.dropna(
                subset=["trustworthiness", SEPARATION_METRIC_KEY]
            ).copy()
            if not scatter_df.empty:
                color_col = "model" if "model" in scatter_df.columns else "analysis_mode"
                modes = scatter_df.get("analysis_mode", pd.Series("", index=scatter_df.index))
                conds = scatter_df.get("condition", pd.Series("", index=scatter_df.index))
                scatter_df["hover_label"] = [
                    f"{mode}/{condition}" for mode, condition in zip(modes, conds)
                ]
                fig = plot_scatter(
                    scatter_df,
                    x="trustworthiness",
                    y=SEPARATION_METRIC_KEY,
                    color=color_col,
                    text="hover_label",
                    hovertemplate=(
                        "%{text}<br>trustworthiness=%{x:.3f}<br>separation=%{y:.3f}<extra></extra>"
                    ),
                    mode="markers",
                    title="Faithful vs discriminative (top-right is best)",
                    xaxis_title="trustworthiness (geometry faithfulness)",
                    yaxis_title=SEPARATION_METRIC_KEY,
                    legend_title=color_col,
                )
                board_sec.add_element(PlotlyElement(fig))
        report.add_section(board_sec)

    link_rows = [
        {
            "analysis_mode": summary.get("analysis_mode", ""),
            "representation": summary.get("representation", ""),
            "run_variant": summary.get("run_variant", ""),
            "report": f"[View Report]({summary['report_path']})"
            if summary.get("report_path")
            else "",
        }
        for summary in summaries
    ]
    if link_rows:
        links_sec = Section("Per-mode reports", icon="🔗")
        links_sec.add_element(
            TableElement(pd.DataFrame(link_rows), title="Per-mode dataset summaries")
        )
        report.add_section(links_sec)

    if task_failures:
        fail_sec = Section("Task Failures", icon="⚠️")
        fail_sec.add_element(
            TableElement(
                pd.DataFrame(list(task_failures)), title="Modes that failed or were skipped"
            )
        )
        report.add_section(fail_sec)

    return report

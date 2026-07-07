#!/usr/bin/env python3
"""Direct EEG foundation-model probing, full fine-tuning, and LoRA."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import coco_pipe.report
import numpy as np
import pandas as pd
from coco_pipe.decoding import (
    CVConfig,
    DecodingUnit,
    ExperimentConfig,
    execute_decoding_sweep,
    get_foundation_model_spec,
    grouped_chance_assessment,
    load_sweep_records,
    redact_sensitive,
)
from coco_pipe.utils import slug

from eeg_adhd_epilepsy.analysis.dataset import build_dataset
from eeg_adhd_epilepsy.analysis.utils.common import pool_containers, require_config
from eeg_adhd_epilepsy.analysis.utils.decoding import (
    build_loader_args,
    foundation_provenance,
    prepare_decoding_scope,
    resolve_decoding_paths,
)
from eeg_adhd_epilepsy.reports.decoding import (
    generate_foundation_decoding_report,
    generate_head_to_head_report,
)
from eeg_adhd_epilepsy.utils.config import resolve_cli_config

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Foundation unit construction
# ---------------------------------------------------------------------------


def _build_foundation_unit(
    condition: str,
    target_name: str,
    model_key: str,
    train_mode: str,
    model_cfg: dict[str, Any],
    sfreq: float,
    channels: list[str],
    target_container: Any,
    y: pd.Series,
    groups: np.ndarray,
    sample_metadata: pd.DataFrame,
    n_splits: int,
    provenance: dict[str, Any],
    capability: Any,
    config: dict[str, Any],
    derivative_root: Path,
) -> Any:
    from coco_pipe.decoding import (
        CheckpointConfig,
        DeviceConfig,
        LoRAConfig,
        NeuralFineTuneConfig,
        TrainerConfig,
    )

    segment_duration = float(model_cfg["segment_duration"])
    window_source = str(model_cfg["window_source"])

    # capability record is always top-level in every returned record
    capability_record = {
        "condition": condition,
        "target": target_name,
        **provenance,
        **capability.to_dict(),
    }
    context = {
        "condition": condition,
        "target": target_name,
        "model_key": model_key,
        "train_mode": train_mode,
        "primary": train_mode == "linear_probe",
        "segment_duration": segment_duration,
        "window_source": window_source,
        "window_mismatch_policy": str(model_cfg["window_mismatch_policy"]),
        "cv_strategy": "stratified_group_kfold",
        "effective_n_splits": n_splits,
        "cv_random_state": int(config["random_state"]),
        "n_samples": int(len(y)),
        "n_groups": int(np.unique(groups).size),
        "capability": capability_record,
    }
    stem = (
        f"fit_{slug(condition)}_{slug(target_name)}_foundation_{slug(model_key)}_{slug(train_mode)}"
    )
    output_dir = derivative_root / "artifacts" / "fits" / stem

    if capability.status != "available":
        if config["on_unsupported"] == "error":
            raise RuntimeError(capability.reason)
        return {
            **context,
            "status": "skipped",
            "reason": capability.reason,
            "output_dir": str(output_dir),
        }

    mode_defaults = config["training_defaults"][train_mode]
    trainer = {
        **mode_defaults,
        **model_cfg["trainer"][train_mode],
    }
    optimizer = {
        "name": "adamw",
        "lr": trainer["lr"],
    }
    neural = NeuralFineTuneConfig(
        model_key=model_key,
        backend=model_cfg["backend"],
        train_mode=train_mode,
        input_kind="epoched",
        optimizer=optimizer,
        trainer=TrainerConfig(
            max_epochs=int(trainer["max_epochs"]),
            early_stopping_patience=trainer["early_stopping_patience"],
            batch_size=int(trainer["batch_size"]),
            validation_fraction=float(trainer["validation_fraction"]),
        ),
        device=DeviceConfig(
            device=model_cfg.get("device", config["device"]),
            precision=model_cfg.get("precision", config["precision"]),
        ),
        checkpoints=CheckpointConfig(
            save="best",
            output_dir=output_dir / "checkpoints",
        ),
        lora=(LoRAConfig(**model_cfg["lora"]) if train_mode == "lora" else None),
        sfreq=sfreq,
        ch_names=channels,
        backend_kwargs=model_cfg["backend_kwargs"],
        class_weight=config["class_weight"],
    )
    experiment_config = ExperimentConfig(
        task="classification",
        tag=f"{target_name}_{model_key}_{train_mode}",
        random_state=int(config["random_state"]),
        models={f"{model_key}_{train_mode}": neural},
        cv=CVConfig(
            strategy="stratified_group_kfold",
            n_splits=n_splits,
            shuffle=True,
            random_state=int(config["random_state"]),
            group_key="group_id",
        ),
        statistical_assessment=grouped_chance_assessment(
            config["chance_method"],
            n_permutations=int(config["n_permutations"]),
            store_null=bool(config["store_null_distribution"]),
        ),
        metrics=config["metrics"],
        use_scaler=False,
        n_jobs=1,
        verbose=bool(config["verbose"]),
    )

    return DecodingUnit(
        experiment_config=experiment_config,
        X=np.asarray(target_container.X, dtype=np.float32),
        y=y,
        output_dir=output_dir,
        context=context,
        run_config={**dict(config), **context},
        groups=groups,
        feature_names=channels,
        sample_ids=sample_metadata["sample_id"].astype(str),
        sample_metadata=sample_metadata,
        inferential_unit="group_id",
        overwrite=bool(config["overwrite"]),
        include_p_values=True,
    )


def _enumerate_foundation_scope(
    condition: str,
    model_cfg: dict[str, Any],
    container: Any,
    spec: Any,
    provenance: dict[str, Any],
    evals: list[dict[str, Any]],
    train_modes: list[str],
    config: dict[str, Any],
    derivative_root: Path,
) -> Iterator[Any]:
    from coco_pipe.decoding.foundation_models import (
        check_capability,
        normalize_inclusive_endpoint,
    )

    model_key = str(model_cfg["model_key"])
    segment_duration = float(model_cfg["segment_duration"])
    sfreq = float(spec.pretrained_sfreq)

    container, window_reason = normalize_inclusive_endpoint(
        container,
        segment_duration=segment_duration,
        expected_sfreq=sfreq,
        model_key=model_key,
        on_mismatch=model_cfg["window_mismatch_policy"],
    )
    if container is None:
        for train_mode in train_modes:
            yield {
                **provenance,
                "condition": condition,
                "target": "N/A",
                "train_mode": train_mode,
                "status": "skipped",
                "reason": f"Window length mismatch for {model_key}: {window_reason}",
                "primary": train_mode == "linear_probe",
            }
        return

    channels = [str(value) for value in container.coords["channel"]]
    for eval_spec in evals:
        target_name = eval_spec.get("name", eval_spec["target_col"])
        try:
            target_container, y, groups, sample_metadata, n_splits = prepare_decoding_scope(
                container,
                eval_spec,
                scope=condition,
                group_col=eval_spec["group_col"],
                session_col=config["session_col"],
                subject_col=config["subject_col"],
                requested_splits=int(config["cv"]["n_splits"]),
            )
        except Exception as exc:
            yield {
                "condition": condition,
                "target": target_name,
                "model_key": model_key,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                **provenance,
            }
            continue

        for train_mode in train_modes:
            capability = check_capability(
                model_key,
                train_mode=train_mode,
                sfreq=sfreq,
                ch_names=channels,
                n_times=int(target_container.X.shape[-1]),
                backend=model_cfg["backend"],
                backend_kwargs=model_cfg["backend_kwargs"],
            )
            yield _build_foundation_unit(
                condition=condition,
                target_name=target_name,
                model_key=model_key,
                train_mode=train_mode,
                model_cfg=model_cfg,
                sfreq=sfreq,
                channels=channels,
                target_container=target_container,
                y=y,
                groups=groups,
                sample_metadata=sample_metadata,
                n_splits=n_splits,
                provenance=provenance,
                capability=capability,
                config=config,
                derivative_root=derivative_root,
            )


def enumerate_foundation_units(
    scopes: list[tuple[str, dict[str, Any], Any, Any, dict[str, Any]]],
    *,
    config: dict[str, Any],
    derivative_root: Path,
) -> tuple[list[DecodingUnit], list[dict[str, Any]]]:
    """Flatten the condition × model × eval × train_mode sweep into units.

    Returns ``(units, failures)`` where failures are enumeration-time skips
    that never become sweep rows.
    """
    units: list[DecodingUnit] = []
    failures: list[dict[str, Any]] = []

    train_modes = config["train_modes"]
    evals = require_config(config, "evals", expected_type=list)

    for condition, model_cfg, container, spec, provenance in scopes:
        for item in _enumerate_foundation_scope(
            condition=condition,
            model_cfg=model_cfg,
            container=container,
            spec=spec,
            provenance=provenance,
            evals=evals,
            train_modes=train_modes,
            config=config,
            derivative_root=derivative_root,
        ):
            if isinstance(item, DecodingUnit):
                units.append(item)
            else:
                failures.append(item)

    return units, failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _foundation_primary_mask(frame: pd.DataFrame) -> pd.Series:
    """Successful linear-probe rows — the primary foundation result."""
    if not {"status", "train_mode"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    return (frame["status"] == "success") & (frame["train_mode"] == "linear_probe")


def _load_capability_matrix(derivative_root: Path) -> list[dict[str, Any]]:
    """Read the persisted capability matrix (for reports/reuse), or []."""
    path = derivative_root / "capability_matrix.csv"
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _build_foundation_scopes(config: dict[str, Any], metadata, cfg_hash: str) -> list:
    """Build (condition, model_cfg, container, spec, provenance) scopes per model."""
    conditions = require_config(config, "conditions", expected_type=list, cast_str=True)
    model_configs = require_config(config, "models", expected_type=list)

    scopes = []
    for model_cfg in model_configs:
        model_key = str(model_cfg["model_key"])
        window_source = str(model_cfg["window_source"])
        if window_source not in ("re_epoch", "derivative"):
            raise ValueError(
                f"model '{model_key}' window_source must be 're_epoch' or 'derivative', "
                f"got: '{window_source}'"
            )
        loader_args = build_loader_args(
            config,
            input_mode="raw",
            layout_mode="sensor",
            segment_duration=float(model_cfg["segment_duration"]),
            overlap=float(model_cfg["overlap"]),
            use_derivatives=bool(model_cfg["use_derivatives"]),
            window_source=window_source,
        )
        model_scopes = []
        for condition in conditions:
            container = build_dataset(loader_args, metadata, condition, target_col=None)
            spec = get_foundation_model_spec(model_key)
            provenance = foundation_provenance(model_cfg, spec, config_hash=cfg_hash)
            model_scopes.append((condition, model_cfg, container, spec, provenance))
        if config.get("run_pooled", False) and len(model_scopes) > 1:
            pooled = pool_containers([item[2] for item in model_scopes])
            _, model_cfg, _container, spec, provenance = model_scopes[0]
            model_scopes.append(("pooled", model_cfg, pooled, spec, provenance))
        scopes.extend(model_scopes)
    return scopes


def run(config: dict[str, Any]) -> Path:
    (
        bids_root,
        derivative_root,
        report_root,
        metadata,
        cfg_hash,
        reports_root,
        dataset_name_slug,
    ) = resolve_decoding_paths(config, input_mode="foundation")
    compare_only = bool(config.get("compare_only", False))
    reports_only = bool(config.get("reports_only", False))

    if not compare_only:
        if reports_only:
            records = load_sweep_records(derivative_root)
        else:
            scopes = _build_foundation_scopes(config, metadata, cfg_hash)
            units, failures = enumerate_foundation_units(
                scopes,
                config=config,
                derivative_root=derivative_root,
            )

            def _write_capability_matrix(
                frame: pd.DataFrame, recs: list[dict[str, Any]], output_root: Path
            ) -> None:
                # capability is a top-level dict on every unit record and every
                # enumeration-time skip (set in _build_foundation_unit / scope).
                caps = [r["capability"] for r in recs if isinstance(r.get("capability"), dict)]
                caps += [f["capability"] for f in failures if isinstance(f.get("capability"), dict)]
                if caps:
                    pd.DataFrame(caps).to_csv(output_root / "capability_matrix.csv", index=False)

            records, _ = execute_decoding_sweep(
                units,
                failures,
                config=config,
                output_root=derivative_root,
                results_filename="foundation_results.csv",
                primary_mask=_foundation_primary_mask,
                leaderboard_group_fields=("condition", "target", "model_key", "train_mode"),
                reallocate_inner_jobs=False,
                extra_outputs=_write_capability_matrix,
                run_metadata={
                    "dataset_name": config["dataset_name"],
                    "input_mode": "foundation",
                    "config_hash": cfg_hash,
                    "run_variant": derivative_root.name,
                },
            )

        capability_records = _load_capability_matrix(derivative_root)
        if config.get("detailed_unit_reports", False):
            coco_pipe.report.render_unit_reports(
                records,
                modes=config.get("detailed_unit_report_modes", ["flat"]),
                feature_metadata=None,
                asset_urls=config.get("report_asset_urls", "inline"),
                title_fn=lambda record: (
                    f"{record.get('target')}: {record.get('analysis_mode')}/"
                    f"{record.get('unit_name')} ({record.get('selection_mode')})"
                ),
            )
        generate_foundation_decoding_report(
            report_root / "dataset_summary.html",
            records,
            title=f"Foundation Decoding: {config['dataset_name']}",
            config=redact_sensitive(config),
            capability_records=capability_records,
            figures_dir=report_root / "figures",
        )

    generate_head_to_head_report(
        bids_root=bids_root,
        reports_root=reports_root,
        dataset_name=dataset_name_slug,
        asset_urls=config["report_asset_urls"],
    )
    return derivative_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort_config",
        required=True,
        help="Cohort/dataset config: subjects + clinical question (configs/cohorts/).",
    )
    parser.add_argument(
        "--analysis_config",
        required=True,
        help=(
            "Analysis/method config: models, train_modes "
            "(configs/analyses/decoding/foundation.yaml)."
        ),
    )
    parser.add_argument("--bids_root", default=None, help="Override BIDS root (else from config).")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override worker count.")
    parser.add_argument("--reports_root", default=None, help="Override reports root (else config).")
    parser.add_argument(
        "--representation",
        choices=["epoch", "recording", "subject"],
        default=None,
        help="Override the raw representation granularity for foundation decoding.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's overwrite flag.",
    )
    parser.add_argument(
        "--reports-only",
        dest="reports_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Regenerate reports from the saved runs/ inventory without refitting.",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Skip decoding and regenerate only the head-to-head comparison report.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = resolve_cli_config(
        cohort_config=args.cohort_config,
        analysis_config=args.analysis_config,
        bids_root=args.bids_root,
        metadata=args.metadata,
        n_jobs=args.n_jobs,
        reports_root=args.reports_root,
        representation=args.representation,
        overwrite=args.overwrite,
        reports_only=args.reports_only,
        compare_only=args.compare_only,
    )
    run(config)


if __name__ == "__main__":
    main()

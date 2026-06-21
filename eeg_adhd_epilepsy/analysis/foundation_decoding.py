#!/usr/bin/env python3
"""Direct EEG foundation-model probing, full fine-tuning, and LoRA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from coco_pipe.decoding import (
    CheckpointConfig,
    CVConfig,
    DeviceConfig,
    Experiment,
    ExperimentConfig,
    LoRAConfig,
    NeuralFineTuneConfig,
    TrainerConfig,
)
from coco_pipe.decoding.foundation_models import (
    check_capability,
    normalize_inclusive_endpoint,
)
from coco_pipe.io import read_table
from coco_pipe.report import make_foundation_decoding_report

from eeg_adhd_epilepsy.analysis.utils.decoding import (
    DEFAULT_METRICS,
    cohort_signature,
    completed_for_config,
    grouped_accuracy_assessment,
    load_completed_result_records,
    load_yaml_config,
    prepare_decoding_scope,
    redact_sensitive,
    require_conditions,
    result_records,
    slug,
    write_run_status,
)
from eeg_adhd_epilepsy.analysis.utils.foundation import (
    FoundationInputPlan,
    default_foundation_models,
    resolve_foundation_input_plan,
)
from eeg_adhd_epilepsy.io.analysis import load_container
from eeg_adhd_epilepsy.io.bids import get_reports_root
from eeg_adhd_epilepsy.reports.decoding import (
    generate_decoding_summary_report,
    generate_foundation_decoding_report,
    generate_head_to_head_report,
)

LOGGER = logging.getLogger(__name__)


def _skip_records(
    evals: list[dict[str, Any]],
    train_modes: list[str],
    *,
    condition: str,
    model_key: str,
    reason: str | None,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one skipped row per (target, train_mode) for a skipped model."""
    rows: list[dict[str, Any]] = []
    for eval_spec in evals:
        target_name = eval_spec.get("name", eval_spec["target_col"])
        for train_mode in train_modes:
            rows.append(
                {
                    "condition": condition,
                    "target": target_name,
                    "model_key": model_key,
                    "train_mode": train_mode,
                    "status": "skipped",
                    "reason": reason,
                    "primary": train_mode == "linear_probe",
                    **provenance,
                }
            )
    return rows


def _raw_loader_args(config: dict[str, Any], plan: FoundationInputPlan):
    filters = config.get("filters", {})
    return SimpleNamespace(
        input_mode="raw",
        analysis_mode="sensor",
        bids_root=config["bids_root"],
        use_derivatives=plan.use_derivatives,
        task=config.get("task", "clinical"),
        segment_duration=plan.segment_duration,
        overlap=plan.overlap,
        subject_col=config.get("subject_col", "study_id"),
        desc=config.get("desc", "base"),
        filter_col=list(filters),
        filter_val=[filters[column] for column in filters],
        group_filters=config.get("group_filters"),
        balance_target=None,
        balance_strategy="undersample",
        representation="epoch_native",
        aggregation_unit="recording",
        window_source=plan.window_source,
    )


def run(config: dict[str, Any]) -> Path:
    bids_root = Path(config["bids_root"]).expanduser()
    metadata = (
        read_table(Path(config["metadata"]).expanduser(), sep=None)
        if config.get("metadata")
        else None
    )
    derivative_root = (
        bids_root
        / "derivatives"
        / "foundation_decoding"
        / str(config.get("output_group", "default"))
        / str(config.get("dataset_name", "dataset"))
    )
    reports_root = Path(config.get("reports_root", get_reports_root(bids_root))).expanduser()
    report_root = (
        reports_root
        / "summary"
        / "foundation_decoding"
        / str(config.get("output_group", "default"))
        / str(config.get("dataset_name", "dataset"))
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    capability_records: list[dict[str, Any]] = []
    model_configs = config.get("models", default_foundation_models())
    train_modes = config.get("train_modes", ["linear_probe", "full", "lora"])
    evals = config.get("evals", [])
    if not evals:
        raise ValueError("At least one target specification is required in evals.")

    for condition in require_conditions(config):
        for model_cfg in model_configs:
            model_key = str(model_cfg["model_key"])
            plan = resolve_foundation_input_plan(config, model_cfg)
            if plan.skip_reason is not None:
                skipped = _skip_records(
                    evals,
                    train_modes,
                    condition=condition,
                    model_key=model_key,
                    reason=plan.skip_reason,
                    provenance=plan.to_provenance(),
                )
                records.extend(skipped)
                failures.extend(skipped)
                capability_records.extend(skipped)
                continue
            container = load_container(
                _raw_loader_args(config, plan),
                config.get("subjects"),
                metadata,
                condition,
                target_col=None,
            )
            container, window_reason = normalize_inclusive_endpoint(
                container,
                segment_duration=plan.segment_duration,
                expected_sfreq=plan.expected_sfreq,
                model_key=plan.model_key,
                on_mismatch=plan.window_mismatch_policy,
            )
            if container is None:
                skipped = _skip_records(
                    evals,
                    train_modes,
                    condition=condition,
                    model_key=model_key,
                    reason=window_reason,
                    provenance=plan.to_provenance(),
                )
                records.extend(skipped)
                failures.extend(skipped)
                capability_records.extend(skipped)
                continue
            sfreq = float(container.meta.get("sfreq", config.get("sfreq", 200.0)))
            channels = [str(value) for value in container.coords["channel"]]
            for eval_spec in evals:
                target_name = eval_spec.get("name", eval_spec["target_col"])
                try:
                    (
                        target_container,
                        y,
                        groups,
                        sample_metadata,
                        n_splits,
                    ) = prepare_decoding_scope(
                        container,
                        eval_spec,
                        scope=condition,
                        group_col=config.get("group_col", "patient_group_id"),
                        session_col=config["session_col"],
                        subject_col=config.get("subject_col", "study_id"),
                        requested_splits=int(config.get("cv", {}).get("n_splits", 5)),
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "condition": condition,
                            "target": target_name,
                            "model_key": model_key,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            **plan.to_provenance(),
                        }
                    )
                    continue

                for train_mode in train_modes:
                    capability = check_capability(
                        model_key,
                        train_mode=train_mode,
                        sfreq=sfreq,
                        ch_names=channels,
                        n_times=int(target_container.X.shape[-1]),
                        backend=model_cfg.get("backend", "auto"),
                        backend_kwargs=model_cfg.get("backend_kwargs", {}),
                    )
                    capability_record = {
                        "condition": condition,
                        "target": target_name,
                        **plan.to_provenance(),
                        **capability.to_dict(),
                    }
                    capability_records.append(capability_record)
                    context = {
                        "condition": condition,
                        "target": target_name,
                        "model_key": model_key,
                        "train_mode": train_mode,
                        "primary": train_mode == "linear_probe",
                        "segment_duration": plan.segment_duration,
                        "window_source": plan.window_source,
                        "window_mismatch_policy": plan.window_mismatch_policy,
                        "cv_strategy": "stratified_group_kfold",
                        "effective_n_splits": n_splits,
                        "cv_random_state": int(config.get("random_state", 42)),
                        "n_samples": int(len(y)),
                        "n_groups": int(np.unique(groups).size),
                        "cohort_signature": cohort_signature(
                            sample_metadata[config.get("subject_col", "study_id")]
                        ),
                    }
                    output_dir = (
                        derivative_root
                        / slug(condition)
                        / slug(target_name)
                        / slug(model_key)
                        / slug(train_mode)
                    )
                    if capability.status != "available":
                        skipped = {
                            **context,
                            "status": "skipped",
                            "reason": capability.reason,
                            "output_dir": str(output_dir),
                        }
                        records.append(skipped)
                        failures.append(skipped)
                        if config.get("on_unsupported", "skip") == "error":
                            raise RuntimeError(capability.reason)
                        continue
                    unit_config = {
                        **config,
                        "effective_n_splits": n_splits,
                        **context,
                    }
                    if not config.get("overwrite", False) and completed_for_config(
                        output_dir, unit_config
                    ):
                        records.extend(
                            load_completed_result_records(
                                output_dir,
                                context=context,
                            )
                        )
                        continue
                    try:
                        mode_defaults = config.get("training_defaults", {}).get(train_mode, {})
                        trainer = {
                            **mode_defaults,
                            **model_cfg.get("trainer", {}).get(train_mode, {}),
                        }
                        optimizer = {
                            "name": "adamw",
                            "lr": trainer.get(
                                "lr",
                                1e-3
                                if train_mode == "linear_probe"
                                else 1e-5
                                if train_mode == "full"
                                else 1e-4,
                            ),
                        }
                        neural = NeuralFineTuneConfig(
                            model_key=model_key,
                            backend=model_cfg.get("backend", "auto"),
                            train_mode=train_mode,
                            input_kind="epoched",
                            optimizer=optimizer,
                            trainer=TrainerConfig(
                                max_epochs=int(trainer.get("max_epochs", 20)),
                                early_stopping_patience=trainer.get("early_stopping_patience", 5),
                                batch_size=int(trainer.get("batch_size", 16)),
                                validation_fraction=float(trainer.get("validation_fraction", 0.2)),
                            ),
                            device=DeviceConfig(
                                device=model_cfg.get("device", config.get("device", "auto")),
                                precision=model_cfg.get(
                                    "precision", config.get("precision", "fp32")
                                ),
                            ),
                            checkpoints=CheckpointConfig(
                                save="best",
                                output_dir=output_dir / "checkpoints",
                            ),
                            lora=(
                                LoRAConfig(**model_cfg.get("lora", {}))
                                if train_mode == "lora"
                                else None
                            ),
                            sfreq=sfreq,
                            ch_names=channels,
                            backend_kwargs=model_cfg.get("backend_kwargs", {}),
                            class_weight=config.get("class_weight", "balanced"),
                        )
                        experiment_config = ExperimentConfig(
                            task="classification",
                            tag=f"{target_name}_{model_key}_{train_mode}",
                            random_state=int(config.get("random_state", 42)),
                            models={f"{model_key}_{train_mode}": neural},
                            cv=CVConfig(
                                strategy="stratified_group_kfold",
                                n_splits=n_splits,
                                shuffle=True,
                                random_state=int(config.get("random_state", 42)),
                                group_key="group_id",
                            ),
                            statistical_assessment=grouped_accuracy_assessment(
                                method=config.get("chance_method", "permutation"),
                                n_permutations=int(config.get("n_permutations", 100)),
                                store_null=bool(config.get("store_null_distribution", False)),
                            ),
                            metrics=config.get("metrics", DEFAULT_METRICS),
                            use_scaler=False,
                            allow_transductive_input=bool(
                                config.get("allow_transductive_input", False)
                            ),
                            n_jobs=1,
                            verbose=bool(config.get("verbose", False)),
                        )
                        result = Experiment(experiment_config).run(
                            np.asarray(target_container.X, dtype=np.float32),
                            y,
                            groups=groups,
                            feature_names=channels,
                            sample_ids=sample_metadata["sample_id"].astype(str),
                            sample_metadata=sample_metadata,
                            observation_level="epoch",
                            inferential_unit="group_id",
                        )
                        result.export(output_dir, config=unit_config)
                        if config.get("detailed_unit_reports", False):
                            make_foundation_decoding_report(
                                result,
                                capability_records=[capability_record],
                                title=f"{model_key} {train_mode}: {target_name}",
                                sections="compact",
                                on_error="placeholder",
                                asset_urls=config.get("report_asset_urls", "inline"),
                                output_path=str(output_dir / "report.html"),
                            )
                        records.extend(
                            result_records(
                                result,
                                context=context,
                                output_dir=output_dir,
                            )
                        )
                    except Exception as exc:
                        LOGGER.exception("Foundation decoding failed: %s", context)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        (output_dir / "_FAILED").write_text("", encoding="utf-8")
                        failure = {
                            **context,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "output_dir": str(output_dir),
                        }
                        records.append(failure)
                        failures.append(failure)

    derivative_root.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(records)
    result_frame.to_csv(derivative_root / "foundation_results.csv", index=False)
    pd.DataFrame(capability_records).to_csv(derivative_root / "capability_matrix.csv", index=False)
    pd.DataFrame(failures).to_csv(derivative_root / "failures.csv", index=False)
    (derivative_root / "config_used.yaml").write_text(
        yaml.safe_dump(redact_sensitive(config), sort_keys=False),
        encoding="utf-8",
    )
    primary_success = bool(
        not result_frame.empty
        and {"status", "train_mode"}.issubset(result_frame.columns)
        and (
            (result_frame["status"] == "success") & (result_frame["train_mode"] == "linear_probe")
        ).any()
    )
    any_success = bool(
        not result_frame.empty
        and "status" in result_frame
        and (result_frame["status"] == "success").any()
    )
    status = (
        "SUCCESS" if primary_success and not failures else "PARTIAL" if any_success else "FAILED"
    )
    write_run_status(derivative_root, status)
    summary_records = result_frame.to_dict("records")
    generate_decoding_summary_report(
        report_root / "dataset_summary.html",
        summary_records,
        title=f"Foundation Decoding: {config.get('dataset_name', 'dataset')}",
        config=redact_sensitive(config),
    )
    generate_foundation_decoding_report(
        report_root / "dataset_visual_report.html",
        summary_records,
        title=f"Foundation Decoding Figures: {config.get('dataset_name', 'dataset')}",
        config=redact_sensitive(config),
        capability_records=capability_records,
        figures_dir=report_root / "figures",
    )
    generate_head_to_head_report(
        bids_root=bids_root,
        reports_root=reports_root,
        output_group=str(config.get("output_group", "default")),
        dataset_name=str(config.get("dataset_name", "dataset")),
        asset_urls=config.get("report_asset_urls", "inline"),
    )
    return derivative_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(load_yaml_config(args.config))


if __name__ == "__main__":
    main()

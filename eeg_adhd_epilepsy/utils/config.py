"""Two-part (cohort + analysis) configuration loading and validation.

The analysis CLIs in this project run a *cohort* against an *analysis*:

* a **cohort config** answers *which subjects and which clinical question*
  (``dataset_name``, ``output_group``, ``group_filters``, ``filter_col`` /
  ``filter_val``, ``conditions``, ``evals`` / ``label_map``); and
* an **analysis config** answers *which method and hyperparameters*
  (dim-reduction ``reducers`` / ``n_components_sweep``; decoding ``models`` /
  ``cv`` / ``feature_selection``; foundation ``models`` / ``train_modes`` …),
  plus input-shaping (``input_mode``, ``qc``, ``descriptor_table_path`` …) and
  run controls (``n_jobs``, ``overwrite`` …).

Historically a single YAML conflated both, which duplicated the (identical)
method block across every cohort variant and made the ``configs/`` tree
ambiguous. This module loads the two files, validates each with actionable
errors, and deep-merges them — **the analysis config overrides the cohort
config** on overlap (so an analysis may, e.g., narrow ``conditions``) — into the
single mapping the existing ``run()`` functions already consume.

Dataset-level *paths* (``bids_root`` / ``metadata``) intentionally live in
**neither** config; callers supply them via CLI/env and layer them on with
:func:`apply_overrides`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .yaml import load_yaml_config

__all__ = [
    "ConfigError",
    "load_cohort_analysis_config",
    "resolve_cli_config",
    "validate_cohort_config",
    "validate_analysis_config",
    "apply_overrides",
]

# Keys a cohort config must define (the "who + which question").
_COHORT_REQUIRED_KEYS = ("dataset_name", "output_group", "evals")

# A recognizable analysis config declares at least one method block.
_ANALYSIS_METHOD_MARKERS = ("reducers", "models", "train_modes")


class ConfigError(ValueError):
    """Raised for a missing or invalid field in a cohort or analysis config."""


def _format_missing(path: str | Path, role: str, missing: list[str], hint: str) -> str:
    return (
        f"{role.capitalize()} config {Path(path)} is missing required key(s): "
        f"{', '.join(missing)}.\n{hint}"
    )


def validate_cohort_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Validate a cohort config, raising :class:`ConfigError` with a fix hint."""
    if not isinstance(config, Mapping):
        raise ConfigError(f"Cohort config {Path(path)} must be a YAML mapping.")
    missing = [k for k in _COHORT_REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(
            _format_missing(
                path,
                "cohort",
                missing,
                "A cohort config defines the dataset selection and clinical "
                "question. Copy one from configs/cohorts/ and edit it. "
                "(bids_root/metadata are NOT cohort keys — pass them via "
                "--bids_root/--metadata or the cluster env vars.)",
            )
        )
    evals = config.get("evals")
    if not isinstance(evals, list) or not evals:
        raise ConfigError(
            f"Cohort config {Path(path)}: 'evals' must be a non-empty list of "
            "{name, target_col, ...} entries."
        )
    for index, spec in enumerate(evals):
        if not isinstance(spec, Mapping) or "name" not in spec:
            raise ConfigError(
                f"Cohort config {Path(path)}: evals[{index}] must be a mapping "
                "with at least a 'name' key."
            )


def validate_analysis_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Validate an analysis config, raising :class:`ConfigError` with a fix hint."""
    if not isinstance(config, Mapping):
        raise ConfigError(f"Analysis config {Path(path)} must be a YAML mapping.")
    if not config:
        raise ConfigError(f"Analysis config {Path(path)} is empty.")
    if not any(marker in config for marker in _ANALYSIS_METHOD_MARKERS):
        raise ConfigError(
            f"Analysis config {Path(path)} does not declare a method block. "
            f"Expected at least one of: {', '.join(_ANALYSIS_METHOD_MARKERS)}. "
            "See configs/analyses/ for examples (dim_reduction uses 'reducers', "
            "decoding/foundation use 'models', foundation adds 'train_modes')."
        )
    if "evals" in config:
        raise ConfigError(
            f"Analysis config {Path(path)} defines 'evals'. Targets/evals belong "
            "to the cohort config (the clinical question), not the analysis."
        )


def _validate_merged(merged: Mapping[str, Any], cohort_path: str | Path) -> None:
    """Cross-config checks that only make sense after merging."""
    selection_eval = merged.get("selection_eval_name")
    if selection_eval is not None:
        eval_names = {
            spec.get("name")
            for spec in (merged.get("evals") or [])
            if isinstance(spec, Mapping)
        }
        if selection_eval not in eval_names:
            raise ConfigError(
                f"selection_eval_name={selection_eval!r} (from the analysis "
                f"config) is not defined in the cohort's evals "
                f"({sorted(n for n in eval_names if n)}). Add an evals entry "
                f"named {selection_eval!r} to the cohort config "
                f"({Path(cohort_path)}), or fix selection_eval_name."
            )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Nested mappings merge key-by-key; lists and scalars from ``override`` replace
    those in ``base`` wholesale (so e.g. an analysis ``conditions`` list cleanly
    overrides the cohort default rather than concatenating).
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def apply_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Layer non-``None`` keyword overrides (e.g. CLI ``--bids_root``) onto a config.

    Returns the same dict for convenience. ``None`` values are ignored so an
    unset CLI flag never clobbers a config value.
    """
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def resolve_cli_config(
    *,
    cohort_config: str | Path | None,
    analysis_config: str | Path | None,
    legacy_config: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Resolve a consumer CLI's config from the two-config flags (or legacy single).

    Preferred: ``--cohort_config`` + ``--analysis_config``. A single
    ``--config`` (``legacy_config``) is still accepted for back-compat. Non-``None``
    ``overrides`` (e.g. CLI ``--bids_root``) are layered on last.
    """
    if legacy_config is not None:
        if cohort_config is not None or analysis_config is not None:
            raise ConfigError(
                "Pass either --config (single, deprecated) or "
                "--cohort_config + --analysis_config, not both."
            )
        config = load_yaml_config(legacy_config)
    elif cohort_config is not None and analysis_config is not None:
        config = load_cohort_analysis_config(cohort_config, analysis_config)
    else:
        raise ConfigError(
            "Provide --cohort_config and --analysis_config "
            "(the cohort defines the dataset/question, the analysis defines the "
            "method). A single --config is still accepted for back-compat."
        )
    apply_overrides(config, **overrides)
    if not config.get("bids_root"):
        raise ConfigError(
            "bids_root is required but not set. Dataset paths live on the CLI/env, "
            "not in the cohort/analysis configs — pass --bids_root (and --metadata) "
            "or set BIDS_ROOT/METADATA_PATH in the cluster environment."
        )
    return config


def load_cohort_analysis_config(
    cohort_path: str | Path, analysis_path: str | Path
) -> dict[str, Any]:
    """Load + validate a cohort and an analysis config and return the merged dict.

    The analysis config overrides the cohort config on overlapping keys. The
    result is the single mapping the ``run()`` entry points consume.
    """
    cohort = load_yaml_config(cohort_path)
    analysis = load_yaml_config(analysis_path)
    validate_cohort_config(cohort, cohort_path)
    validate_analysis_config(analysis, analysis_path)
    merged = _deep_merge(cohort, analysis)
    _validate_merged(merged, cohort_path)
    return merged

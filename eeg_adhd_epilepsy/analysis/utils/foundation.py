"""Shared foundation-model input-window policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from coco_pipe.decoding import get_foundation_model_spec

_WINDOW_POLICIES = {"error", "skip", "re_epoch"}
_WINDOW_SOURCES = {"auto", "derivative", "re_epoch"}


def default_foundation_models() -> list[dict[str, Any]]:
    """Return a fresh default model roster shared by extraction and decoding."""
    return [
        {"model_key": "cbramod"},
        {"model_key": "labram", "backend_kwargs": {"interpolate_channels": True}},
        {"model_key": "reve"},
        {"model_key": "luna"},
    ]


@dataclass(frozen=True)
class FoundationInputPlan:
    """Resolved EEG loading policy for one foundation model."""

    model_key: str
    segment_duration: float
    overlap: float
    use_derivatives: bool
    window_source: str
    window_mismatch_policy: str
    expected_n_times: int | None
    expected_sfreq: float
    expected_duration: float | None
    skip_reason: str | None = None

    def to_provenance(self) -> dict[str, Any]:
        """Return JSON/YAML-safe provenance fields."""
        return asdict(self)


def resolve_foundation_input_plan(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> FoundationInputPlan:
    """Resolve per-model duration and fail-loudly mismatch behavior."""
    model_key = str(model_config["model_key"])
    spec = get_foundation_model_spec(model_key)
    global_duration = float(config.get("segment_duration", 10.0))
    requested_duration = float(model_config.get("segment_duration", global_duration))
    overlap = float(model_config.get("overlap", config.get("overlap", 0.0)))
    policy = str(
        model_config.get(
            "window_mismatch_policy",
            config.get("window_mismatch_policy", "error"),
        )
    )
    source = str(
        model_config.get(
            "window_source",
            config.get("window_source", "auto"),
        )
    )
    if policy not in _WINDOW_POLICIES:
        raise ValueError(
            f"Invalid window_mismatch_policy={policy!r}; "
            f"expected one of {sorted(_WINDOW_POLICIES)}."
        )
    if source not in _WINDOW_SOURCES:
        raise ValueError(
            f"Invalid window_source={source!r}; expected one of {sorted(_WINDOW_SOURCES)}."
        )
    expected_duration = spec.pretrained_window_seconds
    mismatch = expected_duration is not None and abs(requested_duration - expected_duration) > 1e-9
    if mismatch:
        reason = (
            f"{spec.display_name or model_key} requires "
            f"{spec.pretrained_n_times} samples at {spec.pretrained_sfreq:g} Hz "
            f"({expected_duration:g} s), but {requested_duration:g} s was configured."
        )
        if policy == "error":
            raise ValueError(reason)
        if policy == "skip":
            return FoundationInputPlan(
                model_key=model_key,
                segment_duration=requested_duration,
                overlap=overlap,
                use_derivatives=bool(config.get("use_derivatives", True)),
                window_source=source,
                window_mismatch_policy=policy,
                expected_n_times=spec.pretrained_n_times,
                expected_sfreq=float(spec.pretrained_sfreq),
                expected_duration=expected_duration,
                skip_reason=reason,
            )
        requested_duration = expected_duration
        source = "re_epoch"

    use_derivatives = bool(config.get("use_derivatives", True))
    if source == "re_epoch":
        use_derivatives = False
    elif source == "derivative" and not use_derivatives:
        raise ValueError(
            f"{model_key} requests window_source='derivative', but use_derivatives is false."
        )
    elif source == "auto" and use_derivatives and abs(requested_duration - global_duration) > 1e-9:
        reason = (
            f"{model_key} requests a {requested_duration:g} s model-specific "
            f"window while derivative epochs use the global {global_duration:g} s "
            "window."
        )
        if policy == "error":
            raise ValueError(reason)
        if policy == "skip":
            return FoundationInputPlan(
                model_key=model_key,
                segment_duration=requested_duration,
                overlap=overlap,
                use_derivatives=True,
                window_source=source,
                window_mismatch_policy=policy,
                expected_n_times=spec.pretrained_n_times,
                expected_sfreq=float(spec.pretrained_sfreq),
                expected_duration=expected_duration,
                skip_reason=reason,
            )
        use_derivatives = False
        source = "re_epoch"

    return FoundationInputPlan(
        model_key=model_key,
        segment_duration=requested_duration,
        overlap=overlap,
        use_derivatives=use_derivatives,
        window_source=source,
        window_mismatch_policy=policy,
        expected_n_times=spec.pretrained_n_times,
        expected_sfreq=float(spec.pretrained_sfreq),
        expected_duration=expected_duration,
    )


__all__ = [
    "FoundationInputPlan",
    "default_foundation_models",
    "resolve_foundation_input_plan",
]

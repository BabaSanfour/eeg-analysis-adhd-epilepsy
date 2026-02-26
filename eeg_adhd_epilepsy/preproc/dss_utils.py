"""Shared DSS helpers for Stage 1 artifact correction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from mne_denoise.dss import (
    DSS,
    IterativeDSS,
    AverageBias,
    BandpassBias,
    QuasiPeriodicDenoiser,
)

from eeg_adhd_epilepsy.viz import qc as viz_qc
from . import thresholds


LOGGER = logging.getLogger(__name__)


def _resolve_eog_detection_channels(raw: mne.io.BaseRaw) -> Tuple[Optional[List[str]], str]:
    """Pick channels for blink detection: prefer true EOG, else Fp1+Fp2."""
    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    if len(eog_picks) > 0:
        eog_chs = [raw.ch_names[i] for i in eog_picks]
        return eog_chs, "eog"

    if "Fp1" in raw.ch_names and "Fp2" in raw.ch_names:
        return ["Fp1", "Fp2"], "frontal_pair"
    return None, "missing"


def _init_dss(
    config: Any,
    *,
    bias: Optional[Any] = None,
    denoiser: Optional[Any] = None,
    beta: Optional[Any] = None,
    method: str = "deflation",
    max_iter: Optional[int] = None,
) -> Union[DSS, IterativeDSS]:
    """Initialize linear DSS or IterativeDSS from a shared factory."""
    if denoiser is not None:
        kwargs: Dict[str, Any] = {
            "denoiser": denoiser,
            "n_components": config.dss_n_components,
            "method": method,
        }
        if beta is not None:
            kwargs["beta"] = beta
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        return IterativeDSS(**kwargs)

    if bias is None:
        raise ValueError("Linear DSS initialization requires `bias`.")
    return DSS(n_components=config.dss_n_components, bias=bias)


def _remove_source_dss(
    raw: mne.io.BaseRaw,
    dss: Union[DSS, IterativeDSS],
    n_remove: int,
    *,
    expected_ch_names: Optional[List[str]] = None,
    use_single_epoch_3d: bool = False,
    subtract_artifact: bool = False,
    safe_subtraction: bool = False,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Standardized DSS source removal for all artifact types."""
    n_remove = max(1, int(n_remove))
    target_eeg = raw.copy().pick_types(eeg=True, exclude="bads")

    if expected_ch_names is not None and target_eeg.ch_names != expected_ch_names:
        return raw, {"skipped": True, "error": "channel mismatch"}

    # 1. Transform to sources
    target_data = target_eeg.get_data()
    if use_single_epoch_3d:
        sources = dss.transform(target_data[np.newaxis, :, :]) # (1, sources, samples)
        n_available = sources.shape[1]
    else:
        sources = dss.transform(target_data) # (sources, samples)
        n_available = sources.shape[0]

    n_use = min(n_remove, n_available)

    # 2. Modify sources (zero out or keep only artifact)
    sources_out = np.zeros_like(sources)
    if not subtract_artifact:
        # Keep everything except the first n_use components
        sources_out[..., n_use:, :] = sources[..., n_use:, :]
    else:
        # Keep only the first n_use components (the artifact) for subtraction
        sources_out[..., :n_use, :] = sources[..., :n_use, :]

    # 3. Project back
    model_data_raw = dss.inverse_transform(sources_out)
    model_data = model_data_raw[0] if use_single_epoch_3d else model_data_raw

    # 4. Variance Check & Application
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if model_data.shape[0] != len(picks):
        return raw, {"skipped": True, "reason": "shape_mismatch"}

    if subtract_artifact and safe_subtraction:
        orig_std = float(np.std(raw._data[picks, :]))
        art_std = float(np.std(model_data))
        if art_std > 10 * orig_std: # Heuristic threshold
            return raw, {"skipped": True, "reason": "unsafe_variance"}

    raw_out = raw.copy()
    if subtract_artifact:
        raw_out._data[picks, :] -= model_data
    else:
        raw_out._data[picks, :] = model_data

    return raw_out, {"n_components_removed": n_use}


# --- DSS Registry & Profiles ---

DSS_PROFILES = {
    ("eog", "dss"): {
        "fit_mode": "eog_epochs",
        "estimator": "dss",
        "bias_kind": "average",
        "merge_mode": "replace",
        "transform_mode": "2d",
        "plot_dir": "dss_eog",
        "plot_prefix": "eog",
        "overlay_title": "EOG DSS: Signal Overlay (30-90s)",
    },
    ("eog", "blind-dss"): {
        "fit_mode": "continuous",
        "estimator": "iterative",
        "use_config_denoiser": True,
        "merge_mode": "replace",
        "transform_mode": "2d",
        "plot_dir": "dss_blind_eog",
        "plot_prefix": "blind",
        "overlay_title": "EOG Blind DSS: Signal Overlay (30-90s)",
        "fit_max_seconds": 60.0,
        "include_component_summary": False,
    },
    ("ecg", "dss"): {
        "fit_mode": "ecg_epochs",
        "estimator": "dss",
        "bias_kind": "average",
        "merge_mode": "replace",
        "transform_mode": "3d_single_epoch",
        "plot_dir": "dss_ecg",
        "plot_prefix": "ecg",
        "overlay_title": "ECG DSS: Signal Overlay (30-90s)",
    },
    ("ecg", "quasiperiodic"): {
        "fit_mode": "continuous",
        "estimator": "iterative",
        "denoiser_kind": "quasiperiodic",
        "merge_mode": "replace",
        "transform_mode": "2d",
        "plot_dir": "dss_quasiperiodic_ecg",
        "plot_prefix": "qp_ecg",
        "overlay_title": "ECG QuasiPeriodic: Signal Overlay (30-90s)",
        "fit_max_seconds": 60.0,
        "include_score": False,
        "include_component_summary": False,
    },
    ("emg", "dss"): {
        "fit_mode": "continuous",
        "estimator": "dss",
        "bias_kind": "bandpass",
        "merge_mode": "subtract",
        "transform_mode": "2d",
        "plot_dir": "dss_emg",
        "plot_prefix": "emg",
        "overlay_title": "EMG DSS: Signal Overlay (30-90s)",
        "fit_max_seconds": 60.0,
        "safe_subtraction": True,
    },
}


def _get_dss_profile(
    artifact: str,
    method: str,
    config: Any,
    sfreq: float,
) -> Dict[str, Any]:
    """Return a runtime profile for one DSS artifact path via registry."""
    key = (artifact, method)
    if key not in DSS_PROFILES:
        raise ValueError(f"Unsupported DSS profile: artifact={artifact}, method={method}")

    # Standard defaults
    profile = {
        "artifact": artifact,
        "method": method,
        "n_remove": 1,
        "fmax": 50.0,
        "include_score": True,
        "include_component_summary": True,
        "include_spatial_patterns": True,
        "include_component_time_series": True,
        "denoiser_kind": None,
        "bias_kind": None,
        "fit_max_seconds": None,
    }

    # Merge registry values
    reg_val = DSS_PROFILES[key]
    profile.update(reg_val)

    # Instance-specific overrides from config
    if artifact == "eog":
        profile["n_remove"] = config.dss_n_remove_eog or 1
    elif artifact == "ecg":
        profile["n_remove"] = config.dss_n_remove_ecg or 1
    elif artifact == "emg":
        profile["n_remove"] = config.dss_n_remove_emg or 2
        profile["fmax"] = min(100.0, sfreq / 2.0 - 1.0)

    if reg_val.get("use_config_denoiser"):
        profile["denoiser_kind"] = config.blind_nonlinearity

    return profile


def _prepare_continuous_fit(train_raw: mne.io.BaseRaw) -> Dict[str, Any]:
    """Prepare fit data for continuous signals (e.g. EMG or Blind DSS)."""
    train_eeg = train_raw.copy().pick_types(eeg=True, exclude="bads")
    if len(train_eeg.ch_names) == 0:
        return {"skipped": True, "reason": "no_eeg_channels"}
    return {
        "fit_data": train_eeg.get_data(),
        "expected_ch_names": train_eeg.ch_names,
    }


def _prepare_eog_epochs_fit(train_raw: mne.io.BaseRaw) -> Dict[str, Any]:
    """Prepare fit data for EOG-triggered DSS."""
    from mne.preprocessing import create_eog_epochs

    eog_ch_names, eog_source = _resolve_eog_detection_channels(train_raw)
    if eog_ch_names is None:
        return {"skipped": True, "reason": "no_eog_or_frontal"}

    try:
        eog_epochs = create_eog_epochs(
            train_raw,
            ch_name=eog_ch_names,
            baseline=(-0.5, -0.2),
            tmin=-0.5,
            tmax=0.5,
            reject_by_annotation=True,
            verbose="ERROR",
        )
    except Exception:
        return {"skipped": True, "reason": "blink_detection_failed"}

    n_events = len(eog_epochs)
    if n_events < 5:
        return {"skipped": True, "reason": "too_few_blinks", "n_blinks": n_events}

    eog_epochs.pick_types(eeg=True, eog=False, exclude="bads")
    return {
        "fit_data": eog_epochs.get_data(),
        "expected_ch_names": eog_epochs.ch_names,
        "n_events": n_events,
        "n_blinks": n_events,
        "eog_source": eog_source,
    }


def _prepare_ecg_epochs_fit(train_raw: mne.io.BaseRaw) -> Dict[str, Any]:
    """Prepare fit data for ECG-triggered DSS (QRS complex)."""
    from mne.preprocessing import create_ecg_epochs

    try:
        ecg_epochs = create_ecg_epochs(
            train_raw,
            baseline=(-0.2, -0.05),
            tmin=-0.3,
            tmax=0.3,
            reject_by_annotation=True,
            verbose="ERROR",
        )
    except Exception:
        return {"skipped": True, "reason": "no_ecg_channel"}

    n_events = len(ecg_epochs)
    if n_events < 10:
        return {"skipped": True, "reason": "too_few_qrs", "n_qrs": n_events}

    ecg_epochs.pick_types(eeg=True, ecg=False, exclude="bads")
    return {
        "fit_data": ecg_epochs.get_data(),
        "expected_ch_names": ecg_epochs.ch_names,
        "n_events": n_events,
        "n_qrs": n_events,
    }


def _prepare_dss_fit_data(
    train_raw: mne.io.BaseRaw,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Entry point for DSS fit data preparation."""
    fit_mode = profile["fit_mode"]
    if fit_mode == "continuous":
        return _prepare_continuous_fit(train_raw)
    if fit_mode == "eog_epochs":
        return _prepare_eog_epochs_fit(train_raw)
    if fit_mode == "ecg_epochs":
        return _prepare_ecg_epochs_fit(train_raw)
    return {"skipped": True, "reason": f"unsupported_fit_mode:{fit_mode}"}


def _build_dss_estimator(
    config: Any,
    profile: Dict[str, Any],
    sfreq: float,
) -> Tuple[Union[DSS, IterativeDSS], Dict[str, Any]]:
    """Create DSS/IterativeDSS estimator and extra provenance fields."""
    estimator_kind = profile["estimator"]
    bias_kind = profile.get("bias_kind")

    if estimator_kind == "dss":
        if bias_kind == "average":
            return _init_dss(config, bias=AverageBias(axis="epochs")), {}
        if bias_kind == "bandpass":
            nyquist = sfreq / 2.0
            high = max(1.0, nyquist - 1.0)
            low = min(30.0, max(1.0, high - 1.0))
            return _init_dss(
                config,
                bias=BandpassBias(freq_band=(low, high), sfreq=sfreq),
            ), {"bias_type": "high_frequency_bandpass"}
        raise ValueError(f"Unsupported DSS bias kind: {bias_kind}")

    if estimator_kind == "iterative":
        denoiser_kind = profile.get("denoiser_kind")
        if denoiser_kind == "quasiperiodic":
            denoiser = QuasiPeriodicDenoiser(
                peak_distance=int(0.5 * sfreq),
                peak_height_percentile=85,
                smooth_template=True,
            )
            return _init_dss(config, denoiser=denoiser, max_iter=5), {}

        from mne_denoise.dss.denoisers import (
            GaussDenoiser,
            KurtosisDenoiser,
            SmoothTanhDenoiser,
            TanhMaskDenoiser,
            beta_gauss,
            beta_pow3,
            beta_tanh,
        )

        blind_alpha = float(config.blind_alpha)
        blind_window = max(3, int(config.blind_smooth_window))

        if denoiser_kind == "cube":
            denoiser = KurtosisDenoiser(nonlinearity="cube")
            beta = beta_pow3
        elif denoiser_kind == "tanh":
            denoiser = TanhMaskDenoiser(alpha=blind_alpha)
            beta = beta_tanh
        elif denoiser_kind == "gauss":
            denoiser = GaussDenoiser(a=blind_alpha)
            beta = lambda source: beta_gauss(source, a=blind_alpha)
        elif denoiser_kind == "smooth_tanh":
            denoiser = SmoothTanhDenoiser(alpha=blind_alpha, window=blind_window)
            beta = beta_tanh
        else:
            raise ValueError(
                f"Unknown nonlinearity '{denoiser_kind}'. "
                "Expected one of: cube, tanh, gauss, smooth_tanh."
            )

        extra = {
            "nonlinearity": denoiser_kind,
            "alpha": blind_alpha,
        }
        if denoiser_kind == "smooth_tanh":
            extra["smooth_window"] = blind_window
        return _init_dss(config, denoiser=denoiser, beta=beta, method="deflation"), extra

    raise ValueError(f"Unsupported estimator type: {estimator_kind}")


def _run_dss_artifact(
    raw: mne.io.BaseRaw,
    config: Any,
    profile: Dict[str, Any],
    raw_fit: Optional[mne.io.BaseRaw] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Run a single DSS artifact pipeline (Simplified Orchestration)."""
    train_raw = raw_fit if raw_fit is not None else raw
    fit_bundle = _prepare_dss_fit_data(train_raw, profile)

    if fit_bundle.get("skipped"):
        return raw, {
            "method": profile["method"],
            "artifact": profile["artifact"],
            "fit_mode": profile["fit_mode"],
            "skipped": True,
            "reason": fit_bundle.get("reason", "fit_data_unavailable"),
            **{k: v for k, v in fit_bundle.items() if k != "fit_data"},
        }

    # 1. Build & Fit
    dss, extra_stats = _build_dss_estimator(config, profile, float(train_raw.info["sfreq"]))
    fit_data = fit_bundle["fit_data"]
    dss.fit(fit_data)

    # 2. Diagnostic Plots (Pre-cleaning)
    plot_paths: Dict[str, str] = {}
    if output_dir:
        eeg_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude="bads"))
        plot_paths.update(
            viz_qc.save_dss_pre_plots(
                estimator=dss,
                fit_data=fit_data,
                eeg_info=eeg_info,
                fig_dir=output_dir / "figures" / profile["plot_dir"],
                subject_id=subject_id,
                file_prefix=profile["plot_prefix"],
                sfreq=float(train_raw.info["sfreq"]),
                fit_max_seconds=profile.get("fit_max_seconds"),
                include_score=profile.get("include_score", True),
                include_component_summary=profile.get("include_component_summary", True),
                include_spatial_patterns=profile.get("include_spatial_patterns", True),
                include_component_time_series=profile.get("include_component_time_series", True),
            )
        )

    # 3. Dynamic n_remove selection
    scores = getattr(dss, "scores_", None)
    n_remove_auto = profile["n_remove"]
    if scores is not None and len(scores) > 0:
        n_remove_auto = thresholds.select_n_components_dss(
            scores=scores,
            max_n=profile["n_remove"],
            threshold_ratio=0.5,
            method="ratio"
        )

    # 4. Remove Artifact
    raw_out, remove_stats = _remove_source_dss(
        raw=raw,
        dss=dss,
        n_remove=n_remove_auto,
        expected_ch_names=fit_bundle.get("expected_ch_names"),
        use_single_epoch_3d=(profile["transform_mode"] == "3d_single_epoch"),
        subtract_artifact=(profile["merge_mode"] == "subtract"),
        safe_subtraction=bool(profile.get("safe_subtraction", False)),
    )

    # 5. Diagnostic Plots (Post-cleaning)
    if output_dir and not remove_stats.get("skipped"):
        plot_paths.update(
            viz_qc.save_dss_post_plots(
                raw_before=raw,
                raw_after=raw_out,
                fig_dir=output_dir / "figures" / profile["plot_dir"],
                subject_id=subject_id,
                file_prefix=profile["plot_prefix"],
                overlay_title=profile["overlay_title"],
                fmax=float(profile.get("fmax", 50.0)),
            )
        )

    # 6. Final Provenance
    stats = {
        "method": profile["method"],
        "artifact": profile["artifact"],
        "fit_mode": profile["fit_mode"],
        "plot_paths": plot_paths,
        **remove_stats,
        **extra_stats,
        **{k: v for k, v in fit_bundle.items() if k not in ("fit_data", "expected_ch_names")},
    }
    
    return (raw_out if not remove_stats.get("skipped") else raw), stats


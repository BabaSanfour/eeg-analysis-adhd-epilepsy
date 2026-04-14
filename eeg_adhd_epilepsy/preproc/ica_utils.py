"""Shared ICA helpers for Stage 1 artifact correction."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Optional
from pathlib import Path

import mne
import eeg_adhd_epilepsy.viz.clean_qc as viz_qc


def fit_ica_context(train_raw: mne.io.BaseRaw, config: Any) -> Dict[str, Any]:
    """Fit ICA once and run ICLabel once for a training raw."""
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    sfreq = float(train_raw.info["sfreq"])
    h_freq = min(100.0, sfreq / 2.0 - 0.5)
    train_raw_filt = train_raw.copy().filter(l_freq=1.0, h_freq=h_freq, verbose="ERROR")

    ica_picks = mne.pick_types(train_raw_filt.info, eeg=True, exclude="bads")
    n_components = min(config.ica_n_components, len(ica_picks))
    
    ica = ICA(
        n_components=n_components,
        method="fastica",
        max_iter="auto",
        random_state=config.random_state,
        verbose="ERROR",
    )
    ica.fit(train_raw_filt, reject_by_annotation=True)
    labels = label_components(train_raw_filt, ica, method="iclabel")

    return {
        "ica": ica,
        "labels": labels,
    }


from . import thresholds

def apply_ica_artifact(
    raw: mne.io.BaseRaw,
    ica_context: Dict[str, Any],
    *,
    target_labels: Sequence[str],
    exclude_probability: float,
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown",
    artifact_label: str = "ICA",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Apply a shared ICA model by excluding ICs that match target labels."""
    ica = ica_context["ica"]
    labels = ica_context["labels"]

    exclude_idx = thresholds.select_ica_components(
        labels=labels,
        target_labels=target_labels,
        exclude_probability=exclude_probability,
        adaptive=True
    )
    
    probas = [float(labels["y_pred_proba"][i]) for i in exclude_idx]

    raw_clean = ica.apply(raw.copy(), exclude=exclude_idx)
    
    # Save ICA plots
    plot_paths = {}
    if output_dir is not None:
        fig_dir = output_dir / "figures" / "ica"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Spatial patterns of excluded components
        if exclude_idx:
            fig_topo = ica.plot_components(picks=exclude_idx, show=False)
            topo_path = fig_dir / f"{subject_id}_ica_excluded_topo.png"
            fig_topo.savefig(topo_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig_topo)
            plot_paths["spatial_patterns"] = str(topo_path)
            
            # 2. Component time series
            # Use custom snapshot for cleaner, non-red, non-compressed plots
            ts_path = viz_qc.save_ica_sources_snapshot(
                ica, raw, fig_dir, subject_id, picks=exclude_idx, label=artifact_label, 
                start=30.0, duration=20.0
            )
            plot_paths["component_time_series"] = ts_path

    return raw_clean, {
        "method": "ica",
        "n_components_removed": len(exclude_idx),
        "probabilities": probas,
        "total_components": len(labels["labels"]),
        "plot_paths": plot_paths,
    }

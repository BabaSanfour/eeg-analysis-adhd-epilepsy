# Artifact Preprocessing Strategies & "Part 2" Implementation Plan

## Part 1: Summary

### 1. AIM: Non-Destructive Annotation (Current Baseline)
*   **Philosophy**: Prioritize marking artifacts (time × channels) over dropping data entirely. We automatically learn per-channel thresholds via cross-validation to preserve as much data as possible, especially crucial for clinical populations (ADHD/Epilepsy).
*   **Implementation Status**: Currently implemented in `base.py`.
*   **Process**:
    1.  **Global Repair (RANSAC)**: Detects sensors that consistently behave abnormally by correlating each channel with robust re-reference estimates (originally from PREP).
    2.  **Local Repair (AutoReject)**: Operates at the epoch level, finding optimal rejection thresholds per channel. Segments exceeding these are marked "bad" and interpolated.
*   **Tools**: MNE-Python's `autoreject` + pyPREP `RANSAC` facilitates this by marking segments rather than outright rejection.

---

## Part 2: Strategic Options for Next Steps

### 1. Hybrid Pipeline: RELAX / RELAX-Jr (Platinum Standard)
*   **Overview**: The RELAX/RELAX-Jr pipeline combines multi-channel filtering and ICA-based methods to handle difficult clinical data with high artifact loads.
*   **Methodology**:
    *   **Multi-Channel Wiener Filtering (MWF)**: Applied sequentially to suppress stereotyped artifacts.
    *   **Wavelet-Enhanced ICA (wICA)**: Labels independent component time segments as artifactual using wavelet thresholding to remove residual broad-band noise (EMG).
    *   **Automated Classification**: Uses `annotate_muscle_zscore`, `ICLabel` or `ADJUST` python equivalent to identify and remove artifact-dominant components or segments.
*   **Shape/Spectral Features**: Differentiates artifacts by frequency content (e.g., muscle = high freq, HV delta = low freq). RELAX-Jr uses log-frequency slope metrics and sets high absolute amplitude thresholds (±100 µV) to preserve alpha activity and activation responses.
*   **High Thresholds**: Uses high rejection thresholds to pass most abnormal brain waves (e.g., ±100 µV), removing only clear outliers.

### 2. Adaptive Filtering: ASR / ANF (Continuous Cleaning)
*   **Artifact Subspace Reconstruction (ASR)**:
    *   **Mechanism**: Calibrates on a clean reference segment, projects sliding windows onto the learned covariance subspace, and removes components whose variance bursts beyond a dynamic threshold (e.g., 20 standard deviations).
    *   **Use Case**: Superior for handling non-stationary, high-amplitude artifacts (movement, pops).
*   **Adaptive Noise Filters (ANF)**: Uses LMS-based adaptive filtering to continuously subtract noise while retaining seizure-related information. Adapts to preserve non-stationary brain activity without affecting seizure-relevant signals.

### 3. Robust Statistics & Multi-Feature Criteria
*   **Metrics**: Uses Median and IQR instead of Mean/Variance to determine thresholds, mitigating the impact of non-stationary epochs.
*   **Hybrid Workflow**: Fully automated spike detectors often have low specificity (~3-63%). Human expert review of AI-flagged events improves specificity to ~95%.
*   **Component Classification**: Validates ICA components to ensure they do not map to spike-like scalp topographies before removal.
*   **Spatial/Temporal Signatures**: Exploits the distinct field distribution of true IEDs (dipolar) vs. widespread noise.

---

## Part 3: Python Ecosystem for Implementation

### A. Channel & Segment Repair (Implemented)
*   **AutoReject**:
    *   Implements the "RANSAC + AutoReject" strategy.
    *   `autoreject.Ransac` allows detecting and interpolating outlier channels in a single call.
    *   Provides fit/transform interface for local bad segment repair.
*   **PyPREP**:
    *   Python implementation of the PREP pipeline (Bigdely-Shamlo et al., 2015).
    *   Handles robust re-referencing and bad channel identification.

### B. ICA-Based Handling (Next Steps)
*   **MNE-Python**: Core ICA functions (Picard, Infomax, FastICA).
*   **mne-icalabel**:
    *   Port of the `ICLabel` classifier (originally MATLAB).
    *   Auto-tags independent components (Brain, Muscle, Eye, etc.) using pretrained models.

### C. ASR Implementations (Future Phase)
*   **mne-clean-raw / EEGCleaner**:
    *   Community implementations to apply ASR on continuous data.
    *   **pyASR**: Unofficial port of the ASR algorithm.

### D. RELAX Components (Next Steps)
*   **MWF**: Can be implemented via spatial filters or libraries like `meegkit`.
*   **wICA**: Achievable by combining `PyWavelets` (wavelet denoising) with MNE ICA.

---

## Part 4: Detailed Implementation Roadmap

### Step 1: Baseline Pipeline (Completed)
*   **Goal**: Establish a robust foundation for artifact handling.
*   **Mechanism**:
    1.  **Global Repair**: Uses RANSAC to interpolate bad channels based on sensor consensus.
    2.  **Local Repair**: Uses AutoReject to learn optimal peak-to-peak thresholds per channel via cross-validation to interpolate specific bad segments.
*   **Status**: Implemented in `base.py`.

### Step 2: RELAX-style Hybrid Pipeline (Next Priority)
*   **Goal**: Implement the "Platinum Standard" for clinical data.
*   **Mechanism**: A sequential cleaning pipeline:
    1.  **MWF**: Suppresses stereotyped artifacts (blinks, muscle) spatially.
    2.  **wICA**: Removes residual broad-band noise (EMG) from independent components using wavelet thresholding.
    3.  **ICLabel/ADJUST**: Automatically classifies and removes artifact-dominant components.

### Step 3: ASR + ICA Refinement (Future Phase)
*   **Goal**: Address continuous cleaning of non-stationary artifacts.
*   **Mechanism**:
    1.  **ASR**: Attenuates high-variance bursts (movement, pops).
    2.  **ICA Refinement**: Removes remaining stereotyped artifacts (blinks, cardiac) that ASR might miss.

### Step 4: Second-Stage AutoReject (Post-Cleaning)
*   **Goal**: Final cleanup pass.
*   **Mechanism**:
    1.  **Local Repair**: Re-run AutoReject to tidy up any artifacts remaining after complex cleaning.
    2.  **Global Repair**: Ensure channel integrity.
    3.  **Annotate**: Update annotations based on interpolated segments.

---

## Part 5: Alternative / Augmentation: DSS-Centric Pipeline

A unified, data-driven framework using **Denoising Source Separation (DSS)** can replace or enhance specific cleaning steps by exploiting known signal biases (event timing, frequency).

### Concept
*   **Unified Framework**: Instead of ad-hoc heuristics, use DSS to find spatial filters that maximize a specific "bias" (reproducibility etc.).
*   **Implementation**: Leveraging `mne-denoise` in Python.

### Components
*   **EOG Removal**:
    *   **Bias**: Blink timing (from EOG channel or thresholding).
    *   **Process**: DSS finds components time-locked to blinks. These are projected out.
    *   **Benefit**: "Nephew of ICA" specifically for eyes; no manual component selection/labelling needed.
*   **ECG Removal**:
    *   **Bias**: Heartbeat timing (QRS complexes).
    *   **Process**: DSS finds components phase-locked to ECG.
    *   **Benefit**: Replaces regression/OBS; robustly isolates cardiac artifact.
*   **Muscle/EMG**:
    *   **Bias**: High-frequency power (>30Hz) or kurtosis (non-Gaussianity).
    *   **Process**: Iterative DSS or generalized eigen-decomposition (GEVD) to isolate noisy, broadband components.
*   **Signal Enhancement**:
    *   **Alpha/Oscillations**: Narrow-band bias (8-12Hz) to boost alpha SNR.
    *   **SSVEP**: Periodic bias at stimulus frequency to isolate steady-state responses.
*   **Supervised De-Spiking**:
    *   **Bias**: Epileptiform spike timing or template match.
    *   **Process**: Separate spikes from background EEG into parallel streams (one for spike analysis, one for background spectral analysis).

### Selective Augmentation: Hybrid Pipeline with DSS Enhancements
In some cases, an all-DSS pipeline might not be desirable. A hybrid approach runs standard effective methods first, then inserts DSS at strategic points.

#### A. DSS after Standard Cleaning
*   **Strategy**: Run AutoReject/RANSAC first to fix gross errors, then run DSS on the "mostly clean" data.
*   **Evidence**: The APICE pipeline (infant EEG) showed DSS removed ~13-33% additional variance *after* artifact rejection, targeting residual noise without altering earlier steps.
*   **TODO**: Run DSS after `base.py` processing.

#### B. Reducing Reliance on Rejection
*   **Benefit**: Instead of rejecting trials with minor artifacts (e.g., small eye movements), use a mild DSS cleaning to subtract the artifact component.

### Limitations: When DSS Might Not Be Suitable

While DSS is powerful, there are scenarios where it offers little benefit or could even be counterproductive.

#### A. Potential Signal Loss
*   **Limitation**: Aggressive filtering based on bias might remove neural components that don't match the bias perfectly.
*   **Safeguard**: Check variance removed by DSS to ensure it's reasonable.

#### B. Not a Fix for Bad Channels
*   **Limitation**: DSS assumes all channels carry source mixtures; it doesn't fix a dead/flat channel.
*   **Requirement**: Must run bad-channel detection (RANSAC) *before* DSS to prevent artifacts from dominating the covariance.

---

## Part 6: Technical Implementation Plan

This entire phase runs **after** the non-destructive baseline (`base.py`). The input is the `raw` object returned by `base.py` (which has RANSAC-corrected bad channels and initial AutoReject annotations from Part 1).

**Philosophy**: `base.py` provides a conservative foundation. This phase implements advanced cleaning in **2 sequential stages** across **3 new modules**, drawing inspiration from Parts 2-5. Each step allows choosing between multiple methods.

**Key Integration Points**:
1.  Uses `base.py` annotations to exclude bad segments during ICA fitting
2.  Follows `base.py` structure with `benchmark_step()` and provenance tracking
3.  Tracks correction effectiveness by comparing pre/post artifact annotations

### 1. Architecture: Strict 2-Stage Pipeline

We enforce a strict separation of concerns where `denoise.py` consumes the output of `correct.py`.

```
[Raw Data] → base.py (Detect) → [Raw + Bad Annotations] 
                                        ↓
                                   correct.py (Stage 1: Source Correction)
                                   Main: run_source_correction()
                                        ↓
                                   [Raw Corrected]
                                        ↓
                                   denoise.py (Stage 2: Residual Denoising)
                                   Main: run_residual_denoising()
                                        ↓
                                   [Raw Final]
```

**Benchmarking (`compare.py`)**: Now compares `Raw Corrected` vs `Raw Final` to quantify the added value of stage 2.

### 1b. Module Imports

#### `correct.py` imports
```python
"""
Source Correction Module (Stage 1).
Main Entry: run_source_correction()
"""
import logging
from typing import Any, Dict, Optional, Tuple
import mne
import numpy as np
import scipy.linalg
from .utils import benchmark_step

LOGGER = logging.getLogger(__name__)

# Optional dependencies...
```

#### `denoise.py` imports
```python
"""
Residual Denoising Module (Stage 2).
Main Entry: run_residual_denoising()
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mne
import numpy as np

from .utils import benchmark_step, PreprocConfig
from .base import annotate_artifacts_blockwise, _collect_block_windows

# Import Stage 1 for the unified pipeline wrapper (optional)
# But denoise.py logic primarily operates on the OUTPUT of Stage 1
from .correct import run_source_correction

LOGGER = logging.getLogger(__name__)
# Optional dependencies...
```

#### `compare.py` imports
```python
"""
Pipeline Benchmarking.
Compares: Base vs Corrected (Stage 1) vs Final (Stage 2)
"""
import logging
import mne
import pandas as pd
from .correct import run_source_correction
from .denoise import run_residual_denoising

LOGGER = logging.getLogger(__name__)
```


### 2. Core Orchestrators

#### 2a. Source Correction (`correct.py` Main)

Runs Stage 1: Removal of specific physiological artifacts (EOG, ECG, EMG).

```python
def run_source_correction(
    raw: mne.io.BaseRaw, 
    config: 'ArtifactCorrectionConfig'
) -> Tuple[mne.io.BaseRaw, Dict]:
    """
    Stage 1: Remove specific physiological artifacts.
    Output is fed into denoise.py.
    """
    provenance = {
        "steps_completed": [],
        "correction_stats": {},
        "timings": {}
    }
    
    # Extract base.py bad segments to exclude from fitting
    bad_segments = _extract_bad_segments(raw)
    
    # Step 1.1: EOG (Eye) Removal
    if config.eog_method:
        with benchmark_step("eog_removal", provenance):
            if config.eog_method == "dss":
                raw, s = _remove_eog_dss(raw, config, bad_segments)
            elif config.eog_method == "ica":
                raw, s = _remove_eog_ica(raw, config, bad_segments)
            provenance["correction_stats"]["eog"] = s
        provenance["steps_completed"].append("eog_removal")
    
    # Step 1.2: ECG (Heart) Removal
    if config.ecg_method:
        with benchmark_step("ecg_removal", provenance):
            if config.ecg_method == "dss":
                raw, s = _remove_ecg_dss(raw, config, bad_segments)
            elif config.ecg_method == "ica":
                raw, s = _remove_ecg_ica(raw, config, bad_segments)
            elif config.ecg_method == "quasiperiodic":
                raw, s = _remove_ecg_quasiperiodic(raw, config, bad_segments)
            provenance["correction_stats"]["ecg"] = s
        provenance["steps_completed"].append("ecg_removal")

    # Step 1.3: EMG (Muscle) Removal
    if config.emg_method:
        with benchmark_step("emg_removal", provenance):
            if config.emg_method == "mwf":
                raw, s = _remove_emg_mwf(raw, config, bad_segments)
            elif config.emg_method == "wica":
                raw, s = _remove_emg_wica(raw, config, bad_segments)
            elif config.emg_method == "ica":
                raw, s = _remove_emg_ica(raw, config, bad_segments)
            elif config.emg_method == "dss":
                raw, s = _remove_emg_dss(raw, config, bad_segments)
            provenance["correction_stats"]["emg"] = s
        provenance["steps_completed"].append("emg_removal")
        
    return raw, provenance

def _extract_bad_segments(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """Helper: Extract bad segment timestamps from base.py annotations."""
    return [
        (a['onset'], a['duration']) 
        for a in raw.annotations 
        if a['description'].startswith('BAD_')
    ]
```

#### 2b. Residual Denoising (`denoise.py` Main)

Runs Stage 2: Removal of transients and final refinement.

```python
def run_residual_denoising(
    raw: mne.io.BaseRaw, 
    config: 'ArtifactCorrectionConfig', 
    subject_id: str,
    output_dir: Optional[Path] = None,
    stage1_provenance: Optional[Dict] = None
) -> Tuple[mne.io.BaseRaw, Dict]:
    """
    Stage 2: Remove transients and perform final refinement.
    Consumes output of correct.py.
    """
    provenance = {
        "steps_completed": [],
        "correction_stats": {},
        "timings": {},
        "stage1_provenance": stage1_provenance or {}
    }
    
    # Extract base.py bad segments again (needed for final comparison)
    # We can reuse the helper logic or import it
    bad_segments = [
        (a['onset'], a['duration']) 
        for a in raw.annotations 
        if a['description'].startswith('BAD_')
    ]
    
    # Step 2.1: Transient/Movement Removal
    if config.transient_method:
        with benchmark_step("transient_removal", provenance):
            if config.transient_method == "asr":
                raw, s = _remove_transients_asr(raw, config)
            elif config.transient_method == "dss":
                raw, s = _remove_transients_dss(raw, config)
            elif config.transient_method == "wiener_mask":
                raw, s = _remove_transients_wiener_mask(raw, config)
            provenance["correction_stats"]["transients"] = s
        provenance["steps_completed"].append("transient_removal")
    
    # Step 2.2: Final Refinement (AutoReject)
    if config.final_autoreject:
        with benchmark_step("autoreject_refinement", provenance):
            raw, s = _refine_autoreject(raw, config, bad_segments)
            provenance["correction_stats"]["autoreject_stage2"] = s
        provenance["steps_completed"].append("autoreject_refinement")
    
    # Calculate overall effectiveness
    provenance["correction_effectiveness"] = _calculate_correction_effectiveness(
        bad_segments, 
        provenance["correction_stats"]
    )
    
    # Generate Report
    if output_dir:
        plot_paths = {} 
        # ... (Collect plots from output_dir and generate report) ...
        # logic for plot collection goes here
        
        try:
            provenance['report_path'] = str(generate_cleaning_html_report(
                provenance, plot_paths, output_dir, subject_id
            ))
        except Exception as e:
            LOGGER.warning(f"Failed to generate report: {e}")
         
    return raw, provenance
```




### 4. Stage 1 Implementation: Specific Artifacts (ECG, EOG, EMG) → `correct.py`

Drawing from Parts 2 and 5, each physiological artifact can be removed using DSS or ICA.

#### Step 1.1: EOG (Eye) Removal

##### Option A: DSS-Based (Part 5)
```python
def _remove_eog_dss(raw, config, base_bad_segments):
    """
    Remove eye blinks using DSS with blink event bias.
    Part 5: Exploits repetitiveness of blinks.
    
    Excludes base.py bad segments when finding blinks.
    """
    from mne_denoise.dss import DSS, AverageBias
    from mne.preprocessing import create_eog_epochs
    
    # 1. Create EOG epochs (bias = blink timing)
    # Use reject_by_annotation to exclude base.py bad segments
    eog_epochs = create_eog_epochs(
        raw, 
        baseline=(-0.5, -0.2), 
        tmin=-0.5, 
        tmax=0.5,
        reject_by_annotation=True  # Exclude base.py bad segments
    )
    
    if len(eog_epochs) < 5:
        LOGGER.warning("Too few blinks detected, skipping EOG-DSS")
        return raw, {'n_blinks': 0, 'skipped': True}
    
    eog_epochs.pick_types(eeg=True, eog=False)
    
    # 2. Fit DSS with Trial Average Bias
    dss = DSS(n_components=config.dss_n_components, bias=AverageBias(axis='epochs'))
    dss.fit(eog_epochs.get_data())
    
    # 3. Transform → Zero out → Inverse Transform (correct pattern)
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    sources = dss.transform(raw_eeg.get_data().T).T  # (n_components, n_times)
    sources[0, :] = 0  # Zero out blink component
    cleaned_data = dss.inverse_transform(sources.T).T
    # 4. Create new Raw with cleaned EEG data
    raw = mne.io.RawArray(cleaned_data, raw_eeg.info, first_samp=raw.first_samp)
    
    return raw, {
        'method': 'dss', 
        'n_blinks': len(eog_epochs), 
        'n_components_removed': 1,
        'excluded_base_segments': True
    }
```

##### Option B: ICA-Based (Part 2)
```python
def _remove_eog_ica(raw, config, base_bad_segments):
    """
    Remove eye artifacts using ICA + ICLabel.
    Part 2: Standard approach with automated classification.
    
    Fits ICA only on clean segments (excluding base.py annotations).
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    # 1. Fit ICA (MNE automatically excludes annotated bad segments)
    ica = ICA(
        n_components=config.ica_n_components, 
        method='picard',
        max_iter='auto'
    )
    ica.fit(raw, reject_by_annotation=True)  # Key: exclude base.py bads
    
    # 2. Classify components
    labels = label_components(raw, ica, method='iclabel')
    
    # 3. Exclude 'eye' components with high probability
    exclude_idx = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'eye' and prob[i] > config.exclude_probability:
            exclude_idx.append(i)
    
    ica.exclude = exclude_idx
    raw = ica.apply(raw)
    
    return raw, {
        'method': 'ica', 
        'n_components_removed': len(exclude_idx),
        'excluded_base_segments': True,
        'total_components': config.ica_n_components
    }
```

#### Step 1.2: ECG (Heart) Removal

##### Option A: DSS-Based (Part 5)
```python
def _remove_ecg_dss(raw, config, base_bad_segments):
    """
    Remove heartbeat using DSS with QRS event bias.
    Part 5: Exploits periodicity of ECG.
    """
    from mne_denoise.dss import DSS, AverageBias
    from mne.preprocessing import create_ecg_epochs
    
    # 1. Create ECG epochs (bias = QRS timing)
    ecg_epochs = create_ecg_epochs(
        raw, 
        baseline=(-0.2, -0.05), 
        tmin=-0.3, 
        tmax=0.3,
        reject_by_annotation=True
    )
    
    if len(ecg_epochs) < 10:
        LOGGER.warning("Too few QRS events detected, skipping ECG-DSS")
        return raw, {'n_qrs': 0, 'skipped': True}
    
    ecg_epochs.pick_types(eeg=True, ecg=False)
    
    # 2. Fit DSS
    dss = DSS(n_components=config.dss_n_components, bias=AverageBias(axis='epochs'))
    dss.fit(ecg_epochs.get_data())
    
    # 3. Transform → Zero out → Inverse Transform (correct pattern)
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    sources = dss.transform(raw_eeg.get_data().T).T
    sources[0, :] = 0  # Zero out cardiac component
    cleaned_data = dss.inverse_transform(sources.T).T
    # 4. Create new Raw with cleaned EEG data
    raw = mne.io.RawArray(cleaned_data, raw_eeg.info, first_samp=raw.first_samp)
    
    return raw, {
        'method': 'dss', 
        'n_qrs': len(ecg_epochs), 
        'n_components_removed': 1,
        'excluded_base_segments': True
    }
```

##### Option B: ICA-Based (Part 2)
```python
def _remove_ecg_ica(raw, config, base_bad_segments):
    """
    Remove cardiac artifacts using ICA + ICLabel.
    Part 2: Standard approach.
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    ica = ICA(n_components=config.ica_n_components, method='picard', max_iter='auto')
    ica.fit(raw, reject_by_annotation=True)
    
    labels = label_components(raw, ica, method='iclabel')
    
    exclude_idx = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'heart' and prob[i] > config.exclude_probability:
            exclude_idx.append(i)
    
    ica.exclude = exclude_idx
    raw = ica.apply(raw)
    
    return raw, {
        'method': 'ica', 
        'n_components_removed': len(exclude_idx),
        'excluded_base_segments': True
    }
```

##### Option C: QuasiPeriodicDenoiser (Template-Based)
```python
def _remove_ecg_quasiperiodic(raw, config, base_bad_segments):
    """
    Remove cardiac artifacts using template-based QuasiPeriodicDenoiser.
    Best when: ECG channel available with clear R-peaks.
    Algorithm: Detect R-peaks → Build adaptive template → Subtract
    """
    from mne_denoise.dss import IterativeDSS
    from mne_denoise.dss.denoisers import QuasiPeriodicDenoiser
    
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        LOGGER.warning("No ECG channel, falling back to DSS")
        return _remove_ecg_dss(raw, config, base_bad_segments)
    
    sfreq = raw.info['sfreq']
    
    qp_denoiser = QuasiPeriodicDenoiser(
        peak_distance=int(0.5 * sfreq),  # 120 BPM max
        peak_height_percentile=85,
        smooth_template=True
    )
    
    raw_eeg = raw.copy().pick_types(eeg=True)
    idss = IterativeDSS(denoiser=qp_denoiser, n_components=config.dss_n_components, max_iter=5)
    idss.fit(raw_eeg)
    
    sources = idss.transform(raw_eeg)
    sources[0, :] = 0  # Zero cardiac component
    cleaned_data = idss.inverse_transform(sources)
    
    # Create new Raw with cleaned EEG data
    raw = mne.io.RawArray(cleaned_data, raw_eeg.info, first_samp=raw.first_samp)
    
    return raw, {'method': 'quasiperiodic', 'n_components_removed': 1}
```


#### Step 1.3: EMG (Muscle) Removal


##### Option A: Multi-Channel Wiener Filter (Part 2.1, RELAX)
```python
def _remove_emg_mwf(raw, config, base_bad_segments):
    """
    Remove muscle artifacts using MWF.
    Part 2.1 (RELAX): Spatially suppress EMG while preserving brain signals.
    
    Uses base.py muscle annotations to define artifact vs clean segments.
    """
    # 1. Extract muscle annotations from base.py
    muscle_annot = [a for a in raw.annotations if 'muscle' in a['description']]
    if len(muscle_annot) == 0:
        LOGGER.warning("No muscle annotations from base.py, skipping MWF")
        return raw, {'skipped': True, 'reason': 'no_muscle_annotations'}
    
    # 2. Compute covariances
    clean_mask = _get_clean_segments(raw, muscle_annot)
    artifact_mask = ~clean_mask
    
    data = raw.get_data()
    C_signal = np.cov(data[:, clean_mask])
    C_artifact = np.cov(data[:, artifact_mask])
    
    # 3. GEVD: maximize signal/artifact ratio
    eigenvalues, W = scipy.linalg.eigh(C_signal, C_signal + C_artifact)
    
    # 4. Keep top components
    n_keep = config.mwf_n_components
    data_clean = W[:, -n_keep:].T @ data
    raw._data = W[:, -n_keep:] @ data_clean
    
    variance_removed = 1 - np.var(raw._data) / np.var(data)
    return raw, {
        'method': 'mwf', 
        'variance_removed': variance_removed,
        'n_muscle_annotations': len(muscle_annot)
    }
```

##### Option B: Wavelet-ICA (Part 2.1, RELAX-Jr)
```python
def _remove_emg_wica(raw, config, base_bad_segments):
    """
    Remove muscle artifacts using wavelet-enhanced ICA.
    Part 2.1 (RELAX-Jr): Denoise ICA components with wavelet thresholding.
    """
    from sklearn.decomposition import FastICA
    import pywt
    
    # 1. Get clean data (exclude base.py bad segments)
    clean_data = _get_clean_data(raw, base_bad_segments)
    
    # 2. Run ICA on clean data
    ica = FastICA(n_components=config.ica_n_components, max_iter=1000)
    sources = ica.fit_transform(clean_data.T).T
    
    # 3. Wavelet denoise each IC
    denoised_sources = []
    for ic in sources:
        coeffs = pywt.wavedec(ic, config.wavelet_type, level=config.wavelet_level)
        
        # Universal threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(ic)))
        coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        denoised_sources.append(pywt.waverec(coeffs_thresh, config.wavelet_type))
    
    # 4. Inverse ICA (apply to full data)
    raw._data = ica.inverse_transform(np.array(denoised_sources).T).T
    
    return raw, {
        'method': 'wica', 
        'n_components_denoised': len(sources),
        'excluded_base_segments': True
    }
```

##### Option C: Standard ICA (Part 2)
```python
def _remove_emg_ica(raw, config, base_bad_segments):
    """
    Remove muscle artifacts using standard ICA + ICLabel.
    Part 2: Simplest approach.
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    
    ica = ICA(n_components=config.ica_n_components, method='picard', max_iter='auto')
    ica.fit(raw, reject_by_annotation=True)
    
    labels = label_components(raw, ica, method='iclabel')
    
    exclude_idx = []
    for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        if label == 'muscle' and prob[i] > config.exclude_probability:
            exclude_idx.append(i)
    
    ica.exclude = exclude_idx
    raw = ica.apply(raw)
    
    return raw, {
        'method': 'ica', 
        'n_components_removed': len(exclude_idx),
        'excluded_base_segments': True
    }
```

##### Option D: DSS-Based (Part 5)
```python
def _remove_emg_dss(raw, config, base_bad_segments):
    """
    Remove muscle artifacts using DSS with high-frequency bias.
    Part 5: Exploits high-frequency spectral signature of EMG.
    """
    from mne_denoise.dss import DSS, PowerBias
    
    # 1. Use power in high-frequency band (>30 Hz) as bias
    # This targets muscle artifacts which are broadband but strong at high freq
    
    # Filter data to high-freq band for bias computation
    raw_hf = raw.copy().filter(l_freq=30, h_freq=None, verbose='ERROR')
    
    # 2. Compute power bias (high variance in HF = muscle)
    hf_power = np.var(raw_hf.get_data(), axis=1)
    
    # 3. Fit DSS
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    dss = DSS(n_components=config.dss_n_components, bias=PowerBias(bias_signal=hf_power[:len(raw_eeg.ch_names)]))
    dss.fit(raw_eeg.get_data().T)
    
    # 4. Transform → Zero out → Inverse Transform
    sources = dss.transform(raw_eeg.get_data().T).T
    n_remove = config.dss_emg_n_remove or 2
    for i in range(n_remove):
        sources[i, :] = 0  # Zero out muscle components
    cleaned_data = dss.inverse_transform(sources.T).T
    # 5. Create new Raw with cleaned EEG data
    raw = mne.io.RawArray(cleaned_data, raw_eeg.info, first_samp=raw.first_samp)
    
    return raw, {
        'method': 'dss',
        'n_components_removed': n_remove,
        'bias_type': 'high_frequency_power'
    }
```

### 5. Stage 2 Implementation: Aggressive Cleaning → `denoise.py`

#### Step 2.1: Transient/Movement Removal

##### Option A: ASR (Part 2.2 & Part 4 Step 3)
```python
def _remove_transients_asr(raw, config):
    """
    Remove transient artifacts (pops, movement) using ASR.
    Part 2.2: Calibrate on clean reference, remove high-variance bursts.
    """
    from meegkit import asr # could also use asrpy 
    
    # 1. Find clean reference window (lowest variance, avoiding annotated bads)
    data = raw.get_data()
    window_len = int(config.asr_calibration_window * raw.info['sfreq'])
    
    # Get mask of clean periods (no annotations)
    clean_mask = np.ones(data.shape[1], dtype=bool)
    for annot in raw.annotations:
        if annot['description'].startswith('BAD_'):
            start_idx = raw.time_as_index(annot['onset'])[0]
            end_idx = raw.time_as_index(annot['onset'] + annot['duration'])[0]
            clean_mask[start_idx:end_idx] = False
    
    # Find lowest variance window in clean periods
    variances = []
    window_indices = []
    for i in range(0, data.shape[1] - window_len, window_len):
        if clean_mask[i:i+window_len].all():  # Fully clean window
            variances.append(np.var(data[:, i:i+window_len]))
            window_indices.append(i)
    
    if len(variances) == 0:
        LOGGER.warning("No clean calibration window found for ASR")
        return raw, {'skipped': True, 'reason': 'no_clean_calibration_window'}
    
    clean_window_idx = window_indices[np.argmin(variances)]
    clean_data = data[:, clean_window_idx:clean_window_idx+window_len]
    
    # 2. Calibrate ASR
    asr_state = asr.ASR(sfreq=raw.info['sfreq'], cutoff=config.asr_cutoff)
    asr_state.fit(clean_data.T)
    
    # 3. Process with sliding window
    data_clean = asr_state.transform(data.T).T
    raw._data = data_clean
    
    variance_removed = 1 - np.var(data_clean) / np.var(data)
    return raw, {
        'variance_removed': variance_removed, 
        'cutoff': config.asr_cutoff,
        'calibration_used_clean_segments': True
    }
```

##### Option B: DSS-Based (Part 5)
```python
def _remove_transients_dss(raw, config):
    """
    Remove transients using DSS with temporal structure bias.
    Part 5: Alternative to ASR for burst removal.
    
    Uses temporal discontinuity as bias (high derivative = transient).
    """
    from mne_denoise.dss import DSS
    
    # 1. Compute temporal derivative (high = transients/bursts)
    data = raw.get_data()
    derivative = np.diff(data, axis=1)
    derivative_power = np.sum(derivative**2, axis=0)
    
    # 2. Use high-derivative segments as "bias" for artifacts
    # DSS will find components that maximize variance in high-derivative periods
    
    # Create binary mask: high derivative = artifact
    threshold = np.percentile(derivative_power, 95)  # Top 5% = transients
    transient_mask = derivative_power > threshold
    
    # 3. Fit DSS with temporal bias
    # (Implementation would require custom Bias class in mne-denoise)
    # For now, use standard DSS on high-variance segments
    
    LOGGER.info("DSS for transients: experimental feature")
    return raw, {
        'method': 'dss_transients',
        'note': 'experimental'
    }
```

##### Option C: Wiener Mask Denoiser (Adaptive for Bursty Signals)
```python
def _remove_transients_wiener_mask(raw, config):
    """
    Remove transients using Adaptive Wiener Masking from mne-denoise.
    
    IDEAL for your EEG data with:
    - Bursty activity (hyperventilation, photic responses)
    - Epileptic spikes (preserves transient clinical markers)
    - Intermittent artifacts (movement bursts)
    
    The denoiser adapts to LOCAL variance, preserving bursty signals
    while dampening stationary noise.
    """
    from mne_denoise.dss import IterativeDSS
    from mne_denoise.dss.denoisers import WienerMaskDenoiser
    
    LOGGER.info("Applying Adaptive Wiener Mask Denoiser for transients...")
    
    # 1. Get EEG data
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    data = raw_eeg.get_data()
    
    # 2. Configure denoiser
    # Window should capture expected burst duration
    # For bursty activity (beta bursts, spindles): ~200ms window
    # For spikes: shorter window (~50-100ms)
    sfreq = raw.info['sfreq']
    window_samples = int(config.wiener_window_duration * sfreq)
    
    denoiser = WienerMaskDenoiser(
        window_samples=window_samples,
        noise_percentile=config.wiener_noise_percentile
    )
    
    # 3. Apply Iterative DSS with Wiener Masking
    # This finds components that are BURSTY (high local variance)
    # and suppresses components that are STATIONARY (low local variance)
    dss = IterativeDSS(
        denoiser=denoiser,
        n_components=config.wiener_n_components,
        max_iter=config.wiener_max_iter,
        random_state=42
    )
    
    dss.fit(data.T)
    data_denoised = dss.transform(data.T).T
    
    # 4. Reconstruct using only denoised components
    # This preserves bursty clinical signals while removing stationary artifacts
    data_clean = dss.inverse_transform(data_denoised.T).T
    
    # 5. Create new Raw with cleaned EEG data
    raw = mne.io.RawArray(data_clean, raw_eeg.info, first_samp=raw.first_samp)
    
    # 6. Calculate effectiveness
    variance_original = np.var(data)
    variance_clean = np.var(data_clean)
    variance_removed = 1 - (variance_clean / variance_original)
    
    return raw, {
        'method': 'wiener_mask',
        'variance_removed': variance_removed,
        'window_duration': config.wiener_window_duration,
        'noise_percentile': config.wiener_noise_percentile,
        'n_components': config.wiener_n_components,
        'note': 'Preserves bursty clinical signals (spikes, bursts)'
    }
```


#### Step 2.2: Final Refinement (AutoReject)

**Compare against base.py annotations to track correction effectiveness**

```python
def _refine_autoreject(raw, config, base_bad_segments):
    """
    Final cleanup using AutoReject.
    Part 4 Step 4: Re-run local repair after advanced cleaning.
    
    Compares Stage 2 annotations to base.py annotations to measure effectiveness.
    """
    # ===== REUSE base.py's annotate_artifacts_blockwise =====
    # This gives us condition-level AutoReject with chunking for free!
    from .base import annotate_artifacts_blockwise
    
    # Create a minimal config dict for the function
    ar_config = PreprocConfig({
        'artifacts': {
            'epoch_duration_s': config.epoch_duration,
            'n_interpolate': [1, 4, 8],
            'ar_max_chunk_minutes': config.get('ar_max_chunk_minutes', 30),
        }
    })
    
    # Run condition-level AutoReject (same as base.py)
    raw, ar_stats = annotate_artifacts_blockwise(
        raw, ar_config, 
        figures_dir=output_dir / 'figures' if output_dir else None,
        subject_id=subject_id
    )
    
    # Extract Stage 2 bad segments for comparison
    bad_segments_stage2 = [
        (a['onset'], a['duration']) 
        for a in raw.annotations 
        if a['description'].startswith('BAD_')
    ]
    
    # Compare to base.py: How many base.py bad segments are now clean?
    n_corrected = _count_corrected_segments(base_bad_segments, bad_segments_stage2)
    
    return raw, {
        'n_epochs_rejected': ar_stats.get('bad_epochs', 0),
        'percent_rejected': ar_stats.get('bad_epochs', 0) / max(1, ar_stats.get('blocks_processed', 1)) * 100,
        'n_base_segments_corrected': n_corrected,
        'correction_rate': n_corrected / len(base_bad_segments) if base_bad_segments else 0,
        'blocks_processed': ar_stats.get('blocks_processed', 0),
        'reused_base_function': True  # Flag for provenance
    }
```


### 6. Helper: Calculate Correction Effectiveness → `denoise.py`

```python
def _count_corrected_segments(base_segments, stage2_segments):
    """
    Count how many base.py bad segments are NOT in stage2 (i.e., were corrected).
    
    A segment is "corrected" if it was marked bad in base.py but NOT in stage2.
    """
    corrected = 0
    for base_onset, base_dur in base_segments:
        base_end = base_onset + base_dur
        
        # Check if this base segment overlaps with any stage2 segment
        is_still_bad = False
        for s2_onset, s2_dur in stage2_segments:
            s2_end = s2_onset + s2_dur
            # Check overlap
            if not (base_end <= s2_onset or s2_end <= base_onset):
                is_still_bad = True
                break
        
        if not is_still_bad:
            corrected += 1
    
    return corrected

def _calculate_correction_effectiveness(base_segments, correction_stats):
    """
    Calculate overall effectiveness metrics across all cleaning steps.
    """
    total_variance_removed = 0
    total_components_removed = 0
    
    for step, stats in correction_stats.items():
        if 'variance_removed' in stats:
            total_variance_removed += stats['variance_removed']
        if 'n_components_removed' in stats:
            total_components_removed += stats['n_components_removed']
    
    return {
        'total_variance_removed': total_variance_removed,
        'total_components_removed': total_components_removed,
        'n_base_bad_segments': len(base_segments),
        'correction_details': correction_stats
    }
```

### 6b. Visualization & Plotting for Reports → `denoise.py`

**Following `base.py` pattern for report generation**

```python
def plot_cleaning_summary(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    provenance: Dict,
    output_dir: Path,
    subject_id: str
):
    """
    Generate comprehensive cleaning summary plots for HTML report.
    
    Creates:
    1. PSD comparison (before/after cleaning)
    2. Component scores for DSS/ICA
    3. Correction effectiveness bar chart
    4. Evoked comparison for EOG/ECG
    """
    import matplotlib.pyplot as plt
    from mne_denoise.viz import plot_evoked_comparison, plot_score_curve
    
    figures_dir = output_dir / 'figures' / 'advanced_cleaning'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PSD Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Before
    psd_before = raw_before.compute_psd(fmin=0.5, fmax=60)
    psd_before.plot(axes=axes[0], show=False)
    axes[0].set_title('PSD Before Advanced Cleaning')
    
    # After
    psd_after = raw_after.compute_psd(fmin=0.5, fmax=60)
    psd_after.plot(axes=axes[1], show=False)
    axes[1].set_title('PSD After Advanced Cleaning')
    
    psd_fig_path = figures_dir / f'{subject_id}_psd_comparison.png'
    plt.savefig(psd_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Correction Effectiveness Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    correction_stats = provenance.get('correction_stats', {})
    methods = []
    components_removed = []
    
    for step, stats in correction_stats.items():
        if 'n_components_removed' in stats and stats.get('n_components_removed', 0) > 0:
            methods.append(f"{step}\n({stats.get('method', 'unknown')})")
            components_removed.append(stats['n_components_removed'])
    
    if methods:
        ax.bar(methods, components_removed, color='steelblue', alpha=0.7)
        ax.set_ylabel('Components Removed', fontsize=12)
        ax.set_title('Artifact Components Removed by Step', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        effectiveness_fig_path = figures_dir / f'{subject_id}_correction_effectiveness.png'
        plt.savefig(effectiveness_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Variance Removed Summary
    fig, ax = plt.subplots(figsize=(6, 4))
    
    variance_steps = []
    variance_removed = []
    
    for step, stats in correction_stats.items():
        if 'variance_removed' in stats:
            variance_steps.append(step)
            variance_removed.append(stats['variance_removed'] * 100)  # Convert to %
    
    if variance_steps:
        ax.barh(variance_steps, variance_removed, color='coral', alpha=0.7)
        ax.set_xlabel('Variance Removed (%)', fontsize=12)
        ax.set_title('Variance Reduction by Step', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        variance_fig_path = figures_dir / f'{subject_id}_variance_removed.png'
        plt.savefig(variance_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'psd_comparison': str(psd_fig_path),
        'correction_effectiveness': str(effectiveness_fig_path) if methods else None,
        'variance_removed': str(variance_fig_path) if variance_steps else None
    }

def plot_wiener_mask_summary(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    dss_model,
    config,
    output_dir: Path,
    subject_id: str,
    channel_idx: int = 0
):
    """
    Visualize Wiener Mask Denoiser results.
    
    Shows:
    1. Original signal (one channel)
    2. Denoised signal
    3. Estimated Wiener mask over time (adaptive weighting)
    4. Component reconstruction
    
    This demonstrates how the mask adapts to local variance,
    preserving bursty clinical signals (spikes, bursts).
    """
    from scipy import ndimage
    import matplotlib.pyplot as plt
    
    figures_dir = output_dir / 'figures' / 'wiener_mask'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data
    data_before = raw_before.get_data()[channel_idx]
    data_after = raw_after.get_data()[channel_idx]
    sfreq = raw_before.info['sfreq']
    time = np.arange(len(data_before)) / sfreq
    
    # Reconstruct mask for visualization
    window_samples = int(config.wiener_window_duration * sfreq)
    
    # Compute local variance
    source_sq = data_after**2
    local_mean_sq = ndimage.uniform_filter1d(source_sq, size=window_samples, mode='reflect')
    local_mean = ndimage.uniform_filter1d(data_after, size=window_samples, mode='reflect')
    local_var = np.maximum(local_mean_sq - local_mean**2, 0)
    
    # Estimate noise floor and compute mask
    noise_var = np.percentile(local_var, config.wiener_noise_percentile)
    signal_var = np.maximum(local_var - noise_var, 0)
    mask = signal_var / (signal_var + noise_var + 1e-15)
    
    # Create 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Only plot first 30 seconds for clarity
    n_samples_plot = min(int(30 * sfreq), len(time))
    t = time[:n_samples_plot]
    
    # 1. Original Signal
    axes[0].plot(t, data_before[:n_samples_plot], 'k', alpha=0.7)
    axes[0].set_title('Original Signal (Before Wiener Mask Denoising)', fontsize=12)
    axes[0].set_ylabel('Amplitude (µV)', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # 2. Denoised Signal
    axes[1].plot(t, data_after[:n_samples_plot], 'b')
    axes[1].set_title('Denoised Signal (After Wiener Mask)', fontsize=12)
    axes[1].set_ylabel('Amplitude (µV)', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    # 3. Difference (Removed Artifacts)
    difference = data_before[:n_samples_plot] - data_after[:n_samples_plot]
    axes[2].plot(t, difference, 'r', alpha=0.6)
    axes[2].set_title('Removed Components (Difference)', fontsize=12)
    axes[2].set_ylabel('Amplitude (µV)', fontsize=10)
    axes[2].grid(alpha=0.3)
    
    # 4. Adaptive Wiener Mask
    axes[3].plot(t, mask[:n_samples_plot], 'g', lw=2)
    axes[3].fill_between(t, 0, mask[:n_samples_plot], color='g', alpha=0.3)
    axes[3].set_title('Adaptive Wiener Mask (0=suppress, 1=preserve)', fontsize=12)
    axes[3].set_ylabel('Mask Value', fontsize=10)
    axes[3].set_xlabel('Time (s)', fontsize=10)
    axes[3].set_ylim([-0.05, 1.05])
    axes[3].grid(alpha=0.3)
    
    # Add annotation explaining mask behavior
    axes[3].text(
        0.02, 0.95, 
        'High mask values = bursty activity preserved\\nLow mask values = stationary noise suppressed',
        transform=axes[3].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    mask_fig_path = figures_dir / f'{subject_id}_wiener_mask_summary.png'
    plt.savefig(mask_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(mask_fig_path)


def plot_dss_component_summary(
    dss_model,
    epochs_data,
    component_idx: int,
    artifact_type: str,
    output_dir: Path,
    subject_id: str
):
    """
    Plot DSS component summary using mne-denoise visualization.
    
    Shows:
    - Score curve (component quality)
    - Spatial pattern (topography)
    - Time series
    - PSD
    """
    from mne_denoise.viz import plot_component_summary, plot_score_curve
    import matplotlib.pyplot as plt
    
    figures_dir = output_dir / 'figures' / 'dss_components'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Score curve
    fig_score = plot_score_curve(dss_model, mode='ratio', show=False)
    score_path = figures_dir / f'{subject_id}_{artifact_type}_score_curve.png'
    plt.savefig(score_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Component summary
    fig_comp = plot_component_summary(
        dss_model, 
        data=epochs_data, 
        n_components=[component_idx], 
        show=False
    )
    comp_path = figures_dir / f'{subject_id}_{artifact_type}_component_{component_idx}.png'
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'score_curve': str(score_path),
        'component_summary': str(comp_path)
    }

def plot_evoked_artifact_comparison(
    epochs_before: mne.Epochs,
    epochs_after: mne.Epochs,
    artifact_type: str,
    output_dir: Path,
    subject_id: str
):
    """
    Compare evoked responses before/after artifact correction.
    
    Useful for visualizing EOG/ECG removal effectiveness.
    """
    from mne_denoise.viz import plot_evoked_comparison
    import matplotlib.pyplot as plt
    
    figures_dir = output_dir / 'figures' / 'evoked_comparison'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_evoked_comparison(
        epochs_before, 
        epochs_after, 
        labels=('Before Cleaning', 'After Cleaning'),
        show=False
    )
    
    evoked_path = figures_dir / f'{subject_id}_{artifact_type}_evoked_comparison.png'
    plt.savefig(evoked_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(evoked_path)
```

### 6c. HTML Report Generation → `denoise.py`

```python
def generate_cleaning_html_report(
    provenance: Dict,
    plot_paths: Dict[str, str],
    output_dir: Path,
    subject_id: str
) -> Path:
    """
    Generate HTML report for advanced cleaning results.
    Matches base.py report structure for consistency.
    """
    report_dir = output_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Cleaning Report - {subject_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .success {{ color: #27ae60; }}
            .warning {{ color: #f39c12; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🧠 Advanced Artifact Cleaning Report</h1>
        <h2>Subject: {subject_id}</h2>
        
        <h2>Pipeline Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>EOG Method</td><td>{provenance.get('config', {}).eog_method or 'None'}</td></tr>
            <tr><td>ECG Method</td><td>{provenance.get('config', {}).ecg_method or 'None'}</td></tr>
            <tr><td>EMG Method</td><td>{provenance.get('config', {}).emg_method or 'None'}</td></tr>
            <tr><td>Transient Method</td><td>{provenance.get('config', {}).transient_method or 'None'}</td></tr>
        </table>
        
        <h2>Correction Statistics</h2>
        <table>
            <tr><th>Step</th><th>Method</th><th>Components Removed</th><th>Variance Removed</th></tr>
    """
    
    for step, stats in provenance.get('correction_stats', {}).items():
        method = stats.get('method', 'N/A')
        n_comp = stats.get('n_components_removed', '-')
        var_rm = f"{stats.get('variance_removed', 0)*100:.1f}%" if 'variance_removed' in stats else '-'
        html_content += f"<tr><td>{step}</td><td>{method}</td><td>{n_comp}</td><td>{var_rm}</td></tr>"
    
    html_content += """
        </table>
        
        <h2>Effectiveness Summary</h2>
    """
    
    eff = provenance.get('correction_effectiveness', {})
    html_content += f"""
        <ul>
            <li>Total Components Removed: <strong>{eff.get('total_components_removed', 0)}</strong></li>
            <li>Total Variance Removed: <strong>{eff.get('total_variance_removed', 0)*100:.1f}%</strong></li>
            <li>Base.py Bad Segments: <strong>{eff.get('n_base_bad_segments', 0)}</strong></li>
        </ul>
        
        <h2>Visualizations</h2>
    """
    
    # Add plots
    for plot_name, plot_path in plot_paths.items():
        if plot_path:
            html_content += f"""
            <h3>{plot_name.replace('_', ' ').title()}</h3>
            <img src="{plot_path}" alt="{plot_name}">
            """
    
    html_content += """
        <h2>Processing Times</h2>
        <table>
            <tr><th>Step</th><th>Duration (s)</th></tr>
    """
    
    for step in provenance.get('steps_completed', []):
        duration = provenance.get('timings', {}).get(step, 0)
        html_content += f"<tr><td>{step}</td><td>{duration:.2f}</td></tr>"
    
    html_content += """
        </table>
        
        <footer>
            <hr>
            <p><em>Generated by eeg_adhd_epilepsy.preproc.clean</em></p>
        </footer>
    </body>
    </html>
    """
    
    report_path = report_dir / f'{subject_id}_cleaning_report.html'
    report_path.write_text(html_content)
    LOGGER.info(f"HTML report saved: {report_path}")
    
    return report_path
```

### 6d. Condition-Aware Processing → `denoise.py`

```python
def run_condition_aware_cleaning(
    raw: mne.io.BaseRaw,
    config: 'ArtifactCorrectionConfig',
    subject_id: str,
    output_dir: Path,
    conditions: Optional[List[str]] = None
) -> Tuple[mne.io.BaseRaw, Dict]:
    """
    Process each experimental condition separately.
    Matches base.py condition-aware processing pattern.
    
    Args:
        raw: Raw data with condition annotations (from base.py)
        config: Cleaning configuration
        subject_id: Subject identifier
        output_dir: Output directory
        conditions: List of conditions to process (default: auto-detect from annotations)
    
    Returns:
        Concatenated cleaned raw and combined provenance
    """
    # Auto-detect conditions from annotations if not provided
    if conditions is None:
        conditions = _detect_conditions(raw)
    
    if not conditions:
        LOGGER.info("No condition annotations found, processing as single block")
        return run_advanced_artifact_removal(raw, config, subject_id, output_dir)
    
    LOGGER.info(f"Processing {len(conditions)} conditions: {conditions}")
    
    combined_provenance = {
        'subject_id': subject_id,
        'conditions_processed': conditions,
        'condition_stats': {}
    }
    
    cleaned_raws = []
    
    for condition in conditions:
        LOGGER.info(f"\n{'='*40}")
        LOGGER.info(f"Processing condition: {condition}")
        LOGGER.info(f"{'='*40}")
        
        # Extract condition segment
        raw_cond = _extract_condition_segment(raw, condition)
        
        if raw_cond is None:
            LOGGER.warning(f"Could not extract condition: {condition}")
            continue
        
        # Run cleaning on this condition
        raw_cond_clean, prov = run_advanced_artifact_removal(
            raw_cond,
            config,
            f"{subject_id}_{condition}",
            output_dir / condition
        )
        
        cleaned_raws.append(raw_cond_clean)
        combined_provenance['condition_stats'][condition] = prov
    
    # Concatenate cleaned conditions
    if len(cleaned_raws) > 1:
        raw_final = mne.concatenate_raws(cleaned_raws)
    elif len(cleaned_raws) == 1:
        raw_final = cleaned_raws[0]
    else:
        raise ValueError("No conditions could be processed")
    
    return raw_final, combined_provenance


def _detect_conditions(raw: mne.io.BaseRaw) -> List[str]:
    """
    Detect condition annotations from raw.
    REUSES base.py's _collect_block_windows for consistency.
    """
    # Use base.py's block window collection (already imported)
    block_windows = _collect_block_windows(raw)
    
    if not block_windows:
        # Fallback to simple annotation search
        conditions = set()
        condition_prefixes = ['baseline', 'hyperventilation', 'photic', 'rest', 'task']
        for annot in raw.annotations:
            desc = annot['description'].lower()
            for prefix in condition_prefixes:
                if prefix in desc:
                    conditions.add(prefix)
        return sorted(conditions)
    
    # Extract unique condition names from block windows
    return sorted(set(bw.condition for bw in block_windows))


def _extract_condition_segment(
    raw: mne.io.BaseRaw, 
    condition: str
) -> Optional[mne.io.BaseRaw]:
    """
    Extract raw segment for a specific condition.
    REUSES base.py's _collect_block_windows for consistency.
    """
    block_windows = _collect_block_windows(raw)
    
    # Find matching block
    for bw in block_windows:
        if condition.lower() in bw.condition.lower():
            try:
                return raw.copy().crop(tmin=bw.start, tmax=bw.end)
            except Exception as e:
                LOGGER.warning(f"Could not crop for {condition}: {e}")
                return None
    
    return None
```

### 7. Configuration → `denoise.py`

**Step-wise configuration allows method selection per artifact type**

```python
@dataclass
class ArtifactCorrectionConfig:
    # ===== STAGE 1: Specific Artifacts =====
    
    # EOG Removal (Part 5 vs Part 2)
    eog_method: Optional[str] = "dss"  # 'dss', 'ica', None
    
    # ECG Removal (Part 5 vs Part 2)
    ecg_method: Optional[str] = "dss"  # 'dss', 'ica', None
    
    # EMG Removal (Part 2.1 options + Part 5)
    emg_method: Optional[str] = "mwf"  # 'mwf', 'wica', 'ica', 'dss', None
    
    # Shared ICA Parameters (when ICA is used)
    ica_n_components: int = 20
    exclude_probability: float = 0.8
    
    # DSS Parameters (when DSS is used)
    dss_n_components: int = 10
    dss_emg_n_remove: int = 2  # For DSS-EMG: how many components to remove
    
    # MWF Parameters (Part 2.1, RELAX)
    mwf_n_components: int = 30
    
    # wICA Parameters (Part 2.1, RELAX-Jr)
    wavelet_type: str = 'db4'
    wavelet_level: int = 5
    
    # ===== STAGE 2: Aggressive Cleaning =====
    
    # Transient Removal (Part 2.2 ASR, Part 5 DSS, or Wiener Mask)
    transient_method: Optional[str] = "wiener_mask"  # 'asr', 'dss', 'wiener_mask', None
    
    # ASR Parameters (Part 2.2, Part 4 Step 3)
    asr_cutoff: float = 20.0
    asr_calibration_window: float = 10.0
    
    # Wiener Mask Parameters (Adaptive denoising for bursty signals)
    wiener_window_duration: float = 0.2  # Window in seconds (200ms for beta bursts)
    wiener_noise_percentile: int = 25  # Percentile for noise floor estimation
    wiener_n_components: int = 10  # Number of DSS components
    wiener_max_iter: int = 20  # Max iterations for Iterative DSS
    
    # Final AutoReject (Part 4 Step 4)
    final_autoreject: bool = True
    epoch_duration: float = 1.0
```

### 8. Integration & Execution Plan → `denoise.py`

**Example configurations for different use cases:**

```python
# Example 1: Recommended for Epilepsy/ADHD (Bursty EEG)
config_wiener = ArtifactCorrectionConfig(
    eog_method='dss',
    ecg_method='dss',
    emg_method='dss',
    transient_method='wiener_mask',  # BEST for bursty clinical signals
    wiener_window_duration=0.15,  # 150ms for epileptic spikes
    final_autoreject=True
)

# Example 2: DSS-based cleaning (Part 5)
config_dss = ArtifactCorrectionConfig(
    eog_method='dss',
    ecg_method='dss',
    emg_method='dss',
    transient_method='dss',  # Experimental DSS for transients
    final_autoreject=True
)

# Example 3: RELAX-inspired (Part 2.1)
config_relax = ArtifactCorrectionConfig(
    eog_method='ica',
    ecg_method='ica',
    emg_method='mwf',  # Multi-Channel Wiener Filter
    transient_method=None,  # RELAX doesn't use ASR/DSS for transients
    final_autoreject=True
)

# Example 4: Hybrid (DSS for EOG/ECG, MWF for EMG, Wiener for transients)
config_hybrid = ArtifactCorrectionConfig(
    eog_method='dss',
    ecg_method='dss',
    emg_method='mwf',
    transient_method='wiener_mask',  # Adaptive for bursty artifacts
    final_autoreject=True
)

# Usage (following base.py pattern)
from eeg_adhd_epilepsy.preproc import base, correct, denoise
from eeg_adhd_epilepsy.preproc.utils import benchmark_step

# 1. Run Base Pipeline
raw, base_prov = base.run_base_pipeline(raw_original, config, subject_id)

# 2. Run Stage 1: Source Correction (EOG/ECG/EMG)
raw_corrected, corr_prov = correct.run_source_correction(
    raw, 
    config
)

# 3. Run Stage 2: Residual Denoising (Transients/AutoReject)
raw_final, denoise_prov = denoise.run_residual_denoising(
    raw_corrected, 
    config, 
    subject_id,
    output_dir=output_dir,
    stage1_provenance=corr_prov  # Pass Stage 1 stats for final report
)

# Save result
raw_final.save(output_dir / f'{subject_id}_clean_eeg.fif', overwrite=True)

# Provenance is automatically saved/reported by run_residual_denoising()
```



**Implementation Steps:**
1.  **Module Creation**: Create `preproc/clean.py` with all step functions.
2.  **Import Pattern**: Follow `base.py` imports (benchmark_step, LOGGER, etc.).
3.  **Dependencies**: Add `meegkit`, `PyWavelets`, `mne-icalabel`, `mne-denoise`.
4.  **Testing**: Test each step independently, verify base.py annotation usage.
5.  **Comparison**: Run different configurations, track correction rates.


### 9. Pipeline Comparison Framework → `compare.py`

**Compare multiple cleaning strategies on a data subset before full processing**

```python
def compare_pipelines(
    subject_ids: List[str],
    configs: Dict[str, ArtifactCorrectionConfig],
    data_dir: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compare multiple pipeline configurations on a subset of subjects.
    
    Args:
        subject_ids: List of subject IDs to test (e.g., ['sub-001', 'sub-002'])
        configs: Dict of {config_name: config_object} to compare
        data_dir: Path to preprocessed data (from base.py)
        output_dir: Where to save comparison results
        
    Returns:
        DataFrame with comparison metrics for each config
    """
    import time
    import pandas as pd
    
    results = []
    
    for subject in subject_ids:
        LOGGER.info(f"\\n{'='*60}")
        LOGGER.info(f"Processing {subject}")
        LOGGER.info(f"{'='*60}")
        
        # Load base.py output
        raw_path = data_dir / subject / 'eeg' / f'{subject}_eeg.fif'
        raw_original = mne.io.read_raw_fif(raw_path, preload=True)
        
        for config_name, config in configs.items():
            LOGGER.info(f"\\n--- Testing: {config_name} ---")
            
            # Copy raw for this configuration
            raw = raw_original.copy()
            
            # Time the execution
            start_time = time.time()
            
            # Run pipeline
            raw_clean, provenance = run_advanced_artifact_removal(
                raw,
                config,
                subject,
                output_dir=output_dir / config_name
            )
            
            duration = time.time() - start_time
            
            # Extract metrics
            correction_stats = provenance.get('correction_effectiveness', {})
            
            result = {
                'subject': subject,
                'config': config_name,
                'duration_sec': duration,
                'total_components_removed': correction_stats.get('total_components_removed', 0),
                'total_variance_removed': correction_stats.get('total_variance_removed', 0),
                'n_base_bad_segments': correction_stats.get('n_base_bad_segments', 0),
                'eog_method': config.eog_method,
                'ecg_method': config.ecg_method,
                'emg_method': config.emg_method,
                'transient_method': config.transient_method,
            }
            
            # Add step-specific metrics
            for step, stats in provenance.get('correction_stats', {}).items():
                if 'n_components_removed' in stats:
                    result[f'{step}_components'] = stats['n_components_removed']
                if 'variance_removed' in stats:
                    result[f'{step}_variance'] = stats['variance_removed']
            
            results.append(result)
            
            LOGGER.info(f"  Duration: {duration:.1f}s")
            LOGGER.info(f"  Components removed: {result['total_components_removed']}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(comparison_dir / 'pipeline_comparison.csv', index=False)
    
    return df
```

**Usage Example: Test DSS vs ICA on 3 Subjects**

```python
# Define configurations to compare
configs_to_compare = {
    'dss_wiener': ArtifactCorrectionConfig(
        eog_method='dss', ecg_method='dss', emg_method='dss', 
        transient_method='wiener_mask'
    ),
    'ica_wiener': ArtifactCorrectionConfig(
        eog_method='ica', ecg_method='ica', emg_method='ica',
        transient_method='wiener_mask'
    ),
    'hybrid': ArtifactCorrectionConfig(
        eog_method='dss', ecg_method='dss', emg_method='mwf',
        transient_method='wiener_mask'
    )
}

# Test on subset (mix of epilepsy + ADHD subjects)
test_subjects = ['sub-001', 'sub-002', 'sub-003']

# Run comparison
df_results = compare_pipelines(
    subject_ids=test_subjects,
    configs=configs_to_compare,
    data_dir=Path('derivatives/preproc'),
    output_dir=Path('output/comparison_test')
)

# View results
print("\nSpeed Comparison:")
print(df_results.groupby('config')['duration_sec'].mean())
print("\nEffectiveness Comparison:")
print(df_results.groupby('config')['total_components_removed'].mean())

# Choose best config based on results
# Example output:
# dss_wiener: 45.2s, 12.3 components
# ica_wiener: 62.8s, 11.7 components
# hybrid: 48.5s, 13.1 components

# → Choose 'dss_wiener' for full analysis (fastest + effective)
```

**Output Files:**
- `output/comparison_test/comparison/pipeline_comparison.csv` - Full metrics
- Individual pipeline outputs in `output/comparison_test/{config_name}/`

#### DSS vs ICA Correlation Validation

**Validates DSS ≈ ICA using component correlation (from BSS Appendix)**

```python
def validate_dss_ica_correlation(
    raw: mne.io.BaseRaw,
    output_dir: Path,
    subject_id: str,
    n_components: int = 15
) -> Dict:
    """
    Prove DSS extracts same components as ICA but faster.
    
    Based on Appendix: Blind Source Separation (ICA Equivalence).
    Uses component correlation to validate equivalence.
    
    Returns metrics showing DSS speedup and correlation.
    """
    import time
    from sklearn.decomposition import FastICA
    from mne_denoise.dss import IterativeDSS, TanhMaskDenoiser, beta_tanh
    from scipy.optimize import linear_sum_assignment
    
    LOGGER.info(f"Validating DSS vs ICA on {subject_id}...")
    
    data = raw.get_data()
    
    # 1. Fit ICA (sklearn FastICA)
    t_start = time.time()
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    ica_components = ica.fit_transform(data.T).T
    time_ica = time.time() - t_start
    
    # 2. Fit DSS with Tanh (equivalent to Robust ICA)
    t_start = time.time()
    dss = IterativeDSS(
        denoiser=TanhMaskDenoiser(),
        method='deflation',
        n_components=n_components,
        beta=beta_tanh,  # Newton step (from appendix)
        random_state=42,
        verbose=False
    )
    dss.fit(data.T)
    dss_components = dss.transform(data.T).T
    time_dss = time.time() - t_start
    
    # 3. Compute correlation matrix between all component pairs
    corr_matrix = np.abs([[np.corrcoef(ica_components[i], dss_components[j])[0, 1] 
                           for j in range(n_components)] 
                          for i in range(n_components)])
    
    # 4. Find best matches (Hungarian assignment)
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    matched_corrs = corr_matrix[row_ind, col_ind]
    avg_correlation = matched_corrs.mean()
    
    # 5. Identify artifact components (correlate with EOG/ECG)
    artifact_correlations = {}
    if 'EOG' in [ch['kind'] for ch in raw.info['chs']]:
        eog_data = raw.copy().pick_types(eog=True).get_data()[0]
        dss_eog_corr = [np.abs(np.corrcoef(c, eog_data)[0, 1]) for c in dss_components]
        ica_eog_corr = [np.abs(np.corrcoef(c, eog_data)[0, 1]) for c in ica_components]
        artifact_correlations['eog'] = {
            'dss_best': (np.argmax(dss_eog_corr), np.max(dss_eog_corr)),
            'ica_best': (np.argmax(ica_eog_corr), np.max(ica_eog_corr))
        }
    
    # 6. Results
    speedup = time_ica / time_dss
    n_high_corr = (matched_corrs > 0.9).sum()
    
    LOGGER.info(f"  ICA: {time_ica:.2f}s | DSS: {time_dss:.2f}s | Speedup: {speedup:.2f}x ⚡")
    LOGGER.info(f"  Avg correlation: {avg_correlation:.3f} | High corr: {n_high_corr}/{n_components}")
    
    if avg_correlation > 0.85:
        LOGGER.info(f"  ✅ DSS ≈ ICA (equivalent) → USE DSS for {speedup:.1f}x speedup!")
    
    return {
        'subject': subject_id,
        'speedup': speedup,
        'avg_correlation': avg_correlation,
        'n_high_correlation': n_high_corr,
        'artifact_correlations': artifact_correlations
    }

# Example: Validate on test subjects
for subject in test_subjects:
    raw = mne.io.read_raw_fif(f'derivatives/preproc/{subject}/eeg/{subject}_eeg.fif')
    result = validate_dss_ica_correlation(raw, Path('output/validation'), subject)
    print(f"{subject}: {result['speedup']:.1f}x faster, {result['avg_correlation']:.0%} corr")
```

**Key Points:**
- DSS with `TanhMaskDenoiser` = Robust ICA (mathematically equivalent)
- Uses absolute correlation (components can be sign-flipped)
- Hungarian assignment finds optimal component matching
- **Expected**: 2-3x speedup with >90% correlation

---

**END OF PART 6 IMPLEMENTATION PLAN**
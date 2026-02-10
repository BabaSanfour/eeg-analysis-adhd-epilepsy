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

## Part 6: Technical Implementation Plan (`preproc/clean.py`)

This entire phase runs **after** the non-destructive baseline (`base.py`). The input is the `raw` object returned by `base.py` (which has RANSAC-corrected bad channels and initial AutoReject annotations from Part 1).

**Philosophy**: `base.py` provides a conservative foundation. This phase implements advanced cleaning in **2 sequential stages**, drawing inspiration from Parts 2-5. Each step allows choosing between multiple methods.

**Key Integration Points**:
1.  Uses `base.py` annotations to exclude bad segments during ICA fitting
2.  Follows `base.py` structure with `benchmark_step()` and provenance tracking
3.  Tracks correction effectiveness by comparing pre/post artifact annotations

### 1. Architecture: The `clean.py` Module

We will create `eeg_adhd_epilepsy/preproc/clean.py` as a standalone module. It does **not** modify `base.py`.

**Goal**: Implement a **step-wise** pipeline (not pipeline-wise) with method selection per step, organized into 2 major stages:
1.  **Stage 1**: Specific Physiological Artifacts (ECG, EOG, EMG)
2.  **Stage 2**: Aggressive/General Cleaning (transients, final repair)

### 2. Core Orchestrator

Following `base.py` structure with provenance tracking:

```python
def run_advanced_artifact_removal(
    raw: mne.io.BaseRaw, 
    config: ArtifactCorrectionConfig, 
    subject_id: str,
    output_dir: Optional[Path] = None,
) -> Tuple[mne.io.BaseRaw, Dict]:
    """
    Step-wise artifact removal pipeline.
    Operates on the output of base.py.
    
    Draws inspiration from:
    - Part 2: RELAX (MWF, wICA, ICLabel) & ASR strategies
    - Part 4: Sequential roadmap
    - Part 5: DSS as alternative to ICA
    """
    LOGGER.info(f"Starting advanced cleaning for {subject_id}")
    
    # Initialize provenance (following base.py structure)
    provenance: Dict[str, Any] = {
        "subject_id": subject_id,
        "config": config,
        "steps_completed": [],
        "correction_stats": {},
        "base_annotations_used": True,  # Using base.py annotations
    }
    
    # Extract base.py annotations for ICA exclusion
    base_bad_segments = _extract_bad_segments(raw)
    provenance["n_base_bad_segments"] = len(base_bad_segments)
    
    # ===== STAGE 1: Specific Physiological Artifacts =====
    # Remove well-defined, stereotyped artifacts
    
    # Step 1.1: EOG (Eye) Removal
    if config.eog_method:
        with benchmark_step("eog_removal", provenance):
            if config.eog_method == "dss":
                # Part 5: DSS with blink bias
                raw, s = _remove_eog_dss(raw, config, base_bad_segments)
            elif config.eog_method == "ica":
                # Part 2: ICA + ICLabel (exclude base.py bad segments)
                raw, s = _remove_eog_ica(raw, config, base_bad_segments)
            provenance["correction_stats"]["eog"] = s
        provenance["steps_completed"].append("eog_removal")
    
    # Step 1.2: ECG (Heart) Removal
    if config.ecg_method:
        with benchmark_step("ecg_removal", provenance):
            if config.ecg_method == "dss":
                # Part 5: DSS with QRS bias
                raw, s = _remove_ecg_dss(raw, config, base_bad_segments)
            elif config.ecg_method == "ica":
                # Part 2: ICA + ICLabel (exclude base.py bad segments)
                raw, s = _remove_ecg_ica(raw, config, base_bad_segments)
            provenance["correction_stats"]["ecg"] = s
        provenance["steps_completed"].append("ecg_removal")
    
    # Step 1.3: EMG (Muscle) Removal
    if config.emg_method:
        with benchmark_step("emg_removal", provenance):
            if config.emg_method == "mwf":
                # Part 2: Multi-Channel Wiener Filter (from RELAX)
                raw, s = _remove_emg_mwf(raw, config, base_bad_segments)
            elif config.emg_method == "wica":
                # Part 2: Wavelet-ICA (from RELAX-Jr)
                raw, s = _remove_emg_wica(raw, config, base_bad_segments)
            elif config.emg_method == "ica":
                # Part 2: Standard ICA + ICLabel
                raw, s = _remove_emg_ica(raw, config, base_bad_segments)
            elif config.emg_method == "dss":
                # Part 5: DSS with high-frequency bias
                raw, s = _remove_emg_dss(raw, config, base_bad_segments)
            provenance["correction_stats"]["emg"] = s
        provenance["steps_completed"].append("emg_removal")
    
    # ===== STAGE 2: Aggressive/General Cleaning =====
    # Handle non-stationary and residual artifacts
    
    # Step 2.1: Transient/Movement Removal
    if config.transient_method:
        with benchmark_step("transient_removal", provenance):
            if config.transient_method == "asr":
                # Part 2.2 & Part 4 Step 3: ASR for bursts
                raw, s = _remove_transients_asr(raw, config)
            elif config.transient_method == "dss":
                # Part 5: DSS for non-stationary artifacts
                raw, s = _remove_transients_dss(raw, config)
            elif config.transient_method == "wiener_mask":
                # Adaptive Wiener Masking for bursty signals (BEST for epilepsy/ADHD)
                raw, s = _remove_transients_wiener_mask(raw, config)
            provenance["correction_stats"]["transients"] = s
        provenance["steps_completed"].append("transient_removal")
    
    # Step 2.2: Final Refinement (AutoReject)
    if config.final_autoreject:
        with benchmark_step("autoreject_refinement", provenance):
            # Part 4 Step 4: Second-pass AutoReject
            # Compare against base.py annotations
            raw, s = _refine_autoreject(raw, config, base_bad_segments)
            provenance["correction_stats"]["autoreject_stage2"] = s
        provenance["steps_completed"].append("autoreject_refinement")
    
    # Calculate overall correction effectiveness
    provenance["correction_effectiveness"] = _calculate_correction_effectiveness(
        base_bad_segments, 
        provenance["correction_stats"]
    )
         
    return raw, provenance
```

### 3. Helper: Extract Bad Segments from base.py

```python
def _extract_bad_segments(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """
    Extract bad segment timestamps from base.py annotations.
    Used to exclude these segments when fitting ICA.
    
    Returns:
        List of (onset, duration) tuples for bad segments
    """
    bad_segments = []
    for annot in raw.annotations:
        if annot['description'].startswith('BAD_'):
            bad_segments.append((annot['onset'], annot['duration']))
    
    LOGGER.info(f"Extracted {len(bad_segments)} bad segments from base.py")
    return bad_segments
```

### 4. Stage 1 Implementation: Specific Artifacts (ECG, EOG, EMG)

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
    raw_eeg._data = cleaned_data
    
    # 4. Update original raw object
    raw.pick_types(eeg=True).load_data()
    raw._data = raw_eeg._data
    
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
    raw_eeg._data = cleaned_data
    
    # 4. Update original raw object
    raw.pick_types(eeg=True).load_data()
    raw._data = raw_eeg._data
    
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
    raw_eeg._data = cleaned_data
    
    # 5. Update original raw object
    raw.pick_types(eeg=True).load_data()
    raw._data = raw_eeg._data
    
    return raw, {
        'method': 'dss',
        'n_components_removed': n_remove,
        'bias_type': 'high_frequency_power'
    }
```

### 5. Stage 2 Implementation: Aggressive Cleaning

#### Step 2.1: Transient/Movement Removal

##### Option A: ASR (Part 2.2 & Part 4 Step 3)
```python
def _remove_transients_asr(raw, config):
    """
    Remove transient artifacts (pops, movement) using ASR.
    Part 2.2: Calibrate on clean reference, remove high-variance bursts.
    """
    from meegkit import asr
    
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
    
    # 5. Update raw object
    raw_eeg._data = data_clean
    raw.pick_types(eeg=True).load_data()
    raw._data = raw_eeg._data
    
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
    from autoreject import AutoReject
    
    # 1. Epoch
    events = mne.make_fixed_length_events(raw, duration=config.epoch_duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=config.epoch_duration, baseline=None)
    
    # 2. AutoReject
    ar = AutoReject(n_interpolate=[1, 4, 8], random_state=42, n_jobs=-1, verbose=False)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    # 3. Update annotations
    bad_segments_stage2 = []
    bad_segments = reject_log.bad_epochs
    for i, is_bad in enumerate(bad_segments):
        if is_bad:
            onset = events[i, 0] / raw.info['sfreq']
            duration = config.epoch_duration
            raw.annotations.append(onset, duration, 'BAD_autoreject_stage2')
            bad_segments_stage2.append((onset, duration))
    
    # 4. Compare to base.py: How many base.py bad segments are now clean?
    n_corrected = _count_corrected_segments(base_bad_segments, bad_segments_stage2)
    
    return raw, {
        'n_epochs_rejected': bad_segments.sum(),
        'percent_rejected': 100 * bad_segments.sum() / len(epochs),
        'n_base_segments_corrected': n_corrected,
        'correction_rate': n_corrected / len(base_bad_segments) if base_bad_segments else 0
    }
```

### 6. Helper: Calculate Correction Effectiveness

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

### 6b. Visualization & Plotting for Reports

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


### 7. Configuration Structure

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

### 8. Integration & Execution Plan

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
from eeg_adhd_epilepsy.preproc.utils import benchmark_step
from pathlib import Path

# Load data from base.py
raw_before = mne.io.read_raw_fif(f'derivatives/preproc/{subject}/eeg/{subject}_eeg.fif')
raw = raw_before.copy()  # Keep original for comparison

# Run advanced cleaning (RECOMMENDED: config_wiener for epilepsy/ADHD)
raw_clean, provenance = run_advanced_artifact_removal(
    raw, 
    config_wiener,  # Use Wiener Mask for bursty clinical signals
    subject,
    output_dir=Path('./output')
)

# Save with provenance (like base.py)
output_dir = Path('./output')
derivatives_dir = output_dir / 'derivatives' / 'cleaned' / subject / 'eeg'
derivatives_dir.mkdir(parents=True, exist_ok=True)

raw_clean.save(derivatives_dir / f'{subject}_clean_eeg.fif')
with open(derivatives_dir / f'{subject}_provenance.json', 'w') as f:
    json.dump(provenance, f, indent=2)

# Generate visualization for reports
plot_paths = plot_cleaning_summary(
    raw_before=raw_before,
    raw_after=raw_clean,
    provenance=provenance,
    output_dir=output_dir,
    subject_id=subject
)

# Save plot paths to provenance
provenance['figures'] = plot_paths
with open(derivatives_dir / f'{subject}_provenance.json', 'w') as f:
    json.dump(provenance, f, indent=2)

# Log correction effectiveness
LOGGER.info(f"Correction effectiveness: {provenance['correction_effectiveness']}")
LOGGER.info(f"Figures saved: {plot_paths}")
```



**Implementation Steps:**
1.  **Module Creation**: Create `preproc/clean.py` with all step functions.
2.  **Import Pattern**: Follow `base.py` imports (benchmark_step, LOGGER, etc.).
3.  **Dependencies**: Add `meegkit`, `PyWavelets`, `mne-icalabel`, `mne-denoise`.
4.  **Testing**: Test each step independently, verify base.py annotation usage.
5.  **Comparison**: Run different configurations, track correction rates.


### 9. Pipeline Comparison Framework

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

---


## Appendix: DSS Implementation Examples (mne-denoise)

The following Python code demonstrates how to use `mne-denoise` for EOG (Blink) and ECG (Heartbeat) artifact correction. This serves as a reference for integrating DSS into the `DSSStrategy` and `HybridDSSStrategy`.

### EOG and ECG Correction with DSS

```python
"""
Artifact Correction with DSS.
=============================
DSS is a powerful tool for removing artifacts (ECG, EOG) from data.
The core idea is: **Artifacts are repetitive**.
If we can define when artifacts happen (e.g., using EOG/ECG channels),
we can use **Trial Average Bias** (or Cycle Average Bias) to find the artifact source and remove it.
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

# mne-denoise imports
from mne_denoise.dss import DSS, AverageBias, CycleAverageBias
from mne_denoise.viz import (
    plot_component_summary,
    plot_component_time_series,
    plot_evoked_comparison,
    plot_psd_comparison,
    plot_score_curve,
    plot_spatial_patterns,
)

def run_eog_correction(raw):
    """
    Part 1: EOG (Blink) Correction
    Goal: Remove eye blinks using DSS and Trial Average Bias.
    """
    print("\n--- Part 1: EOG (Blink) Correction ---")

    # 1. Create EOG Epochs
    # We epoch around blink events found via the EOG channel.
    eog_epochs = create_eog_epochs(
        raw, ch_name="EOG 061", baseline=(-0.5, -0.2), tmin=-0.5, tmax=0.5, verbose=False
    )
    # IMPORTANT: DSS should be fitted on the data channels (MEG/EEG) we want to clean.
    # Exclude the EOG channel itself from the model.
    eog_epochs.pick_types(meg="grad", eeg=True, eog=False, ecg=False)
    print(f"Found {len(eog_epochs)} blink events.")

    # 2. Fit DSS with Trial Average Bias
    # AverageBias(axis='epochs') works on pre-epoched data.
    dss_eog = DSS(n_components=10, bias=AverageBias(axis="epochs"), return_type="sources")
    dss_eog.fit(eog_epochs)

    # 3. Visualize & Diagnosis
    # Score Curve: Comp 0 should have a high score.
    plot_score_curve(dss_eog, mode="ratio", show=False)
    # Component Summary: Shows Time Series, Topography, and PSD
    plot_component_summary(dss_eog, data=eog_epochs, n_components=[0, 1], show=False)
    plt.show(block=False)

    # 4. Remove Blink Component
    print("Removing blink component (Comp 0)...")
    raw_meg = raw.copy().pick_types(meg="grad", eeg=True, eog=False, ecg=False)
    sources = dss_eog.transform(raw_meg) # Project continuous data to DSS space
    sources[0, :] = 0  # Zero out the blink component
    
    # Reconstruction
    cleaned_data = dss_eog.inverse_transform(sources)
    raw_clean = mne.io.RawArray(cleaned_data, raw_meg.info)
    
    # 5. Verification (Evoked Comparison)
    eog_epochs_clean = mne.Epochs(raw_clean, eog_epochs.events, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.2))
    plot_evoked_comparison(eog_epochs, eog_epochs_clean, labels=("Original", "Cleaned"))
    
    return raw_clean

def run_ecg_correction(raw):
    """
    Part 2: ECG (Heartbeat) Correction
    Goal: Remove cardiac field artifact using DSS.
    """
    print("\n--- Part 2: ECG (Heartbeat) Correction ---")

    # 1. Create ECG Epochs
    ecg_epochs = create_ecg_epochs(
        raw, ch_name=None, tmin=-0.1, tmax=0.1, baseline=(None, 0), verbose=False
    )
    ecg_epochs.pick_types(meg="grad", eeg=True, eog=False, ecg=False)

    # 2. Fit DSS
    dss_ecg = DSS(n_components=8, bias=AverageBias(axis="epochs"))
    dss_ecg.fit(ecg_epochs)

    # 3. Visualize
    plot_component_summary(dss_ecg, data=ecg_epochs, n_components=[0], show=False)
    
    # 4. Remove Cardiac Component
    print("Removing cardiac component (Comp 0)...")
    # Apply to continuous data (assuming we have raw_meg from somewhere or just use raw pick types)
    raw_meg = raw.copy().pick_types(meg="grad", eeg=True, eog=False, ecg=False)
    sources_ecg = dss_ecg.transform(raw_meg)
    sources_ecg[0, :] = 0 
    
    raw_clean_ecg = mne.io.RawArray(dss_ecg.inverse_transform(sources_ecg), raw_meg.info)
    
    return raw_clean_ecg
```

### Periodic Signals (SSVEP & Quasi-Periodic)

```python
"""
=====================================================
Example 5: Periodic Signals (SSVEP & Quasi-Periodic).
=====================================================

This example comprehensively demonstrates periodic signal extraction using DSS.

**Periodic Module Functions:**
- `PeakFilterBias`: Single frequency (narrow bandpass)
- `CombFilterBias`: Fundamental + harmonics (SSVEP)
- `QuasiPeriodicDenoiser`: Template-based (ECG, respiration)

**DSS Types:**
- `DSS`: Linear spatial filtering (bias functions)
- `IterativeDSS`: Nonlinear denoising (iterative refinement)

**Structure**:
- Part 0: Single Frequency (PeakFilterBias + DSS)
- Part 1: SSVEP Harmonics (CombFilterBias + DSS)
- Part 2: Quasi-Periodic Synthetic (QuasiPeriodicDenoiser + IterativeDSS)
- Part 3: Real ECG Artifact (QuasiPeriodicDenoiser, single-channel)
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import detrend

from mne_denoise.dss import DSS, IterativeDSS
from mne_denoise.dss.denoisers import (
    CombFilterBias,
    PeakFilterBias,
    QuasiPeriodicDenoiser,
)
from mne_denoise.dss.variants import ssvep_dss
from mne_denoise.viz import (
    plot_component_summary,
    plot_psd_comparison,
    plot_spectral_psd_comparison,
    plot_time_course_comparison,
)

# Part 0: Single Frequency (PeakFilterBias)
# ==========================================
# PeakFilterBias applies a narrow bandpass filter at a single frequency.
# Simpler than CombFilterBias, useful for extracting single oscillations.

print("\n--- Part 0: Single Frequency Extraction (PeakFilterBias) ---")

rng = np.random.default_rng(42)
sfreq = 250
n_seconds = 10
n_times = n_seconds * sfreq
n_channels = 16

# Simulate alpha rhythm (10 Hz) embedded in noise
alpha_freq = 10.0
times = np.arange(n_times) / sfreq

# Pink noise background
noise = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
noise = detrend(noise, axis=1)

# Alpha source
alpha_source = np.sin(2 * np.pi * alpha_freq * times)

# Mix into occipital channels
mixing = rng.standard_normal(n_channels) * 0.1
mixing[0:2] = [2.0, 1.5]  # Strong in channels 0, 1
alpha_component = np.outer(mixing, alpha_source)

data_alpha = noise + alpha_component

# Create MNE Raw with montage
ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
info = mne.create_info(ch_names, sfreq, "eeg")
montage = mne.channels.make_standard_montage("standard_1020")
# info.set_montage(montage) # Skip montage for dummy channels to avoid errors
raw_alpha = mne.io.RawArray(data_alpha, info)

print(f"Simulated {n_seconds}s, {n_channels} channels, {sfreq} Hz")

# Apply DSS with PeakFilterBias
peak_bias = PeakFilterBias(freq=alpha_freq, sfreq=sfreq, q_factor=30)
dss_peak = DSS(n_components=5, bias=peak_bias)
dss_peak.fit(raw_alpha)

sources_peak = dss_peak.transform(raw_alpha)

# Part 1: SSVEP (CombFilterBias)
# ================================
# CombFilterBias filters fundamental + harmonics, ideal for SSVEP.

print("\n--- Part 1: SSVEP with Harmonics (CombFilterBias) ---")

# Simulate 12 Hz SSVEP with harmonics
f_stim = 12.0
ssvep_source = (
    1.0 * np.sin(2 * np.pi * f_stim * times)  # 12 Hz
    + 0.5 * np.sin(2 * np.pi * 2 * f_stim * times)  # 24 Hz
    + 0.2 * np.sin(2 * np.pi * 3 * f_stim * times)  # 36 Hz
)

noise_ssvep = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
noise_ssvep = detrend(noise_ssvep, axis=1)

ssvep_component = np.outer(mixing, ssvep_source)
data_ssvep = noise_ssvep + ssvep_component

raw_ssvep = mne.io.RawArray(data_ssvep, info)

# Method 1: Manual DSS + CombFilterBias
comb_bias = CombFilterBias(
    fundamental_freq=f_stim, sfreq=sfreq, n_harmonics=3, q_factor=30
)
dss_comb = DSS(n_components=5, bias=comb_bias)
dss_comb.fit(raw_ssvep)
sources_comb = dss_comb.transform(raw_ssvep)

# Method 2: Convenience Wrapper (ssvep_dss)
dss_wrapper = ssvep_dss(sfreq=sfreq, stim_freq=f_stim, n_harmonics=3, n_components=5)
dss_wrapper.fit(raw_ssvep)

# Part 2: Quasi-Periodic (Iterative DSS + Multi-Channel)
# =======================================================
# QuasiPeriodicDenoiser works with IterativeDSS for multi-channel spatial denoising.

print("\n--- Part 2: Quasi-Periodic Denoising (IterativeDSS) ---")

# Simulate multi-channel heartbeat-like signal
n_beats = 10
beat_interval = 0.8  # seconds (~75 BPM)
beat_samples = int(beat_interval * sfreq)

# Single beat template (QRS complex)
t_beat = np.linspace(0, 1, beat_samples)
beat_template = (
    -0.2 * np.exp(-((t_beat - 0.2) ** 2) / 0.001)  # Q wave
    + 1.0 * np.exp(-((t_beat - 0.25) ** 2) / 0.0005)  # R wave
    + -0.3 * np.exp(-((t_beat - 0.3) ** 2) / 0.001)  # S wave
    + 0.15 * np.exp(-((t_beat - 0.55) ** 2) / 0.005)  # T wave
)

# Multi-channel quasi-periodic signal
n_times_qp = n_beats * beat_samples
quasi_periodic_mc = np.zeros((n_channels, n_times_qp))

for i in range(n_beats):
    start_idx = i * beat_samples
    amplitude = 1.0 + rng.normal(0, 0.1)
    
    actual_start = start_idx
    actual_end = min(n_times_qp, actual_start + beat_samples)
    beat_len = actual_end - actual_start

    # Mix into channels with random strengths
    for ch in range(n_channels):
        quasi_periodic_mc[ch, actual_start:actual_end] += (
            rng.random() * amplitude * beat_template[:beat_len]
        )

# Add noise
noisy_qp_mc = quasi_periodic_mc + 0.3 * rng.standard_normal((n_channels, n_times_qp))
raw_qp = mne.io.RawArray(noisy_qp_mc, info)

# Apply IterativeDSS with QuasiPeriodicDenoiser
qp_denoiser = QuasiPeriodicDenoiser(
    peak_distance=int(beat_interval * sfreq * 0.7),
    peak_height_percentile=70,
    smooth_template=True,
)

idss_qp = IterativeDSS(n_components=3, denoiser=qp_denoiser, max_iter=3)
idss_qp.fit(raw_qp)
sources_qp = idss_qp.transform(raw_qp)

print(f"\nIterativeDSS converged in {len(idss_qp.convergence_info_)} iterations")

# Part 3: Real ECG Artifact (Single-Channel Denoising)
# =====================================================
print("\n--- Part 3: Real ECG Artifact (Sample Dataset) ---")

# (Assumes access to MNE sample data, code provided for reference)
try:
    from mne.datasets import sample
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw_ecg = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw_ecg.pick_types(meg="grad", eog=False, stim=False, exclude="bads")
    raw_ecg.filter(0.5, 30, fir_design="firwin", verbose=False)
    raw_ecg.crop(10, 30)  # 20 seconds
    
    channel_data = raw_ecg.get_data()[0]
    
    ecg_denoiser = QuasiPeriodicDenoiser(
        peak_distance=int(0.6 * raw_ecg.info["sfreq"]),  # ~100 BPM
        peak_height_percentile=85,
        smooth_template=True,
    )
    
    denoised_channel = ecg_denoiser.denoise(channel_data)
    print("Applied QuasiPeriodicDenoiser to MEG channel")
    
except ImportError:
    print("MNE sample data not found or mne not installed.")
    print("MNE sample data not found or mne not installed.")
```

### Temporal DSS (Time-Shift Regression & Smoothness)

```python
"""
=================================================
Temporal DSS: Time-Shift Regression & Smoothness.
=================================================

This example demonstrates DSS for extracting **temporally structured** signals:
autocorrelated components, slow drifts, and smooth waveforms.

We cover both **linear biases** (TimeShiftBias, SmoothingBias) and
**nonlinear denoisers** (DCTDenoiser, TemporalSmoothnessDenoiser).

**Structure**:
- Part 0: Synthetic Slow Drift (Random Walk)
- Part 1: TimeShiftBias + TSR (Time-Shift Regression)
- Part 2: SmoothingBias (Temporal Smoothing)
- Part 3: DCTDenoiser + IterativeDSS (DCT Domain)
- Part 4: Real EEG Slow Cortical Potentials
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

from mne_denoise.dss import DSS, IterativeDSS
from mne_denoise.dss.denoisers import (
    DCTDenoiser,
    SmoothingBias,
    TimeShiftBias,
)
from mne_denoise.dss.variants import smooth_dss, time_shift_dss
from mne_denoise.viz import (
    plot_component_summary,
    plot_psd_comparison,
    plot_time_course_comparison,
)

# Part 0: Synthetic Slow Drift (Random Walk)
# ===========================================
print("--- Part 0: Synthetic Slow Drift ---")

rng = np.random.default_rng(42)
sfreq = 250  # Hz
n_seconds = 30
n_times = n_seconds * sfreq
n_channels = 16
times = np.arange(n_times) / sfreq

# Slow Drift (Random Walk)
drift_seed = rng.normal(0, 0.05, n_times)
drift = np.cumsum(drift_seed)
drift = signal.detrend(drift)
drift *= 2.0

# Fast Noise (White)
noise = rng.normal(0, 1.5, (n_channels, n_times))

# Mix: First 4 channels get the drift
data = noise.copy()
data[:4] += drift * np.array([[1.0], [0.8], [0.6], [0.4]])

# Create MNE Raw
ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
info_sim = mne.create_info(ch_names, sfreq, "eeg")
# info_sim.set_montage(montage) # Skip montage for dummy channels
raw_sim = mne.io.RawArray(data, info_sim)

# Part 1: Time-Shift Regression (TimeShiftBias)
# ==============================================
print("\n--- Part 1: Time-Shift Regression ---")

# Manual TimeShiftBias
bias_tsr = TimeShiftBias(shifts=10, method="autocorrelation")
dss_tsr = DSS(n_components=5, bias=bias_tsr)
dss_tsr.fit(raw_sim)

# Part 2: Temporal Smoothing (SmoothingBias)
# ===========================================
print("\n--- Part 2: Temporal Smoothing ---")

bias_smooth = SmoothingBias(window=50)  # 50 samples = 200ms
dss_smooth = DSS(n_components=5, bias=bias_smooth)
dss_smooth.fit(raw_sim)

# Part 3: DCT Denoiser (Nonlinear, IterativeDSS)
# ===============================================
print("\n--- Part 3: DCT Denoiser + IterativeDSS ---")

# DCTDenoiser keeps first 30% of DCT coefficients
dct_denoiser = DCTDenoiser(cutoff_fraction=0.3)
idss_dct = IterativeDSS(denoiser=dct_denoiser, n_components=3, max_iter=5)
idss_dct.fit(raw_sim)

# Part 4: Real EEG Data (Slow Cortical Potentials)
# =================================================
print("\n--- Part 4: Real EEG Data ---")

try:
    from mne.datasets import eegbci
    subject = 1
    runs = [1]  # Baseline, eyes open
    raw_path = eegbci.load_data(subject, runs)[0]
    raw_eeg = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
    
    # Preproc for slow waves
    raw_eeg.filter(0.1, 10, fir_design="firwin", verbose=False)
    raw_eeg.set_eeg_reference("average", projection=True, verbose=False)
    raw_eeg.apply_proj()
    raw_eeg.crop(0, 30)

    # Apply TSR
    dss_eeg_tsr = time_shift_dss(shifts=10, n_components=5)
    dss_eeg_tsr.fit(raw_eeg)
    sources_eeg = dss_eeg_tsr.transform(raw_eeg)
    
    print("TSR extracted components from real EEG.")
    
except ImportError:
    print("MNE eegbci data not found.")
except Exception as e:
    print(f"Skipping real data example: {e}")
```

### Time-Frequency DSS (Spectrogram Masking)

```python
"""
========================================
Time-Frequency DSS: Spectrogram Masking.
========================================

This example demonstrates DSS for extracting **transient oscillatory bursts**
using time-frequency (TF) domain constraints via spectrogram masking.

We cover **SpectrogramBias** (linear) and **SpectrogramDenoiser** (nonlinear)
for isolating activity that is sparse in the TF domain.

**Structure**:
- Part 0: Synthetic Transient Bursts (Spindles)
- Part 1: SpectrogramBias with Fixed TF Mask
- Part 2: SpectrogramDenoiser (Adaptive Masking) + IterativeDSS
- Part 3: Real MEG Gamma Bursts (Somato Dataset)
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import somato
from scipy import signal as sp_signal

from mne_denoise.dss import DSS, IterativeDSS
from mne_denoise.dss.denoisers import SpectrogramBias, SpectrogramDenoiser
from mne_denoise.viz import (
    plot_component_spectrogram,
    plot_component_summary,
    plot_spectrogram_comparison,
    plot_tf_mask,
    plot_time_course_comparison,
)

# Part 0: Synthetic Transient Bursts (Spindles)
# ====================================================
print("--- Part 0: Synthetic Spindle Bursts ---")

rng = np.random.default_rng(42)
sfreq = 250
n_seconds = 10
n_times = n_seconds * sfreq
times = np.arange(n_times) / sfreq

# Background noise
noise = rng.normal(0, 1.0, n_times)

# Spindle bursts (12 Hz)
spindle_freq = 12.0
envelope = np.zeros_like(times)
# Burst 1: 2-3s, Burst 2: 7-8s
mask1 = (times >= 2) & (times < 3)
mask2 = (times >= 7) & (times < 8)
envelope[mask1] = np.hanning(mask1.sum())
envelope[mask2] = np.hanning(mask2.sum())

signal_spindle = envelope * np.sin(2 * np.pi * spindle_freq * times) * 3.0
data_mixed = signal_spindle + noise

# Part 1: SpectrogramBias with Fixed Mask
# ========================================
print("\n--- Part 1: Fixed TF Mask (SpectrogramBias) ---")

nperseg = 128
noverlap = 96
_, t_grid, _ = sp_signal.spectrogram(data_mixed, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
n_freqs = nperseg // 2 + 1
freq_axis = np.fft.rfftfreq(nperseg, 1 / sfreq)

# Mask: 10-15 Hz during burst times
mask_fixed = np.zeros((n_freqs, len(t_grid)))
freq_mask = (freq_axis >= 10) & (freq_axis <= 15)
time_mask = (t_grid >= 2) & (t_grid < 3) | (t_grid >= 7) & (t_grid < 8)
mask_fixed[freq_mask[:, None] & time_mask] = 1.0

# Create multi-channel data for DSS
n_channels = 8
data_multichan = np.tile(data_mixed, (n_channels, 1)) + rng.normal(0, 0.2, (n_channels, n_times))
ch_names = [f"EEG{i}" for i in range(n_channels)]
info = mne.create_info(ch_names, sfreq, "eeg")
raw_spindle = mne.io.RawArray(data_multichan, info)

# Apply SpectrogramBias
bias_tf = SpectrogramBias(mask=mask_fixed, nperseg=nperseg, noverlap=noverlap)
dss_tf = DSS(n_components=3, bias=bias_tf)
dss_tf.fit(raw_spindle)

# Part 2: SpectrogramDenoiser + IterativeDSS (Adaptive)
# ======================================================
print("\n--- Part 2: Adaptive TF Masking (SpectrogramDenoiser) ---")

# Keep top 10% of TF energy
spec_denoiser = SpectrogramDenoiser(threshold_percentile=90, nperseg=128, noverlap=96)
idss_spec = IterativeDSS(denoiser=spec_denoiser, n_components=2, max_iter=3)
idss_spec.fit(raw_spindle)

# Part 3: Real MEG Gamma Bursts (Somato Dataset)
# ===============================================
print("\n--- Part 3: Real MEG Data (Gamma Bursts) ---")

try:
    data_path = somato.data_path(verbose=False)
    raw_path = data_path / "sub-01" / "meg" / "sub-01_task-somato_meg.fif"
    raw_somato = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    raw_somato.pick_types(meg="grad", exclude="bads")
    raw_somato.filter(1, 100, fir_design="firwin", verbose=False)
    raw_somato.crop(0, 60)

    # Apply SpectrogramDenoiser (Top 5% energy)
    spec_denoiser_meg = SpectrogramDenoiser(threshold_percentile=95, nperseg=256)
    idss_meg = IterativeDSS(denoiser=spec_denoiser_meg, n_components=3, max_iter=3)
    idss_meg.fit(raw_somato)
    
    print("IterativeDSS extracted gamma bursts from MEG.")
    
    print(f"Skipping real data example: {e}")
```

### Blind Source Separation (ICA Equivalence)

```python
"""
=============================================================================
08. Blind Source Separation & ICA Equivalence.
=============================================================================

This example demonstrates how **Nonlinear DSS** can perform Blind Source Separation (BSS),
effectively recovering independent sources from mixed signals. It explicitly shows
the equivalence between DSS with specific nonlinearities and Independent Component Analysis (ICA).

We cover:
1.  **Synthetic BSS**: Separating mixed Super-Gaussian sources (speech/bursts) and Sub-Gaussian sources.
2.  **ICA Equivalence**: Comparing DSS (`TanhMaskDenoiser`, `KurtosisDenoiser`) against `sklearn.decomposition.FastICA`.
3.  **Real MEG Data**: Performing blind decomposition of the MNE Sample dataset to find artifacts (EOG/ECG) and brain sources.
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
from scipy import stats

from mne_denoise.dss import IterativeDSS, KurtosisDenoiser, TanhMaskDenoiser, beta_tanh
from mne_denoise.viz import (
    plot_component_summary,
    plot_component_time_series,
    plot_overlay_comparison,
)

# Part 1: Synthetic Blind Source Separation
# -----------------------------------------
print("\n--- 1. Creating Synthetic Mixed Data ---")

n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Sources: Laplace (sparse), Square Wave (kurtotic), Sinusoid (sub-gaussian), Gaussian
s1 = stats.laplace.rvs(size=n_samples); s1 /= s1.std()
s2 = np.sign(np.sin(3 * time)); s2 /= s2.std()
s3 = np.sin(10 * time); s3 /= s3.std()
s4 = np.random.randn(n_samples)

S_true = np.c_[s1, s2, s3, s4].T
n_sources = S_true.shape[0]

# Mix sources
np.random.seed(42)
A = np.random.randn(n_sources, n_sources)
X = np.dot(A, S_true)

# Run DSS with Tanh Nonlinearity (Robust ICA)
# -------------------------------------------
print("\nRunning DSS with Tanh Nonlinearity (Robust)...")

# Newton Method (Fast - FastICA style)
dss_tanh = IterativeDSS(
    denoiser=TanhMaskDenoiser(),
    method="deflation",
    n_components=n_sources,
    beta=beta_tanh,  # Newton step
    random_state=42,
    verbose=False,
)
dss_tanh.fit(X)
S_dss_tanh = dss_tanh.transform(X)

# Run DSS with Kurtosis Nonlinearity (Standard FastICA)
# -----------------------------------------------------
print("Running DSS with Kurtosis Nonlinearity (FastICA standard)...")
dss_kurt = IterativeDSS(
    denoiser=KurtosisDenoiser(nonlinearity="cube"),
    method="deflation",
    n_components=n_sources,
    beta=-3.0,
    random_state=42,
    verbose=False,
)
dss_kurt.fit(X)
S_dss_kurt = dss_kurt.transform(X)

# Comparison with sklearn FastICA
# -------------------------------
from sklearn.decomposition import FastICA
print("Running sklearn FastICA (Benchmark)...")
ica = FastICA(n_components=n_sources, algorithm="deflation", fun="logcosh", random_state=42)
S_fastica = ica.fit_transform(X.T).T

# Part 2: Blind Separation of Real MEG Data
# -----------------------------------------
print("\n--- 2. Real MEG Data (Blind Separation) ---")

try:
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw.crop(0, 60).pick_types(meg=True, eeg=False, eog=True, stim=False).load_data()
    raw.filter(1, 40, verbose=False)

    # Prepare MEG-only data
    raw_meg = raw.copy().pick_types(meg=True, eeg=False, eog=False, stim=False)
    
    # Fit DSS-Tanh (Blind Decomposition)
    print("Fitting Blind DSS (this may take a moment)...")
    n_components = 15
    dss_meg = IterativeDSS(
        denoiser=TanhMaskDenoiser(),
        method="deflation",
        n_components=n_components,
        beta=beta_tanh,
        verbose=True,
    )
    dss_meg.fit(raw_meg)
    
    # Identify Artifacts by correlation with EOG
    eog_ch = raw.get_data(picks="eog")[0]
    sources = dss_meg.transform(raw_meg)
    corrs = [np.abs(np.corrcoef(s, eog_ch)[0, 1]) for s in sources]
    blink_idx = np.argmax(corrs)
    print(f"\nMost likely EOG component: #{blink_idx} (Corr: {corrs[blink_idx]:.3f})")

except ImportError:
    print("MNE sample data not found or mne not installed.")
except Exception as e:
    print(f"Skipping real data example: {e}")
```



#### DSS vs ICA Correlation Validation

**Based on Appendix BSS/ICA Equivalence - Validates DSS ≈ ICA**

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
    
    data = raw.get_data()  # (n_channels, n_times)
    
    # 1. Fit ICA (sklearn FastICA)
    t_start = time.time()
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    ica_components = ica.fit_transform(data.T).T  # (n_components, n_times)
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
    dss_components = dss.transform(data.T).T  # (n_components, n_times)
    time_dss = time.time() - t_start
    
    # 3. Compute correlation matrix between all component pairs
    corr_matrix = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            # Absolute correlation (components can be sign-flipped)
            corr = np.abs(np.corrcoef(ica_components[i], dss_components[j])[0, 1])
            corr_matrix[i, j] = corr
    
    # 4. Find best matches (Hungarian assignment)
    # Maximizes sum of correlations by matching components optimally
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    matched_corrs = corr_matrix[row_ind, col_ind]
    avg_correlation = matched_corrs.mean()
    
    # 5. Identify artifact components (correlate with reference channels if available)
    # This follows the appendix pattern of correlating with EOG/ECG
    artifact_correlations = {}
    if 'EOG' in [ch['kind'] for ch in raw.info['chs']]:
        eog_data = raw.copy().pick_types(eog=True).get_data()[0]
        dss_eog_corr = [np.abs(np.corrcoef(c, eog_data)[0, 1]) for c in dss_components]
        ica_eog_corr = [np.abs(np.corrcoef(c, eog_data)[0, 1]) for c in ica_components]
        artifact_correlations['eog'] = {
            'dss_best_idx': np.argmax(dss_eog_corr),
            'dss_best_corr': np.max(dss_eog_corr),
            'ica_best_idx': np.argmax(ica_eog_corr),
            'ica_best_corr': np.max(ica_eog_corr)
        }
    
    # 6. Results
    speedup = time_ica / time_dss
    n_high_corr = (matched_corrs > 0.9).sum()
    
    results = {
        'subject': subject_id,
        'n_components': n_components,
        'time_ica': time_ica,
        'time_dss': time_dss,
        'speedup': speedup,
        'avg_correlation': avg_correlation,
        'min_correlation': matched_corrs.min(),
        'max_correlation': matched_corrs.max(),
        'n_high_correlation': n_high_corr,
        'pct_high_correlation': 100 * n_high_corr / n_components,
        'artifact_correlations': artifact_correlations
    }
    
    # 7. Logging
    LOGGER.info(f"  ICA time: {time_ica:.2f}s")
    LOGGER.info(f"  DSS time: {time_dss:.2f}s")
    LOGGER.info(f"  Speedup: {speedup:.2f}x ⚡")
    LOGGER.info(f"  Avg correlation: {avg_correlation:.3f}")
    LOGGER.info(f"  Components with >0.9 corr: {n_high_corr}/{n_components}")
    
    if avg_correlation > 0.85:
        LOGGER.info("  ✅ DSS and ICA are EQUIVALENT (high correlation)")
        LOGGER.info(f"  💡 USE DSS for {speedup:.1f}x speedup!")
    else:
        LOGGER.warning("  ⚠️  Low correlation - methods may extract different components")
    
    return results

# Example Usage in Pipeline Comparison
comparison_results = []
for subject in test_subjects:
    raw = mne.io.read_raw_fif(f'derivatives/preproc/{subject}/eeg/{subject}_eeg.fif')
    
    val_result = validate_dss_ica_correlation(
        raw, 
        Path('output/validation'), 
        subject,
        n_components=15
    )
    comparison_results.append(val_result)

# Summary
df_val = pd.DataFrame(comparison_results)
print(f"\nAverage speedup: {df_val['speedup'].mean():.1f}x")
print(f"Average correlation: {df_val['avg_correlation'].mean():.2%}")
print(f"\n→ Conclusion: DSS is {df_val['speedup'].mean():.1f}x faster with {df_val['avg_correlation'].mean():.0%} equivalent components")
```

**Key Insights from BSS Appendix:**
- DSS with `TanhMaskDenoiser` = Robust ICA
- DSS with `KurtosisDenoiser` = Standard FastICA
- Components can be sign-flipped → use absolute correlation
- Artifact detection: correlate components with reference channels (EOG/ECG)

**Expected Results:**
- Speedup: **2-3x faster** than ICA
- Correlation: **>90%** (proves equivalence)
- **Recommendation**: Use DSS for artifact removal (faster, equivalent results)


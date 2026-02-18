
import numpy as np
import torch
from scipy.signal import resample, butter, filtfilt



def preprocess_signal(raw_eeg, sfreq, target_sfreq=200):
    """
    Input: (Channels, Time) numpy array or tensor
    Apply: Resample(200Hz) -> Bandpass(0.5-99.5) -> Z-score -> Clip(15std)
    """
    if isinstance(raw_eeg, torch.Tensor):
        raw_eeg = raw_eeg.cpu().numpy()
        
    n_channels, n_samples = raw_eeg.shape
    
    # 1. Resample
    if sfreq != target_sfreq:
        num_samples = int(n_samples * target_sfreq / sfreq)
        raw_eeg = resample(raw_eeg, num_samples, axis=-1)
        sfreq = target_sfreq
        
    # 2. Bandpass Filter (0.5 - 99.5 Hz)
    nyquist = sfreq / 2
    low = 0.5 / nyquist
    high = 99.5 / nyquist
    if high >= 1.0: high = 0.999 # Avoid instability if sfreq is exactly 200 (Nyquist 100)
    
    b, a = butter(N=2, Wn=[low, high], btype='band')
    raw_eeg = filtfilt(b, a, raw_eeg, axis=-1)
    
    # 3. Z-score Normalization (per recording)
    # Mean and Std over the entire recording (Time dimension)
    mean = np.mean(raw_eeg, axis=-1, keepdims=True)
    std = np.std(raw_eeg, axis=-1, keepdims=True)
    
    raw_eeg = (raw_eeg - mean) / (std + 1e-6)
    
    # 4. Clip values (15 std)
    raw_eeg = np.clip(raw_eeg, -15, 15)
    
    return torch.tensor(raw_eeg, dtype=torch.float32)

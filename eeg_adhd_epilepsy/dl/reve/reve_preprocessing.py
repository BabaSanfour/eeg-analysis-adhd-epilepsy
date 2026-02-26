import numpy as np
import torch




def preprocess_signal(raw_eeg):
    """
    Input: (Channels, Time) numpy array or tensor
    Apply: Z-score -> Clip(15std)
    """
    if isinstance(raw_eeg, torch.Tensor):
        raw_eeg = raw_eeg.cpu().numpy()

    
    # 3. Z-score Normalization (per recording)
    # Mean and Std over the entire recording (Time dimension)
    mean = np.mean(raw_eeg, axis=-1, keepdims=True)
    std = np.std(raw_eeg, axis=-1, keepdims=True)
    
    raw_eeg = (raw_eeg - mean) / (std + 1e-6)
    
    # 4. Clip values (15 std)
    raw_eeg = np.clip(raw_eeg, -15, 15)
    
    return torch.tensor(raw_eeg, dtype=torch.float32)

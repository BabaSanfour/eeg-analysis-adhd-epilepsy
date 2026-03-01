
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_attn = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x: (Batch, SeqLen, Dim)
        attn_logits = self.to_attn(x) # (Batch, SeqLen, 1)
        attn_weights = self.softmax(attn_logits) # (Batch, SeqLen, 1)
        out = (x * attn_weights).sum(dim=1) # (Batch, Dim)
        return out

class REVEFeatureExtractor(nn.Module):
    def __init__(self, model_size='base'):
        super().__init__()
        
        # Determine model ID
        if model_size == 'base':
            self.model_id = "brain-bzh/reve-base"
        elif model_size == 'large':
            self.model_id = "brain-bzh/reve-large"
        else:
            raise ValueError(f"Unknown model size: {model_size}")
            
        print(f"Loading REVE model from {self.model_id}...")
        self.encoder = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            torch_dtype="auto",
            token=True
        )
        
        # Load Position Bank
        # Note: Position bank is usually global / per-channel.
        # It's a small model that embeds strings ("Fp1") to (1, 3) coords.
        print("Loading REVE position bank...")
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", 
            trust_remote_code=True, 
            torch_dtype="auto",
            token=True
        )
        
        # Dimension is usually 512 for base, 1216 for large
        self.dim = getattr(self.encoder.config, "hidden_size", None)
        if self.dim is None:
             self.dim = getattr(self.encoder.config, "embed_dim", None)

        if self.dim is None:
             # Fallback
             if model_size == 'large':
                 self.dim = 1216
             else:
                 self.dim = 512
        
        # Add Pooling Layer
        self.pooler = AttentionPooling(self.dim)

    def forward(self, x, channel_names=None, pool=True):
        """
        x: (Batch, Channels, Time) 
        channel_names: List of strings (e.g. ['Fp1', 'Fp2', ...])
        """
        
        # 1. Get Coordinates
        
        if channel_names is None:
             # For now error.
             raise ValueError("channel_names must be provided")

        # Get positions using the bank
        
        # Move pos_bank to same device as x temporarily or permanently
        self.pos_bank.to(x.device)
        
        coords = self.pos_bank(channel_names) # (C, 3)
        coords = coords.to(x.device)
        
        # Expand for batch: (B, C, 3)
        pos = coords.unsqueeze(0).expand(x.shape[0], -1, -1)
        
        # Pass return_output=True to natively extract all intermediate transformer layers
        outputs = self.encoder(x, pos=pos, return_output=True)
        
        # If it returned a list of tensors (all layers)
        if isinstance(outputs, (list, tuple)):
            clean_layer_outputs = list(outputs)
            
            B = x.shape[0]
            C = x.shape[1]

            if pool:
                # Concatenate all tokens from all layers together
                # Each layer_out is [Batch, C*P, Dim]. 
                # Concatenating them along dim=1 makes a massive sequence [Batch, NumLayers * C * P, Dim]
                massive_sequence = torch.cat(clean_layer_outputs, dim=1)
                
                # The pooler squashes all layers, channels, and patches into ONE vector
                global_token = self.pooler(massive_sequence) # -> [Batch, Dim]
                
                return global_token
            else:
                # The transformer returns [Batch, C*P, Dim].
                # Stack the layers to return [Batch, Layers, Tokens, Dim]
                return torch.stack(clean_layer_outputs, dim=1)
                
        else:
            # If return_output=True is not supported by this model version, we fail hard
            raise RuntimeError(
                f"REVE model returned a single tensor instead of a list of layer outputs. "
                f"return_output=True may not be supported by this model version. "
                f"Output type: {type(outputs)}, shape: {getattr(outputs, 'shape', 'N/A')}"
            )


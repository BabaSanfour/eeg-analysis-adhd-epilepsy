
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
            torch_dtype="auto"
        )
        
        # Load Position Bank
        # Note: Position bank is usually global / per-channel.
        # It's a small model that embeds strings ("Fp1") to (1, 3) coords.
        print("Loading REVE position bank...")
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", 
            trust_remote_code=True, 
            torch_dtype="auto"
        )
        
        # Dimension is usually 512 for base
        try:
             self.dim = self.encoder.config.hidden_size 
        except:
             self.dim = 512 # Fallback
        
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
        
        layer_outputs = []
        hooks = []
        if hasattr(self.encoder, 'transformer') and hasattr(self.encoder.transformer, 'layers'):
            def get_hook():
                def hook(module, input, output):
                    layer_outputs.append(output)
                return hook
            for layer in self.encoder.transformer.layers:
                hooks.append(layer.register_forward_hook(get_hook()))

        try:
            outputs = self.encoder(x, pos=pos)
            
            if len(layer_outputs) > 0:
                # PAPER ALIGNMENT: Clean tuples if necessary
                clean_layer_outputs = [out[0] if isinstance(out, tuple) else out for out in layer_outputs]
                
                if pool:
                    # Concatenate the tokens from ALL layers along the sequence dimension
                    # Resulting shape: (Batch, NumLayers * SeqLen, Dim)
                    concatenated_tokens = torch.cat(clean_layer_outputs, dim=1) 
                    
                    #Pool the massive sequence into a SINGLE global token
                    # Resulting shape: (Batch, Dim)
                    embedding = self.pooler(concatenated_tokens) 
                    
                    return embedding
                else:
                    return torch.stack(clean_layer_outputs, dim=1)
            else:
                 # Fallback if no hooks could be registered
                 if hasattr(outputs, 'last_hidden_state'):
                    sequence_output = outputs.last_hidden_state
                 else:
                    sequence_output = outputs
                    
                 if pool:
                     embedding = self.pooler(sequence_output)
                     return embedding
                 else:
                     return sequence_output
            
        finally:
            for h in hooks:
                h.remove()


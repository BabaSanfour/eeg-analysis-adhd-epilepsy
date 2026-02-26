import sys
import torch
sys.path.append('/home/mat/projects/coco-pipe')
from CBraMod.models.cbramod import CBraMod
model = CBraMod()
model.eval()

# Let's say C=19 channels, S=10 segments, P=200 points
x = torch.randn(1, 19, 10, 200)

layer_outputs = []
def hook(module, input, output):
    layer_outputs.append(output)

for name, module in model.named_modules():
    if "TransformerEncoderLayer" in module.__class__.__name__:
        module.register_forward_hook(hook)

with torch.no_grad():
    out = model(x)

print(f"Input shape: {x.shape}")
print(f"First hook output shape: {layer_outputs[0].shape}")
print(f"Final model output shape: {out.shape}")

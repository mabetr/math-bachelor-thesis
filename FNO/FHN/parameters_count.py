#parameters_count.py
import torch
from neuralop.models import FNO

model = FNO(
    n_modes=(32,),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    n_layers=5
)

# trainable parameters count
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total number of trainable parameters: {num_params:,}")
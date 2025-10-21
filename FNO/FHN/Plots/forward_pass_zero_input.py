# Plots for forward pass for zero input with/without positional embedding and bias modifications
from neuralop.models import FNO
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- functions ---
def zero_all_biases(model):
    for m in model.modules():
        if hasattr(m, "bias") and m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()
def forward_zero_input(model, n_points, device):
    model.eval()
    x = torch.zeros(1, 1, n_points, device=device)
    with torch.no_grad():
        y_pred = model(x)
    return y_pred.squeeze().cpu().numpy()

# --- setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_points = 100

# --- model 1 : with positional embedding, biases as usual ---
model1 = FNO(
    n_modes=(32,),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    n_layers=5
).to(device)
y1 = forward_zero_input(model1, n_points, device)

# --- model 2 : without positional embedding, biases as usual ---
model2 = FNO(
    n_modes=(32,),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    positional_embedding=None,
    n_layers=5
).to(device)
y2 = forward_zero_input(model2, n_points, device)

# --- model 3 : without positional embedding, zero biases ---
model3 = FNO(
    n_modes=(32,),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    positional_embedding=None,
    n_layers=5
).to(device)
zero_all_biases(model3)
y3 = forward_zero_input(model3, n_points, device)

# --- visualization ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = [
    "PE, biases intact",
    "No PE, biases intact",
    "No PE, biases = 0"
]
ys = [y1, y2, y3]
labels = ["(a)", "(b)", "(c)"]

for ax, y, label, title in zip(axes, ys, labels, titles):
    ax.plot(np.arange(n_points), y)
    ax.set_xlabel("t")
    ax.set_ylabel("$V(t)$")
    ax.grid(True)
    ax.ticklabel_format(style='plain', axis='y', useOffset=True)

    ax.text(
        0.5, -0.25, f"{label} {title}",
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=12
    )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
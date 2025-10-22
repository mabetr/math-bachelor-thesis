# Test_FNO.py
# FNO for data_tensors with minmax normalization -> Evaluation of the best model on the test set
# plotting for a few representatives of each category

from neuralop.models import FNO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random

#seed
seed = 71
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Use of the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataloader
data = torch.load("../../Data/FitzHugh_Nagumo/data_tensors/dataset_fno1d_minmax_red.pt")

X_train_norm, Y_train_norm = data['X_train_norm'], data['Y_train_norm']
X_val_norm, Y_val_norm     = data['X_val_norm'], data['Y_val_norm']
X_test_norm, Y_test_norm   = data['X_test_norm'], data['Y_test_norm']

#Batch creation
batch_size = 16
train_loader = DataLoader(TensorDataset(X_train_norm, Y_train_norm), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_norm, Y_val_norm), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_norm, Y_test_norm), batch_size=batch_size, shuffle=True)

# relative L2-loss
def relative_L2_loss(y_pred, y_true) :
    diff = y_true - y_pred
    diff_norm = torch.sqrt(torch.sum(diff**2, dim=-1))
    true_norm = torch.sqrt(torch.sum(y_true**2, dim=-1))
    rel_loss = diff_norm / (true_norm + 1e-8)
    return torch.mean(rel_loss) #mean over batch


### Evaluation on the test set
print("\n--- Evaluation of the best model on the test set ---")
best_model = FNO(
    n_modes=(32,),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    #positional_embedding=None,
    n_layers = 5,
)

best_model.to(device)
best_model.load_state_dict(torch.load('../../best_paths/best_fno_model_minmax_mode32_hc64_layers5_bz16_seed71.pth', map_location=device, weights_only=False))
best_model.eval()

# test loss
test_loss = 0
with torch.no_grad():
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
        y_pred_test = best_model(x_test_batch)
        test_loss += relative_L2_loss(y_pred_test, y_test_batch).item()
test_loss /= len(test_loader)
print(f"Final loss on the test set : {test_loss:.6f}")

# ------------------------------------------------
# Plot for a few representatives of each category
# ------------------------------------------------
# Indices to visualize
indices_to_plot = [0, 21, 297, 348]
plot_labels = ["t0", "gen", "nap", "tfin"]

# Extract only these samples from the test set
x_visu = X_test_norm[indices_to_plot].to(device)
y_visu = Y_test_norm[indices_to_plot].to(device)

# Predictions
with torch.no_grad():
    pred_visu = best_model(x_visu)

# Denormalization
normalization_param = torch.load("../../Data/FitzHugh_Nagumo/data_tensors/normalization_minmax_stats.pt", map_location=device)
X_min, X_max = normalization_param["X_min"].to(device), normalization_param["X_max"].to(device)
Y_min, Y_max = normalization_param["Y_min"].to(device), normalization_param["Y_max"].to(device)
y_visu_denorm = Y_min + y_visu * (Y_max - Y_min)
pred_visu_denorm = Y_min + pred_visu * (Y_max - Y_min)
x_visu_denorm = X_min + x_visu * (X_max - X_min)

# time scale for the visualization
n_points_new = x_visu.shape[-1]
t_max = 100.0
t = np.linspace(0, t_max, n_points_new)

#plot
plt.figure(figsize=(15, 15))
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=False)
handles = []
labels = []
for col_idx, label in enumerate(plot_labels):
    axes[0, col_idx].set_title(f"{label}", fontsize=14)

for col_idx, label in enumerate(plot_labels):
    ax = axes[0, col_idx]
    x_sample = x_visu_denorm[col_idx, 0, :].cpu().numpy()
    ax.plot(t, x_sample, color="black", linewidth=1.5, label="Input function")
    ax.set_ylabel("$I_{app}(t)$")
    ax.set_xlabel("t")
    ax.grid(True)

for col_idx, label in enumerate(plot_labels):
    ax = axes[1, col_idx]
    y_true_sample = y_visu_denorm[col_idx, 0, :].cpu().numpy()
    y_pred_sample = pred_visu_denorm[col_idx, 0, :].cpu().numpy()

    ax.plot(t, y_true_sample, '--', color="blue", linewidth=1.5, label="True solution")
    ax.plot(t, y_pred_sample, color="orange", linewidth=1.5, label="Predicted solution")

    ax.set_xlabel("t")
    ax.set_ylabel("$V(t)$")
    ax.grid(True)

if not handles:
    handles, labels = ax.get_legend_handles_labels()
    handles_top, labels_top = axes[0, col_idx].get_legend_handles_labels()
    handles = handles_top + handles
    labels = labels_top + labels

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

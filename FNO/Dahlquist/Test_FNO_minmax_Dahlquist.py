# FNO for Dahlquist for data_tensors with minmax normalization -> Evaluation of the best model on the test set
# plotting for a few representatives of each category

from neuralop.models import FNO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random

#seed
seed = 42
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
data = torch.load("../../Data/Dahlquist/data_tensors_Dahlquist/dataset_fno1d_minmax.pt")

X_train_norm, Y_train_norm = data['X_train_norm'], data['Y_train_norm']
X_val_norm, Y_val_norm     = data['X_val_norm'], data['Y_val_norm']
X_test_norm, Y_test_norm   = data['X_test_norm'], data['Y_test_norm']

# point number reduction
n_points_new = 500
n_points_old = X_train_norm.shape[-1]  # 10001
indices = torch.linspace(0, n_points_old-1, n_points_new).long()

X_train_norm = X_train_norm[:, :, indices]
Y_train_norm = Y_train_norm[:, :, indices]
X_val_norm   = X_val_norm[:, :, indices]
Y_val_norm   = Y_val_norm[:, :, indices]
X_test_norm  = X_test_norm[:, :, indices]
Y_test_norm  = Y_test_norm[:, :, indices]

print("New shape of the dataset:", X_train_norm.shape)  # (batch, 1, 500)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_norm, Y_train_norm), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_norm, Y_val_norm), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_norm, Y_test_norm), batch_size=batch_size, shuffle=False)

# relative L2-loss
def relative_L2_loss(y_pred, y_true) : # y_pred, y_true shape: (batch, ndim, n_points)
    diff = y_true - y_pred
    diff_norm = torch.sqrt(torch.sum(diff**2, dim=-1))      # L2 over the points
    true_norm = torch.sqrt(torch.sum(y_true**2, dim=-1))
    rel_loss = diff_norm / (true_norm + 1e-8)
    return torch.mean(rel_loss) #mean over batch

### Evaluation on the test set
print("\n--- Evaluation of the best model on the test set ---")
best_model = FNO(
    n_modes=(32,),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    #positional_embedding=None,
    n_layers = 4,
)

best_model.to(device)
best_model.load_state_dict(torch.load('../../best_paths/Dahlquist/best_fno_model_minmax_Dahlquist_mode32_hc32.pth', map_location=device, weights_only=False))
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

# indices for visualization
indices_to_plot = [0, 10, 297, 348]

# samples from test set
x_visu = X_test_norm[indices_to_plot].to(device)
y_visu = Y_test_norm[indices_to_plot].to(device)

# predictions
with torch.no_grad():
    pred_visu = best_model(x_visu)

# denormalization
normalization_param = torch.load("../../Data/Dahlquist/data_tensors_Dahlquist/normalization_minmax_stats.pt", map_location=device)
X_min, X_max = normalization_param["X_min"].to(device), normalization_param["X_max"].to(device)
Y_min, Y_max = normalization_param["Y_min"].to(device), normalization_param["Y_max"].to(device)
y_visu_denorm = Y_min + y_visu * (Y_max - Y_min)
pred_visu_denorm = Y_min + pred_visu * (Y_max - Y_min)
x_visu_denorm = X_min + x_visu * (X_max - X_min)

# time scale
t_max = 100.0
t = np.linspace(0, t_max, n_points_new)

plt.figure(figsize=(15, 15))

# visualization
for plot_idx, sample_idx in enumerate(indices_to_plot):
    # input
    plt.subplot(4, 2, 2 * plot_idx + 1)
    x_sample = x_visu_denorm[plot_idx, 0, :].cpu().numpy()
    plt.plot(t, x_sample, label="Input function")
    plt.title(f"Input #{sample_idx}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.grid(True)

    # output
    plt.subplot(4, 2, 2 * plot_idx + 2)
    y_true_sample = y_visu_denorm[plot_idx, 0, :].cpu().numpy()
    y_pred_sample = pred_visu_denorm[plot_idx, 0, :].cpu().numpy()
    plt.plot(t, y_true_sample, '--', label="True solution")
    plt.plot(t, y_pred_sample, label="Predicted solution")
    plt.title(f"Output #{sample_idx}")
    plt.xlabel("t")
    plt.ylabel("v(t)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

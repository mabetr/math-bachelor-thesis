#normalization_minmax.py
#minmax normalization of dataset_fno1d.pt
#->save into data_tensors/dataset_fno1d_minmax.pt and data_tensors/normalization_minmax_stats.pt

import torch
import os

# Save directory
folder = "data_tensors"

# Load the dataset already converted into tensors
dataset_path = os.path.join(folder, "dataset_fno1d.pt")
data = torch.load(dataset_path)

X_train, Y_train = data['X_train'], data['Y_train']
X_val,   Y_val   = data['X_val'],   data['Y_val']
X_test,  Y_test  = data['X_test'],  data['Y_test']

# -----------------------------
# MIN-MAX NORMALIZATION [0,1]
# -----------------------------
def minmax_normalize(tensor, eps=1e-8):
    t_min = tensor.min()
    t_max = tensor.max()
    t_norm = (tensor - t_min) / (t_max - t_min + eps)
    return t_norm, t_min, t_max

# Normalize X
X_train_norm, X_min, X_max = minmax_normalize(X_train)
X_val_norm   = (X_val - X_min) / (X_max - X_min + 1e-8)
X_test_norm  = (X_test - X_min) / (X_max - X_min + 1e-8)

# Normalize Y
Y_train_norm, Y_min, Y_max = minmax_normalize(Y_train)
Y_val_norm   = (Y_val - Y_min) / (Y_max - Y_min + 1e-8)
Y_test_norm  = (Y_test - Y_min) / (Y_max - Y_min + 1e-8)

# Save the normalized data
os.makedirs(folder, exist_ok=True)
torch.save({
    'X_train_norm': X_train_norm, 'Y_train_norm': Y_train_norm,
    'X_val_norm': X_val_norm, 'Y_val_norm': Y_val_norm,
    'X_test_norm': X_test_norm, 'Y_test_norm': Y_test_norm
}, os.path.join(folder, "dataset_fno1d_minmax.pt"))

# Save bounds for denormalization
stats = {
    "X_min": X_min, "X_max": X_max,
    "Y_min": Y_min, "Y_max": Y_max
}
torch.save(stats, os.path.join(folder, "normalization_minmax_stats.pt"))

# Check of the shapes
print("Train:", X_train_norm.shape, Y_train_norm.shape)
print("Val  :", X_val_norm.shape, Y_val_norm.shape)
print("Test :", X_test_norm.shape, Y_test_norm.shape)

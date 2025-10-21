# reduction_number_points.py
# reduction of the number of points (to accelerate learning)
# ->save in data_tensors/dataset_fno1d_minmax_red.pt

import torch
import torch.nn as nn

#Dataloader
data = torch.load("../data_tensors/dataset_fno1d_minmax.pt")

X_train_norm, Y_train_norm = data['X_train_norm'], data['Y_train_norm']
X_val_norm, Y_val_norm     = data['X_val_norm'], data['Y_val_norm']
X_test_norm, Y_test_norm   = data['X_test_norm'], data['Y_test_norm']

# reduction of the number of points
n_points_new = 500
n_points_old = X_train_norm.shape[-1]  # 10001
indices = torch.linspace(0, n_points_old-1, n_points_new).long()

X_train_norm = X_train_norm[:, :, indices]
Y_train_norm = Y_train_norm[:, :, indices]
X_val_norm   = X_val_norm[:, :, indices]
Y_val_norm   = Y_val_norm[:, :, indices]
X_test_norm  = X_test_norm[:, :, indices]
Y_test_norm  = Y_test_norm[:, :, indices]

# Creation of the reduced dictionary
data_reduced = {
    'X_train_norm': X_train_norm,
    'Y_train_norm': Y_train_norm,
    'X_val_norm': X_val_norm,
    'Y_val_norm': Y_val_norm,
    'X_test_norm': X_test_norm,
    'Y_test_norm': Y_test_norm
}

# Save
torch.save(data_reduced, "../data_tensors/dataset_fno1d_minmax_red.pt")

# Check of the shape
print("New shape of the dataset:", X_train_norm.shape)  # (batch, 1, 500)
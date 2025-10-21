# transformation of the dictionnary in tensor and normalization of the datas

import numpy as np
import torch
import os

def prepare_FNO_tensors_vectorized(dataset, resolution):
    # dataset : dict list {'amplitude', 'duration', 'time', 'V'}
    # resolution : lenght of time vector
    # return: X (batch,1,res), Y (batch,1,res) format torch.float32 (and 1 is the channel)


    amplitudes = np.array([sim["amplitude"] for sim in dataset], dtype=np.float32)[:, None] # (N,1)
    durations = np.array([sim['duration'] for sim in dataset], dtype=np.float32)[:, None]  # (N,1)
    time_grid = np.array([sim['time'] for sim in dataset], dtype=np.float32)  # (N,res)
    V_values = np.array([sim['V'] for sim in dataset], dtype=np.float32)  # (N,res)

    # square-function
    I_stim = np.where(time_grid <= durations, amplitudes, 0.0)  # (N,res)

    # channel
    X = I_stim[:, None, :]  # (N,1,res)
    Y = V_values[:, None, :]  # (N,1,res)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# dataset loading
data_train = np.load('../datasets_dict_Dahlquist/dataset_train.npz', allow_pickle=True)
data_test = np.load('../datasets_dict_Dahlquist/dataset_test.npz', allow_pickle=True)
data_val = np.load('../datasets_dict_Dahlquist/dataset_val.npz', allow_pickle=True)

# Concatenation
train_data = np.concatenate([data_train["t0"], data_train["gen"], data_train["nap"]])
test_data = np.concatenate([data_test["t0"], data_test["gen"], data_test["nap"], data_test["tfin"]])
val_data = np.concatenate([data_val["t0"], data_val["gen"], data_val["nap"], data_val["tfin"]])

# Resolution
time_vec = data_train["t0"][0]["time"]
resolution = len(time_vec)

# Tensor conversion
X_train, Y_train = prepare_FNO_tensors_vectorized(train_data, resolution)
X_test, Y_test = prepare_FNO_tensors_vectorized(test_data, resolution)
X_val, Y_val = prepare_FNO_tensors_vectorized(val_data, resolution)

# Save .pt
folder = "data_tensors_Dahlquist"
os.makedirs(folder, exist_ok=True)
torch.save({
    'X_train': X_train,
    'Y_train': Y_train,
    'X_val': X_val,
    'Y_val': Y_val,
    'X_test': X_test,
    'Y_test': Y_test
}, os.path.join(folder, "dataset_fno1d.pt"))

# check
print("Train:", X_train.shape, Y_train.shape)
print("Test :", X_test.shape, Y_test.shape)
print("Val  :", X_val.shape, Y_val.shape)


# MIN-MAX NORMALIZATION [0,1]
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

# Save tensor
os.makedirs(folder, exist_ok=True)
torch.save({
    'X_train_norm': X_train_norm, 'Y_train_norm': Y_train_norm,
    'X_val_norm': X_val_norm, 'Y_val_norm': Y_val_norm,
    'X_test_norm': X_test_norm, 'Y_test_norm': Y_test_norm
}, os.path.join(folder, "dataset_fno1d_minmax.pt"))

# Save stats for denormalization
stats = {
    "X_min": X_min, "X_max": X_max,
    "Y_min": Y_min, "Y_max": Y_max
}
torch.save(stats, os.path.join(folder, "normalization_minmax_stats.pt"))

# shapes verification
print("Train:", X_train_norm.shape, Y_train_norm.shape)
print("Val  :", X_val_norm.shape, Y_val_norm.shape)
print("Test :", X_test_norm.shape, Y_test_norm.shape)
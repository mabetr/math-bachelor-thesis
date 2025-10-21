# conversion_into_data_tensors.py :
# Transformation of the dictionnary (datasets_dict) in tensors (data_tensors)
# -> save in data_tensors/dataset_fno1d.pt

import numpy as np
import torch
import os

def prepare_FNO_tensors_vectorized(dataset, resolution):
    #dataset : dict list {'amplitude', 'duration', 'time', 'V'}
    #resolution : lenght of time vector
    #return: X (batch,1,res), Y (batch,1,res) format torch.float32 (and 1 is the channel)

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
data_train = np.load('../datasets_dict/dataset_train.npz', allow_pickle=True)
data_test = np.load('../datasets_dict/dataset_test.npz', allow_pickle=True)
data_val = np.load('../datasets_dict/dataset_val.npz', allow_pickle=True)

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
folder = "data_tensors"
os.makedirs(folder, exist_ok=True)
torch.save({
    'X_train': X_train,
    'Y_train': Y_train,
    'X_val': X_val,
    'Y_val': Y_val,
    'X_test': X_test,
    'Y_test': Y_test
}, os.path.join(folder, "dataset_fno1d.pt"))

# Check the shapes
print("Train:", X_train.shape, Y_train.shape)
print("Test :", X_test.shape, Y_test.shape)
print("Val  :", X_val.shape, Y_val.shape)



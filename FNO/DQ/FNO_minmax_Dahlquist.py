# FNO for Dahlquist for data_tensors with minmax normalization -> training and validation
# best parameters save in best_paths

from neuralop.models import FNO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import time

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

#batch creation
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_norm, Y_train_norm), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_norm, Y_val_norm), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_norm, Y_test_norm), batch_size=batch_size, shuffle=True)

# FNO model
model = FNO(
    n_modes=(32,),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    #positional_embedding=None,
    n_layers = 4,
)

model.to(device)

# relative L2-loss
def relative_L2_loss(y_pred, y_true) : # y_pred, y_true shape: (batch, ndim, n_points)
    diff = y_true - y_pred
    diff_norm = torch.sqrt(torch.sum(diff**2, dim=-1))      # L2 over the points
    true_norm = torch.sqrt(torch.sum(y_true**2, dim=-1))
    rel_loss = diff_norm / (true_norm + 1e-8)
    return torch.mean(rel_loss) #mean over batch

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-7
)

# Training loop
num_epochs = 1000
best_val_loss = float('inf') # positive infinity
model_path = "../../best_paths/Dahlquist/best_fno_model_minmax_Dahlquist_mode32_hc32.pth"

# Losses lists
train_losses = []
val_losses = []

start_time = time.time()

for epoch in range(num_epochs) :
    model.train()
    train_loss = 0
    total_samples = 0
    for x_batch, y_batch in train_loader :
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        current_batch_size=y_batch.size(0)
        optimizer.zero_grad()
        y_pred=model(x_batch)
        loss=relative_L2_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_samples += current_batch_size
        train_loss += loss.item() * current_batch_size
    train_loss /= total_samples #divide by number of elements in order to obtain the mean for the epoch
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    total_samples = 0
    with torch.no_grad() :
        for x_val_batch, y_val_batch in val_loader :
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            current_batch_size = y_val_batch.size(0)
            y_pred_val = model(x_val_batch)
            total_samples += current_batch_size
            val_loss += relative_L2_loss(y_pred_val, y_val_batch).item() * current_batch_size
    val_loss /= total_samples
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Print the results
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save the best model
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model...")
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)

print("End of the training")
print(f"Best model save {model_path}")
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal time of the training : {elapsed:.2f} secondes")

np.save("train_losses.npy", np.array(train_losses))
np.save("val_losses.npy", np.array(val_losses))

# plot of the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Relative L2 loss')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig("losses_plot.png")
plt.show()
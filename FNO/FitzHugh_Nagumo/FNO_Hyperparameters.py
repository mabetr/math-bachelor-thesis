# FNO_hyperparameters.py
# Hyperparameter search in two steps: 1st = quick scan over the whole set, 2nd = fine-tuned search among the top 10.
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train

# Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataloading
data = torch.load("../../Data/FitzHugh_Nagumo/data_tensors/dataset_fno1d_minmax_red.pt")
X_train, Y_train = data['X_train_norm'], data['Y_train_norm']
X_val, Y_val     = data['X_val_norm'], data['Y_val_norm']

# Use of the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Loss function
def relative_L2_loss(y_pred, y_true):
    diff = y_true - y_pred
    diff_norm = torch.sqrt(torch.sum(diff**2, dim=-1))
    true_norm = torch.sqrt(torch.sum(y_true**2, dim=-1))
    return torch.mean(diff_norm / (true_norm + 1e-8))

#------------------------------------------------------------
# STEP 1: Search in all configs with few epochs
#------------------------------------------------------------
# train_fno
trial_counter = 0

def train_fno(config):
    global trial_counter
    trial_counter += 1
    print(f"Starting trial #{trial_counter} with config: {config}")

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=config["batch_size"], shuffle=False)

    # model
    model = FNO(
        n_modes=(config["modes"],),
        hidden_channels=config["hidden_channels"],
        in_channels=1,
        out_channels=1,
        n_layers=config["n_layers"]
    ).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 100
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = relative_L2_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
        train_loss /= total_samples
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0
        total_samples = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                y_pred_val = model(x_val_batch)
                val_loss += relative_L2_loss(y_pred_val, y_val_batch).item() * y_val_batch.size(0)
                total_samples += y_val_batch.size(0)
        val_loss /= total_samples
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        tune.report({"val_loss": val_loss})


#  Hyperparameter search space
search_space = {
    "modes": tune.grid_search([8, 16, 32, 64]),
    "hidden_channels": tune.grid_search([8, 16, 32, 64, 128]),
    "n_layers": tune.grid_search([3,4,5,6]),
    "batch_size": tune.grid_search([16, 32, 64])
}

import itertools
num_combos = np.prod([len(v) for v in [ [8,16,32,64], [8,16,32,64,128], [3,4,5,6], [16,32,64] ]])
print(f"Total number of combinations to test : {num_combos}")


# Scheduler for the hyperparameter search
scheduler1 = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=20,
    grace_period=5,
    reduction_factor=2
)

# ray analysis
analysis1 = tune.run(
    train_fno,
    config=search_space,
    scheduler=scheduler1,
    resources_per_trial={"cpu": 1, "gpu": 1},
    name="fno1d_grid_stage1"
)

df1 = analysis1.results_df
df1 = df1.rename(columns={
    "config/modes": "modes",
    "config/hidden_channels": "hidden_channels",
    "config/n_layers": "n_layers",
    "config/batch_size": "batch_size"
})
df1["trial_id"] = range(len(df1))
df1.to_csv("gridsearch_stage1.csv", index=False)
np.savez("gridsearch_stage1.npz", df=df1.to_dict('list'))

#------------------------------------------------------------
# STEP 2: FINE TUNING (TOP 10 CONFIGS) WITH ReduceLROnPlateau
#------------------------------------------------------------
# train_fno2 (with scheduler ReduceLROnPlateau)
trial_counter = 0
def train_fno2(config):
    global trial_counter
    trial_counter += 1
    print(f"\nStarting trial #{trial_counter} with config: {config}")

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=config["batch_size"], shuffle=False)

    # Initialize FNO model
    model = FNO(
        n_modes=(config["modes"],),
        hidden_channels=config["hidden_channels"],
        in_channels=1,
        out_channels=1,
        n_layers=config["n_layers"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )

    num_epochs = 100
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        total_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = relative_L2_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
        train_loss /= total_samples

        # Validation
        model.eval()
        val_loss = 0
        total_samples = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                y_pred_val = model(x_val_batch)
                val_loss += relative_L2_loss(y_pred_val, y_val_batch).item() * y_val_batch.size(0)
                total_samples += y_val_batch.size(0)
        val_loss /= total_samples

        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

#top 10 of the best configs
top_k = 10
top_configs = df1.nsmallest(top_k, "val_loss")[["modes", "hidden_channels", "n_layers", "batch_size"]].to_dict(orient="records")

search_space2 = tune.grid_search(top_configs)

analysis2 = tune.run(
    train_fno2,
    config=search_space2,
    resources_per_trial={"cpu": 1, "gpu": 1},
    name="fno1d_grid_stage2"
)

df2 = analysis2.results_df
df2 = df2.rename(columns={
    "config/modes": "modes",
    "config/hidden_channels": "hidden_channels",
    "config/n_layers": "n_layers",
    "config/batch_size": "batch_size",
})
df2["trial_id"] = range(len(df2))
df2.to_csv("gridsearch_stage2.csv", index=False)
np.savez("gridsearch_stage2.npz", df=df2.to_dict('list'))

best_cfg = analysis2.get_best_config(metric="val_loss", mode="min")
np.savez("best_config_final.npz", **best_cfg)
print("Best configuration found:")
print(best_cfg)

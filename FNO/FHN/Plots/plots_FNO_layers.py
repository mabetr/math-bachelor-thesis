# Plots of some representative examples for the variation of the number of Fourier layers
import torch
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# General parameters
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices_to_plot = [0, 21, 297, 348]
plot_labels = ["t0", "gen", "nap", "tfin"]

# dataload
data = torch.load("../../../Data/FitzHugh_Nagumo/data_tensors/dataset_fno1d_minmax_red.pt", map_location=device)
X_test_norm, Y_test_norm = data['X_test_norm'], data['Y_test_norm']

# normalization parameters
normalization_param = torch.load("../../../Data/FitzHugh_Nagumo/data_tensors/normalization_minmax_stats.pt", map_location=device)
X_min, X_max = normalization_param["X_min"].to(device), normalization_param["X_max"].to(device)
Y_min, Y_max = normalization_param["Y_min"].to(device), normalization_param["Y_max"].to(device)

# Samples for visualization
x_visu = X_test_norm[indices_to_plot].to(device)
y_visu = Y_test_norm[indices_to_plot].to(device)
x_visu_denorm = X_min + x_visu * (X_max - X_min)
y_visu_denorm = Y_min + y_visu * (Y_max - Y_min)

# temporal axis
n_points_new = x_visu.shape[-1]
t_max = 100.0
t = np.linspace(0, t_max, n_points_new)

# -----------------------
# List of models to compare
# -----------------------
layer_list = [1, 2, 3, 4, 5, 6]
model_paths = [
    f"../../../best_paths/best_fno_model_minmax_mode32_hc64_layers{m}_bz16_seed71.pth" for m in layer_list
]

from neuralop.models import FNO

# -----------------------
# Figure
# -----------------------
n_models = len(layer_list)
n_rows = n_models + 1
n_cols = len(indices_to_plot)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows +1), sharex=True)
handles = []
labels = []

# 1st row : inputs
for col_idx, label in enumerate(plot_labels):
    ax = axes[0, col_idx]
    x_sample = x_visu_denorm[col_idx, 0, :].cpu().numpy()
    ax.plot(t, x_sample, color="black", label="Input function")
    ax.set_title(f"{label}", fontsize=12)
    ax.set_xlabel("t")
    ax.set_ylabel("$I_{app(t)}$")
    ax.grid(True)


# outputs
for row_idx, (m, model_path) in enumerate(zip(layer_list, model_paths), start=1):
    # model
    model = FNO(
        n_modes=(32,),
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        n_layers=m,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Predictions
    with torch.no_grad():
        y_pred = model(x_visu)
    y_pred_denorm = Y_min + y_pred * (Y_max - Y_min)

    # Plot for each input
    for col_idx, label in enumerate(plot_labels):
        ax = axes[row_idx, col_idx]
        y_true_sample = y_visu_denorm[col_idx, 0, :].cpu().numpy()
        y_pred_sample = y_pred_denorm[col_idx, 0, :].cpu().numpy()

        ax.plot(t, y_true_sample, "--", color="blue", label="True solution")
        ax.plot(t, y_pred_sample, color="orange", label="Prediction")

        ax.set_xlim(0, t_max)
        ax.set_xlabel("t")
        if col_idx == 0:
            ax.set_ylabel(r"$\mathbf{n\_layers = " + str(m) + "}$", fontsize=14)
        else:
            ax.set_ylabel("V(t)")
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
    fontsize=12
)

plt.tight_layout(rect=[0, 0.015, 1, 1])
plt.subplots_adjust(hspace=0.4)
plt.show()
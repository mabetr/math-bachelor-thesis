import torch
import numpy as np
import os
from datetime import date
from neuralop.models import FNO1d

# --- PARAMETERS ---
latex_file = "FNO_testset_minmax_Dahlquist_plots.tex"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_points_new = 500  # nombre de points pour les plots
t_max = 100.0  # temps max pour l'axe x

# --- LOAD DATASETS ---
data = torch.load("../../Data/Dahlquist/data_tensors_Dahlquist/dataset_fno1d_minmax.pt")
X_test_norm, Y_test_norm = data['X_test_norm'], data['Y_test_norm']

# --- LOAD MODEL ---
modes = 32
hidden_channels = 32
best_model = FNO1d(n_modes_height=modes, hidden_channels=hidden_channels,
                   in_channels=1, out_channels=1).to(device)
best_model.load_state_dict(torch.load('../../best_paths/trials_fno1d/best_fno1d_model_minmax_Dahlquist.pth', map_location=device, weights_only=False))
best_model.eval()

# --- LOAD MIN-MAX STATS ---
normalization_param = torch.load("../../Data/Dahlquist/data_tensors_Dahlquist/normalization_minmax_stats.pt", map_location=device)
X_min, X_max = normalization_param["X_min"].to(device), normalization_param["X_max"].to(device)
Y_min, Y_max = normalization_param["Y_min"].to(device), normalization_param["Y_max"].to(device)

# --- TIME VECTOR ---
t_old = np.linspace(0, t_max, X_test_norm.shape[-1])
indices = np.linspace(0, len(t_old) - 1, n_points_new).astype(int)
t = t_old[indices]

# --- GENERATE LATEX FILE ---
with open(latex_file, "w", encoding="utf-8") as f:
    # Préambule
    f.write(r"\documentclass[a4paper,12pt]{article}" + "\n")
    f.write(r"\usepackage{pgfplots}" + "\n")
    f.write(r"\pgfplotsset{compat=1.18}" + "\n")
    f.write(r"\usepgfplotslibrary{groupplots}" + "\n")
    f.write(r"\usepackage[margin=2cm]{geometry}" + "\n")
    f.write(r"\usepackage{fancyhdr}" + "\n")
    f.write(r"\pagestyle{fancy}" + "\n")
    f.write(r"\fancyhf{}" + "\n")
    f.write(r"\rfoot{\thepage}" + "\n")
    f.write(r"\begin{document}" + "\n\n")

    # Page de titre
    f.write(r"\begin{titlepage}" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"{\Huge FNO for Data_generation with minmax normalization - test set visualisation \par}" + "\n")
    f.write(r"\vspace{2cm}" + "\n")
    f.write(r"{\Large Marine Betrisey \par}" + "\n")
    f.write(r"\vspace{1cm}" + "\n")
    f.write("{" + date.today().strftime("%B %d, %Y") + "}\n")
    f.write(r"\end{titlepage}" + "\n\n")


# --- LOOP OVER TEST SET ---
    for i in range(X_test_norm.shape[0]):
        x_sample = X_test_norm[i:i + 1].to(device)
        y_true = Y_test_norm[i:i + 1].to(device)
        with torch.no_grad():
            y_pred = best_model(x_sample)

        # Denormalization
        x_denorm = X_min + x_sample * (X_max - X_min)
        y_true_denorm = Y_min + y_true * (Y_max - Y_min)
        y_pred_denorm = Y_min + y_pred * (Y_max - Y_min)

        # points selection for the plot
        x_plot = x_denorm[0, 0, indices].cpu().numpy()
        y_true_plot = y_true_denorm[0, 0, indices].cpu().numpy()
        y_pred_plot = y_pred_denorm[0, 0, indices].cpu().numpy()

        # --- FIGURE ---
        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tikzpicture}" + "\n")
        f.write(
            r"\begin{groupplot}[group style={group size=1 by 2, vertical sep=1.5cm}, width=\textwidth, height=0.35\textheight]" + "\n")

        # Input plot
        f.write(r"\nextgroupplot[ylabel={Input}, xlabel={$t$ [ms]}, grid=major]" + "\n")
        f.write(r"\addplot[black, very thin] coordinates {" + "\n")
        for tt, xx in zip(t, x_plot):
            f.write(f"({tt},{xx}) ")
        f.write("};\n")

        # Output plot
        f.write(r"\nextgroupplot[ylabel={y(t)}, xlabel={$t$ [ms]}, grid=major]" + "\n")
        f.write(r"\addplot[dashed, blue] coordinates {" + "\n")
        for tt, yy in zip(t, y_true_plot):
            f.write(f"({tt},{yy}) ")
        f.write("};\n")
        f.write(r"\addlegendentry{True solution}" + "\n")
        f.write(r"\addplot[solid, red] coordinates {" + "\n")
        for tt, yy in zip(t, y_pred_plot):
            f.write(f"({tt},{yy}) ")
        f.write("};\n")
        f.write(r"\addlegendentry{Predicted solution}" + "\n")

        f.write(r"\end{groupplot}" + "\n")
        f.write(r"\end{tikzpicture}" + "\n")
        f.write(r"\caption{Test sample ID: " + str(i) + "}" + "\n")
        f.write(r"\end{figure}" + "\n\n")
        f.write(r"\clearpage" + "\n")

    f.write(r"\end{document}" + "\n")

print(f"LaTeX file generated: {latex_file}")
#Plots for the perf of FNO with variable number of Fourier modes
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# data
modes = [1, 2, 4, 8, 16, 32, 64, 128]
test_error = [0.06745, 0.024194, 0.020276, 0.013339, 0.018282, 0.015124, 0.017256, 0.019191]
time_min = [22.00, 22.20, 22.05, 22.52, 22.15, 22.98, 22.16, 22.69]
parameters = [79649, 100129, 120609, 161569, 243489, 407329, 735009, 1390369]

# Subplot creation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

xticks = modes

def format_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.grid(True, which="both")

# (a) Test Error
axes[0].plot(modes, test_error, marker='o')
format_axis(axes[0])
axes[0].set_ylim(0.01, 0.085)
axes[0].set_xlabel('Fourier Modes')
axes[0].set_ylabel('Test Error')
axes[0].text(0.5, -0.2, '(a) Test Error', ha='center', va='top', transform=axes[0].transAxes)

# (b) Time in minutes
axes[1].plot(modes, time_min, marker='o', color='orange')
format_axis(axes[1])
axes[1].set_ylim(7, 45)
axes[1].set_xlabel('Fourier Modes')
axes[1].set_ylabel('Time (min)')
axes[1].text(0.5, -0.2, '(b) Time in min', ha='center', va='top', transform=axes[1].transAxes)

# (c) Number of Trainable Parameters
axes[2].plot(modes, parameters, marker='o', color='green')
format_axis(axes[2])
axes[2].set_ylim(6000, 1700000)
axes[2].set_xlabel('Fourier Modes')
axes[2].set_ylabel('Number of Trainable Parameters')
axes[2].text(0.5, -0.2, '(c) Number of Trainable Parameters', ha='center', va='top', transform=axes[2].transAxes)

plt.tight_layout()
plt.show()
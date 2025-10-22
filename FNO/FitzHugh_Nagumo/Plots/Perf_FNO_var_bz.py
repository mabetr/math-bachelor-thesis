#Plots for the perf of FNO with variable batch size
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Data
batch = [8, 16, 32, 64, 128]
test_error = [0.015369, 0.015124, 0.014885, 0.014988, 0.01661]
time_min = [43.83, 22.98, 12.44, 9.71, 9.27]
parameters = [407329, 407329, 407329, 407329, 407329]

# Subplots creation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

xticks = batch

def format_axis(ax):
    ax.set_xscale('log', base=2)  # log scale
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.grid(True, which="both")

# (a) Test Error
axes[0].plot(batch, test_error, marker='o')
format_axis(axes[0])
axes[0].set_ylim(0.01, 0.085)
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Test Error')
axes[0].text(0.5, -0.2, '(a) Test Error', ha='center', va='top', transform=axes[0].transAxes)

# (b) Time in minutes
axes[1].plot(batch, time_min, marker='o', color='orange')
format_axis(axes[1])
axes[1].set_ylim(7, 45)
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Time (min)')
axes[1].text(0.5, -0.2, '(b) Time in min', ha='center', va='top', transform=axes[1].transAxes)

# (c) Number of Trainable Parameters
axes[2].plot(batch, parameters, marker='o', color='green')
format_axis(axes[2])
axes[2].set_ylim(6000, 1700000)
axes[2].set_xlabel('Batch Size')
axes[2].set_ylabel('Number of Trainable Parameters')
axes[2].text(0.5, -0.2, '(c) Number of Trainable Parameters', ha='center', va='top', transform=axes[2].transAxes)

plt.tight_layout()
plt.show()
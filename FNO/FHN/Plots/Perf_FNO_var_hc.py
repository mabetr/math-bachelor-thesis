#Plots for the perf of FNO with variable hidden chanels
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Data
hc = [8, 16, 32, 64, 128]
test_error = [0.030795, 0.038922, 0.017542, 0.015124, 0.019994]
time_min = [22.48, 22.41, 22.45, 22.98, 28.16]
parameters = [6565, 25801, 102289, 407329, 1625665]

# Subplots creation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

xticks = hc

def format_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.grid(True, which="both")

# (a) Test Error
axes[0].plot(hc, test_error, marker='o')
format_axis(axes[0])
axes[0].set_ylim(0.01, 0.085)
axes[0].set_xlabel('Hidden Channels (hc)')
axes[0].set_ylabel('Test Error')
axes[0].text(0.5, -0.2, '(a) Test Error', ha='center', va='top', transform=axes[0].transAxes)

# (b) Time in minutes
axes[1].plot(hc, time_min, marker='o', color='orange')
format_axis(axes[1])
axes[1].set_ylim(7, 45)
axes[1].set_xlabel('Hidden Channels (hc)')
axes[1].set_ylabel('Time (min)')
axes[1].text(0.5, -0.2, '(b) Time in min', ha='center', va='top', transform=axes[1].transAxes)

# (c) Number of Trainable Parameters
axes[2].plot(hc, parameters, marker='o', color='green')
format_axis(axes[2])
axes[2].set_ylim(6000, 1700000)
axes[2].set_xlabel('Hidden Channels (hc)')
axes[2].set_ylabel('Number of Trainable Parameters')
axes[2].text(0.5, -0.2, '(c) Number of Trainable Parameters', ha='center', va='top', transform=axes[2].transAxes)

plt.tight_layout()
plt.show()
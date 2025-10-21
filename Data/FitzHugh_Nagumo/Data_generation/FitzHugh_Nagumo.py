# FitzHugh_Nagumo.py
# fhn_system

import numpy as np

def I_app(t, stim_amplitude, T_stim):
    return stim_amplitude if t <= T_stim else 0.0

def fhn_system(t, y, stim_amplitude, T_stim):
    V, w = y
    I = I_app(t, stim_amplitude, T_stim)

    dVdt = 5 * V * (V - 0.1) * (1 - V) - w + I
    dwdt = V - 0.25 * w
    return [dVdt, dwdt]
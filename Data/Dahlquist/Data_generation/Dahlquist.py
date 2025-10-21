# Data_generation for Dahlquist equation with lambda=20
import numpy as np

def I_app(t, stim_amplitude, T_stim):
    return stim_amplitude if t <= T_stim else 0.0

def Dahlquist_ODE(t, y, stim_amplitude, T_stim):
    S = I_app(t, stim_amplitude, T_stim)
    dydt = -20*y[0] + S
    return np.array([dydt])
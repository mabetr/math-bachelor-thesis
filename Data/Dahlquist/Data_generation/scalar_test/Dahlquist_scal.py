#Data_generation with lambda=20 and scalar stimulation
import numpy as np

def Dahlquist_ODE(t, y, S):
    dydt = -20*y[0] + S
    return np.array([dydt])
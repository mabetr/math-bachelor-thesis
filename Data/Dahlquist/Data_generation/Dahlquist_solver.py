#Data_generation solver
import numpy as np
from scipy.integrate import solve_ivp
from Dahlquist import I_app, Dahlquist_ODE

def solve_Dahlquist_radau(T, dt, stim_amplitude, T_stim):
    t_eval = np.linspace(0, T, int(T/dt) + 1)
    y0 = np.array([0.0])

    # solve_ivp call
    sol = solve_ivp(
        Dahlquist_ODE, (0, T), y0,
        method='Radau',
        t_eval=t_eval,
        args=(stim_amplitude, T_stim),
        rtol=1e-6, atol=1e-8
    )
    return sol.t, sol.y[0]
# solve_fhn_radau.py
import numpy as np
from scipy.integrate import solve_ivp
from FitzHugh_Nagumo import I_app, fhn_system

def solve_fhn_radau(T, dt, stim_amplitude, T_stim):
    t_eval = np.linspace(0, T, int(T/dt) + 1)
    y0 = [0.0, 0.0] # initial conditions for V and w

    # solve_ivp call
    sol = solve_ivp(
        fhn_system, (0, T), y0,
        method='Radau',
        t_eval=t_eval,
        args=(stim_amplitude, T_stim),
        rtol=1e-6, atol=1e-8 # Solver tolerances
    )
    return sol.t, sol.y[0], sol.y[1]
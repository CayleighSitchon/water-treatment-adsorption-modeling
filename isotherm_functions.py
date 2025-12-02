# isotherm_functions.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

# Isotherm functions
def langmuir_isotherm(C, q_max, b):
    """Langmuir isotherm q*(C) = q_max * b C / (1 + b C)"""
    Cpos = np.maximum(C, 0.0)
    return (q_max * b * Cpos) / (1.0 + b * Cpos)

def freundlich_isotherm(C, Kf, n):
    """Freundlich isotherm q*(C) = Kf * C^(1/n)"""
    Cpos = np.maximum(C, 0.0)
    return Kf * (Cpos ** (1.0 / n))


# ODE system
def adsorption_odes(t, y, params, isotherm='langmuir'):
    """
    Returns [dC/dt, dq/dt] for the kinetically-limited batch model.
    State y = [C (mg/L), q (mg/g)]
    params: dict with keys 'V', 'm', 'k_ads', and isotherm params
    """
    C, q = y
    V = params['V']
    m = params['m']
    k_ads = params['k_ads']
    
    # compute equilibrium q_star per chosen isotherm
    if isotherm == 'langmuir':
        q_star = langmuir_isotherm(C, params['q_max'], params['b'])
    elif isotherm == 'freundlich':
        q_star = freundlich_isotherm(C, params['Kf'], params['n'])
    else:
        raise ValueError("isotherm must be 'langmuir' or 'freundlich'")
    
    dqdt = k_ads * (q_star - q)
    dCdt = - (m / V) * dqdt
    return [dCdt, dqdt]


def run_simulation(params, C0=50.0, q0=0.0, t_span=(0,200), t_eval=None, isotherm='langmuir', solver='BDF'):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    y0 = [C0, q0]
    sol = solve_ivp(adsorption_odes, t_span, y0, args=(params, isotherm),
                    t_eval=t_eval, method=solver, rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + str(sol.message))
    return sol
# Will run the examples and save results on file
def example_and_save():
    params = {
        'V': 1.0,         # L
        'm': 1.0,         # g
        'k_ads': 0.1,     # 1/min
        # Langmuir
        'q_max': 200.0,   # mg/g
        'b': 0.05,        # L/mg
        # Freundlich
        'Kf': 30.0,
        'n': 2.0
    }
    C0 = 50.0  # mg/L
    q0 = 0.0
    t_span = (0, 200)
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Langmuir
    solL = run_simulation(params, C0=C0, q0=q0, t_span=t_span, t_eval=t_eval, isotherm='langmuir')
    C_final_L = solL.y[0, -1]
    removal_pct_L = 100.0 * (C0 - C_final_L) / C0

    # Freundlich
    solF = run_simulation(params, C0=C0, q0=q0, t_span=t_span, t_eval=t_eval, isotherm='freundlich')
    C_final_F = solF.y[0, -1]
    removal_pct_F = 100.0 * (C0 - C_final_F) / C0
    print(f"Final removal (Langmuir)  : {removal_pct_L:.2f} % (C_final = {C_final_L:.3f} mg/L)")
    print(f"Final removal (Freundlich): {removal_pct_F:.2f} % (C_final = {C_final_F:.3f} mg/L)")


    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(solL.t, solL.y[0], label='C (mg/L)')
    plt.plot(solL.t, solL.y[1], label='q (mg/g)')
    plt.title('Langmuir')
    plt.xlabel('Time (min)'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(solF.t, solF.y[0], label='C (mg/L)')
    plt.plot(solF.t, solF.y[1], label='q (mg/g)')
    plt.title('Freundlich')
    plt.xlabel('Time (min)'); plt.legend()
plt.tight_layout()
    plt.show()


    out_df = pd.DataFrame([{
        'isotherm': 'Langmuir',
        'C0': C0, 'C_final': float(C_final_L), 'removal_pct': float(removal_pct_L),
        **params
    }, {
        'isotherm': 'Freundlich',
        'C0': C0, 'C_final': float(C_final_F), 'removal_pct': float(removal_pct_F),
        **params
    }])
    out_file = os.path.join(os.getcwd(), 'isotherm_example_results.csv')
    out_df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")

if __name__ == '__main__':
    example_and_save()


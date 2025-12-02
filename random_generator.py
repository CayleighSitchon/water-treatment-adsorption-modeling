import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from isotherm_functions import adsorption_odes

def generate_synthetic_dataset(num_samples=500, isotherm='langmuir'):
    results = []

    for _ in range(num_samples):
        params = {
            'V': 1.0,
            'm': np.random.uniform(0.5, 5.0),
            'k_ads': np.random.uniform(0.01, 0.2),
            'q_max': 200.0,
            'b': np.random.uniform(0.01, 0.1),
            'Kf': np.random.uniform(10.0, 50.0),
            'n': np.random.uniform(1.0, 3.0)
        }

        C0 = np.random.uniform(10, 100)
        q0 = 0.0
        y0 = [C0, q0]
        t_span = (0, 200)
        t_eval = np.linspace(t_span[0], t_span[1], 500)

        sol = solve_ivp(adsorption_odes, t_span, y0, args=(params, isotherm),
                        t_eval=t_eval, method='BDF')

        if not sol.success:
            print("ODE solver failed for sample, skipping...")
            continue

        C_final = sol.y[0, -1]
        removal_pct = 100 * (C0 - C_final) / C0

        data_point = {
            'C0': C0,
            'm': params['m'],
            'k_ads': params['k_ads'],
            'removal_pct': removal_pct
        }

        if isotherm == 'langmuir':
            data_point.update({'q_max': params['q_max'], 'b': params['b']})
        elif isotherm == 'freundlich':
            data_point.update({'Kf': params['Kf'], 'n': params['n']})

        results.append(data_point)

    df = pd.DataFrame(results)
    return df

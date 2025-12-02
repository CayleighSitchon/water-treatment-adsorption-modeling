import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from isotherm_functions import adsorption_odes

params = {
    'V': 1.0,
    'm': 1.0,
    'k_ads': 0.1,
    'q_max': 200.0,
    'b': 0.05,
    'Kf': 30.0,
    'n': 2.0}
C0 = 50.0
q0 = 0.0
y0 = [C0, q0]
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 500)
sotherm_choice = 'langmuir'


sol = solve_ivp(adsorption_odes, t_span, y0, args=(params, isotherm_choice), t_eval=t_eval, method='BDF')
df_results = pd.DataFrame({
    'time_min': sol.t,
    'C_mg_per_L': sol.y[0],
    'q_mg_per_g': sol.y[1]})


output_filename = f'adsorption_simulation_{isotherm_choice}.csv'
df_results.to_csv(output_filename, index=False)
print(f"Simulation results saved to {output_filename}")

plt.plot(sol.t, sol.y[0], label='C (mg/L)')
plt.plot(sol.t, sol.y[1], label='q (mg/g)')
plt.xlabel('Time (min)')
plt.ylabel('Concentration / Adsorbed Amount')
plt.title(f'Batch Adsorption Simulation ({isotherm_choice.title()})')
plt.legend()


plt.show()


from random_generator import generate_synthetic_dataset
if __name__ == '__main__':
    # Generates Langmuir
    df_langmuir = generate_synthetic_dataset(num_samples=500, isotherm='langmuir')
    df_langmuir.to_csv('synthetic_dataset_langmuir.csv', index=False)
    print("Langmuir synthetic dataset saved to synthetic_dataset_langmuir.csv")
    # Generates Freundlich
    df_freundlich = generate_synthetic_dataset(num_samples=500, isotherm='freundlich')
    df_freundlich.to_csv('synthetic_dataset_freundlich.csv', index=False)
    print("Freundlich synthetic dataset saved to synthetic_dataset_freundlich.csv")


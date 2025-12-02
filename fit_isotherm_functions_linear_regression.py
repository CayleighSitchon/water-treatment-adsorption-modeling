
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Isotherm models ---
def langmuir_q(C, q_max, b):
    return (q_max * b * C) / (1 + b * C)

def freundlich_q(C, Kf, n):
    return Kf * (C ** (1/n))

def calc_metrics(y_true, y_pred, num_params):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    aic = n * np.log(rss / n) + 2 * num_params
    return r2, rmse, aic

def fit_dataset(filepath, V=1.0):
    print(f"\n=== Processing: {filepath} ===")
    df = pd.read_csv(filepath)

    # Compute equilibrium concentration Ce and adsorbed amount q_eq
    df['C'] = df['C0'] * (1 - df['removal_pct'] / 100.0)
    df['q_eq'] = (df['C0'] - df['C']) * V / df['m']

    # Remove bad data
    mask = np.isfinite(df['C']) & np.isfinite(df['q_eq']) & (df['q_eq'] > 0)
    C_data = df.loc[mask, 'C'].values
    q_eq = df.loc[mask, 'q_eq'].values

    # Langmuir fit
    try:
        p0_L = [np.max(q_eq), 0.1]  # initial guess
        popt_L, _ = curve_fit(
            langmuir_q, C_data, q_eq,
            bounds=(0, np.inf), p0=p0_L, maxfev=20000
        )
        q_pred_L = langmuir_q(C_data, *popt_L)
        r2_L, rmse_L, aic_L = calc_metrics(q_eq, q_pred_L, 2)
        print(f"Langmuir fit: q_max={popt_L[0]:.4f}, b={popt_L[1]:.4f}, R²={r2_L:.4f}, RMSE={rmse_L:.4f}, AIC={aic_L:.2f}")
    except Exception as e:
        print("Langmuir fit failed:", e)
        popt_L = None

    # Freundlich fit
    try:
        p0_F = [1, 1]  # initial guess
        popt_F, _ = curve_fit(
            freundlich_q, C_data, q_eq,
            bounds=(0, np.inf), p0=p0_F, maxfev=20000
        )
        q_pred_F = freundlich_q(C_data, *popt_F)
        r2_F, rmse_F, aic_F = calc_metrics(q_eq, q_pred_F, 2)
        print(f"Freundlich fit: Kf={popt_F[0]:.4f}, n={popt_F[1]:.4f}, R²={r2_F:.4f}, RMSE={rmse_F:.4f}, AIC={aic_F:.2f}")
    except Exception as e:
        print("Freundlich fit failed:", e)
        popt_F = None

    # Plot data and fits
    plt.figure(figsize=(8,6))
    plt.scatter(C_data, q_eq, label='Data', alpha=0.6)
    C_fit = np.linspace(min(C_data), max(C_data), 200)
    if popt_L is not None:
        plt.plot(C_fit, langmuir_q(C_fit, *popt_L), 'r-', label='Langmuir fit')
    if popt_F is not None:
        plt.plot(C_fit, freundlich_q(C_fit, *popt_F), 'g--', label='Freundlich fit')
    plt.xlabel('Equilibrium Concentration, C (mg/L)')
    plt.ylabel('Equilibrium Adsorbed Amount, q_eq (mg/g)')
    plt.title(f'Isotherm Fits for {filepath}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    datasets = [
        'synthetic_dataset_langmuir.csv',
        'synthetic_dataset_freundlich.csv'
    ]
    for ds in datasets:
        try:
            fit_dataset(ds)
        except Exception as e:
            print(f"Error processing {ds}: {e}")


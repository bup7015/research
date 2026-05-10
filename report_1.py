# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =========================
# 1. データ生成
# =========================
n_samples = 10000
seed = 42

x_config = [
    ["X01_Ethylene_Feed", 4500, 5500], ["X02_Oxygen_Feed", 2200, 2800],
    ["X03_Methane_Ballast", 9000, 11000], ["X04_Ethylene_Purity", 99.5, 100.0],
    ["X05_Oxygen_Purity", 99.0, 100.0], ["X06_Inlet_Temp", 220, 250],
    ["X07_Inlet_Pres", 1.8, 2.2], ["X08_Cat_Selectivity", 80, 90],
    ["X09_Inhibitor_Conc", 2, 8], ["X10_Coolant_Flow", 13000, 17000],
    ["X11_Recycle_Flow", 35000, 45000], ["X12_Purge_Ratio", 0.5, 2.0],
    ["X13_CO2_Abs_Flow", 150, 200], ["X14_CO2_Abs_Temp", 35, 45],
    ["X15_EO_Abs_Water_Temp", 30, 40], ["X16_EO_Abs_L_V", 1.0, 1.5],
    ["X17_Stripper_Duty", 4.0, 6.0], ["X18_Reflux_Ratio", 2.0, 3.5],
    ["X19_Dist_Pres", 130, 170], ["X20_Cat_Volume", 45, 55],
    ["X21_HEX_UA", 400, 600]
]

sampler = qmc.LatinHypercube(d=len(x_config), seed=seed)
sample_raw = sampler.random(n=n_samples)
x_data = [sample_raw[:, i] * (v[2] - v[1]) + v[1] for i, v in enumerate(x_config)]
df = pd.DataFrame(np.array(x_data).T, columns=[v[0] for v in x_config])

# トレードオフ設計
conv = 5 + 0.3*(df['X06_Inlet_Temp']-220) + 0.0002*(df['X11_Recycle_Flow']-35000) + 2.0*np.sqrt(df['X20_Cat_Volume']-44)
cost = 14000 + 0.8*(df['X06_Inlet_Temp']-220)**2.5 + 0.2*(df['X11_Recycle_Flow']-35000) + 15*(df['X20_Cat_Volume'] - 45)**2 + df['X17_Stripper_Duty']*100

df['Y3_Ethylene_Conversion'] = conv.clip(0, 30) + np.random.normal(0, conv.std()*0.002, n_samples)
df['Y5_Total_Utility_Cost'] = cost + np.random.normal(0, cost.std()*0.002, n_samples)
df['Y5_Neg_Cost'] = -df['Y5_Total_Utility_Cost']

# =========================
# 2. ベイズ最適化
# =========================
X_cols = [c for c in df.columns if c.startswith('X')]

initial = df.sample(20, random_state=42)
pool = df.drop(initial.index).sample(5000, random_state=42)

X_train = initial[X_cols].values
y_train = initial[['Y3_Ethylene_Conversion', 'Y5_Neg_Cost']].values

X_search = pool[X_cols].values
y_search = pool[['Y3_Ethylene_Conversion', 'Y5_Neg_Cost']].values

n_initial = X_train.shape[0]

n_iter = 50
acq_method = 'std_ei_sum'  # 'pi_log_sum' or 'std_ei_sum'
w_conv = 10.0
w_cost = 1.0

for i in range(n_iter):
    sc_x = StandardScaler()
    X_t_scaled = sc_x.fit_transform(X_train)
    X_s_scaled = sc_x.transform(X_search)

    combined_acq = np.zeros(X_search.shape[0])

    if acq_method == 'pi_log_sum':
        for j in range(y_train.shape[1]):
            sc_y = StandardScaler()
            y_t_scaled = sc_y.fit_transform(y_train[:, j].reshape(-1, 1)).flatten()

            gp = GaussianProcessRegressor(
                kernel=ConstantKernel()*RBF()+WhiteKernel(),
                alpha=1e-2
            )
            gp.fit(X_t_scaled, y_t_scaled)

            mu, std = gp.predict(X_s_scaled, return_std=True)
            best = np.max(y_t_scaled)

            Z = (mu - best) / np.maximum(std, 1e-9)
            pi = np.clip(norm.cdf(Z), 1e-10, 1.0)

            weight = w_conv if j == 0 else w_cost
            combined_acq += weight * np.log(pi)

    elif acq_method == 'std_ei_sum':
        for j in range(y_train.shape[1]):
            sc_y = StandardScaler()
            y_t_scaled = sc_y.fit_transform(y_train[:, j].reshape(-1, 1)).flatten()

            gp = GaussianProcessRegressor(
                kernel=ConstantKernel()*RBF()+WhiteKernel(),
                alpha=1e-2
            )
            gp.fit(X_t_scaled, y_t_scaled)

            mu_s, std_s = gp.predict(X_s_scaled, return_std=True)
            best = np.max(y_t_scaled)

            Z_s  = (mu_s - best) / np.maximum(std_s, 1e-9)
            ei_s = (mu_s - best) * norm.cdf(Z_s) + std_s * norm.pdf(Z_s)

            mu_t, std_t = gp.predict(X_t_scaled, return_std=True)
            Z_t  = (mu_t - best) / np.maximum(std_t, 1e-9)
            ei_t = (mu_t - best) * norm.cdf(Z_t) + std_t * norm.pdf(Z_t)

            std_ei = np.std(ei_t)
            if std_ei < 1e-9:
                ei_s_std = ei_s - np.mean(ei_t)
            else:
                ei_s_std = (ei_s - np.mean(ei_t)) / std_ei

            weight = w_conv if j == 0 else w_cost
            combined_acq += weight * ei_s_std

    else:
        raise ValueError("acq_method must be 'pi_log_sum' or 'std_ei_sum'")

    idx = np.argmax(combined_acq)

    X_train = np.vstack([X_train, X_search[idx]])
    y_train = np.vstack([y_train, y_search[idx]])

    X_search = np.delete(X_search, idx, axis=0)
    y_search = np.delete(y_search, idx, axis=0)

    print(f"Iter {i+1}/{n_iter} completed. ({acq_method})")

# =========================
# 3. 可視化
# =========================
def find_pareto(df, maximize_mask):
    vals = df.values
    is_pareto = np.ones(vals.shape[0], dtype=bool)
    for i, v in enumerate(vals):
        better = np.all((vals >= v) if maximize_mask else (vals <= v), axis=1) & \
                 np.any((vals > v) if maximize_mask else (vals < v), axis=1)
        if np.any(better):
            is_pareto[i] = False
    return np.where(is_pareto)[0]

res_df = pd.DataFrame(y_train, columns=['Conv', 'NegCost'])
res_df['Cost'] = -res_df['NegCost']

p_idx = find_pareto(res_df[['Conv', 'Cost']], maximize_mask=[True, False])

ideal_conv = df['Y3_Ethylene_Conversion'].max()
ideal_cost = df['Y5_Total_Utility_Cost'].min()

plt.figure(figsize=(6, 5))
plt.rcParams.update({'font.size': 15})
plt.scatter(df['Y3_Ethylene_Conversion'],
            df['Y5_Total_Utility_Cost'],
            c='silver', alpha=0.1,
          )

plt.scatter(res_df.iloc[:n_initial]['Conv'],
            res_df.iloc[:n_initial]['Cost'],
            c='blue', label='Initial')

plt.scatter(res_df.iloc[n_initial:]['Conv'],
            res_df.iloc[n_initial:]['Cost'],
            c='red', marker='x', label='BO Proposed')

plt.scatter(res_df.iloc[p_idx]['Conv'],
            res_df.iloc[p_idx]['Cost'],
            c='black', s=40, label='Pareto')

plt.scatter(ideal_conv, ideal_cost,
            c='green', marker='*', s=200, label='Ideal Target')

plt.xlabel('Conversion (%)')
plt.ylabel('Cost (USD/h)')
plt.title(f'BO Results: {acq_method} (w_cost={w_cost})')

plt.gca().invert_yaxis()

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:58:35 2026

@author: hikosakatatsuya
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# ===== Rastrigin関数 =====
def rastrigin(x):
    n = x.shape[1]
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1))

# ===== 全候補生成 =====
n_samples = 1000000
n_features = 5
lower_bound = -5.12
upper_bound = 5.12

X_all = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
y_all = rastrigin(X_all)

# ===== D最適設計 =====
n_select = 30
n_trials = 1000

best_det = -np.inf
best_indices = None

for _ in range(n_trials):
    idx = np.random.choice(n_samples, n_select, replace=False)
    X_sub = X_all[idx]

    X_design = np.hstack([np.ones((n_select, 1)), X_sub])
    XtX = X_design.T @ X_design

    sign, logdet = np.linalg.slogdet(XtX)
    if sign > 0 and logdet > best_det:
        best_det = logdet
        best_indices = idx

# ===== 初期データ =====
X = X_all[best_indices]
y = y_all[best_indices]

print("初期 best y:", y.max())

# ===== 候補集合 =====
mask = np.ones(n_samples, dtype=bool)
mask[best_indices] = False
X_candidate = X_all[mask]

# ===== GPR（安定化のみ追加：問題設定は変更していない）=====
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
         Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + \
         WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

# ===== BO設定 =====
max_iter = 100
target_value = -28

best_y_history = [y.max()]

# ===== BOループ =====
for iteration in range(max_iter):

    # ===== スケーリング =====
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0)
    x_std[x_std == 0] = 1

    y_mean = y.mean()
    y_std  = y.std()
    if y_std == 0:
        y_std = 1

    X_scaled = (X - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    Xc_scaled = (X_candidate - x_mean) / x_std

    # ===== GPR =====
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
    model.fit(X_scaled, y_scaled)

    # ===== 予測 =====
    mu, sigma = model.predict(Xc_scaled, return_std=True)
    mu = mu * y_std + y_mean
    sigma = sigma * y_std

    sigma[sigma < 1e-10] = 1e-10  # 安定化

    # ===== PTR（上限なし・正しい形）=====
    acquisition = 1 - norm.cdf(target_value, loc=mu, scale=sigma)
    acquisition[sigma <= 0] = 0

    # ===== 次サンプル =====
    next_idx = np.argmax(acquisition)
    x_next = X_candidate[next_idx].reshape(1, -1)
    y_next = rastrigin(x_next)

    # ===== 更新 =====
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)

    X_candidate = np.delete(X_candidate, next_idx, axis=0)

    current_best = y.max()
    best_y_history.append(current_best)

    # ===== 出力 =====
    print(f"Iteration {iteration+1}")
    print("y_next:", y_next[0])
    print("current best y:", current_best)
    print("----------------------")

    # ===== 終了条件（厳守）=====
    if current_best > target_value:
        print("目標達成！")
        break

print("探索終了")
print("最終 best y:", y.max())

# ===== 可視化 =====
plt.figure()
plt.plot(best_y_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Best y so far")
plt.title("BO Progress (PTR, target stopping)")
plt.grid()
plt.show()
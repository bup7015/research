# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dcekit.generative_model import GMR
from dcekit.variable_selection import cvpfi_gmr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- グラフの全体設定 ---
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 30,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.figsize': [18, 12]
})

# 1. データ準備
try:
    df_all = pd.read_csv('EO_Process_Data_Ordered.csv')
except Exception as e:
    print("CSVファイルが見つかりません。")

X_cols = [c for c in df_all.columns if c.startswith('X')]
Y_cols = ['Y3_Ethylene_Conversion', 'Y5_Total_Utility_Cost']
n_initial = 30
n_iter = 50

train_df = df_all.sample(n_initial, random_state=0)
search_pool = df_all.drop(train_df.index)
X_train = train_df[X_cols].values
Y_train = train_df[Y_cols].values
target_Y = np.array([df_all[Y_cols[0]].max(), df_all[Y_cols[1]].min()])

# 2. GMR最適化（モデル構築）
print("--- モデル最適化を開始 ---")
for i in range(n_iter):
    current_data = np.c_[X_train, Y_train]
    scaler = StandardScaler()
    autoscaled_data = scaler.fit_transform(current_data)
    
    # 複雑すぎるモデルによる過学習を防ぐためコンポーネント数は5に設定
    model = GMR(n_components=5, covariance_type='full', rep='mode')
    model.fit(autoscaled_data)
    
    target_Y_scaled = (target_Y - scaler.mean_[-2:]) / scaler.scale_[-2:]
    indices_x = list(range(len(X_cols)))
    indices_y = [len(X_cols), len(X_cols) + 1]
    
    estimated_x_scaled = model.predict_rep(target_Y_scaled.reshape(1, -1), indices_y, indices_x)
    X_proposed = estimated_x_scaled * scaler.scale_[:len(X_cols)] + scaler.mean_[:len(X_cols)]
    
    dists = np.linalg.norm(search_pool[X_cols].values - X_proposed, axis=1)
    idx_nearest = np.argmin(dists)
    
    X_train = np.vstack([X_train, search_pool.iloc[idx_nearest][X_cols].values])
    Y_train = np.vstack([Y_train, search_pool.iloc[idx_nearest][Y_cols].values])
    search_pool = search_pool.drop(search_pool.index[idx_nearest])

# 3. 重要度評価と描画ロジック（RMSEベースに修正）
def evaluate_and_plot_rmse(target_idx, title_name):
    print(f"[{title_name}] の重要度を評価中 (RMSE Increase)...")
    
    current_data_final = np.c_[X_train, Y_train]
    scaler_final = StandardScaler()
    autoscaled_df = pd.DataFrame(scaler_final.fit_transform(current_data_final))
    
    # 指標を 'r2' から 'rmse' に変更し、5-foldで安定的に計算
    imp_mean, imp_std, _ = cvpfi_gmr(
        model, autoscaled_df, indices_x, [target_idx], 
        fold_number=5, scoring='rmse', n_repeats=5, alpha_r=0.999, random_state=0
    )
    
    imp_df = pd.DataFrame({
        'Feature': X_cols,
        'Importance': imp_mean,
        'Std': imp_std
    }).sort_values(by='Importance', ascending=False).head(10)

    # グラフ描画
    plt.figure(figsize=(18, 12)) 
    sns.barplot(data=imp_df, x='Feature', y='Importance', palette='coolwarm', edgecolor='black')
    
    plt.errorbar(x=range(len(imp_df)), y=imp_df['Importance'], yerr=imp_df['Std'], 
                 fmt='none', c='black', capsize=12, elinewidth=3)
    
    # 基準となる「0」のライン
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    
    plt.title(f'Top 10 Variable Importance: {title_name}', pad=30)
    plt.xlabel('Process Parameters (X)', labelpad=20)
    # ラベルを RMSE Increase に変更
    plt.ylabel('Importance (RMSE Increase)', labelpad=20)
    
    # バーが確実に見えるように、最大値の1.2倍をY軸の上限にする
    max_val = imp_df['Importance'].max()
    if max_val > 0:
        plt.ylim(min(0, imp_df['Importance'].min() * 1.1), max_val * 1.2)
    
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    plt.show()

# 4. 出力実行
print("--- グラフ描画を開始 ---")

# 探索軌跡
plt.figure(figsize=(12, 10))
plt.scatter(df_all[Y_cols[0]], df_all[Y_cols[1]], c='silver', alpha=0.1)
plt.scatter(Y_train[:n_initial, 0], Y_train[:n_initial, 1], c='blue', label='Initial', s=120)
plt.scatter(Y_train[n_initial:, 0], Y_train[n_initial:, 1], c='red', marker='x', label='Proposed', s=120)
plt.scatter(target_Y[0], target_Y[1], c='green', marker='*', s=700, label='Target', edgecolors='black')
plt.xlabel('Conversion (%)')
plt.ylabel('Utility Cost (USD/h)')
plt.title('Objective Space: Optimization Path')
plt.gca().invert_yaxis()
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.show()

# 個別プロット（2つ）
evaluate_and_plot_rmse(len(X_cols), "Ethylene Conversion (Y3)")
evaluate_and_plot_rmse(len(X_cols) + 1, "Utility Cost (Y5)")
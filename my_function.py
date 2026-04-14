#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sample_functions
import warnings
from dcekit.variable_selection import cvpfi
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.model_selection import KFold
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 18})



# =========================
# 可視化系
# =========================

def plot_hist(dataset, bins=None, fontsize=None):
    """
    各変数ごとのヒストグラムを表示する
    """
    if fontsize is not None:
        plt.rcParams['font.size'] = fontsize

    for i in range(dataset.shape[1]):
        x = dataset.iloc[:, i]

        if bins is None:
            b = int(np.round(np.sqrt(len(x))))
        else:
            b = bins

        plt.hist(x, bins=b)
        plt.xlabel(dataset.columns[i])
        plt.ylabel("frequency")
        plt.show()


def plot_box(dataset, fontsize=None):
    """
    全変数の箱ひげ図をまとめて表示する
    """
    if fontsize is not None:
        plt.rcParams['font.size'] = fontsize

    n = dataset.shape[1]
    plt.figure(figsize=(n, n))
    sns.boxplot(data=dataset)

    plt.xticks(fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)
    plt.show()


def plot_box_by_category(dataset, x, y, fontsize=None):
    """
    カテゴリごとの箱ひげ図を表示する（x:カテゴリ, y:数値）
    """
    if fontsize is not None:
        plt.rcParams['font.size'] = fontsize

    n = dataset.shape[1]
    plt.figure(figsize=(n, n))
    sns.boxplot(data=dataset, x=x, y=y)
    plt.show()


def plot_scatter(dataset, var1, var2, figsize=None, fontsize=None):
    """
    2変数の散布図と相関係数を表示する
    """
    if fontsize is not None:
        plt.rcParams['font.size'] = fontsize

    if figsize is not None:
        plt.figure(figsize=figsize)

    x = dataset.iloc[:, var1]
    y = dataset.iloc[:, var2]

    r = np.corrcoef(x, y)[0, 1]

    plt.scatter(x, y)
    plt.xlabel(dataset.columns[var1])
    plt.ylabel(dataset.columns[var2])
    plt.title(f"相関係数 = {r:.3g}")
    plt.show()


def plot_corr_heatmap(dataset, fontsize=None):
    """
    相関係数行列を計算しヒートマップとして表示・保存する
    """
    if fontsize is not None:
        plt.rcParams['font.size'] = fontsize
    n = dataset.shape[1]
    plt.figure(figsize=(n, n))  
    corr = dataset.corr(numeric_only=True)
    corr.to_csv('correlation_coefficients.csv')

    sns.heatmap(corr, cmap='seismic', annot=True, fmt='.2f',
                vmin=-1, vmax=1, square=True)
    plt.show()

    return corr


# =========================
# PCA
# =========================

def analyze_pca(dataset, n_components=None):
    """
    PCAを実行し、モデル・スケーラー・結果を辞書で返す。
    """
    autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()

    pca = PCA(n_components=n_components)
    pca.fit(autoscaled_dataset)

    pc_names = [f'PC{i+1}' for i in range(pca.n_components_)]

    results = {
        "model": pca,
        "score": pd.DataFrame(
            pca.transform(autoscaled_dataset),
            index=dataset.index,
            columns=pc_names
        ),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=dataset.columns,
            columns=pc_names
        ),
        "contribution": pd.DataFrame({
            'ratio': pca.explained_variance_ratio_,
            'cumulative': pca.explained_variance_ratio_.cumsum()
        }, index=pc_names)
    }

    return results


def visualize_pca(pca_results, pc_x=1, pc_y=2):
    """
    スコアプロットと寄与率プロットを表示する
    """
    score = pca_results['score']
    cont = pca_results['contribution']

    # --- スコア ---
    plt.figure(figsize=(6, 5))
    plt.scatter(score.iloc[:, pc_x-1],
                score.iloc[:, pc_y-1],
                c='blue', alpha=0.5,
                edgecolors='white', s=50)

    plt.xlabel(f'PC{pc_x}')
    plt.ylabel(f'PC{pc_y}')
    plt.title('PCA Score Plot')
    plt.grid(True)
    plt.show()

    # --- 寄与率 ---
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x_axis = range(1, len(cont) + 1)

    ax1.bar(x_axis, cont['ratio'])
    ax1.set_xlabel('PC')
    ax1.set_ylabel('ratio')

    ax2 = ax1.twinx()
    ax2.plot(x_axis, cont['cumulative'], 'ro-')
    ax2.set_ylabel('cumulative')

    plt.title('Scree Plot')
    plt.show()


def run_pca_pipeline(dataset, index_col=0,
                     n_components=None,
                     pc_x=1, pc_y=2,
                     save=True):
    """
    PCAの計算・可視化・保存を一括で実行する
    """
    results = analyze_pca(dataset, n_components=n_components)

    visualize_pca(results, pc_x=pc_x, pc_y=pc_y)

    if save:
        results['score'].to_csv('pca_score.csv')
        results['loadings'].to_csv('pca_loadings.csv')
        results['contribution'].to_csv('pca_contribution_metrics.csv')

    return results


# =========================
# t-SNE
# =========================

def analyze_tsne_with_k3n(dataset,
                          k_in_k3n_error=10,
                          candidates_of_perplexity=None,
                          random_state=10):
    """
    k3n-error による perplexity 最適化と t-SNE
    """
    if candidates_of_perplexity is None:
        candidates_of_perplexity = np.arange(5, 105, 5)

    autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()

    k3n_errors = []
    for index, perplexity in enumerate(candidates_of_perplexity):
        print(index + 1, '/', len(candidates_of_perplexity))
        t = TSNE(perplexity=perplexity,
                 n_components=2,
                 init='pca',
                 random_state=random_state).fit_transform(autoscaled_dataset)

        scaled_t = (t - t.mean(0)) / t.std(0, ddof=1)

        k3n_errors.append(
              sample_functions.k3n_error(autoscaled_dataset, scaled_t, k_in_k3n_error)
            + sample_functions.k3n_error(scaled_t, autoscaled_dataset, k_in_k3n_error)
        )

    optimal = candidates_of_perplexity[np.argmin(k3n_errors)]

    print('optimal perplexity:', optimal)

    t = TSNE(perplexity=optimal,
             n_components=2,
             init='pca',
             random_state=random_state).fit_transform(autoscaled_dataset)

    t = pd.DataFrame(t, index=dataset.index,columns=['t1', 't2'])

    return {
        "t": t,
        "k3n_errors": k3n_errors,
        "perplexities": candidates_of_perplexity,
        "optimal_perplexity": optimal
    }


def visualize_tsne_results(results):
    """ 
    k3n-errorとt-SNE結果を可視化する
    """
    plt.scatter(results["perplexities"],results["k3n_errors"])
    plt.xlabel("perplexity")
    plt.ylabel("k3n-error")
    plt.show()

    t = results["t"]

    plt.scatter(t.iloc[:, 0], t.iloc[:, 1],c="blue")
    plt.xlabel('t1')
    plt.ylabel('t2')
    plt.show()


def run_tsne_pipeline(dataset, save=True):
    """
    t-SNEの計算・可視化・保存を一括で実行する
　　"""
    results = analyze_tsne_with_k3n(dataset)

    visualize_tsne_results(results)

    if save:
        results["t"].to_csv('tsne_t.csv')

    return results



def double_cross_validation_regression(
    run_pipeline_func,
    x,
    y,
    model_type,
    outer_fold_number=10,
    fold_number=5):
    
    # ここに追加！！
    if model_type.endswith("_ensemble"):
        raise ValueError("DCVではensembleは使えません")

    estimated_y = pd.Series(index=y.index, dtype=float)

    kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=0)

    for fold, (train_idx, test_idx) in enumerate(kf.split(x), 1):
        print(f'\n===== Outer Fold {fold} / {outer_fold_number} =====')

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # autoscaling（回帰のみyもスケーリング）
        ms_x, ss_x = x_train.mean(), x_train.std()
        ss_x[ss_x == 0] = 1.0

        autoscaled_x_train = (x_train - ms_x) / ss_x
        autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
        autoscaled_x_test  = (x_test - ms_x) / ss_x

        results = run_pipeline_func(
            model_type=model_type,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            autoscaled_x_train=autoscaled_x_train,
            autoscaled_y_train=autoscaled_y_train,
            autoscaled_x_test=autoscaled_x_test,
            fold_number=fold_number
        )

        model = results["model"]

        # 予測（逆スケーリング）
        y_pred = model.predict(autoscaled_x_test).ravel()
        y_pred = y_pred * y_train.std() + y_train.mean()

        estimated_y.iloc[test_idx] = y_pred

    return estimated_y

# =========================
# 評価
# =========================
def evaluate_dcv(y, estimated_y):

    # =========================
    # 指標計算
    # =========================
    r2 = 1 - sum((y - estimated_y) ** 2) / sum((y - y.mean()) ** 2)
    rmse = (sum((y - estimated_y) ** 2) / len(y)) ** 0.5
    mae = sum(abs(y - estimated_y)) / len(y)

    print(f'r2dcv: {float(r2)}')
    print(f'RMSEdcv: {float(rmse)}')
    print(f'MAEdcv: {float(mae)}')

    # =========================
    # yy-plot
    # =========================
    plt.figure(figsize=(6, 6))

    plt.scatter(y, estimated_y)

    # 範囲設定
    y_max = np.max(np.array([np.array(y), estimated_y]))
    y_min = np.min(np.array([np.array(y), estimated_y]))

    margin = 0.05 * (y_max - y_min)

    # 理想線（y=x）
    plt.plot(
        [y_min - margin, y_max + margin],
        [y_min - margin, y_max + margin],
        'k-'
    )

    plt.ylim(y_min - margin, y_max + margin)
    plt.xlim(y_min - margin, y_max + margin)

    plt.xlabel('Actual Y')
    plt.ylabel('Estimated Y in DCV')
    plt.title('YY-plot (DCV)')
    plt.grid(True)

    plt.show()


# =========================
# 共通前処理
# =========================

def prepare_regression_data(dataset, number_of_test_samples, target_column_index=0,add_non_linear_flag=False,
                            random_number=0):

    y = dataset.iloc[:, target_column_index]
    x = dataset.drop(dataset.columns[target_column_index], axis=1)
    
    y = dataset.iloc[:, target_column_index]
    original_x = dataset.drop(dataset.columns[target_column_index], axis=1)
    if add_non_linear_flag:
        # 説明変数の二乗項や交差項を追加
        x = original_x.copy()  # 元の説明変数のデータセット
        x_square = original_x ** 2  # 二乗項
        # 追加
        for i in range(original_x.shape[1]):
            for j in range(original_x.shape[1]):
                if i == j:  # 二乗項
                    x = pd.concat(
                        [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],
                        axis=1)
                elif i < j:  # 交差項
                    x = pd.concat([x, original_x.iloc[:, i] * original_x.iloc[:, j]], axis=1)
                    x = x.rename(columns={0: '{0}*{1}'.format(x_square.columns[i], x_square.columns[j])})

    # 分割
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=number_of_test_samples,shuffle=True,
                                                        random_state=random_number)

    # オートスケーリング
    std = x_train.std()
    std[std == 0] = 1

    autoscaled_x_train = (x_train - x_train.mean()) / std
    autoscaled_x_test  = (x_test - x_train.mean()) / std
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
  

    return x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_y_train, autoscaled_x_test

def prepare_classification_data(dataset, number_of_test_samples,
                                target_column_index=0,
                                random_number=0):

    # 目的変数・説明変数
    y = dataset.iloc[:, target_column_index]
    x = dataset.drop(dataset.columns[target_column_index], axis=1)

    # stratifyでクラス比維持
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=number_of_test_samples,
        shuffle=True,
        random_state=random_number,
        stratify=y)

    # オートスケーリング（xのみ）
    autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
    autoscaled_x_test  = (x_test - x_train.mean()) / x_train.std()

    return x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_x_test

def prepare_regression_data_for_dcv(dataset, target_column_index=0):
    y = dataset.iloc[:, target_column_index]
    x = dataset.drop(dataset.columns[target_column_index], axis=1)
    return x, y



def calculate_ad_knn(autoscaled_x_train,autoscaled_x_test,k_in_knn=5,
                     rate_of_training_samples_inside_ad=0.8,save_csv=None):

    # =========================
    # モデル構築
    # =========================
    ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
    ad_model.fit(autoscaled_x_train)

    # =========================
    # train の距離
    # =========================
    knn_distance_train, _ = ad_model.kneighbors(autoscaled_x_train,n_neighbors=k_in_knn + 1)
    knn_distance_train = pd.DataFrame(knn_distance_train, index=autoscaled_x_train.index)
    mean_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                       columns=['mean_of_knn_distance'])

    # =========================
    # 閾値決定
    # =========================
    sorted_dist = mean_distance_train.iloc[:, 0].sort_values()
    ad_threshold = sorted_dist.iloc[round(len(sorted_dist) * rate_of_training_samples_inside_ad) - 1]

    # =========================
    # train 判定
    # =========================
    inside_ad_flag_train = (mean_distance_train <= ad_threshold)
    inside_ad_flag_train.columns = ['inside_ad_flag']

    # =========================
    # test の距離
    # =========================
    knn_distance_test, _ = ad_model.kneighbors(autoscaled_x_test)
    knn_distance_test = pd.DataFrame(knn_distance_test, index=autoscaled_x_test.index)
    mean_distance_test = pd.DataFrame(knn_distance_test.mean(axis=1),columns=['mean_of_knn_distance'])

    # =========================
    # test 判定
    # =========================
    inside_ad_flag_test = (mean_distance_test <= ad_threshold)
    inside_ad_flag_test.columns = ['inside_ad_flag']

    # =========================
    # 保存（任意）
    # =========================
    if save_csv:
        mean_distance_train.to_csv('ad_mean_distance_train.csv')
        mean_distance_test.to_csv('ad_mean_distance_test.csv')
        inside_ad_flag_train.to_csv('ad_inside_flag_train.csv')
        inside_ad_flag_test.to_csv('ad_inside_flag_test.csv')

    # =========================
    # まとめて返す
    # =========================
    return {
        "ad_threshold": ad_threshold,
        "mean_distance_train": mean_distance_train,
        "mean_distance_test": mean_distance_test,
        "inside_ad_flag_train": inside_ad_flag_train,
        "inside_ad_flag_test": inside_ad_flag_test
    }

def calculate_ad_ocsvm(autoscaled_x_train, autoscaled_x_test,
                      ocsvm_nu=0.045,
                      ocsvm_gammas=2 ** np.arange(-20, 11, dtype=float),
                      save_csv=None):

    from scipy.spatial.distance import cdist

    # =========================
    # γ最適化（グラム行列分散最大化）
    # =========================
    variance_of_gram_matrix = list()
    for ocsvm_gamma in ocsvm_gammas:
        gram_matrix = np.exp(
            -ocsvm_gamma * cdist(autoscaled_x_train, autoscaled_x_train, metric='sqeuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_ocsvm_gamma = ocsvm_gammas[np.argmax(variance_of_gram_matrix)]

    print('最適化された gamma :', optimal_ocsvm_gamma)

    # =========================
    # モデル構築
    # =========================
    ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)
    ad_model.fit(autoscaled_x_train)

    # =========================
    # train のデータ密度
    # =========================
    data_density_train = ad_model.decision_function(autoscaled_x_train)
    data_density_train = pd.DataFrame(data_density_train, index=autoscaled_x_train.index, 
                                      columns=['ocsvm_data_density'])
    number_of_support_vectors = len(ad_model.support_)
    number_of_outliers_in_training_data = (data_density_train < 0).sum().iloc[0]
    print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
    print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / autoscaled_x_train.shape[0])
    print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
    print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / autoscaled_x_train.shape[0])
   
    inside_ad_flag_train = data_density_train >= 0
    inside_ad_flag_train.columns = ['inside_ad_flag']

    # =========================
    # test のデータ密度
    # =========================
    data_density_test = ad_model.decision_function(autoscaled_x_test)
    data_density_test = pd.DataFrame(
        data_density_test,
        index=autoscaled_x_test.index,
        columns=['ocsvm_data_density']
    )
    number_of_outliers_in_test_data = (data_density_test < 0).sum().iloc[0]
    print('\nテストデータにおける外れサンプル数 :', number_of_outliers_in_test_data)
    print('テストデータにおける外れサンプルの割合 :', number_of_outliers_in_test_data / autoscaled_x_test.shape[0])
    
    inside_ad_flag_test = data_density_test >= 0
    inside_ad_flag_test.columns = ['inside_ad_flag']

    # =========================
    # 保存
    # =========================
    if save_csv:
        data_density_train.to_csv('ocsvm_data_density_train.csv')
        data_density_test.to_csv('ocsvm_data_density_test.csv')
        inside_ad_flag_train.to_csv('ocsvm_inside_flag_train.csv')
        inside_ad_flag_test.to_csv('ocsvm_inside_flag_test.csv')

    # =========================
    # まとめて返す
    # =========================
    return {
        "ocsvm_gamma": optimal_ocsvm_gamma,
        "data_density_train": data_density_train,
        "data_density_test": data_density_test,
        "inside_ad_flag_train": inside_ad_flag_train,
        "inside_ad_flag_test": inside_ad_flag_test
    }
def calculate_ad(method="knn",
                 autoscaled_x_train=None,
                 autoscaled_x_test=None,
                 k_in_knn=5,
                 rate_of_training_samples_inside_ad=0.8,
                 ocsvm_nu=0.045,
                 ocsvm_gammas=2 ** np.arange(-20, 11, dtype=float),
                 save_csv=None):

    # =========================
    # kNN
    # =========================
    if method == "knn":

        return calculate_ad_knn(
            autoscaled_x_train=autoscaled_x_train,
            autoscaled_x_test=autoscaled_x_test,
            k_in_knn=k_in_knn,
            rate_of_training_samples_inside_ad=rate_of_training_samples_inside_ad,
            save_csv=save_csv
        )

    # =========================
    # OCSVM
    # =========================
    elif method == "ocsvm":

        return calculate_ad_ocsvm(
            autoscaled_x_train=autoscaled_x_train,
            autoscaled_x_test=autoscaled_x_test,
            ocsvm_nu=ocsvm_nu,
            ocsvm_gammas=ocsvm_gammas,
            save_csv=save_csv
        )

    # =========================
    # error
    # =========================
    else:
        raise ValueError(f"Unknown AD method: {method}")
        
def evaluate_subset(y_true, y_pred, name=""):
    r2 = 1 - sum((y_true - y_pred) ** 2) / sum((y_true - y_true.mean()) ** 2)
    rmse = (sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5
    mae = sum(abs(y_true - y_pred)) / len(y_true)

    print(f"\n=== {name} ===")
    print(f"n_samples: {len(y_true)}")
    print(f"r2   : {float(r2)}")
    print(f"RMSE : {float(rmse)}")
    print(f"MAE  : {float(mae)}")
    
def run_ensemble_regression_pipeline(
        base_model_type,
        x_train, x_test, y_train, y_test,
        autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
        fold_number,
        number_of_submodels=3,
        rate_of_selected_x_variables=0.7
    ):

    number_of_x_variables = int(np.ceil(x_train.shape[1] * rate_of_selected_x_variables))

    submodels = []
    selected_x_variable_numbers = []
    estimated_y_test_all = pd.DataFrame(index=x_test.index)

    for i in range(number_of_submodels):
        print(f"{i+1}/{number_of_submodels}")

        # =========================
        # 変数ランダム選択（元コード互換）
        # =========================
        random_x_variables = np.random.rand(x_train.shape[1])
        selected_idx = random_x_variables.argsort()[:number_of_x_variables]

        selected_x_variable_numbers.append(selected_idx)

        x_tr = x_train.iloc[:, selected_idx]
        x_te = x_test.iloc[:, selected_idx]

        auto_x_tr = autoscaled_x_train.iloc[:, selected_idx]
        auto_x_te = autoscaled_x_test.iloc[:, selected_idx]

        # =========================
        # モデル構築
        # =========================
        result = run_regression_pipeline(
        model_type=base_model_type,
        x_train=x_tr,
        x_test=x_te,
        y_train=y_train,
        y_test=y_test,
        autoscaled_x_train=auto_x_tr,
        autoscaled_y_train=autoscaled_y_train,
        autoscaled_x_test=auto_x_te,
        fold_number=fold_number,
        run_cvpfi_flag=False)

        model = result["model"]

        submodels.append(model)

        # =========================
        # 予測（スケール戻し）
        # =========================
        y_pred_scaled = model.predict(auto_x_te).ravel()
        y_pred = y_pred_scaled * y_train.std() + y_train.mean()

        estimated_y_test_all[f"model_{i}"] = y_pred

    # =========================
    # 集約（元コード完全一致）
    # =========================
    y_mean = pd.Series(
        estimated_y_test_all.median(axis=1),
        index=x_test.index)

    y_std = pd.Series(
        estimated_y_test_all.std(axis=1),
        index=x_test.index)

    # =========================
    # 可視化
    # =========================
    plt.rcParams['font.size'] = 18

    plt.scatter(y_std, abs(y_test - y_mean), c='blue')
    plt.xlabel('std. of estimated y')
    plt.ylabel('absolute error of y')
    plt.show()

    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "submodels": submodels,
        "selected_x_variable_numbers": selected_x_variable_numbers
    }

# =========================
# PLS
# =========================

def run_pls_pipeline(x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_y_train, 
                     autoscaled_x_test,fold_number,run_cvpfi_flag=True):
    max_number_of_principal_components = min(x_train.shape[0],x_train.shape[1], 20)

    def optimize_and_fit_pls(autoscaled_x_train, autoscaled_y_train, y_train,
                             max_number_of_principal_components):
        components = []
        r2_in_cv_all = []

        for component in range(1, max_number_of_principal_components + 1):
            model = PLSRegression(n_components=component)
            
            # CVによる予測
            estimated_y_in_cv_scaled = cross_val_predict(model, autoscaled_x_train,
                                                         autoscaled_y_train, cv=fold_number)
            
            # 逆スケーリング
            estimated_y_in_cv = estimated_y_in_cv_scaled * y_train.std() + y_train.mean()
            r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)

            r2_in_cv_all.append(r2_in_cv)
            components.append(component)

        optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(components, r2_in_cv_all,
                                                                                         'number of components',
                                                                                         'cross-validated r2')
        print('\nCV で最適化された成分数 :', optimal_component_number)
        model = PLSRegression(n_components=optimal_component_number)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        return model

    def save_and_plot_standard_regression_coefficients(model, x_train):

        # ※標準回帰係数
        standard_regression_coefficients = pd.DataFrame(
                                            model.coef_,
                                            index=x_train.columns,
                                            columns=['standard_regression_coefficients'])

        standard_regression_coefficients.to_csv('pls_standard_regression_coefficients.csv')

        # 絶対値ベースで降順（解釈性向上）
        standard_regression_coefficients_sorted = standard_regression_coefficients.sort_values(
                                                    by='standard_regression_coefficients',
                                                    key=lambda x: x.abs(),
                                                    ascending=False)

        plt.figure()
        standard_regression_coefficients_sorted.plot.bar(legend=False)
        plt.ylabel('standard coef')
        plt.title('PLS')
        plt.tight_layout()
        plt.show()

        return standard_regression_coefficients

    # 最適化と学習
    model = optimize_and_fit_pls(autoscaled_x_train, autoscaled_y_train, y_train,
                                 max_number_of_principal_components)

    # 係数の保存とプロット
    standard_regression_coefficients = save_and_plot_standard_regression_coefficients(model, x_train)

    # 性能チェック
    sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                                   autoscaled_x_test, y_test)
    # ★これを追加
    cvpfi_df = None
    if run_cvpfi_flag:
        cvpfi_df=run_cvpfi_analysis(
                        model,
                        autoscaled_x_train,
                        autoscaled_y_train,
                        fold_number=fold_number,
                        model_name='PLS',
                        alpha_r=0.95)

    return model, cvpfi_df,standard_regression_coefficients


# =========================
# SVR
# =========================

def run_svr_pipeline(x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_y_train, 
                     autoscaled_x_test,fold_number,run_cvpfi_flag=True):
    svr_cs       = 2 ** np.arange(-5, 11, dtype=float)
    svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)
    svr_gammas   = 2 ** np.arange(-20, 11, dtype=float)

    def optimize_and_fit_svr(autoscaled_x_train, autoscaled_y_train, svr_cs, svr_epsilons, svr_gammas):
        # 従来の最適化ロジック
        optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svr_gammas)

        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma),
                                  {'epsilon': svr_epsilons}, cv=fold_number)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_svr_epsilon = model_in_cv.best_params_['epsilon']

        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                  {'C': svr_cs}, cv=fold_number)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_svr_c = model_in_cv.best_params_['C']

        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                  {'gamma': svr_gammas}, cv=fold_number)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_svr_gamma = model_in_cv.best_params_['gamma']
        
        print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))

        model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        return model

    # 最適化と学習
    model = optimize_and_fit_svr(autoscaled_x_train, autoscaled_y_train, svr_cs, svr_epsilons, svr_gammas)

    # 性能チェック
    sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                                   autoscaled_x_test, y_test)
    
    cvpfi_df = None
    if run_cvpfi_flag:
        cvpfi_df=run_cvpfi_analysis(
            model,
            autoscaled_x_train,
            autoscaled_y_train,
            fold_number=fold_number,
            model_name='SVR',
            alpha_r=0.95)


    return model,cvpfi_df

# =========================
# kNN
# =========================

def run_knn_pipeline(x_train, x_test, y_train, y_test,
                    autoscaled_x_train, autoscaled_x_test,
                    fold_number,
                    max_number_of_k=20):

    def optimize_and_fit_knn(autoscaled_x_train, y_train, max_number_of_k):

        accuracy_in_cv_all = []
        ks = []

        for k in range(1, max_number_of_k + 1):
            model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            estimated_y_in_cv = cross_val_predict(model,autoscaled_x_train,y_train, cv=fold_number)
            accuracy = metrics.accuracy_score(y_train, estimated_y_in_cv)
            print(k, accuracy)

            accuracy_in_cv_all.append(accuracy)
            ks.append(k)

        optimal_k = sample_functions.plot_and_selection_of_hyperparameter(ks, accuracy_in_cv_all, 'k',
                                                                          'cross-validated accuracy')
        print('\nCV で最適化された k :', optimal_k)
        model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
        model.fit(autoscaled_x_train, y_train)

        return model

    # 最適化＋学習
    model = optimize_and_fit_knn(autoscaled_x_train, y_train, max_number_of_k)

    # 性能評価（既存関数をそのまま使う）
    sample_functions.estimation_and_performance_check_in_classification_train_and_test(
        model,
        autoscaled_x_train, y_train,
        autoscaled_x_test, y_test
    )

    return model


# =========================
# SVM
# =========================
def run_svc_pipeline(x_train, x_test, y_train, y_test,
                     autoscaled_x_train, autoscaled_x_test,
                     fold_number):

    svm_cs = 2 ** np.arange(-5, 11, dtype=float)
    svm_gammas = 2 ** np.arange(-20, 11, dtype=float)

    # =========================
    # Step1: γ初期値（分散最大化）
    # =========================
    optimal_gamma = sample_functions.gamma_optimization_with_variance(
        autoscaled_x_train, svm_gammas
    )

    # =========================
    # Step2: C最適化
    # =========================
    model_in_cv = GridSearchCV(
        svm.SVC(kernel='rbf', gamma=optimal_gamma),
        {'C': svm_cs},
        cv=fold_number
    )
    model_in_cv.fit(autoscaled_x_train, y_train)
    optimal_c = model_in_cv.best_params_['C']

    # =========================
    # Step3: γ再最適化
    # =========================
    model_in_cv = GridSearchCV(
        svm.SVC(kernel='rbf', C=optimal_c),
        {'gamma': svm_gammas},
        cv=fold_number
    )
    model_in_cv.fit(autoscaled_x_train, y_train)
    optimal_gamma = model_in_cv.best_params_['gamma']

    print('CV で最適化された C :', optimal_c)
    print('CV で最適化された γ:', optimal_gamma)

    # =========================
    # モデル構築
    # =========================
    model = svm.SVC(kernel='rbf', C=optimal_c, gamma=optimal_gamma)
    model.fit(autoscaled_x_train, y_train)

    # =========================
    # 性能評価
    # =========================
    sample_functions.estimation_and_performance_check_in_classification_train_and_test(
        model,
        autoscaled_x_train, y_train,
        autoscaled_x_test, y_test
    )

    return model

# =========================
# Decision Tree Classifier
# =========================
def run_dt_pipeline(x_train, x_test, y_train, y_test,autoscaled_x_train, autoscaled_x_test,
                    fold_number,max_max_depth=10,min_samples_leaf=3):

    # =========================
    # 深さの最適化
    # =========================
    accuracy_cv = []
    max_depthes = []
    for max_depth in range(1, max_max_depth):
        model_in_cv = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
        estimated_y_in_cv = cross_val_predict(model_in_cv, x_train, y_train, cv=fold_number)
        accuracy_cv.append(metrics.accuracy_score(y_train, estimated_y_in_cv))
        max_depthes.append(max_depth)

    optimal_max_depth = sample_functions.plot_and_selection_of_hyperparameter(max_depthes, accuracy_cv,
                                                                              'maximum depth of tree', 'accuracy in CV')
    print('\nCV で最適化された木の深さ :', optimal_max_depth)

    # =========================
    # モデル構築
    # =========================
    model = tree.DecisionTreeClassifier(
        max_depth=optimal_max_depth,
        min_samples_leaf=min_samples_leaf
    )
    model.fit(x_train, y_train)

    # =========================
    # 性能評価
    # =========================
    sample_functions.estimation_and_performance_check_in_classification_train_and_test(model,x_train, y_train,
                                                                                       x_test, y_test)

    # =========================
    # 可視化
    # =========================
    plt.figure(figsize=(12, 10))
    tree.plot_tree(model,feature_names=x_train.columns,class_names=[str(c) for c in model.classes_],
        max_depth=3,filled=True,rounded=True,impurity=False)
    plt.show()

    return model

# =========================
# Random Forest　Classifier
# =========================
def run_rf_pipeline(x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_x_test, fold_number,
                    rf_number_of_trees=300):

    rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10
    # OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
    accuracy_oob = []
    for index, x_variables_rate in enumerate(rf_x_variables_rates):
        print(index + 1, '/', len(rf_x_variables_rates))
        model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees,
            max_features=int(max(math.ceil(x_train.shape[1] * x_variables_rate), 1)),
            oob_score=True, n_jobs=-1)
        model_in_validation.fit(x_train, y_train)
        accuracy_oob.append(model_in_validation.oob_score_)

    optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(
        rf_x_variables_rates, accuracy_oob, 'rate of x-variables', 'accuracy for OOB')
    print('\nOOB で最適化された説明変数の数の割合 :', optimal_x_variables_rate)
    # RF
    model = RandomForestClassifier(n_estimators=rf_number_of_trees,
        max_features=int(max(math.ceil(x_train.shape[1] * optimal_x_variables_rate), 1)),
        oob_score=True, n_jobs=-1)
    model.fit(x_train, y_train)

    x_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=['importance'])
    x_importances.to_csv('rf_x_importances.csv')
    x_importances = x_importances.sort_values('importance', ascending=False)

    plt.figure()
    sns.barplot(x=x_importances.index, y=x_importances['importance'])
    plt.xticks(rotation=90)
    plt.show()

    sample_functions.estimation_and_performance_check_in_classification_train_and_test(
        model, x_train, y_train, x_test, y_test)

    return model

def run_regression_pipeline(
        model_type,
        x_train, x_test, y_train, y_test,
        autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
        fold_number,
        run_cvpfi_flag=True
    ):

    # =========================
    # アンサンブル分岐
    # =========================
    if model_type.endswith("_ensemble"):
        base = model_type.replace("_ensemble", "")

        ensemble_results = run_ensemble_regression_pipeline(
            base,
            x_train, x_test, y_train, y_test,
            autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
            fold_number
        )

        return {
            "model": ensemble_results["submodels"],  # 複数モデル
            "estimated_y_test": ensemble_results["y_mean"],
            "y_std": ensemble_results["y_std"],
            "cvpfi": None,
            "type": "ensemble"
        }

    # =========================
    # 通常モデル
    # =========================
    model_dict = {
        "pls": run_pls_pipeline,
        "svr": run_svr_pipeline
    }

    if model_type not in model_dict:
        raise ValueError(f"[Regression] Unknown model_type: {model_type}")

    results = model_dict[model_type](
        x_train, x_test, y_train, y_test,
        autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
        fold_number,
        run_cvpfi_flag
    )

    # =========================
    # 戻り値の吸収
    # =========================
    if model_type == "pls":
        model, cvpfi_df, coef = results
    else:  # svr
        model, cvpfi_df = results
        coef = None

    # =========================
    # 予測（統一）
    # =========================
    estimated_y_test = model.predict(autoscaled_x_test).ravel()
    estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()

    return {
        "model": model,
        "estimated_y_test": pd.Series(estimated_y_test, index=x_test.index),
        "y_std": None,
        "cvpfi": cvpfi_df,
        "coef": coef,
        "type": "single"
    }

def run_classification_pipeline(
    model_type,
    x_train, x_test, y_train, y_test,
    autoscaled_x_train, autoscaled_x_test,
    fold_number
    ):

    model_dict = {
        "svc": run_svc_pipeline,
        "knn": run_knn_pipeline,
        "dt": run_dt_pipeline,
        "rf": run_rf_pipeline
        
    }

    if model_type not in model_dict:
        raise ValueError(f"[Classification] Unknown model_type: {model_type}")

    return model_dict[model_type](
        x_train, x_test, y_train, y_test,
        autoscaled_x_train, autoscaled_x_test,
        fold_number
    )


def run_cvpfi_analysis(model,
                       autoscaled_x_train,
                       autoscaled_y_train,
                       fold_number,
                       model_name='Model',
                       n_repeats=5,
                       alpha_r=0.999,
                       random_state=9,
                       top_n=None,
                       figsize=(8, 6)):


    # =========================
    # CVPFI計算
    # =========================
    importances_mean, importances_std, _ = cvpfi(
                                            model,
                                            autoscaled_x_train,
                                            autoscaled_y_train,
                                            fold_number=fold_number,
                                            scoring='r2',
                                            n_repeats=n_repeats,
                                            alpha_r=alpha_r,
                                            random_state=random_state,
                                        )

    # =========================
    # DataFrame化
    # =========================
    cvpfi_df = pd.DataFrame({
        'feature': autoscaled_x_train.columns,
        'cvPFI': importances_mean,
        'std': importances_std
    })

    cvpfi_df = cvpfi_df.sort_values(by='cvPFI', ascending=False)

    if top_n is not None:
        cvpfi_df = cvpfi_df.head(top_n)

    # =========================
    # 可視化
    # =========================
    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(cvpfi_df))

    plt.bar(
        cvpfi_df['feature'],
        cvpfi_df['cvPFI'],
        color=palette
    )

    plt.errorbar(
        cvpfi_df['feature'],
        cvpfi_df['cvPFI'],
        yerr=cvpfi_df['std'],
        fmt='none',
        ecolor='black',
        capsize=3
    )

    plt.xticks(rotation=90)
    plt.ylabel('CVPFI')
    plt.xlabel('Feature')
    plt.title(f'CVPFI ({model_name})') 
    plt.tight_layout()
    plt.show()

    # =========================
    # 保存
    # =========================
    cvpfi_df.to_csv(f'{model_name}_cvpfi.csv', index=False)

    return cvpfi_df


# =========================
# 実行モード選択
# =========================
mode = "dcv_regression"  

# =========================
# 共通設定
# =========================
fold_number   = 5
random_number = 99
number_of_test_samples = 150


# =========================
# 可視化モード
# =========================
if mode == "visualization":

    dataset = pd.read_csv('./sample_data/iris_without_species.csv', index_col=0)

    plot_hist(dataset)
    plot_box(dataset)
    plot_scatter(dataset, var1=1, var2=2)

    corr = plot_corr_heatmap(dataset)

    results_pca  = run_pca_pipeline(dataset)
    results_tsne = run_tsne_pipeline(dataset)


# =========================
# 回帰
# =========================
elif mode == "regression":

    dataset = pd.read_csv('./sample_data/boston.csv', index_col=0)

    x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_y_train, autoscaled_x_test = \
        prepare_regression_data(dataset, number_of_test_samples,
                                add_non_linear_flag=False,
                                random_number=random_number)

    results = run_regression_pipeline(
        model_type="pls_ensemble",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        autoscaled_x_train=autoscaled_x_train,
        autoscaled_y_train=autoscaled_y_train,
        autoscaled_x_test=autoscaled_x_test,
        fold_number=fold_number
    )

    # アンサンブル対応
    if results["type"] == "ensemble":
        y_pred = results["estimated_y_test"]
    else:
        model = results["model"]
        y_pred = model.predict(autoscaled_x_test).ravel()
        y_pred = y_pred * y_train.std() + y_train.mean()

    # AD
    ad_results = calculate_ad(
        method="knn",
        autoscaled_x_train=autoscaled_x_train,
        autoscaled_x_test=autoscaled_x_test)
    inside_test = ad_results["inside_ad_flag_test"].iloc[:, 0]

    # 分割評価
    evaluate_subset(y_test[inside_test], y_pred[inside_test], "AD inside")
    evaluate_subset(y_test[~inside_test], y_pred[~inside_test], "AD outside")


# =========================
# 分類
# =========================
elif mode == "classification":

    dataset = pd.read_csv('./sample_data/iris.csv', index_col=0)

    x_train, x_test, y_train, y_test, autoscaled_x_train, autoscaled_x_test = \
        prepare_classification_data(dataset,
                                    number_of_test_samples,
                                    random_number=random_number)

    model = run_classification_pipeline(
        model_type="rf",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        autoscaled_x_train=autoscaled_x_train,
        autoscaled_x_test=autoscaled_x_test,
        fold_number=fold_number)


# =========================
# DCV 回帰
# =========================
elif mode == "dcv_regression":

    dataset = pd.read_csv('./sample_data/boston.csv', index_col=0)

    x, y = prepare_regression_data_for_dcv(dataset)

    estimated_y = double_cross_validation_regression(
        run_pipeline_func=run_regression_pipeline,
        x=x,
        y=y,
        model_type="svr",
        outer_fold_number=5,
        fold_number=fold_number
    )

    evaluate_dcv(y, estimated_y)


# =========================
# エラー
# =========================
else:
    raise ValueError("Unknown mode")

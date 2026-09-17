"""
tools/oi_correction.py

简化最优插值（Optimal Interpolation, OI）误差订正模块。

最优插值是资料同化中最经典的分析方法之一，核心思想是：
给定背景场（first guess，这里取模式预报值）与观测，按两者误差方差之比
分配权重，使分析值（订正后结果）的误差方差最小。

本模块为**单变量、标量权重**的简化演示版：
    - 背景场误差方差 sigma_b^2：训练集上 (forecast - observed) 的样本方差
    - 观测误差方差 sigma_o^2：固定值（默认 0.3），代表观测自身的不确定性
    - OI 权重 w = sigma_b^2 / (sigma_b^2 + sigma_o^2)
    - 虚拟观测新息 innovation = mean(observed - forecast)，取训练集均值
      （等于项目定义的平均偏差 Bias 的相反数；新息为负表示预报系统性偏高）
    - 分析值 analysis = forecast + w * innovation
      即用训练集的平均系统偏差作为「虚拟观测新息」，通过 OI 权重公式做订正

验证方式（与随机森林订正、基线对比实验完全一致）：
    滚动窗口交叉验证，共用 ml_correction 中的 _prepare_model_data 与
    _rolling_splits，保证各方法在完全相同的有效样本集和测试集上可比。

说明：
    真实的资料同化（3D-Var、EnKF 等）处理的是高维空间场的误差协方差
    结构，本模块只以一个标量权重复现其方法论核心，用于展示统计后处理
    与资料同化在方法论上的共同根源。
"""

import numpy as np

from tools.ml_correction import (
    _calc_metrics,
    _mean_std,
    _prepare_model_data,
    _rolling_splits,
)


def optimal_interpolation_correction(df, n_splits=3, sigma_o_squared=0.3):
    """
    简化最优插值（OI）误差订正 + 滚动窗口交叉验证。

    参数：
        df              : pandas.DataFrame，包含观测与预报数据
        n_splits        : int，交叉验证折数（默认 3）
        sigma_o_squared : float，观测误差方差（固定值，代表观测自身的不确定性）

    返回：
        dict，包含：
            method             : str   验证方法名（rolling_window_cv_oi）
            n_splits           : int   折数
            total_size         : int   参与建模的有效样本总数
            test_len_per_fold  : int   每折测试段长度（约为总样本的 20%）
            initial_train_size : int   第 1 折的训练样本数
            sigma_o_squared    : float 观测误差方差（输入参数）
            folds              : list  每折结果（训练/测试样本数、测试时段、
                                        sigma_b_squared、权重 w、新息（观测−预报均值）、
                                        原始与分析值的 MAE/RMSE/Bias、改善率）
            raw_mean / raw_std : dict  原始预报各折指标的均值 / 标准差
            oi_mean / oi_std   : dict  OI 分析值各折指标的均值 / 标准差
            mae_improve_mean / rmse_improve_mean : float 各折改善率的均值（%）
    """
    # 数据准备与滚动窗口划分（与随机森林订正、基线对比实验共用，保证测试集完全一致）
    model_df, features, has_time = _prepare_model_data(df)
    n = len(model_df)
    splits, test_len, initial_train = _rolling_splits(n, n_splits)

    # ---------- 逐折计算 OI 分析值并检验 ----------
    folds = []

    for i, (train_end, test_end) in enumerate(splits):
        train = model_df.iloc[:train_end]  # 训练窗口：从头到 train_end（逐折扩大）
        test = model_df.iloc[train_end:test_end]  # 测试段：紧跟训练窗口之后

        # 背景场误差方差：训练集上 (forecast - observed) 的样本方差（ddof=1）
        train_error = train["forecast_temperature"] - train["observed_temperature"]
        sigma_b_squared = float(np.var(train_error, ddof=1))

        # OI 权重：背景场误差方差占比（背景场误差越大，越信任观测新息）
        w = sigma_b_squared / (sigma_b_squared + sigma_o_squared)

        # 虚拟观测新息：训练集上 (observed - forecast) 的均值
        #（新息 = 观测 - 背景；为负表示预报系统性偏高，分析值应向下订正）
        innovation = float(
            np.mean(train["observed_temperature"] - train["forecast_temperature"])
        )

        # 分析值 = 背景场 + w * 新息
        analysis = test["forecast_temperature"] + w * innovation

        # 该折测试段上的订正前后检验
        raw_metrics = _calc_metrics(
            test["forecast_temperature"], test["observed_temperature"]
        )
        oi_metrics = _calc_metrics(analysis, test["observed_temperature"])

        folds.append(
            {
                "fold": i + 1,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
                "test_start": str(test["time"].iloc[0]) if has_time else None,
                "test_end": str(test["time"].iloc[-1]) if has_time else None,
                "sigma_b_squared": sigma_b_squared,
                "weight": w,
                "innovation": innovation,
                "raw": raw_metrics,
                "oi": oi_metrics,
                "mae_improve_percent": (
                    (raw_metrics["MAE"] - oi_metrics["MAE"])
                    / raw_metrics["MAE"]
                    * 100
                ),
                "rmse_improve_percent": (
                    (raw_metrics["RMSE"] - oi_metrics["RMSE"])
                    / raw_metrics["RMSE"]
                    * 100
                ),
            }
        )

    # ---------- 汇总：均值 ± 标准差 ----------
    raw_mean, raw_std = {}, {}
    oi_mean, oi_std = {}, {}
    for key in ["MAE", "RMSE", "Bias"]:
        raw_mean[key], raw_std[key] = _mean_std(folds, "raw", key)
        oi_mean[key], oi_std[key] = _mean_std(folds, "oi", key)

    # 各折改善率的均值
    mae_improve_mean = float(np.mean([f["mae_improve_percent"] for f in folds]))
    rmse_improve_mean = float(np.mean([f["rmse_improve_percent"] for f in folds]))

    return {
        "method": "rolling_window_cv_oi",
        "n_splits": int(n_splits),
        "total_size": int(n),
        "test_len_per_fold": int(test_len),
        "initial_train_size": int(initial_train),
        "sigma_o_squared": float(sigma_o_squared),
        "folds": folds,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "oi_mean": oi_mean,
        "oi_std": oi_std,
        "mae_improve_mean": mae_improve_mean,
        "rmse_improve_mean": rmse_improve_mean,
    }

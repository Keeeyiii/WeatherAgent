"""
tools/ml_correction.py

机器学习预报订正模块（滚动窗口交叉验证版）。

研究目标：
    学习温度预报订正量 correction = observed - forecast，用随机森林预测该订正量，
    再用 corrected = forecast + predicted_correction 得到订正预报。

验证方式（滚动窗口交叉验证 rolling window cross-validation）：
    相比单次 80/20 时间划分，这里进行多折「时间外推」验证
    （以 240 条数据、n_splits=3 为例）：
      第 1 折：训练 [0, 96)   -> 测试 [96, 144)
      第 2 折：训练 [0, 144)  -> 测试 [144, 192)
      第 3 折：训练 [0, 192)  -> 测试 [192, 240)
    训练窗口逐折扩大、测试段依次后移且互不重叠；每折独立训练模型，
    并在该折测试段上计算 MAE/RMSE/Bias，最后汇总为「均值 ± 标准差」。
    目的：反映订正效果在不同时间段上的稳定性，避免单次划分结果的偶然性。

    任何一折中，训练数据全部位于测试数据之前，不存在数据泄漏。

特征：
    forecast_temperature, humidity, pressure, wind_speed, hour, month

基线对比实验（见 baseline_comparison）：
    在随机森林之外，另提供两种简单基线方法（与随机森林使用完全相同的
    有效样本筛选和滚动窗口划分，保证四种方法在同一测试集上可比）：
      基线A（常数去偏）  ：用训练集整体平均偏差做固定订正
      基线B（按小时去偏）：用训练集各小时的平均偏差分别订正
    用于判断随机森林的收益中，有多少只是来自系统性偏差与日变化偏差的消除。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _calc_metrics(forecast, observed):
    """用 Python 计算 MAE / RMSE / Bias（error = forecast - observed）。"""
    error = forecast - observed
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
    }


def _mean_std(folds, key, metric):
    """汇总某一指标在所有折上的均值与标准差（样本标准差，ddof=1）。"""
    values = [fold[key][metric] for fold in folds]
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _prepare_model_data(df):
    """
    数据准备（随机森林订正与基线对比实验共用）。

    包含：必要字段检查、数值转换、目标变量（订正量）构造、
    无效样本剔除、按时间排序，确保两种实验使用完全相同的有效样本集。

    返回：
        model_df : pandas.DataFrame 有效样本（含特征、target、observed_temperature）
        features : list 特征列名
        has_time : bool 原始数据是否含可解析的 time 字段
    """
    required = [
        "observed_temperature",
        "forecast_temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"数据中缺少字段：{col}")

    df_work = df.copy()

    # 特征列（基础气象要素）
    features = ["forecast_temperature", "humidity", "pressure", "wind_speed"]

    # 时间特征：hour = time.dt.hour, month = time.dt.month
    has_time = "time" in df_work.columns
    if has_time:
        time_parsed = pd.to_datetime(df_work["time"], errors="coerce")
        df_work["hour"] = time_parsed.dt.hour
        df_work["month"] = time_parsed.dt.month
        features += ["hour", "month"]

    # 目标变量：订正量 = observed - forecast
    df_work["observed_temperature"] = pd.to_numeric(
        df_work["observed_temperature"], errors="coerce"
    )
    df_work["forecast_temperature"] = pd.to_numeric(
        df_work["forecast_temperature"], errors="coerce"
    )
    df_work["target"] = (
        df_work["observed_temperature"] - df_work["forecast_temperature"]
    )

    # 只保留特征和目标都完整、且观测/预报有效的样本
    # 注意：forecast_temperature 已在 features 中，无需重复加入
    model_df = df_work[features + ["target", "observed_temperature"]].copy()
    if has_time:
        model_df["time"] = df_work["time"]
    model_df = model_df.dropna()

    if len(model_df) < 20:
        raise ValueError("有效样本过少，无法训练模型。")

    # 按时间排序，确保每一折都满足「训练在前、测试在后」
    if has_time:
        model_df = model_df.sort_values("time").reset_index(drop=True)

    return model_df, features, has_time


def _rolling_splits(n, n_splits):
    """
    计算滚动窗口交叉验证的各折划分（随机森林订正与基线对比实验共用）。

    训练窗口逐折扩大（expanding window）、测试段依次后移且互不重叠，
    每折测试段长度约为总样本的 20%。

    返回：
        splits        : list[(train_end, test_end)] 每折的划分索引，
                        训练段为 [0, train_end)，测试段为 [train_end, test_end)
        test_len      : int 每折测试段长度
        initial_train : int 第 1 折的训练样本数
    """
    test_len = int(round(n * 0.2))  # 每折测试段长度（约为总样本的 20%）
    if test_len < 5:
        raise ValueError("有效样本过少，无法进行滚动窗口交叉验证。")

    initial_train = n - n_splits * test_len  # 第 1 折的训练样本数
    if initial_train < test_len:
        raise ValueError(
            f"样本量不足以支撑 {n_splits} 折滚动验证"
            f"（至少需要 {(n_splits + 1) * test_len} 条有效样本，当前 {n} 条）。"
        )

    splits = []
    for i in range(n_splits):
        train_end = initial_train + i * test_len
        splits.append((train_end, train_end + test_len))
    return splits, test_len, initial_train


def machine_learning_correction(df, n_splits=3, n_estimators=200):
    """
    随机森林误差订正 + 滚动窗口交叉验证。

    参数：
        df           : pandas.DataFrame，包含观测与预报数据
        n_splits     : int，交叉验证折数（默认 3）
        n_estimators : int，随机森林的树数量

    返回：
        dict，包含：
            method             : str   验证方法名（rolling_window_cv）
            n_splits           : int   折数
            total_size         : int   参与建模的有效样本总数
            test_len_per_fold  : int   每折测试段长度（约为总样本的 20%）
            initial_train_size : int   第 1 折的训练样本数
            folds              : list  每折结果（训练/测试样本数、测试时段、
                                        原始与订正后的 MAE/RMSE/Bias、改善率）
            raw_mean / raw_std : dict  原始预报各折指标的均值 / 标准差
            corrected_mean / corrected_std : dict 订正后各折指标的均值 / 标准差
            mae_improve_mean / rmse_improve_mean : float 各折改善率的均值（%）
            importance         : dict  各折特征重要性的平均值（按降序）
            test_results       : pandas.DataFrame 各折测试段按时间顺序拼接
                                 （fold / time / observed_temperature /
                                  forecast_temperature / corrected_forecast）
    """
    # 数据准备与滚动窗口划分（与基线对比实验共用，保证四种方法测试集完全一致）
    model_df, features, has_time = _prepare_model_data(df)
    n = len(model_df)
    splits, test_len, initial_train = _rolling_splits(n, n_splits)

    # ---------- 逐折训练与检验 ----------
    folds = []
    test_parts = []
    importances = []

    for i, (train_end, test_end) in enumerate(splits):
        train = model_df.iloc[:train_end]  # 训练窗口：从头到 train_end（逐折扩大）
        test = model_df.iloc[train_end:test_end]  # 测试段：紧跟训练窗口之后

        # 每折独立训练随机森林：random_state=42 保证可复现
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(train[features], train["target"])

        predicted_correction = model.predict(test[features])
        test = test.copy()
        test["corrected_forecast"] = test["forecast_temperature"] + predicted_correction
        test["fold"] = i + 1

        # 该折测试段上的订正前后检验
        raw_metrics = _calc_metrics(
            test["forecast_temperature"], test["observed_temperature"]
        )
        corrected_metrics = _calc_metrics(
            test["corrected_forecast"], test["observed_temperature"]
        )

        folds.append(
            {
                "fold": i + 1,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
                "test_start": str(test["time"].iloc[0]) if has_time else None,
                "test_end": str(test["time"].iloc[-1]) if has_time else None,
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "mae_improve_percent": (
                    (raw_metrics["MAE"] - corrected_metrics["MAE"])
                    / raw_metrics["MAE"]
                    * 100
                ),
                "rmse_improve_percent": (
                    (raw_metrics["RMSE"] - corrected_metrics["RMSE"])
                    / raw_metrics["RMSE"]
                    * 100
                ),
            }
        )

        importances.append(model.feature_importances_)
        test_parts.append(test)

    # ---------- 汇总：均值 ± 标准差 ----------
    raw_mean, raw_std = {}, {}
    corrected_mean, corrected_std = {}, {}
    for key in ["MAE", "RMSE", "Bias"]:
        raw_mean[key], raw_std[key] = _mean_std(folds, "raw", key)
        corrected_mean[key], corrected_std[key] = _mean_std(folds, "corrected", key)

    # 各折改善率的均值
    mae_improve_mean = float(np.mean([f["mae_improve_percent"] for f in folds]))
    rmse_improve_mean = float(np.mean([f["rmse_improve_percent"] for f in folds]))

    # 特征重要性：各折取平均后降序排列
    avg_importance = np.mean(importances, axis=0)
    importance = dict(
        sorted(
            zip(features, avg_importance),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    importance = {k: float(v) for k, v in importance.items()}

    # 各折测试段按时间顺序拼接（各段互不重叠，用于可视化）
    if has_time:
        result_cols = [
            "fold",
            "time",
            "observed_temperature",
            "forecast_temperature",
            "corrected_forecast",
        ]
    else:
        result_cols = [
            "fold",
            "observed_temperature",
            "forecast_temperature",
            "corrected_forecast",
        ]
    test_results = pd.concat(test_parts)[result_cols].reset_index(drop=True)

    return {
        "method": "rolling_window_cv",
        "n_splits": int(n_splits),
        "total_size": int(n),
        "test_len_per_fold": int(test_len),
        "initial_train_size": int(initial_train),
        "folds": folds,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "corrected_mean": corrected_mean,
        "corrected_std": corrected_std,
        "mae_improve_mean": mae_improve_mean,
        "rmse_improve_mean": rmse_improve_mean,
        "importance": importance,
        "test_results": test_results,
    }


def baseline_comparison(df, n_splits=3):
    """
    基线对比实验：原始预报 + 两种简单去偏基线（滚动窗口交叉验证）。

    与 machine_learning_correction 使用完全相同的有效样本筛选和
    滚动窗口划分（共用 _prepare_model_data / _rolling_splits），
    因此四种方法（原始预报 / 基线A / 基线B / Random Forest）
    在完全相同的测试集上可比。

    基线方法（偏差定义为 error = forecast - observed）：
        基线A（常数去偏）  ：train_bias = 训练集上 mean(forecast - observed)，
                             测试集 corrected = forecast - train_bias
        基线B（按小时去偏）：按 24 个小时分别计算训练集上该小时的平均偏差，
                             测试集按对应小时订正（训练集中未出现的小时
                             退回使用整体平均偏差）

    参数：
        df       : pandas.DataFrame，包含观测与预报数据（需含 time 字段）
        n_splits : int，交叉验证折数（默认 3，与随机森林实验一致）

    返回：
        dict，包含：
            method / n_splits / total_size / test_len_per_fold / initial_train_size
            folds : list 每折结果（训练/测试样本数、测试时段、
                    原始与两种基线的 MAE/RMSE/Bias）
            raw_mean / raw_std               : dict 原始预报各折汇总
            baseline_a_mean / baseline_a_std : dict 基线A 各折汇总
            baseline_b_mean / baseline_b_std : dict 基线B 各折汇总
    """
    # 与随机森林实验共用同一套数据准备与划分，保证测试集完全一致
    model_df, features, has_time = _prepare_model_data(df)
    if not has_time:
        raise ValueError("基线对比实验需要 time 字段（基线B 按小时去偏依赖小时信息）。")

    n = len(model_df)
    splits, test_len, initial_train = _rolling_splits(n, n_splits)

    folds = []
    for i, (train_end, test_end) in enumerate(splits):
        train = model_df.iloc[:train_end]  # 与随机森林实验相同的训练窗口
        test = model_df.iloc[train_end:test_end]  # 与随机森林实验相同的测试段

        # ---------- 原始预报（该折测试段） ----------
        raw_metrics = _calc_metrics(
            test["forecast_temperature"], test["observed_temperature"]
        )

        # ---------- 基线A：常数去偏 ----------
        # 训练集整体平均偏差（error = forecast - observed），对测试集做固定订正
        train_bias = float(
            np.mean(train["forecast_temperature"] - train["observed_temperature"])
        )
        corrected_a = test["forecast_temperature"] - train_bias
        baseline_a_metrics = _calc_metrics(corrected_a, test["observed_temperature"])

        # ---------- 基线B：按小时去偏 ----------
        # 训练集上各小时的平均偏差；测试集中该小时若在训练集未出现，
        # 退回整体平均偏差（避免订正量缺失）
        train_error = train["forecast_temperature"] - train["observed_temperature"]
        bias_by_hour = train_error.groupby(train["hour"]).mean()
        hour_bias = test["hour"].map(bias_by_hour).fillna(train_bias)
        corrected_b = test["forecast_temperature"] - hour_bias
        baseline_b_metrics = _calc_metrics(corrected_b, test["observed_temperature"])

        folds.append(
            {
                "fold": i + 1,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
                "test_start": str(test["time"].iloc[0]),
                "test_end": str(test["time"].iloc[-1]),
                "raw": raw_metrics,
                "baseline_a": baseline_a_metrics,
                "baseline_b": baseline_b_metrics,
            }
        )

    # ---------- 汇总：均值 ± 标准差 ----------
    raw_mean, raw_std = {}, {}
    baseline_a_mean, baseline_a_std = {}, {}
    baseline_b_mean, baseline_b_std = {}, {}
    for key in ["MAE", "RMSE", "Bias"]:
        raw_mean[key], raw_std[key] = _mean_std(folds, "raw", key)
        baseline_a_mean[key], baseline_a_std[key] = _mean_std(folds, "baseline_a", key)
        baseline_b_mean[key], baseline_b_std[key] = _mean_std(folds, "baseline_b", key)

    return {
        "method": "baseline_comparison（与随机森林相同的滚动窗口划分）",
        "n_splits": int(n_splits),
        "total_size": int(n),
        "test_len_per_fold": int(test_len),
        "initial_train_size": int(initial_train),
        "folds": folds,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "baseline_a_mean": baseline_a_mean,
        "baseline_a_std": baseline_a_std,
        "baseline_b_mean": baseline_b_mean,
        "baseline_b_std": baseline_b_std,
    }

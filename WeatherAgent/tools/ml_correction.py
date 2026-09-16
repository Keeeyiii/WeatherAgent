"""
tools/ml_correction.py

机器学习预报订正模块。

研究目标：
    学习温度预报误差 error = observed - forecast，用随机森林预测该误差，
    再用 corrected = forecast + predicted_error 得到订正预报。

训练方式：
    按时间顺序划分数据（前 80% 训练、后 20% 测试），不随机打乱，避免数据泄漏。
    测试集不参与模型训练。

特征：
    forecast_temperature, humidity, pressure, wind_speed, hour, month
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


def machine_learning_correction(df, test_ratio=0.2, n_estimators=200):
    """
    用随机森林对温度预报进行订正，返回订正前后的检验指标与测试集结果。

    参数：
        df           : pandas.DataFrame，包含观测与预报数据
        test_ratio   : float，测试集比例（按时间顺序取末尾部分）
        n_estimators : int，随机森林的树数量

    返回：
        dict，包含：
            raw          : dict  订正前指标 {"MAE","RMSE","Bias"}（测试集）
            corrected    : dict  订正后指标 {"MAE","RMSE","Bias"}（测试集）
            train_size   : int   训练样本数
            test_size    : int   测试样本数
            importance   : dict  特征重要性（按降序）
            test_results : pandas.DataFrame 测试集结果
                           （time / observed_temperature / forecast_temperature / corrected_forecast）
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
    if "time" in df_work.columns:
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
    if "time" in df_work.columns:
        model_df["time"] = df_work["time"]
    model_df = model_df.dropna()

    if len(model_df) < 20:
        raise ValueError("有效样本过少，无法训练模型。")

    # 按时间排序，确保「前训练、后测试」符合时间外推逻辑
    if "time" in model_df.columns:
        model_df = model_df.sort_values("time").reset_index(drop=True)

    n_test = int(len(model_df) * test_ratio)
    if n_test < 1:
        n_test = 1

    train = model_df.iloc[:-n_test]   # 前 80%：训练集
    test = model_df.iloc[-n_test:]    # 后 20%：测试集

    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]

    # 随机森林：random_state=42 保证可复现
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    predicted_error = model.predict(X_test)

    test = test.copy()
    test["corrected_forecast"] = test["forecast_temperature"] + predicted_error

    # 订正前后检验（只在测试集上比较）
    raw = _calc_metrics(test["forecast_temperature"], test["observed_temperature"])
    corrected = _calc_metrics(test["corrected_forecast"], test["observed_temperature"])

    # 特征重要性（按从高到低排序）
    importance = dict(
        sorted(
            zip(features, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # 测试集结果（仅保留展示所需的列）
    result_cols = ["observed_temperature", "forecast_temperature", "corrected_forecast"]
    if "time" in test.columns:
        result_cols = ["time"] + result_cols
    test_results = test[result_cols].reset_index(drop=True)

    return {
        "raw": raw,
        "corrected": corrected,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "importance": importance,
        "test_results": test_results,
    }

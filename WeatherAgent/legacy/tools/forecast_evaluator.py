"""
tools/forecast_evaluator.py

预报误差检验模块。

职责：
    用 Python 真实计算预报误差检验指标。
    本模块只负责数值计算，绝不让大语言模型代替计算。

计算指标（error = forecast - observed）：
    MAE  : mean(abs(error))        平均绝对误差
    RMSE : sqrt(mean(error^2))     均方根误差
    Bias : mean(error)             平均偏差（系统性误差）
"""

import numpy as np
import pandas as pd


def evaluate_forecast(
    df,
    observed_col="observed_temperature",
    forecast_col="forecast_temperature",
):
    """
    计算预报误差检验指标。

    参数：
        df           : pandas.DataFrame，包含观测与预报数据
        observed_col : str，观测值所在列名
        forecast_col : str，预报值所在列名

    返回：
        dict，包含：
            MAE  : float  平均绝对误差
            RMSE : float  均方根误差
            Bias : float  平均偏差（正值 = 预报系统性偏高）
    """
    for col in (observed_col, forecast_col):
        if col not in df.columns:
            raise ValueError(f"数据中缺少字段：{col}")

    observed = pd.to_numeric(df[observed_col], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")

    # 只保留两个字段都有效的样本（剔除 NaN）
    valid = observed.notna() & forecast.notna()
    observed = observed[valid]
    forecast = forecast[valid]

    error = forecast - observed

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
    }

"""Forecast verification and error-correction experiments.

The scientific question this module answers is deliberately narrow:

    机器学习订正 2 米气温预报时，收益究竟来自误差的哪一部分？

To answer it, every correction method is evaluated on *identical*,
strictly chronological splits, and the methods are nested -- each one adds
exactly one new piece of error structure (constant bias -> month -> hour ->
month x hour -> nonlinear).  The MAE reduction at each step therefore
measures how much that piece of structure is worth.

All functions take the real-data tables produced by `tools.real_data` and
return tidy DataFrames, so the Streamlit front end only has to render them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Features mirror the ones used by the original WeatherAgent prototype.
FEATURES = [
    "forecast_temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "hour",
    "month",
]

# `error` uses the same sign convention as the original app:
#     error = forecast - observed   (>0 模式偏暖)
SEASON_OF_MONTH = {
    12: "冬季 DJF", 1: "冬季 DJF", 2: "冬季 DJF",
    3: "春季 MAM", 4: "春季 MAM", 5: "春季 MAM",
    6: "夏季 JJA", 7: "夏季 JJA", 8: "夏季 JJA",
    9: "秋季 SON", 10: "秋季 SON", 11: "秋季 SON",
}


def add_error(df: pd.DataFrame, forecast: str = "forecast_temperature") -> pd.DataFrame:
    out = df.copy()
    out["error"] = out[forecast] - out["observed_temperature"]
    return out


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Beijing-time hour (UTC+8) plus season, so plots read naturally in China."""
    out = df.copy()
    out["bjt"] = (out["hour"] + 8) % 24
    out["season"] = out["month"].map(SEASON_OF_MONTH)
    return out


# --------------------------------------------------------------------------- #
# verification scores
# --------------------------------------------------------------------------- #
def temperature_scores(df: pd.DataFrame) -> pd.Series:
    err = df["error"]
    return pd.Series(
        {
            "n": len(df),
            "bias": err.mean(),
            "MAE": err.abs().mean(),
            "RMSE": float(np.sqrt((err**2).mean())),
            "corr": df["forecast_temperature"].corr(df["observed_temperature"]),
            "qd95": err.abs().quantile(0.95),
        }
    )


def score_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    grouped = df.groupby(by, dropna=False).apply(temperature_scores, include_groups=False)
    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# nested correction methods  (the ablation experiment)
# --------------------------------------------------------------------------- #
def _group_mean(train: pd.DataFrame, keys: list[str]) -> pd.Series:
    return train.groupby(keys)["error"].mean()


def _constant_bias(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train["error"].mean())


def _month_bias(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    table = _group_mean(train, ["month"])
    return test["month"].map(table).fillna(train["error"].mean()).to_numpy()


def _hour_bias(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    table = _group_mean(train, ["hour"])
    return test["hour"].map(table).fillna(train["error"].mean()).to_numpy()


def _month_hour_bias(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Seasonal x diurnal look-up table, falling back to month, then global mean."""
    global_mean = train["error"].mean()
    by_month = _group_mean(train, ["month"])
    by_month_hour = _group_mean(train, ["month", "hour"])
    mapped = test.set_index(["month", "hour"]).index.map(by_month_hour)
    corr = pd.Series(mapped, index=test.index, dtype="float64")
    corr = corr.fillna(test["month"].map(by_month))
    return corr.fillna(global_mean).to_numpy()


def _random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str] = FEATURES,
    n_estimators: int = 200,
    seed: int = 42,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    model.fit(train[features], train["error"])
    return model.predict(test[features])


# Ordered from "no correction" to "full nonlinear correction".  Each entry
# adds one piece of error structure, which is what makes the table an ablation
# rather than a horse race.
METHODS: dict[str, object] = {
    "① 原始预报（不订正）": None,
    "② 常数去偏": _constant_bias,
    "③ 按月去偏": _month_bias,
    "④ 按小时去偏": _hour_bias,
    "⑤ 按(月×小时)去偏": _month_hour_bias,
    "⑥ 随机森林（多变量非线性）": _random_forest,
}

METHOD_NOTE = {
    "① 原始预报（不订正）": "模式直接输出，作为基准",
    "② 常数去偏": "只消除全时段平均偏差",
    "③ 按月去偏": "再叠加季节调制",
    "④ 按小时去偏": "只叠加日变化（与③对照，检验季节 vs 日变化谁重要）",
    "⑤ 按(月×小时)去偏": "同时刻画季节与日变化的耦合",
    "⑥ 随机森林（多变量非线性）": "在⑤之上，检验非线性多变量订正是否还有增量",
}


def default_folds(
    df: pd.DataFrame, n_folds: int = 3
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-window, season-spanning folds: (train_end, test_start, test_end).

    Each test block follows its training block in time, so there is no leakage,
    and every test block covers a different season.
    """
    start, end = df["time"].min(), df["time"].max()
    usable_days = (end - start).days / (n_folds + 1)
    folds = []
    for index in range(n_folds):
        train_end = start + pd.Timedelta(days=usable_days * (index + 1))
        test_start = train_end
        test_end = min(test_start + pd.Timedelta(days=usable_days), end)
        folds.append((train_end, test_start, test_end))
    return folds


def _apply(
    method: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    seed: int,
) -> np.ndarray:
    if method is None:
        return test["forecast_temperature"].to_numpy()
    kwargs = {"features": features, "seed": seed} if method is _random_forest else {}
    correction = method(train, test, **kwargs)  # type: ignore[operator]
    return test["forecast_temperature"].to_numpy() - correction


def correction_experiment(
    df: pd.DataFrame,
    *,
    folds: list[tuple] | None = None,
    features: list[str] = FEATURES,
    seed: int = 42,
) -> pd.DataFrame:
    """Run every correction method on identical chronological splits."""
    data = add_error(df) if "error" not in df.columns else df.copy()
    folds = folds or default_folds(data)
    rows = []

    for fold_index, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train = data[data["time"] < train_end]
        test = data[(data["time"] >= test_start) & (data["time"] < test_end)]
        if len(train) < 200 or len(test) < 100:
            continue

        raw_mae = test["error"].abs().mean()
        raw_rmse = float(np.sqrt((test["error"] ** 2).mean()))

        for name, method in METHODS.items():
            corrected = _apply(method, train, test, features=features, seed=seed)
            err = corrected - test["observed_temperature"].to_numpy()
            rows.append(
                {
                    "fold": fold_index,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_n": len(train),
                    "test_n": len(test),
                    "method": name,
                    "bias": err.mean(),
                    "MAE": np.abs(err).mean(),
                    "RMSE": float(np.sqrt((err**2).mean())),
                    "MAE_improvement_pct": 100 * (raw_mae - np.abs(err).mean()) / raw_mae,
                    "RMSE_improvement_pct": 100
                    * (raw_rmse - float(np.sqrt((err**2).mean())))
                    / raw_rmse,
                }
            )
    return pd.DataFrame(rows)


def summarise_correction(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each method across the folds, in ablation order."""
    summary = (
        results.groupby("method")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            bias_mean=("bias", "mean"),
            MAE_improvement_pct=("MAE_improvement_pct", "mean"),
            RMSE_improvement_pct=("RMSE_improvement_pct", "mean"),
        )
        .reset_index()
    )
    order = {name: index for index, name in enumerate(METHODS)}
    summary["__order"] = summary["method"].map(order)
    summary = summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # Marginal value of each step relative to the previous method.
    gains = summary["MAE_improvement_pct"].diff()
    gains.iloc[0] = np.nan
    summary = summary.assign(step_gain_pct=gains)
    summary["note"] = summary["method"].map(METHOD_NOTE)
    return summary


def cross_station_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str] = FEATURES,
    seed: int = 42,
) -> pd.DataFrame:
    """Train on one station, verify on another: how well does the correction transfer?"""
    train = add_error(train) if "error" not in train.columns else train
    test = add_error(test) if "error" not in test.columns else test
    rows = []
    raw_mae = test["error"].abs().mean()
    raw_rmse = float(np.sqrt((test["error"] ** 2).mean()))

    for name, method in METHODS.items():
        corrected = _apply(method, train, test, features=features, seed=seed)
        err = corrected - test["observed_temperature"].to_numpy()
        rows.append(
            {
                "method": name,
                "train_station": train["station"].iloc[0],
                "test_station": test["station"].iloc[0],
                "train_n": len(train),
                "test_n": len(test),
                "bias": err.mean(),
                "MAE": np.abs(err).mean(),
                "RMSE": float(np.sqrt((err**2).mean())),
                "MAE_improvement_pct": 100 * (raw_mae - np.abs(err).mean()) / raw_mae,
                "RMSE_improvement_pct": 100
                * (raw_rmse - float(np.sqrt((err**2).mean())))
                / raw_rmse,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# case studies
# --------------------------------------------------------------------------- #
def corrected_series(
    df: pd.DataFrame,
    train: pd.DataFrame | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Return observations, raw forecast and corrected forecast for one period."""
    data = add_error(df) if "error" not in df.columns else df.copy()
    reference = train if train is not None else data
    reference = add_error(reference) if "error" not in reference.columns else reference
    out = data[["time", "observed_temperature", "forecast_temperature"]].copy()
    out["corrected_temperature"] = _apply(
        _month_hour_bias, reference, data, features=FEATURES, seed=seed
    )
    out["corrected_rf"] = _apply(
        _random_forest, reference, data, features=FEATURES, seed=seed
    )
    out["raw_error"] = data["error"].to_numpy()
    out["corrected_error"] = out["corrected_temperature"] - out["observed_temperature"]
    out["rf_error"] = out["corrected_rf"] - out["observed_temperature"]
    return out


def case_summary(series: pd.DataFrame) -> pd.DataFrame:
    """MAE / bias / worst error for the raw, (month x hour) and RF correction."""
    rows = []
    for label, column in (
        ("原始预报", "raw_error"),
        ("(月×小时)去偏", "corrected_error"),
        ("随机森林", "rf_error"),
    ):
        err = series[column]
        rows.append(
            {
                "method": label,
                "bias": err.mean(),
                "MAE": err.abs().mean(),
                "RMSE": float(np.sqrt((err**2).mean())),
                "max_abs_error": err.abs().max(),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(
    df: pd.DataFrame, *, features: list[str] = FEATURES, seed: int = 42
) -> pd.DataFrame:
    data = add_error(df) if "error" not in df.columns else df.copy()
    model = RandomForestRegressor(
        n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=5
    )
    model.fit(data[features], data["error"])
    return (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def error_autocorrelation(df: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Does yesterday's error help predict today's? (a proxy for error persistence)"""
    data = add_error(df) if "error" not in df.columns else df.copy()
    series = data.set_index("time")["error"].sort_index()
    rows = [
        {"lag_hour": lag, "autocorr": series.autocorr(lag=lag)} for lag in range(1, max_lag + 1)
    ]
    return pd.DataFrame(rows)

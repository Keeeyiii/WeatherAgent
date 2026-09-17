"""Amplitude-damping diagnostics.

The headline finding of this project is that the GFS 2 m temperature error at
Nanjing is not mainly a warm or cold *offset* -- it is an amplitude problem:
the model systematically flattens the variability of the real atmosphere.

These functions quantify that statement from several independent angles, so
the claim does not rest on a single number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_range(df: pd.DataFrame) -> pd.DataFrame:
    """Daily maximum minus minimum, for observation and forecast side by side."""
    work = df.copy()
    work["date"] = work["time"].dt.normalize()
    daily = work.groupby("date").agg(
        obs_max=("observed_temperature", "max"),
        obs_min=("observed_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
    )
    daily["obs_range"] = daily["obs_max"] - daily["obs_min"]
    daily["fcst_range"] = daily["fcst_max"] - daily["fcst_min"]
    daily["range_error"] = daily["fcst_range"] - daily["obs_range"]
    return daily.reset_index()


def range_summary(df: pd.DataFrame) -> dict:
    daily = daily_range(df)
    obs = daily["obs_range"].mean()
    fcst = daily["fcst_range"].mean()
    return {
        "obs_range": obs,
        "fcst_range": fcst,
        "ratio": fcst / obs,
        "damping_pct": 100 * (1 - fcst / obs),
        "daily_corr": daily["obs_range"].corr(daily["fcst_range"]),
        "n_days": len(daily),
    }


def range_by_month(df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_range(df)
    daily["month"] = daily["date"].dt.month
    table = daily.groupby("month").agg(
        obs_range=("obs_range", "mean"),
        fcst_range=("fcst_range", "mean"),
        n=("obs_range", "size"),
    )
    table["ratio"] = table["fcst_range"] / table["obs_range"]
    table["damping_pct"] = 100 * (1 - table["ratio"])
    return table.reset_index()


def seasonal_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean temperature for observation and forecast, plus the annual amplitude."""
    work = df.copy()
    work["month"] = work["time"].dt.month
    table = work.groupby("month").agg(
        observed=("observed_temperature", "mean"),
        forecast=("forecast_temperature", "mean"),
        n=("observed_temperature", "size"),
    )
    table["error"] = table["forecast"] - table["observed"]
    table["obs_amplitude"] = table["observed"] - table["observed"].mean()
    table["fcst_amplitude"] = table["forecast"] - table["forecast"].mean()
    return table.reset_index()


def seasonal_amplitude_ratio(df: pd.DataFrame) -> dict:
    table = seasonal_cycle(df)
    obs_std = table["observed"].std()
    fcst_std = table["forecast"].std()
    return {
        "obs_std": obs_std,
        "fcst_std": fcst_std,
        "ratio": fcst_std / obs_std,
        "damping_pct": 100 * (1 - fcst_std / obs_std),
    }


def anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Observed anomaly relative to the (month, Beijing-hour) climatology."""
    work = df.copy()
    work["month"] = work["time"].dt.month
    work["bjt"] = (work["time"].dt.hour + 8) % 24
    climatology = work.groupby(["month", "bjt"])["observed_temperature"].transform("mean")
    work["anomaly"] = work["observed_temperature"] - climatology
    return work


def anomaly_regression(df: pd.DataFrame) -> dict:
    """Fit  error = intercept + slope * observed anomaly.

    A negative slope is the signature of amplitude damping: the model runs warm
    when the atmosphere is unusually cold, and cold when it is unusually warm.
    """
    work = anomaly_table(df)
    work["error"] = work["forecast_temperature"] - work["observed_temperature"]
    slope, intercept = np.polyfit(work["anomaly"], work["error"], 1)
    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": work["anomaly"].corr(work["error"]),
        "n": len(work),
        "table": work,
    }


def anomaly_binned(df: pd.DataFrame, bins: list[float] | None = None) -> pd.DataFrame:
    bins = bins or [-20, -8, -6, -4, -2, 0, 2, 4, 6, 8, 20]
    work = anomaly_table(df)
    work["error"] = work["forecast_temperature"] - work["observed_temperature"]
    work["bin"] = pd.cut(work["anomaly"], bins)
    table = work.groupby("bin", observed=True).agg(
        n=("error", "size"),
        mean_anomaly=("anomaly", "mean"),
        mean_error=("error", "mean"),
    )
    return table.reset_index()


def variability_ratio(df: pd.DataFrame) -> dict:
    """Standard deviation of the anomaly (synoptic-scale variability) after removing
    the mean seasonal-diurnal cycle."""
    work = df.copy()
    work["month"] = work["time"].dt.month
    work["bjt"] = (work["time"].dt.hour + 8) % 24
    base = work.groupby(["month", "bjt"])[["observed_temperature", "forecast_temperature"]].transform(
        "mean"
    )
    obs_anom = work["observed_temperature"] - base["observed_temperature"]
    fcst_anom = work["forecast_temperature"] - base["forecast_temperature"]
    return {
        "obs_std": obs_anom.std(),
        "fcst_std": fcst_anom.std(),
        "ratio": fcst_anom.std() / obs_anom.std(),
        "damping_pct": 100 * (1 - fcst_anom.std() / obs_anom.std()),
    }


def extreme_error(df: pd.DataFrame, quantile: float = 0.01) -> pd.DataFrame:
    """Mean error in the coldest / warmest tail of the observed distribution."""
    work = df.copy()
    work["error"] = work["forecast_temperature"] - work["observed_temperature"]
    low = work["observed_temperature"].quantile(quantile)
    high = work["observed_temperature"].quantile(1 - quantile)
    rows = [
        {
            "分组": f"最冷 {quantile:.0%}（≤{low:.1f} ℃）",
            "样本数": int((work["observed_temperature"] <= low).sum()),
            "平均误差 (℃)": work.loc[work["observed_temperature"] <= low, "error"].mean(),
            "平均MAE (℃)": work.loc[work["observed_temperature"] <= low, "error"].abs().mean(),
        },
        {
            "分组": "全部样本",
            "样本数": len(work),
            "平均误差 (℃)": work["error"].mean(),
            "平均MAE (℃)": work["error"].abs().mean(),
        },
        {
            "分组": f"最暖 {quantile:.0%}（≥{high:.1f} ℃）",
            "样本数": int((work["observed_temperature"] >= high).sum()),
            "平均误差 (℃)": work.loc[work["observed_temperature"] >= high, "error"].mean(),
            "平均MAE (℃)": work.loc[work["observed_temperature"] >= high, "error"].abs().mean(),
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# robustness checks
# --------------------------------------------------------------------------- #
def range_vs_resolution(df: pd.DataFrame, steps: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    """Recompute the daily-range ratio with progressively coarser sampling.

    This is the check that rules out the most obvious artefact: if the archived
    forecast were effectively coarser than the hourly observation, the apparent
    "damping" could be a sampling effect rather than a model property.
    Sampling *both* series at the same interval removes that possibility.
    """
    work = df.copy()
    work["date"] = work["time"].dt.normalize()
    work["hour_of_day"] = work["time"].dt.hour
    rows = []
    for step in steps:
        subset = work if step == 1 else work[work["hour_of_day"] % step == 0]
        daily = subset.groupby("date").agg(
            obs_max=("observed_temperature", "max"),
            obs_min=("observed_temperature", "min"),
            fcst_max=("forecast_temperature", "max"),
            fcst_min=("forecast_temperature", "min"),
        )
        obs_range = (daily["obs_max"] - daily["obs_min"]).mean()
        fcst_range = (daily["fcst_max"] - daily["fcst_min"]).mean()
        rows.append(
            {
                "采样间隔": f"{step} 小时",
                "观测日较差": obs_range,
                "预报日较差": fcst_range,
                "比值": fcst_range / obs_range,
                "样本天数": len(daily),
            }
        )
    return pd.DataFrame(rows)


def series_step_fraction(df: pd.DataFrame) -> dict:
    """How often does each series repeat the previous hour's value?

    A large value for the forecast would indicate a coarser archive; a large
    value for the observation is expected because METAR temperature is reported
    as a whole degree.
    """
    forecast = df["forecast_temperature"]
    observed = df["observed_temperature"]
    return {
        "forecast_repeat": float((forecast.diff().abs() < 1e-9).mean()),
        "observed_repeat": float((observed.diff().abs() < 1e-9).mean()),
    }


def minmax_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Split the daily error into a "daily minimum" and a "daily maximum" part.

    This is what pinpoints *which side* of the diurnal cycle is being damped.
    A positive `bias_min` means the model's night-time minimum is too warm,
    i.e. it misses the nocturnal radiative cooling.
    """
    work = df.copy()
    work["date"] = work["time"].dt.normalize()
    daily = work.groupby("date").agg(
        obs_min=("observed_temperature", "min"),
        obs_max=("observed_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
    )
    daily["bias_min"] = daily["fcst_min"] - daily["obs_min"]
    daily["bias_max"] = daily["fcst_max"] - daily["obs_max"]
    daily["month"] = daily.index.month
    table = (
        daily.groupby("month")[["bias_min", "bias_max"]]
        .mean()
        .reset_index()
        .rename(columns={"bias_min": "最低气温偏差", "bias_max": "最高气温偏差"})
    )
    return table


def minmax_summary(df: pd.DataFrame) -> dict:
    """Year-round, winter and summer mean bias of the daily minimum / maximum."""
    work = df.copy()
    work["date"] = work["time"].dt.normalize()
    daily = work.groupby("date").agg(
        obs_min=("observed_temperature", "min"),
        obs_max=("observed_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
    )
    daily["bias_min"] = daily["fcst_min"] - daily["obs_min"]
    daily["bias_max"] = daily["fcst_max"] - daily["obs_max"]
    month = daily.index.month
    winter = daily[month.isin([12, 1, 2])]
    summer = daily[month.isin([6, 7, 8])]
    return {
        "year_min": daily["bias_min"].mean(),
        "year_max": daily["bias_max"].mean(),
        "winter_min": winter["bias_min"].mean(),
        "winter_max": winter["bias_max"].mean(),
        "summer_min": summer["bias_min"].mean(),
        "summer_max": summer["bias_max"].mean(),
        "min_spread_ratio": daily["fcst_min"].std() / daily["obs_min"].std(),
        "max_spread_ratio": daily["fcst_max"].std() / daily["obs_max"].std(),
    }

"""Run the complete real-data verification study and dump every number used in the app."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.real_data import build_leadtime_dataset  # noqa: E402
from tools.verification import (  # noqa: E402
    add_error,
    add_time_columns,
    case_summary,
    correction_experiment,
    corrected_series,
    cross_station_experiment,
    error_autocorrelation,
    feature_importance,
    score_by,
    summarise_correction,
    temperature_scores,
)

OUT = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def title(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def frame(path: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(ROOT, "data", path), parse_dates=["time"], encoding="utf-8-sig"
    )


nj = add_time_columns(add_error(frame("sample_weather_real.csv")))
sh = add_time_columns(add_error(frame("zsss_shanghai_real.csv")))
lead = add_error(frame("gfs_leadtime.csv"))

title("1) 总体检验  ZSNJ 2024-01-01 ~ 2026-09-15")
overall = temperature_scores(nj)
print(overall.to_string())

title("2) 按季节")
by_season = score_by(nj, "season")
print(by_season.round(3).to_string(index=False))

title("3) 按月份")
by_month = score_by(nj, "month")
print(by_month.round(3).to_string(index=False))

title("4) 按北京时小时的误差日变化")
by_hour = score_by(nj, "bjt")
print(by_hour[["bjt", "n", "bias", "MAE", "RMSE"]].round(3).to_string(index=False))

title("5) 按预报时效")
by_lead = score_by(lead, "lead_day")
print(by_lead.round(3).to_string(index=False))

title("6) 订正方法消融实验（3 折严格时序划分）")
results = correction_experiment(nj)
summary = summarise_correction(results)
print("\n各折 MAE：")
print(results.pivot_table(index="method", columns="fold", values="MAE").round(3).to_string())
print("\n汇总：")
print(
    summary[
        [
            "method",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "bias_mean",
            "MAE_improvement_pct",
            "step_gain_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

title("7) 随机森林特征重要性")
importance = feature_importance(nj)
print(importance.round(4).to_string(index=False))

title("8) 跨站点迁移：南京训练 -> 上海虹桥测试")
cross = cross_station_experiment(nj, sh)
print(cross.round(3).to_string(index=False))

title("9) 误差时间自相关（滞后 1-12 小时）")
autocorr = error_autocorrelation(nj)
print(autocorr.round(3).to_string(index=False))

title("10) 真实案例")
CASES = {
    "2024年1月寒潮过程": ("2024-01-19", "2024-01-26"),
    "2024年8月高温过程": ("2024-08-01", "2024-08-12"),
}
case_frames = {}
for name, (start, end) in CASES.items():
    window = nj[(nj["time"] >= start) & (nj["time"] < f"{end} 23:59")]
    # Train on everything strictly before the case, so the case is out-of-sample.
    train = nj[nj["time"] < start]
    series = corrected_series(window, train)
    case_frames[name] = series
    print(f"\n--- {name}  ({start} ~ {end}, n={len(series)}) ---")
    print(f"观测温度范围 {series['observed_temperature'].min():.1f} ~ "
          f"{series['observed_temperature'].max():.1f} ℃")
    print(f"24 小时最大降温 {series['observed_temperature'].diff().min():.1f} ℃")
    print(case_summary(series).round(3).to_string(index=False))
    series.to_csv(
        os.path.join(OUT, f"case_{start.replace('-', '')}.csv"), index=False
    )

title("11) 逐日误差最大的 15 天（误差如何被平均值掩盖）")
daily = (
    nj.set_index("time")
    .resample("1D")
    .agg(obs=("observed_temperature", "mean"), fcst=("forecast_temperature", "mean"))
)
daily["bias"] = daily["fcst"] - daily["obs"]
daily["MAE_proxy"] = daily["bias"].abs()
print(daily.reindex(daily["MAE_proxy"].nlargest(15).index).round(2).to_string())

results.to_csv(os.path.join(OUT, "correction_folds.csv"), index=False)
summary.to_csv(os.path.join(OUT, "correction_summary.csv"), index=False)
importance.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
cross.to_csv(os.path.join(OUT, "cross_station.csv"), index=False)
autocorr.to_csv(os.path.join(OUT, "error_autocorrelation.csv"), index=False)
by_season.to_csv(os.path.join(OUT, "score_by_season.csv"), index=False)
by_month.to_csv(os.path.join(OUT, "score_by_month.csv"), index=False)
by_hour.to_csv(os.path.join(OUT, "score_by_hour.csv"), index=False)
by_lead.to_csv(os.path.join(OUT, "score_by_lead.csv"), index=False)
overall.to_frame().T.to_csv(os.path.join(OUT, "score_overall.csv"), index=False)
print("\n[ok] tables written to analysis/")

"""Generate the figures used in the written report (matplotlib, PNG)."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import amplitude as amp
from tools.verification import (
    add_error,
    add_time_columns,
    case_summary,
    correction_experiment,
    corrected_series,
    score_by,
    summarise_correction,
    temperature_scores,
)

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C_NAVY = "#123a5f"
C_BLUE = "#2f7fb5"
C_ORANGE = "#e07b39"
C_RED = "#c0392b"
C_GREEN = "#2e8b68"
C_GREY = "#7a8794"


def load(name):
    return pd.read_csv(
        os.path.join(ROOT, "data", name), parse_dates=["time"], encoding="utf-8-sig"
    )


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


nanjing = add_time_columns(add_error(load("sample_weather_real.csv")))
lead = add_error(load("gfs_leadtime.csv"))
shanghai = add_time_columns(add_error(load("zsss_shanghai_real.csv")))

# ---------------------------------------------------------------- 图 1
month_range = amp.range_by_month(nanjing)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
width = 0.38
ax.bar(
    month_range["month"] - width / 2,
    month_range["obs_range"],
    width,
    label="观测",
    color=C_ORANGE,
)
ax.bar(
    month_range["month"] + width / 2,
    month_range["fcst_range"],
    width,
    label="GFS 预报",
    color=C_BLUE,
)
ax.set_xlabel("月份")
ax.set_ylabel("日较差 (℃)")
ax.set_title("图 1  逐月平均日较差：模式系统性压扁了温度变化的幅度")
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(axis="y", alpha=0.3)
save(fig, "fig1_diurnal_range.png")

# ---------------------------------------------------------------- 图 2
regression = amp.anomaly_regression(nanjing)
bins = amp.anomaly_binned(nanjing)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
colors = [C_RED if value > 0 else C_BLUE for value in bins["mean_error"]]
ax.bar(bins["mean_anomaly"], bins["mean_error"], width=1.4, color=colors, label="分箱平均误差")
fit_x = np.linspace(bins["mean_anomaly"].min(), bins["mean_anomaly"].max(), 50)
ax.plot(
    fit_x,
    regression["intercept"] + regression["slope"] * fit_x,
    color=C_NAVY,
    linewidth=2.2,
    linestyle="--",
    label=f"线性拟合：斜率 {regression['slope']:+.3f}（r = {regression['correlation']:.2f}）",
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("观测距平（相对月份 × 北京时气候态，℃）")
ax.set_ylabel("平均误差 (℃)")
ax.set_title("图 2  误差与观测距平成反比：该冷时偏暖，该热时偏冷")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save(fig, "fig2_anomaly_error.png")

# ---------------------------------------------------------------- 图 3
results = correction_experiment(nanjing)
summary = summarise_correction(results)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
colors = [C_GREY, C_GREY, C_BLUE, C_GREY, C_BLUE, C_ORANGE]
labels = [
    "① 原始\n预报",
    "② 常数\n去偏",
    "③ 按月\n去偏",
    "④ 按小时\n去偏",
    "⑤ 月×小时\n去偏",
    "⑥ 随机\n森林",
]
bars = ax.bar(
    labels, summary["MAE_mean"], color=colors, yerr=summary["MAE_std"], capsize=4
)
for bar, value in zip(bars, summary["MAE_mean"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.03,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("MAE (℃)")
ax.set_title("图 3  订正方法消融对比（误差棒为三折标准差）")
ax.set_ylim(0, summary["MAE_mean"].max() * 1.15)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig3_ablation.png")

# ---------------------------------------------------------------- 图 4
hour = score_by(nanjing, "bjt")
month = score_by(nanjing, "month")
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
axes[0].bar(hour["bjt"], hour["bias"], color=C_BLUE)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("北京时")
axes[0].set_ylabel("bias (℃)")
axes[0].set_title("(a) 误差的日变化")
axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(
    month["month"],
    month["bias"],
    color=[C_RED if value > 0 else C_BLUE for value in month["bias"]],
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("月份")
axes[1].set_ylabel("bias (℃)")
axes[1].set_title("(b) 误差的季节变化")
axes[1].grid(axis="y", alpha=0.3)
fig.suptitle("图 4  误差既随季节变号，也随日变化变号", y=1.03)
save(fig, "fig4_season_diurnal.png")

# ---------------------------------------------------------------- 图 5
lead_score = score_by(lead, "lead_day")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(lead_score["lead_day"], lead_score["RMSE"], "o-", color=C_NAVY, linewidth=2.2, label="RMSE")
ax.plot(lead_score["lead_day"], lead_score["MAE"], "s-", color=C_BLUE, linewidth=2.2, label="MAE")
ax.plot(lead_score["lead_day"], lead_score["bias"], "^--", color=C_ORANGE, linewidth=1.8, label="bias")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("预报时效（天）")
ax.set_ylabel("℃")
ax.set_title("图 5  误差随时效增长，但偏差基本不变")
ax.set_xticks(lead_score["lead_day"])
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig5_leadtime.png")

# ---------------------------------------------------------------- 图 6
tide = nanjing[(nanjing["time"] >= "2024-01-19") & (nanjing["time"] < "2024-01-27")]
train = nanjing[nanjing["time"] < "2024-01-19"]
series = corrected_series(tide, train)
fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.2), sharex=True, height_ratios=[2.1, 1.0])
axes[0].plot(series["time"], series["observed_temperature"], color=C_NAVY, linewidth=2.3, label="观测")
axes[0].plot(
    series["time"],
    series["forecast_temperature"],
    color=C_ORANGE,
    linewidth=1.7,
    linestyle=":",
    label="GFS 原始预报",
)
axes[0].plot(series["time"], series["corrected_rf"], color=C_GREEN, linewidth=1.7, label="随机森林订正后")
axes[0].set_ylabel("2 米气温 (℃)")
axes[0].set_title("图 6  2024 年 1 月寒潮过程（订正模型仅用过程之前的数据训练）")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(series["time"], series["raw_error"], color=C_ORANGE, linewidth=1.8, label="原始预报误差")
axes[1].plot(series["time"], series["rf_error"], color=C_GREEN, linewidth=1.8, label="订正后误差")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("误差 (℃)")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.autofmt_xdate()
save(fig, "fig6_cold_wave.png")

# ---------------------------------------------------------------- 图 7
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
for ax, frame, label in zip(axes, (nanjing, shanghai), ("南京 ZSNJ", "上海 ZSSS")):
    table = amp.range_by_month(frame)
    ratio = table["fcst_range"].sum() / table["obs_range"].sum()
    ax.plot(table["month"], table["obs_range"], "o-", color=C_ORANGE, label="观测")
    ax.plot(table["month"], table["fcst_range"], "s-", color=C_BLUE, label="预报")
    ax.set_title(f"{label}（全年比值 {ratio:.2f}）")
    ax.set_xlabel("月份")
    ax.set_ylabel("日较差 (℃)")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("图 7  振幅压缩是站点相关的：南京明显，上海几乎没有", y=1.03)
save(fig, "fig7_cross_station.png")

print("\n总体检验:", {k: round(v, 4) for k, v in temperature_scores(nanjing).items()})
print("寒潮案例:", case_summary(series).round(3).to_dict("records"))

# ---------------------------------------------------------------- 图 8
minmax = amp.minmax_bias(nanjing)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
width = 0.38
ax.bar(
    minmax["month"] - width / 2,
    minmax["最低气温偏差"],
    width,
    label="日最低气温偏差",
    color=C_BLUE,
)
ax.bar(
    minmax["month"] + width / 2,
    minmax["最高气温偏差"],
    width,
    label="日最高气温偏差",
    color=C_ORANGE,
)
ax.axhline(0, color="black", linewidth=0.9)
ax.set_xlabel("月份")
ax.set_ylabel("偏差 (℃)")
ax.set_title("图 8  误差的落点：冬季夜间最低气温偏暖，日最高气温几乎无偏差")
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(axis="y", alpha=0.3)
save(fig, "fig8_minmax_bias.png")

"""Concept figure: what 'amplitude damping' looks like in the actual data."""

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

from tools.amplitude import daily_range  # noqa: E402

C_NAVY = "#123a5f"
C_ORANGE = "#e07b39"
C_BLUE = "#2f7fb5"
C_GREY = "#9aa5b1"

data = pd.read_csv(
    os.path.join(ROOT, "data", "sample_weather_real.csv"),
    parse_dates=["time"],
    encoding="utf-8-sig",
)
data["error"] = data["forecast_temperature"] - data["observed_temperature"]
data["date"] = data["time"].dt.normalize()

# 选一个“晴天型”的三天窗口：日变化清楚、压缩明显、又不是极端天气过程
daily = daily_range(data)
daily["month"] = daily["date"].dt.month
candidates = daily[(daily["month"] == 12) & (daily["obs_range"] > 10)].head(30)
best_day = candidates.iloc[1]["date"] if len(candidates) > 1 else daily.iloc[0]["date"]
start = pd.Timestamp(best_day)
window = data[(data["time"] >= start) & (data["time"] < start + pd.Timedelta(days=3))]
print("窗口:", start.date(), "~", (start + pd.Timedelta(days=3)).date(), "样本", len(window))
print(
    "观测日较差均值 %.2f, 预报 %.2f"
    % (
        window.groupby(window["time"].dt.normalize())["observed_temperature"]
        .apply(lambda s: s.max() - s.min())
        .mean(),
        window.groupby(window["time"].dt.normalize())["forecast_temperature"]
        .apply(lambda s: s.max() - s.min())
        .mean(),
    )
)

fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.plot(window["time"], window["observed_temperature"], color=C_NAVY, linewidth=2.6, label="真实观测")
ax.plot(
    window["time"],
    window["forecast_temperature"],
    color=C_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="GFS 预报",
)

# 只在第一天做两组对照标注：峰值被压低、谷值被抬高
first_day = window[window["time"] < start + pd.Timedelta(days=1)]
obs_peak = first_day.loc[first_day["observed_temperature"].idxmax()]
obs_trough = first_day.loc[first_day["observed_temperature"].idxmin()]
fc_at_peak = first_day.loc[(first_day["time"] - obs_peak["time"]).abs().idxmin(), "forecast_temperature"]
fc_at_trough = first_day.loc[(first_day["time"] - obs_trough["time"]).abs().idxmin(), "forecast_temperature"]

for x, y_from, y_to, label, color in (
    (obs_peak["time"], obs_peak["observed_temperature"], fc_at_peak, "峰值被压低", C_NAVY),
    (obs_trough["time"], obs_trough["observed_temperature"], fc_at_trough, "谷值被抬高", C_NAVY),
):
    ax.annotate(
        "",
        xy=(x, y_to),
        xytext=(x, y_from),
        arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.3, shrinkA=3, shrinkB=3),
        zorder=6,
    )
    ax.annotate(
        label,
        xy=(x, (y_from + y_to) / 2),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=7,
    )

obs_range = (
    window.groupby(window["time"].dt.normalize())["observed_temperature"]
    .apply(lambda s: s.max() - s.min())
    .mean()
)
fcst_range = (
    window.groupby(window["time"].dt.normalize())["forecast_temperature"]
    .apply(lambda s: s.max() - s.min())
    .mean()
)
ax.text(
    0.985,
    0.06,
    f"三天平均日较差：观测 {obs_range:.1f} ℃ → 预报 {fcst_range:.1f} ℃",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4f9", edgecolor="#cfdae4"),
)

ax.set_ylabel("2 米气温 (℃)")
ax.set_title("振幅阻尼长什么样：观测的峰谷被模式压平了", fontsize=13)
ax.legend(loc="upper right", ncol=1, fontsize=9.5, framealpha=0.95)
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.savefig(os.path.join(ROOT, "figures", "fig11_damping_concept.png"))
plt.close(fig)
print("wrote fig11_damping_concept.png")

# 第二张：逐日较差散点，点应当落在 1:1 线下方
daily_all = daily_range(data)
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(daily_all["obs_range"], daily_all["fcst_range"], s=8, alpha=0.35, color=C_BLUE)
limit = [0, max(daily_all["obs_range"].max(), daily_all["fcst_range"].max()) + 2]
ax.plot(limit, limit, color=C_GREY, linestyle="--", linewidth=1.4, label="预报 = 观测")
ax.set_xlim(limit)
ax.set_ylim(limit)
ax.set_xlabel("观测日较差 (℃)")
ax.set_ylabel("预报日较差 (℃)")
ax.set_title("每个点是一天：绝大多数落在\n1:1 线下方，即预报起伏更小", fontsize=11)
ratio = daily_all["fcst_range"].sum() / daily_all["obs_range"].sum()
ax.legend(title=f"整体比值 {ratio:.2f}", loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(ROOT, "figures", "fig12_range_scatter.png"))
plt.close(fig)
print("wrote fig12_range_scatter.png")

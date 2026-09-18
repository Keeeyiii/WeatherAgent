"""Figures for the data-assimilation section of the report."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.da import evaluate_methods, sensitivity  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C_NAVY = "#123a5f"
C_BLUE = "#2f7fb5"
C_ORANGE = "#e07b39"
C_GREEN = "#2e8b68"
C_RED = "#c0392b"

dataset = pd.read_csv(
    os.path.join(ROOT, "data", "assimilation_stations.csv"),
    parse_dates=["time"],
    encoding="utf-8-sig",
)

result = evaluate_methods(dataset, length_scale_km=500.0, sigma_o=1.0, split_date="2025-01-01")
table = result["comparison"]
print(table.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.6, 3.5))
bars = ax.bar(
    ["① 背景场\n(GFS 原始预报)", "② 本站气候态\n偏差订正", "③ 最优插值\n(邻站同刻观测)"],
    table["RMSE"],
    color=[C_ORANGE, C_BLUE, C_GREEN],
)
for bar, value, improve in zip(bars, table["RMSE"], table["相对背景场改善 %"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}\n({improve:+.1f}%)" if improve else f"{value:.3f}",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("RMSE (℃)")
ax.set_ylim(0, table["RMSE"].max() * 1.18)
ax.set_title("图 9  资料同化实验：三种做法的检验效果（2025 年起评估）")
ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(FIG, "fig9_da_comparison.png"))
plt.close(fig)
print("wrote fig9")

table2 = sensitivity(dataset, sigma_o_values=(0.5, 1.0, 1.5), split_date="2025-01-01")
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for sigma_o, color in ((0.5, C_NAVY), (1.0, C_BLUE), (1.5, C_ORANGE)):
    subset = table2[table2["sigma_o"] == sigma_o]
    ax.plot(subset["length_scale_km"], subset["oi_RMSE"], "o-", color=color, label=f"σo = {sigma_o} ℃")
ax.axhline(
    float(table.loc[0, "RMSE"]),
    color=C_RED,
    linestyle="--",
    linewidth=1.5,
    label=f"背景场 RMSE = {table.loc[0, 'RMSE']:.2f} ℃",
)
ax.set_xlabel("背景误差相关长度 L (km)")
ax.set_ylabel("留一站检验 RMSE (℃)")
ax.set_title("图 10  OI 分析误差对相关长度与观测误差的敏感性")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "fig10_da_sensitivity.png"))
plt.close(fig)
print("wrote fig10")

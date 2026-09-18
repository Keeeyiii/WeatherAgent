"""Build the multi-station assimilation dataset and run a first OI test."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.da import (  # noqa: E402
    SITES,
    build_assimilation_dataset,
    evaluate_methods,
    innovation_correlation,
    leave_one_out,
    sensitivity,
)

pd.set_option("display.width", 220)

dataset = build_assimilation_dataset("2024-01-01", "2026-09-15")
path = os.path.join(ROOT, "data", "assimilation_stations.csv")
dataset.to_csv(path, index=False, encoding="utf-8-sig")
print(f"数据: {path}  ({len(dataset):,} 行, {dataset['station'].nunique()} 站)")
print("时段:", dataset["time"].min(), "->", dataset["time"].max())
print(dataset.head(3).to_string(index=False))

print("\n各站样本量与背景场误差：")
per = dataset.groupby("station").apply(
    lambda g: pd.Series(
        {
            "n": len(g),
            "name": SITES[g.name].name if g.name in SITES else "",
            "bg_bias": (g["t_bg"] - g["t_obs"]).mean(),
            "bg_MAE": (g["t_bg"] - g["t_obs"]).abs().mean(),
            "bg_RMSE": ((g["t_bg"] - g["t_obs"]) ** 2).mean() ** 0.5,
        }
    ),
    include_groups=False,
)
print(per.round(3).to_string())

print("\n=== 新息的空间相关（判断误差是否存在空间结构）===")
print(innovation_correlation(dataset).round(3).to_string())

print("\n=== 三方法对照（标定期 2024 年，评估期 2025-01 起，L=300 km, σo=1.0 ℃）===")
result = evaluate_methods(
    dataset, length_scale_km=300.0, sigma_o=1.0, split_date="2025-01-01"
)
print(f"标定样本 {result['calibration_size']:,} / 评估样本 {result['evaluation_size']:,}")
print(f"新息总标准差 {result['total_variance'] ** 0.5:.3f} ℃")
print(result["comparison"].round(3).to_string(index=False))
print("\n各站：")
print(result["per_station"].round(3).to_string(index=False))

print("\n=== 相关长度敏感性（评估期同上，方差取标定期）===")
table = sensitivity(dataset, sigma_o_values=(0.5, 1.0, 1.5), split_date="2025-01-01")
print(table.pivot(index="length_scale_km", columns="sigma_o", values="oi_MAE").round(3).to_string())
print("\n对应 RMSE：")
print(table.pivot(index="length_scale_km", columns="sigma_o", values="oi_RMSE").round(3).to_string())
print("\n最优组合:")
best = table.loc[table["oi_RMSE"].idxmin()]
print(best[["length_scale_km", "sigma_o", "oi_MAE", "oi_RMSE"]].round(3).to_string())

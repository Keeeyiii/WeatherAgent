"""Fetch the gridded background field used for the 2-D OI analysis map."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.da import fetch_grid_background, grid_points  # noqa: E402

# 2024 年 1 月寒潮过程（与真实案例页呼应）：2024-01-22 00 UTC 至 2024-01-24 23 UTC
POINTS = grid_points(29.5, 32.5, 116.5, 122.0, 0.25)
print(f"网格点数: {len(POINTS)}")

grid = fetch_grid_background(POINTS, "2024-01-22", "2024-01-24")
path = os.path.join(ROOT, "data", "assimilation_grid.csv")
grid.to_csv(path, index=False, encoding="utf-8-sig")
print(f"已保存 {path}（{len(grid):,} 行, {grid['time'].nunique()} 个时刻）")
print(grid.groupby("time")["t_bg"].agg(["min", "max"]).head(6).round(2).to_string())

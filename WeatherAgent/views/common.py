"""Shared visual helpers and cached data access for the app."""

from __future__ import annotations

import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import amplitude as amp  # noqa: E402
from tools.verification import (  # noqa: E402
    add_error,
    add_time_columns,
    case_summary,
    correction_experiment,
    corrected_series,
    cross_station_experiment,
    feature_importance,
    summarise_correction,
    temperature_scores,
)

DATA = os.path.join(ROOT, "data")

# 配色：以气象常见的蓝-橙对比为主
C_NAVY = "#123a5f"
C_BLUE = "#2f7fb5"
C_ORANGE = "#e07b39"
C_RED = "#c0392b"
C_GREY = "#7a8794"
C_GREEN = "#2e8b68"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2.0rem; max-width: 1180px; }}
          h1, h2, h3 {{ color: {C_NAVY}; }}
          .hero {{
              background: linear-gradient(105deg, {C_NAVY} 0%, {C_BLUE} 100%);
              color: #fff; padding: 1.4rem 1.7rem; border-radius: 12px;
              margin-bottom: 1.0rem;
          }}
          .hero h1 {{ color: #fff; margin: 0 0 .3rem 0; font-size: 1.95rem; }}
          .hero p {{ color: #d8e6f2; margin: 0; font-size: .96rem; line-height: 1.6; }}
          .card {{
              background: #f6f9fc; border: 1px solid #dde7f0;
              border-left: 4px solid {C_BLUE}; border-radius: 8px;
              padding: .8rem 1.0rem; height: 100%;
          }}
          .card .num {{ font-size: 1.55rem; font-weight: 700; color: {C_NAVY}; line-height: 1.2; }}
          .card .lbl {{ font-size: .80rem; color: {C_GREY}; margin-top: .15rem; }}
          .qbox {{
              background: #fff8ef; border: 1px solid #f2d9bd;
              border-left: 4px solid {C_ORANGE}; border-radius: 8px;
              padding: .85rem 1.05rem; line-height: 1.75;
          }}
          .abox {{
              background: #f1f8f4; border: 1px solid #cfe7da;
              border-left: 4px solid {C_GREEN}; border-radius: 8px;
              padding: .85rem 1.05rem; line-height: 1.75;
          }}
          .evi {{ border-left: 3px solid {C_BLUE}; padding-left: .8rem; margin: .6rem 0; line-height: 1.7; }}
          .small {{ color: {C_GREY}; font-size: .84rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(number: str, label: str) -> str:
    return f'<div class="card"><div class="num">{number}</div><div class="lbl">{label}</div></div>'


def cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (number, label) in zip(columns, items):
        column.markdown(card(number, label), unsafe_allow_html=True)


def hero(title: str, subtitle: str, *, large: bool = False) -> None:
    """页面顶部的标题块。`large=True` 用于首页，让标题更突出。"""
    size = "font-size:2.45rem;" if large else ""
    padding = "padding:1.75rem 1.8rem;" if large else ""
    st.markdown(
        f'<div class="hero" style="{padding}">'
        f'<h1 style="{size}">{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        # 顶部留足空间：标题在上、图例在标题下方，避免两者重叠
        margin=dict(l=10, r=10, t=92, b=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        title=dict(x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=14.5)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, x=0.01, font=dict(size=11)
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d7dee6")
    fig.update_yaxes(gridcolor="#eef2f6", zerolinecolor="#c8d2dc")
    return fig


def signed(value: float, digits: int = 2, unit: str = "") -> str:
    """格式化为「−0.21 ℃」的形式（使用真正的减号而不是 hyphen）。"""
    text = f"{value:+.{digits}f}".replace("-", "\u2212")
    return f"{text} {unit}".strip()


def caption(text: str) -> None:
    """小字说明。

    注意：Streamlit 的 `st.caption` 不会解析 `**粗体**`，会把星号原样显示出来，
    所以这里改用 markdown + 自定义样式，保证强调能正常生效。
    """
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(
        '<div style="font-size:0.82rem;color:#7a8794;line-height:1.62;'
        f'margin:0.15rem 0 0.35rem 0;">{html}</div>',
        unsafe_allow_html=True,
    )


def go_to(page_key: str) -> None:
    """切换到 app.py 注册的页面，并立即重跑以显示目标页。"""
    st.session_state["active_page"] = page_key
    st.rerun()


# --------------------------------------------------------------------------- #
# cached data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_nanjing() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA, "sample_weather_real.csv"),
        parse_dates=["time"],
        encoding="utf-8-sig",
    )
    return add_time_columns(add_error(df))


@st.cache_data(show_spinner=False)
def load_shanghai() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA, "zsss_shanghai_real.csv"),
        parse_dates=["time"],
        encoding="utf-8-sig",
    )
    return add_time_columns(add_error(df))


@st.cache_data(show_spinner=False)
def load_leadtime() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA, "gfs_leadtime.csv"),
        parse_dates=["time", "init_time"],
        encoding="utf-8-sig",
    )
    return add_error(df)


@st.cache_data(show_spinner="正在运行订正实验（扩张窗口、严格时序三折验证）…")
def run_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = correction_experiment(load_nanjing())
    return results, summarise_correction(results)


@st.cache_data(show_spinner=False)
def run_importance() -> pd.DataFrame:
    return feature_importance(load_nanjing())


@st.cache_data(show_spinner="正在做跨站点迁移检验…")
def run_cross_station() -> pd.DataFrame:
    return cross_station_experiment(load_nanjing(), load_shanghai())


@st.cache_data(show_spinner=False)
def run_case(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """样本外案例：只用过程开始之前的数据训练订正模型。"""
    nj = load_nanjing()
    window = nj[(nj["time"] >= start) & (nj["time"] < end)]
    train = nj[nj["time"] < start]
    series = corrected_series(window, train)
    return series, case_summary(series)


@st.cache_data(show_spinner=False)
def headline_numbers() -> dict:
    nj = load_nanjing()
    scores = temperature_scores(nj)
    rng = amp.range_summary(nj)
    extremes = amp.extreme_error(nj)
    regression = amp.anomaly_regression(nj)
    return {
        "n": int(scores["n"]),
        "mae": scores["MAE"],
        "rmse": scores["RMSE"],
        "bias": scores["bias"],
        "corr": scores["corr"],
        "obs_range": rng["obs_range"],
        "fcst_range": rng["fcst_range"],
        "damping": rng["damping_pct"],
        "cold_error": extremes.loc[0, "平均误差 (℃)"],
        "warm_error": extremes.loc[2, "平均误差 (℃)"],
        "slope": regression["slope"],
        "intercept": regression["intercept"],
        "reg_r": regression["correlation"],
        "start": nj["time"].min(),
        "end": nj["time"].max(),
    }


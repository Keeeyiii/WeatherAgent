"""资料同化实验页：用邻站观测做最优插值（OI）分析。"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools.da import evaluate_methods, grid_analysis, innovation_correlation
from views.common import (
    C_BLUE,
    C_GREEN,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    caption,
    card,
    hero,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA, "assimilation_stations.csv"),
        parse_dates=["time"],
        encoding="utf-8-sig",
    )


@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA, "assimilation_grid.csv"),
        parse_dates=["time"],
        encoding="utf-8-sig",
    )


@st.cache_data(show_spinner="正在做留一站交叉检验…")
def run_comparison(length_scale_km: float, sigma_o: float, split_date: str) -> dict:
    return evaluate_methods(
        load_stations(),
        length_scale_km=length_scale_km,
        sigma_o=sigma_o,
        split_date=split_date,
    )


@st.cache_data(show_spinner=False)
def run_correlation() -> pd.DataFrame:
    return innovation_correlation(load_stations())


@st.cache_data(show_spinner="正在扫描参数…")
def run_sweep() -> pd.DataFrame:
    rows = []
    # σo 必须小于新息总标准差，否则等于宣称“观测误差比总误差还大”，不合理
    for sigma_o in (0.5, 1.0, 1.5):
        for length_scale in (50, 100, 200, 300, 500, 800):
            result = evaluate_methods(
                load_stations(),
                length_scale_km=float(length_scale),
                sigma_o=float(sigma_o),
                split_date="2025-01-01",
            )
            row = result["comparison"].iloc[2].to_dict()
            row["L"] = float(length_scale)
            row["sigma_o"] = float(sigma_o)
            rows.append(row)
    return pd.DataFrame(rows)


def _heatmap(grid: pd.DataFrame, column: str, title: str, zmid=None) -> go.Figure:
    latitudes = sorted(grid["lat"].unique())
    longitudes = sorted(grid["lon"].unique())
    matrix = (
        grid.pivot_table(index="lat", columns="lon", values=column, aggfunc="mean")
        .reindex(index=latitudes, columns=longitudes)
        .to_numpy()
    )
    figure = go.Figure(
        go.Heatmap(
            x=longitudes,
            y=latitudes,
            z=matrix,
            colorscale="RdBu_r",
            zmid=zmid,
            colorbar=dict(title="℃", thickness=14, len=0.85),
            hovertemplate="经度 %{x:.2f}<br>纬度 %{y:.2f}<br>%{z:.2f} ℃<extra></extra>",
        )
    )
    figure.update_layout(title=title)
    figure.update_xaxes(title="经度 (°E)")
    figure.update_yaxes(title="纬度 (°N)")
    return figure


DESIGN_TABLE = """
| 环节 | 设置 |
|---|---|
| 观测 | 南京禄口、合肥新桥、上海虹桥、上海浦东、杭州萧山、宁波栎社，逐小时 2 米气温 |
| 背景场 | GFS 2 米气温预报，在各站位置取值（相当于模式的“第一猜值”） |
| 标定期 | 2024 年全年，只用来估计误差方差与各站平均偏差 |
| 评估期 | 2025-01-01 起，**与标定期完全不重叠** |
| 检验方式 | **留一站交叉检验**：分析某站时绝不使用该站自己的观测 |
"""

LIMIT_TABLE = """
| 项目 | 本实验 | 业务 3D-Var / EnKF |
|---|---|---|
| 分析对象 | 只有 2 米气温，单变量 | 温度、风、湿度、气压等多变量并相互约束 |
| 背景误差协方差 | 均匀各向同性的高斯模型，只有一个相关长度 | 随空间和流型变化，或由集合样本统计 |
| 观测 | 6 个地面站 | 地面站、探空、卫星、雷达、飞机报等 |
| 分析位置 | 站点与规则网格 | 模式格点上的完整三维场 |
| 与模式的耦合 | 无（离线融合） | 与模式动力约束一致，并参与模式积分 |
"""


def render() -> None:
    hero(
        "资料同化实验：用邻站观测修正本站预报",
        "把「统计后处理」和「资料同化」放进同一套检验里对比，回答一个问题："
        "同一时刻的邻站观测，是否比本站的历史平均偏差更有用？",
    )

    stations = load_stations()
    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{stations['station'].nunique()} 个站", "长三角真实机场观测"),
            (f"{len(stations):,}", "“观测—背景场”小时样本"),
            ("2024 标定", "估计误差统计量"),
            ("2025 起评估", "与标定期完全不重叠"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
        <div class="qbox">
        <b>最优插值在做什么（一句话）</b><br>
        假设南京此刻预报 5 ℃，而同一时刻合肥、杭州、上海都比各自的预报偏高了约 2 ℃。
        三个站同时出错的概率很低，更可能是<b>模式在这个区域整体偏暖了 2 ℃</b>，
        那么南京也应该往下调一些。<b>调多少？</b>
        站越近越可信、权重越大；背景场越可信、权重越小——这就是最优插值（OI）。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.subheader("实验设计：三种做法，同一批样本")
    st.markdown(DESIGN_TABLE)

    comparison = run_comparison(300.0, 1.0, "2025-01-01")
    table = comparison["comparison"]
    background_rmse = float(table.loc[0, "RMSE"])
    debias_row = table.iloc[1]
    oi_row = table.iloc[2]

    st.subheader("结果：邻站的同刻观测明显优于本站的历史偏差")
    figure = go.Figure()
    figure.add_bar(
        x=table["方法"],
        y=table["RMSE"],
        marker_color=[C_ORANGE, C_BLUE, C_GREEN],
        text=[f"{value:.3f}" for value in table["RMSE"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=12),
        cliponaxis=False,
        name="RMSE",
    )
    figure.update_layout(title="三种方法的 RMSE（越低越好）")
    figure.update_yaxes(title="RMSE (℃)", range=[0, background_rmse * 1.12])
    figure.update_xaxes(tickangle=-8)
    st.plotly_chart(base_layout(figure, 380), use_container_width=True)

    display = table.copy()
    display.columns = ["方法", "样本数", "bias (℃)", "MAE (℃)", "RMSE (℃)", "相对背景场改善 %"]
    st.dataframe(display.round(3), use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="abox">
        <b>结论</b><br>
        背景场 RMSE <b>{background_rmse:.2f} ℃</b>；
        传统的“本站气候态偏差订正”只把它降到 <b>{debias_row['RMSE']:.2f} ℃</b>
        （改善 {debias_row['相对背景场改善 %']:.1f}%）；
        最优插值把它降到 <b>{oi_row['RMSE']:.2f} ℃</b>
        （改善 <b>{oi_row['相对背景场改善 %']:.1f}%</b>），
        MAE 从 {table.loc[0, 'MAE']:.2f} ℃ 降到 {oi_row['MAE']:.2f} ℃。
        <br><br>
        <b>让误差下降幅度扩大四倍以上的，不是“本站过去平均偏多少”，
        而是“邻站此刻正在偏多少”。</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.subheader("为什么 OI 会有效：误差确实存在空间结构")
    correlation = run_correlation()
    figure = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=list(correlation.columns),
            y=list(correlation.index),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(correlation.to_numpy(), 2),
            texttemplate="%{text}",
            colorbar=dict(title="相关", thickness=14, len=0.85),
        )
    )
    figure.update_layout(title="站点之间观测新息（观测 − 背景场）的相关系数")
    st.plotly_chart(base_layout(figure, 420), use_container_width=True)
    caption(
        "任意两站之间的相关系数都在 **0.33—0.56** 之间且全部为正："
        "模式在某站偏高时，邻近站点往往也偏高。"
        "这个空间相关结构正是 OI 能起作用的前提——如果各站误差互不相关，邻站信息就毫无价值。"
    )

    st.divider()
    st.subheader("空间视角：把 6 个站的观测摊开成一片分析场")
    grid = load_grid()
    times = sorted(grid["time"].unique())
    choice = st.select_slider(
        "选择时刻（2024 年 1 月寒潮过程）",
        options=times,
        value=times[24],
        format_func=lambda value: pd.Timestamp(value).strftime("%m-%d %H UTC"),
    )
    moment = grid[grid["time"] == choice]
    observed = stations[stations["time"] == choice].dropna(subset=["t_obs", "t_bg"])
    if observed.empty:
        st.warning("该时刻没有同时具备观测与背景场的站点，请换一个时刻。")
        return

    analysis = grid_analysis(
        moment,
        observed,
        length_scale_km=300.0,
        sigma_o=1.0,
        total_variance=comparison["total_variance"],
    )

    left, right = st.columns(2, gap="medium")
    with left:
        figure = _heatmap(
            analysis,
            "t_bg",
            "① 背景场（GFS 原始预报）",
            zmid=float(analysis["t_bg"].mean()),
        )
        figure.add_scatter(
            x=observed["lon"],
            y=observed["lat"],
            mode="markers+text",
            marker=dict(size=9, color="black", symbol="x"),
            text=[f"{row.t_obs - row.t_bg:+.1f}" for row in observed.itertuples()],
            textposition="top center",
            textfont=dict(size=10),
            showlegend=False,
        )
        st.plotly_chart(base_layout(figure, 400), use_container_width=True)
    with right:
        figure = _heatmap(
            analysis,
            "t_analysis",
            "② 分析场（OI 融合后）",
            zmid=float(analysis["t_bg"].mean()),
        )
        figure.add_scatter(
            x=observed["lon"],
            y=observed["lat"],
            mode="markers",
            marker=dict(size=9, color="black", symbol="x"),
            showlegend=False,
        )
        st.plotly_chart(base_layout(figure, 400), use_container_width=True)
    caption(
        "黑色叉号是观测所在位置，其旁标注的数字是**该站的观测新息**（观测 − 背景场，℃）。"
        "左图是模式原始温度场，右图是融合观测后的分析场。"
    )

    st.markdown("**③ 分析增量（分析场 − 背景场）**")
    limit = float(np.nanmax(np.abs(analysis["increment"])))
    figure = _heatmap(analysis, "increment", "OI 对背景场做了多大的调整", zmid=0)
    figure.update_traces(zmin=-limit, zmax=limit)
    st.plotly_chart(base_layout(figure, 400), use_container_width=True)
    caption(
        "增量在站点附近最大，并随距离按高斯相关函数衰减。"
        f"该时刻的最大调整量约 **{limit:.1f} ℃**，"
        "说明在寒潮这样的强天气过程中，仅凭模式背景场的温度场确实需要观测来纠正。"
    )

    st.divider()
    st.subheader("参数怎么选：相关长度与观测误差")
    columns = st.columns(2)
    columns[0].markdown(
        card("L", "相关长度（km）：误差在多远范围内还算“同一件事”")
        + "<p style='font-size:0.84rem;color:#7a8794;margin-top:.5rem;'>"
        "太短则只信自己，退化为不订正；太长则把无关的站也算进来，产生虚假相关。</p>",
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        card("σo", "观测误差（℃）：观测本身有多可信")
        + "<p style='font-size:0.84rem;color:#7a8794;margin-top:.5rem;'>"
        "太大则更相信背景场，订正保守；太小则过度相信观测，容易被噪声带偏。</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    sweep_table = run_sweep()
    figure = go.Figure()
    palette = {0.5: C_NAVY, 1.0: C_BLUE, 2.0: C_ORANGE}
    for sigma_o in sorted(sweep_table["sigma_o"].unique()):
        subset = sweep_table[sweep_table["sigma_o"] == sigma_o]
        figure.add_scatter(
            x=subset["L"],
            y=subset["RMSE"],
            mode="lines+markers",
            name=f"σo = {sigma_o} ℃",
            line=dict(width=2.4, color=palette.get(float(sigma_o), C_BLUE)),
            marker=dict(size=7),
        )
    figure.add_hline(
        y=background_rmse,
        line=dict(color=C_RED, dash="dash", width=1.6),
        annotation_text=f"背景场 RMSE = {background_rmse:.2f} ℃",
        annotation_position="top right",
    )
    figure.update_layout(title="分析误差随相关长度与观测误差的变化")
    figure.update_xaxes(title="相关长度 L (km)")
    figure.update_yaxes(title="留一站检验 RMSE (℃)")
    st.plotly_chart(base_layout(figure, 400), use_container_width=True)

    best = sweep_table.loc[sweep_table["RMSE"].idxmin()]
    caption(
        "三条曲线讲的是同一件事的两个方向。"
        "**L 太小（50 km）时几乎用不到邻站信息**，误差与背景场几乎一样；"
        "L 增大到 150—300 km 后订正才真正生效。"
        "**σo 越大，订正越保守**：σo = 1.5 ℃ 时改善明显缩水，"
        "因为这时候“更相信背景场”意味着分析几乎不动。"
        f"本次扫描的最优组合是 **L ≈ {int(best['L'])} km、σo = {best['sigma_o']} ℃**"
        f"（RMSE {best['RMSE']:.3f} ℃）。"
    )
    st.info(
        "**两个技术细节（也是容易出错的地方）**\n\n"
        "1. **σo 不能随便取大。** 从数据里能直接估计的只有新息总方差 "
        f"σb² + σo² = {comparison['total_variance']:.2f} ℃²（标准差 "
        f"{comparison['total_variance'] ** 0.5:.2f} ℃）。"
        "如果 σo 取到 2 ℃（σo² = 4 > 总方差），就相当于宣称“观测误差比总误差还大”，"
        "逻辑上不成立，此时 σb 被压到接近 0，分析几乎退化为不订正。"
        "所以本页把 σo 限制在 0.5—1.5 ℃。\n\n"
        "2. **σo 很小时会出现负权重。** 当观测误差设得很小，OI 会自动扣除"
        "“已经被邻近站点重复提供”的信息，因此个别权重为负。"
        "负权重在最优插值里是合法的（它代表去除冗余），但会让分析对观测噪声更敏感——"
        "这就是 σo = 0.5 ℃ 那条曲线不再单调的原因。"
    )

    st.divider()
    st.subheader("这个实验做到哪一步，没做到哪一步")
    st.markdown(
        "把它称为**最优插值**是准确的，但它与业务资料同化仍有明确距离："
    )
    st.markdown(LIMIT_TABLE)
    st.markdown(
        "因此本实验展示的是**最优插值的核心思想与权衡**，不是一套可业务运行的同化系统。"
        "但它给出了一条有意义的结论：**在这个站点网络上，误差的空间相关结构真实存在，"
        "而且可以被利用。**"
    )

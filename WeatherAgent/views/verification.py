"""误差检验页：从四个角度给误差画一张画像。"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tools.verification import score_by
from views.common import (
    caption,
    C_BLUE,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    card,
    headline_numbers,
    hero,
    load_leadtime,
    load_nanjing,
)


def render() -> None:
    hero(
        "误差检验",
        "从总体、季节、日变化、预报时效四个角度给误差画一张画像，"
        "并从中找出值得追问的线索",
    )
    nj = load_nanjing()
    lead = load_leadtime()
    head = headline_numbers()

    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{head['bias']:+.2f} ℃", "平均偏差 bias"),
            (f"{head['mae']:.2f} ℃", "MAE"),
            (f"{head['rmse']:.2f} ℃", "RMSE"),
            (f"{head['corr']:.3f}", "预报与观测相关系数"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    st.info(
        f"先注意这一对数字的落差：**bias 只有 {head['bias']:+.2f} ℃，MAE 却有 {head['mae']:.2f} ℃**。"
        "平均偏差很小，说明误差的主导成分不是“整体偏暖或偏冷”。"
        "接下来四个角度，都是为了找出这个落差背后的结构。"
    )

    st.subheader("① 季节：冬季偏暖、夏季偏冷，符号完全相反")
    season = score_by(nj, "season")
    fig = go.Figure()
    fig.add_bar(
        x=season["season"],
        y=season["bias"],
        marker_color=[C_RED if value > 0 else C_BLUE for value in season["bias"]],
        text=[f"{value:+.2f}" for value in season["bias"]],
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(title="各季节平均偏差（正 = 模式偏暖）")
    low = min(0.0, float(season["bias"].min()))
    high = max(0.0, float(season["bias"].max()))
    span = high - low
    fig.update_yaxes(title="bias (℃)", range=[low - 0.18 * span, high + 0.22 * span])
    st.plotly_chart(base_layout(fig, 330), use_container_width=True)
    winter = season.loc[season["season"] == "冬季 DJF", "bias"].iloc[0]
    summer = season.loc[season["season"] == "夏季 JJA", "bias"].iloc[0]
    caption(
        f"冬季 {winter:+.2f} ℃、夏季 {summer:+.2f} ℃。这是一个关键线索："
        "**误差不是一个恒定的偏移量，它随季节改变正负号。**"
        "这意味着任何“减掉一个常数”的订正都不可能有效。"
    )

    st.subheader("② 月份：误差最大的月份最需要解释")
    month = score_by(nj, "month")
    fig = go.Figure()
    fig.add_bar(x=month["month"], y=month["RMSE"], name="RMSE", marker_color=C_NAVY)
    fig.add_scatter(
        x=month["month"],
        y=month["bias"],
        name="bias",
        mode="lines+markers",
        marker_color=C_ORANGE,
        line=dict(width=3),
    )
    fig.update_layout(title="各月 RMSE 与 bias")
    fig.update_xaxes(dtick=1, title="月份")
    fig.update_yaxes(title="℃")
    st.plotly_chart(base_layout(fig, 350), use_container_width=True)
    jan = month.loc[month["month"] == 1].iloc[0]
    aug = month.loc[month["month"] == 8].iloc[0]
    caption(
        f"1 月 RMSE 最大（{jan['RMSE']:.2f} ℃）、bias 也最大（{jan['bias']:+.2f} ℃）；"
        f"8 月 bias 转为 {aug['bias']:+.2f} ℃。"
        "冬季误差既大又偏暖 —— 这是后面真实案例重点检查的时段。"
    )

    st.subheader("③ 日变化：白天偏冷、夜间偏暖")
    hour = score_by(nj, "bjt")
    fig = go.Figure()
    fig.add_bar(x=hour["bjt"], y=hour["bias"], name="bias", marker_color=C_BLUE)
    fig.add_scatter(
        x=hour["bjt"],
        y=hour["MAE"],
        name="MAE",
        mode="lines+markers",
        marker_color=C_ORANGE,
        line=dict(width=3),
    )
    fig.update_layout(title="按北京时小时的误差日变化")
    fig.update_xaxes(dtick=2, title="北京时")
    fig.update_yaxes(title="℃")
    st.plotly_chart(base_layout(fig, 350), use_container_width=True)
    caption(
        "偏差从清晨的 +0.52 ℃ 下降到傍晚的 −0.92 ℃，跨度约 1.4 ℃。"
        "**模式对温度日变化的还原存在系统性的形状偏差**，"
        "这是核心发现的第一个线索：误差不但有大小，还有“形状”。"
    )

    st.subheader("④ 预报时效：误差随时效增长，但偏差几乎不变")
    lead_score = score_by(lead, "lead_day")
    fig = go.Figure()
    fig.add_scatter(
        x=lead_score["lead_day"],
        y=lead_score["RMSE"],
        mode="lines+markers",
        name="RMSE",
        marker_color=C_NAVY,
        line=dict(width=3),
    )
    fig.add_scatter(
        x=lead_score["lead_day"],
        y=lead_score["MAE"],
        mode="lines+markers",
        name="MAE",
        marker_color=C_BLUE,
        line=dict(width=3),
    )
    fig.add_scatter(
        x=lead_score["lead_day"],
        y=lead_score["bias"],
        mode="lines+markers",
        name="bias",
        marker_color=C_ORANGE,
        line=dict(width=2, dash="dot"),
    )
    fig.update_layout(title="误差随预报时效的增长")
    fig.update_xaxes(title="预报时效（天）", dtick=1)
    fig.update_yaxes(title="℃")
    st.plotly_chart(base_layout(fig, 350), use_container_width=True)
    growth = lead_score["RMSE"].iloc[-1] / lead_score["RMSE"].iloc[0]
    caption(
        f"RMSE 从 1 天的 {lead_score['RMSE'].iloc[0]:.2f} ℃ 增长到 7 天的 "
        f"{lead_score['RMSE'].iloc[-1]:.2f} ℃（约 {growth:.1f} 倍），"
        f"相关系数从 {lead_score['corr'].iloc[0]:.2f} 降到 {lead_score['corr'].iloc[-1]:.2f}。"
        "但**偏差基本不随时效变化** —— 说明系统性偏差不是初值误差随时间放大的结果，"
        "更像是模式物理过程本身的固有倾向。这一判断直接影响了后续的订正策略："
        "订正的对象不是“初始误差”，而是“模式的系统性行为”。"
    )

    with st.expander("查看完整数据表"):
        st.markdown("**按季节**")
        st.dataframe(season.round(3), use_container_width=True, hide_index=True)
        st.markdown("**按月份**")
        st.dataframe(month.round(3), use_container_width=True, hide_index=True)
        st.markdown("**按北京时小时**")
        st.dataframe(hour.round(3), use_container_width=True, hide_index=True)
        st.markdown("**按预报时效**")
        st.dataframe(lead_score.round(3), use_container_width=True, hide_index=True)

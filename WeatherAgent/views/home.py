"""首页：一屏之内说清“发现了什么、凭什么、有什么用”。

排版顺序刻意做成 问题 → 结论 → 证据 → 意义 → 目录，
并且把核心概念（振幅阻尼）放在第一眼就能看到的位置。
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from tools import amplitude as amp
from views.common import (
    C_BLUE,
    C_ORANGE,
    base_layout,
    card,
    caption,
    headline_numbers,
    hero,
    load_nanjing,
    run_ablation,
    signed,
)

FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def render() -> None:
    head = headline_numbers()
    _, summary = run_ablation()
    best = summary.iloc[-1]
    both_box = summary.loc[summary["method"].str.contains("月×小时", regex=False)].iloc[0]
    constant = summary.loc[summary["method"].str.contains("常数去偏", regex=False)].iloc[0]
    nj = load_nanjing()
    minmax = amp.minmax_summary(nj)

    hero(
        "WeatherAgent · 用真实数据回答一个具体问题",
        f"以南京禄口站（ZSNJ）{head['n']:,} 小时真实观测为基准检验 GFS 2 米气温预报，"
        f"诊断误差的结构，并检验误差订正与资料同化各自能把误差降到多少。"
        f"数据时段 {head['start']:%Y-%m-%d} — {head['end']:%Y-%m-%d}。",
        large=True,
    )

    st.markdown(
        """
        <div class="qbox">
        <b>要回答的问题</b>　GFS 对南京 2 米气温的预报误差，是随机噪声，还是有结构的系统性偏差？
        如果是结构，它长什么样、能不能被订正？
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # ------------------------------------------------------------------ 核心结论
    st.markdown(
        """
        <div style="border:1px solid #dde7f0;border-left:5px solid #2f7fb5;
                    border-radius:8px;padding:.85rem 1.05rem;background:#f8fbfd;">
          <div style="font-size:.74rem;color:#8a95a0;letter-spacing:.14em;">核 心 结 论</div>
          <div style="font-size:1.16rem;font-weight:700;color:#123a5f;margin:.28rem 0 .3rem 0;">
            误差的形态是「振幅阻尼」：模式把温度的起伏压平了
          </div>
          <div style="font-size:.93rem;color:#33475b;line-height:1.72;">
            不是整体偏暖或偏冷，而是<b>该冷的时候不够冷、该热的时候不够热</b>，越极端越明显；
            压缩有明确的落点：<b>冬季夜间的最低气温</b>。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"−{head['damping']:.0f}%", "日较差被压缩（9.18 → 7.68 ℃）"),
            (f"{signed(minmax['winter_min'], 2, '℃')}", "冬季日最低气温偏差（落点）"),
            (f"{signed(minmax['winter_max'], 2, '℃')}", "冬季日最高气温偏差（几乎无偏）"),
            (f"{signed(head['cold_error'], 2, '℃')}", "最冷 1% 时刻的模式偏暖"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    # ------------------------------------------------------------------ 概念图
    st.subheader("一眼看懂：什么叫被压平")
    concept = os.path.join(FIGURES, "fig11_damping_concept.png")
    if os.path.exists(concept):
        st.image(concept, use_container_width=True)
    caption(
        "2024 年 12 月 2 日起三天。观测峰值 22 ℃ 被预报压到约 19.7 ℃，"
        "观测谷值 7 ℃ 被预报抬到约 10 ℃——**该高的时候不够高，该低的时候不够低**。"
    )

    # ------------------------------------------------------------------ 证据
    st.subheader("四条独立证据")
    evidence = [
        (
            "① 日变化",
            f"日较差压缩 <b>{head['damping']:.0f}%</b>（观测 9.18 ℃ → 预报 7.68 ℃）；"
            "冷季只剩 66%，暖季接近 1.00。",
        ),
        (
            "② 强度响应",
            f"误差 = {signed(head['intercept'])} − {abs(head['slope']):.2f} × 观测距平"
            f"（r = {head['reg_r']:.2f}）。<b>天越冷越偏暖，天越热越偏冷。</b>",
        ),
        (
            "③ 极端时刻",
            f"最冷 1% 偏暖 <b>{signed(head['cold_error'], 2, '℃')}</b>，"
            f"最暖 1% 偏冷 <b>{signed(head['warm_error'], 2, '℃')}</b>，"
            f"而全样本平均偏差只有 {signed(head['bias'])}。",
        ),
        (
            "④ 落点",
            f"冬季日最低气温偏高 <b>{signed(minmax['winter_min'], 2, '℃')}</b>，"
            f"日最高气温仅 {signed(minmax['winter_max'], 2, '℃')}——"
            "<b>模式抓不住夜间强辐射降温。</b>",
        ),
    ]
    for start in range(0, len(evidence), 2):
        for column, (title, text) in zip(st.columns(2, gap="large"), evidence[start : start + 2]):
            column.markdown(
                f'<div class="evi"><b>{title}</b>　{text}</div>', unsafe_allow_html=True
            )
    st.caption("这四条在设计上相互独立，并且已经排除了“采样分辨率造成的假象”。详见「核心发现」页。")

    # ------------------------------------------------------------------ 意义
    st.divider()
    st.subheader("这个发现有什么用")
    st.markdown(
        f"""
        **它直接解释了误差订正的结果。** 因为误差的形态是结构而非偏移：

        - 全时段平均偏差只有 {signed(head['bias'])}，所以“减去平均偏差”几乎没用
          （仅改善 {constant['MAE_improvement_pct']:.1f}%）；
        - 日变化偏差在冬夏符号相反，所以“单独按小时去偏”也没用；
        - 只有同时刻画**季节与日变化的耦合**，才能拿到 {both_box['MAE_improvement_pct']:.1f}% 的改善。
        """
    )

    columns = st.columns(2)
    with columns[0]:
        st.markdown(
            f"""
            <div class="abox">
            <b>统计订正（单站历史数据）</b><br>
            MAE <b>{head['mae']:.2f} → {best['MAE_mean']:.2f} ℃</b>
            （改善 {best['MAE_improvement_pct']:.1f}%）。<br>
            但同一套模型迁移到上海站反而变差 11.2%——
            <b>说明它高度依赖站点。</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            """
            <div class="abox">
            <b>资料同化（邻站同刻观测）</b><br>
            长三角 6 站最优插值把 RMSE
            <b>1.77 → 1.44 ℃</b>（改善 18.6%），<br>
            远好于本站历史偏差订正的 3.9%——
            <b>“邻站此刻偏多少”比“本站过去平均偏多少”有用得多。</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ 图表
    st.write("")
    month_range = amp.range_by_month(nj)
    figure = go.Figure()
    figure.add_bar(
        x=month_range["month"],
        y=month_range["obs_range"],
        name="观测日较差",
        marker_color=C_ORANGE,
    )
    figure.add_bar(
        x=month_range["month"],
        y=month_range["fcst_range"],
        name="预报日较差",
        marker_color=C_BLUE,
    )
    figure.update_layout(barmode="group", title="逐月平均日较差：预报被系统性压扁")
    figure.update_xaxes(dtick=1, title="月份")
    figure.update_yaxes(title="日较差 (℃)")
    st.plotly_chart(base_layout(figure, 340), use_container_width=True)
    caption(
        "夏季模式能还原日变化（比值≈1.0），冬季几乎腰斩（比值 0.66）。"
        "这正是冬季系统性偏暖、夏季系统性偏冷的来源。"
    )

    # ------------------------------------------------------------------ 目录
    st.divider()
    st.subheader("这个应用包含什么")
    st.caption(
        "左侧边栏切换页面。「订正工具」是能直接用的工具，"
        "其余页面是围绕南京这一个数据集做的完整研究，顺序为：数据 → 检验 → 诊断 → 订正 → 同化 → 案例 → 局限。"
    )

    pages = [
        ("🧰 订正工具（用你的数据）", "上传你自己的“预报—观测”数据：自动质检、诊断、比较订正方法，并导出订正后的 CSV"),
        ("📐 数据与方法", "数据来源、指标定义，以及为什么必须用严格时序划分而不是随机划分"),
        ("📊 误差检验", "从总体、季节、日变化、预报时效四个角度给误差画一张画像"),
        ("🔍 核心发现：振幅阻尼", "先定义概念、再列出判据，然后用四组独立证据逐条对照"),
        ("🧪 订正实验", "六个嵌套方法的消融对比：订正收益究竟来自哪一部分误差"),
        ("🛰 资料同化实验（OI）", "6 站长三角真实观测的最优插值、留一站检验与参数敏感性"),
        ("🌀 真实案例", "2024 年 1 月寒潮与 2024 年 8 月高温，严格样本外的过程检验"),
        ("🧭 局限与展望", "被数据推翻的假设、已知边界，以及下一步可以做什么"),
        ("🤖 研究助手", "用自然语言查询本项目的计算结果，不需要任何 API Key"),
    ]
    for start in range(0, len(pages), 2):
        for column, (title, text) in zip(st.columns(2), pages[start : start + 2]):
            with column:
                st.markdown(f"**{title}**")
                st.caption(text)

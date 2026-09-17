"""首页：一屏之内说明这个应用做出了什么成果。"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tools import amplitude as amp
from views.common import (
    caption,
    C_BLUE,
    C_ORANGE,
    base_layout,
    card,
    headline_numbers,
    hero,
    load_nanjing,
    run_ablation,
    signed,
)
from tools import amplitude as amplitude_module


def render() -> None:
    head = headline_numbers()
    _, summary = run_ablation()
    best = summary.iloc[-1]

    hero(
        "WeatherAgent · 用真实数据回答一个具体问题",
        f"以南京禄口站（ZSNJ）<b>{head['n']:,} 小时真实观测</b>为基准检验 GFS 2 米气温预报，"
        f"诊断误差的结构，并检验误差订正究竟订正了什么。"
        f"数据时段 {head['start']:%Y-%m-%d} — {head['end']:%Y-%m-%d}。",
    )

    st.markdown(
        """
        <div class="qbox">
        <b>我要回答的问题</b><br>
        GFS 对南京 2 米气温的预报误差，是<b>随机噪声</b>，还是<b>有结构的系统性偏差</b>？
        如果是结构，它长什么样？能被订正吗？
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown(
        f"""
        <div class="abox">
        <b>一句话结论</b><br>
        误差的核心不是“偏暖”或“偏冷”，而是<b>模式压扁了温度的变幅</b>，
        而且压缩有一个明确的落点：<b>冬季夜间的最低气温</b>。
        日较差被系统性低估 <b>{head['damping']:.0f}%</b>，冷过程报得偏暖、暖过程报得偏冷，
        越极端越明显。这解释了为什么“减去平均偏差”式的订正几乎无效，
        而随机森林能把 MAE 从 <b>{head['mae']:.2f} ℃</b> 降到 <b>{best['MAE_mean']:.2f} ℃</b>
        （改善 <b>{best['MAE_improvement_pct']:.1f}%</b>）。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    minmax = amplitude_module.minmax_summary(load_nanjing())
    formula = f"{signed(head['intercept'])} \u2212 {abs(head['slope']):.2f} \u00d7 \u89c2\u6d4b\u8ddd\u5e73"
    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{head['n']:,}", "小时“预报—观测”真实配对"),
            (f"{head['mae']:.2f} ℃", "原始预报 MAE"),
            (f"−{head['damping']:.0f}%", "日较差被模式压缩"),
            (f"{minmax['winter_min']:+.2f} ℃", "冬季日最低气温偏差"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("五条独立证据")
        st.markdown(
            f"""
            <div class="evi">
            <b>证据 1 ｜ 日较差</b><br>
            观测平均日较差 <b>{head['obs_range']:.2f} ℃</b>，预报只有 <b>{head['fcst_range']:.2f} ℃</b>。
            冬季压缩最严重（只剩观测的 66%），夏季几乎不压缩（比值≈1.0）。
            </div>
            <div class="evi">
            <b>证据 2 ｜ 误差与观测距平成反比</b><br>
            以（月份 × 北京时）气候态为基准：<br>
            误差 = <b>{formula}</b>（r = {head['reg_r']:.2f}）。
            天越冷模式越偏暖，天越热模式越偏冷。
            </div>
            <div class="evi">
            <b>证据 3 ｜ 压缩落在夜间最低气温</b><br>
            冬季日最低气温平均偏高 <b>{signed(minmax['winter_min'], 2, '℃')}</b>，
            而冬季日最高气温偏差只有 <b>{signed(minmax['winter_max'], 2, '℃')}</b>。
            12 月与 1 月最明显。<b>模式抓不住夜间强辐射降温。</b>
            </div>
            <div class="evi">
            <b>证据 4 ｜ 极端时刻最明显</b><br>
            最冷 1% 的时刻平均偏暖 <b>{signed(head['cold_error'], 2, '℃')}</b>，
            最暖 1% 的时刻平均偏冷 <b>{signed(head['warm_error'], 2, '℃')}</b>，
            而全样本平均偏差只有 {signed(head['bias'], 2, '℃')}。
            </div>
            <div class="evi">
            <b>证据 5 ｜ 订正实验</b><br>
            因为平均偏差≈0，“常数去偏”几乎无效（仅 {summary.iloc[1]['MAE_improvement_pct']:.1f}%）；
            因为日变化偏差在冬夏符号相反，“按小时去偏”也几乎无效。
            只有同时刻画季节与日变化，才能拿到 {summary.iloc[4]['MAE_improvement_pct']:.1f}% 的改善。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("一眼看懂的图")
        month_range = amp.range_by_month(load_nanjing())
        fig = go.Figure()
        fig.add_bar(
            x=month_range["month"], y=month_range["obs_range"],
            name="观测日较差", marker_color=C_ORANGE,
        )
        fig.add_bar(
            x=month_range["month"], y=month_range["fcst_range"],
            name="预报日较差", marker_color=C_BLUE,
        )
        fig.update_layout(barmode="group", title="逐月平均日较差：预报被系统性压扁")
        fig.update_xaxes(dtick=1, title="月份")
        fig.update_yaxes(title="日较差 (℃)")
        st.plotly_chart(base_layout(fig, 340), use_container_width=True)
        caption(
            "夏季模式能还原日变化，冬季几乎腰斩。这正是冬季系统性偏暖、夏季系统性偏冷的来源。"
        )

    st.divider()
    st.subheader("这个应用包含什么")
    caption("左侧边栏切换页面。逻辑顺序是：数据 → 检验 → 诊断 → 订正 → 案例 → 局限。")

    pages = [
        ("📐 数据与方法", "数据来源、指标定义，以及为什么必须用严格时序划分而不是随机划分"),
        ("📊 误差检验", "从总体、季节、日变化、预报时效四个角度给误差画一张画像"),
        ("🔍 核心发现：振幅阻尼", "本项目聚焦的细微点——模式怎样压扁温度变幅，用四组独立检验论证"),
        ("🧪 订正实验", "六个嵌套方法的消融对比：订正收益究竟来自哪一部分误差"),
        ("🌀 真实案例", "2024 年 1 月寒潮与 2024 年 8 月高温，严格样本外的过程检验"),
        ("🧭 局限与展望", "被数据推翻的假设、已知边界，以及下一步可以做什么"),
    ]
    for start in range(0, len(pages), 2):
        for column, (title, text) in zip(st.columns(2), pages[start : start + 2]):
            with column:
                st.markdown(f"**{title}**")
                caption(text)

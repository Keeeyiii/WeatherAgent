"""核心发现页：什么是振幅阻尼、凭什么这么说。

页面结构刻意做成“先定义 → 再给判据 → 再逐条对照 → 最后排除假象”，
目的是让不了解这个概念的读者也能顺着逻辑读下去。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools import amplitude as amp
from views.common import (
    C_BLUE,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    caption,
    card,
    hero,
    load_nanjing,
    signed,
)

FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def render() -> None:
    hero(
        "核心发现：模式把温度的起伏压平了",
        "这一页只回答一个问题：<b>什么是“振幅阻尼”，凭什么说它存在？</b>",
    )
    nj = load_nanjing()
    rng = amp.range_summary(nj)
    regression = amp.anomaly_regression(nj)
    extremes = amp.extreme_error(nj)
    minmax = amp.minmax_bias(nj)
    minmax_overall = amp.minmax_summary(nj)

    # ---------------------------------------------------------------- 一、定义
    st.subheader("一、先明确：什么是“振幅阻尼”")
    st.markdown(
        """
        真实大气的温度一直在起伏：白天升高、夜间降低，冬天冷、夏天热，
        寒潮来时骤降、回暖时骤升。**模式的预报曲线虽然大体跟着走，但起伏总是偏小**
        ——峰值报低了，谷值报高了，整条曲线像被“压扁”了一样。

        这种现象就叫**振幅阻尼**（amplitude damping）：模式对温度变化的**响应不足**。

        注意它和“整体偏暖/偏冷”是两回事：整体偏移是整条曲线上下平移，
        而振幅阻尼是曲线被压扁——**中间的部分可能很准，两端却总是差得多**。
        """
    )

    concept = os.path.join(FIGURES, "fig11_damping_concept.png")
    if os.path.exists(concept):
        st.image(concept, width="stretch")
    caption(
        "2024 年 12 月 2 日起的三天。观测的峰值 22 ℃ 被预报压到约 19.7 ℃，"
        "观测的谷值 7 ℃ 被预报抬到约 10 ℃——**该高时不够高，该低时不够低**。"
    )

    st.info(
        "**一句话判据**：如果误差真的来自“模式响应不足”，那么它不应该只在一处出现，"
        "而应该在**日变化、强度、极端、以及最低/最高气温的分解**这几个相互独立的角度上同时出现。"
        "下面就用这四组证据来检验。"
    )

    # ---------------------------------------------------------------- 二、判据
    st.subheader("二、如果这个判断成立，应该看到什么")
    st.markdown(
        "先写下预测，再对照数据——这样才不会被“看起来差不多”的结果误导。"
    )
    predictions = pd.DataFrame(
        [
            (
                "① 日变化尺度",
                "预报的日较差应小于观测",
                f"{rng['obs_range']:.2f} → {rng['fcst_range']:.2f} ℃（−{rng['damping_pct']:.0f}%）"
                "，冷季仅剩 66%",
                "成立",
            ),
            (
                "② 强度响应",
                "误差应与观测距平成反比",
                f"斜率 {regression['slope']:+.3f}（r = {regression['correlation']:.2f}）",
                "成立",
            ),
            (
                "③ 极端时刻",
                "越极端的时刻偏差应越大",
                f"最冷 1% {signed(extremes.loc[0, '平均误差 (℃)'], 2, '℃')}／"
                f"最暖 1% {signed(extremes.loc[2, '平均误差 (℃)'], 2, '℃')}",
                "成立",
            ),
            (
                "④ 误差落在哪一侧",
                "压缩应集中在变化幅度最大的那一侧",
                f"冬季最低气温 {signed(minmax_overall['winter_min'])}／"
                f"最高气温 {signed(minmax_overall['winter_max'])}",
                "成立",
            ),
        ],
        columns=["检验角度", "如果“振幅阻尼”成立，应该看到", "数据里的结果", "判断"],
    )
    st.dataframe(predictions, width="stretch", hide_index=True)
    st.caption("四条全部成立，而且在后面的稳健性检验中还排除了“采样造成的假象”。")

    st.divider()
    st.subheader("三、四组证据")

    # ---------------------------------------------------------- 证据 1 日变化
    st.markdown("#### 证据 1 ｜ 日变化尺度：每天的起伏被压小")
    st.markdown(
        "**看什么**：把每天的最高气温减最低气温，得到“日较差”，比较观测与预报。"
    )
    month_range = amp.range_by_month(nj)
    figure = go.Figure()
    figure.add_scatter(
        x=month_range["month"],
        y=month_range["obs_range"],
        name="观测",
        mode="lines+markers",
        marker_color=C_ORANGE,
        line=dict(width=3),
    )
    figure.add_scatter(
        x=month_range["month"],
        y=month_range["fcst_range"],
        name="预报",
        mode="lines+markers",
        marker_color=C_BLUE,
        line=dict(width=3),
    )
    figure.update_layout(title="逐月平均日较差（日最高 − 日最低）")
    figure.update_xaxes(dtick=1, title="月份")
    figure.update_yaxes(title="日较差 (℃)")
    st.plotly_chart(base_layout(figure, 340), width="stretch")
    st.markdown(
        f"""
        **结果**：全年平均日较差观测 {rng['obs_range']:.2f} ℃、预报 {rng['fcst_range']:.2f} ℃，
        压缩 **{rng['damping_pct']:.0f}%**。

        **关键细节**：压缩并不均匀。冷季（11—1 月）模式只报出观测日较差的
        **66%—69%**，相当于把一天的起伏砍掉三分之一；而暖季（6—9 月）比值接近 1.0，
        几乎没有偏差。**这为后面解释季节偏差提供了线索。**
        """
    )

    # ---------------------------------------------------------- 证据 2 距平
    st.markdown("#### 证据 2 ｜ 强度尺度：偏离常态越多，误差越大")
    st.markdown(
        "**看什么**：以“同一月份、同一时刻”的气候平均值为基准算出观测距平"
        "（今天比常年同期冷多少或热多少），再看误差与它的关系。"
    )
    bins = amp.anomaly_binned(nj)
    figure = go.Figure()
    figure.add_bar(
        x=bins["mean_anomaly"],
        y=bins["mean_error"],
        marker_color=[C_RED if value > 0 else C_BLUE for value in bins["mean_error"]],
        text=[f"{value:+.2f}" for value in bins["mean_error"]],
        textposition="outside",
        cliponaxis=False,
        name="分箱平均误差",
    )
    fit_x = [bins["mean_anomaly"].min(), bins["mean_anomaly"].max()]
    figure.add_scatter(
        x=fit_x,
        y=[
            regression["intercept"] + regression["slope"] * fit_x[0],
            regression["intercept"] + regression["slope"] * fit_x[1],
        ],
        mode="lines",
        name=f"线性拟合（斜率 {regression['slope']:+.3f}）",
        line=dict(color=C_NAVY, width=3, dash="dash"),
    )
    figure.add_hline(y=0, line=dict(color="#999", width=1))
    figure.update_layout(title="观测距平 vs 平均预报误差")
    figure.update_xaxes(title="观测距平（相对月份 × 北京时气候态，℃）")
    low = min(0.0, float(bins["mean_error"].min()))
    high = max(0.0, float(bins["mean_error"].max()))
    span = high - low
    figure.update_yaxes(title="平均误差 (℃)", range=[low - 0.22 * span, high + 0.26 * span])
    st.plotly_chart(base_layout(figure, 360), width="stretch")
    st.markdown(
        f"""
        **结果**：关系几乎单调。观测比常年冷约 9 ℃ 时，模式平均偏暖 +0.93 ℃；
        比常年暖约 10 ℃ 时，模式平均偏冷 −0.93 ℃。
        回归得到 **误差 = {signed(regression['intercept'])} − {abs(regression['slope']):.2f} × 观测距平**
        （r = {regression['correlation']:.2f}）。

        **怎么读这个斜率**：负号意味着“观测越冷、模式越偏暖”。
        斜率为 −0.14 说明观测距平每变化 1 ℃，误差就朝相反方向变化 0.14 ℃——
        模式只还原了约 **{100 * (1 - abs(regression['slope'])):.0f}%** 的距平信号，
        剩下的被“吃掉”了。这就是振幅压缩的定量表达。
        """
    )

    # ---------------------------------------------------------- 证据 3 极端
    st.markdown("#### 证据 3 ｜ 极端时刻：偏差最大，平均指标最容易掩盖它")
    st.markdown(
        "**看什么**：把样本按观测温度排序，取最冷的 1% 和最暖的 1%，分别看平均误差。"
    )
    figure = go.Figure()
    figure.add_bar(
        x=extremes["分组"],
        y=extremes["平均误差 (℃)"],
        marker_color=[
            C_RED if value > 0 else C_BLUE for value in extremes["平均误差 (℃)"]
        ],
        text=[f"{value:+.2f} ℃" for value in extremes["平均误差 (℃)"]],
        textposition="outside",
        cliponaxis=False,
        name="平均误差",
    )
    figure.add_hline(y=0, line=dict(color="#999", width=1))
    figure.update_layout(title="最冷 / 全样本 / 最暖时刻的平均误差")
    low = min(0.0, float(extremes["平均误差 (℃)"].min()))
    high = max(0.0, float(extremes["平均误差 (℃)"].max()))
    span = high - low
    figure.update_yaxes(title="平均误差 (℃)", range=[low - 0.18 * span, high + 0.22 * span])
    st.plotly_chart(base_layout(figure, 330), width="stretch")
    st.dataframe(extremes.round(2), width="stretch", hide_index=True)
    st.markdown(
        f"""
        **结果**：全样本平均偏差只有 {signed(float(extremes.loc[1, '平均误差 (℃)']))}，
        但最冷 1% 的时刻平均偏暖 **{signed(extremes.loc[0, '平均误差 (℃)'], 2, '℃')}**，
        最暖 1% 的时刻平均偏冷 **{signed(extremes.loc[2, '平均误差 (℃)'], 2, '℃')}**。

        **这说明什么**：偏差并不是恒定的小量，而是**随温度的极端程度迅速放大**。
        这正是“响应不足”的必然结果：越极端的天气，越需要模式做出大响应，
        而模式恰恰在这里跟不上。同时它也提醒：**平均指标会掩盖最值得关注的部分。**
        """
    )

    # ---------------------------------------------------------- 证据 4 分解
    st.markdown("#### 证据 4 ｜ 定位：压缩落在“最低气温”这一侧")
    st.markdown(
        "**看什么**：把每天的误差拆成两部分——日最低气温的误差和日最高气温的误差。"
        "如果压缩是均匀的，两者应该差不多；如果偏向某一侧，就会露出来。"
    )
    figure = go.Figure()
    figure.add_bar(
        x=minmax["month"] - 0.2,
        y=minmax["最低气温偏差"],
        width=0.4,
        name="日最低气温偏差",
        marker_color=C_BLUE,
    )
    figure.add_bar(
        x=minmax["month"] + 0.2,
        y=minmax["最高气温偏差"],
        width=0.4,
        name="日最高气温偏差",
        marker_color=C_ORANGE,
    )
    figure.add_hline(y=0, line=dict(color="#999", width=1))
    figure.update_layout(title="逐月的日最低 / 日最高气温偏差")
    figure.update_xaxes(dtick=1, title="月份")
    figure.update_yaxes(title="偏差 (℃)")
    st.plotly_chart(base_layout(figure, 340), width="stretch")
    columns = st.columns(3)
    for column, (number, label) in zip(
        columns,
        [
            (signed(minmax_overall["winter_min"], 2, "℃"), "冬季日最低气温偏差"),
            (signed(minmax_overall["winter_max"], 2, "℃"), "冬季日最高气温偏差"),
            (
                f"{minmax_overall['min_spread_ratio']:.0%}",
                "预报最低气温的日际变率（占观测）",
            ),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f"""
        **结果**：12 月与 1 月模式把日最低气温分别报高了
        {signed(minmax.loc[11, '最低气温偏差'], 2, '℃')} 和
        {signed(minmax.loc[0, '最低气温偏差'], 2, '℃')}，
        而同期日最高气温几乎没有偏差
        （{signed(minmax.loc[11, '最高气温偏差'])} / {signed(minmax.loc[0, '最高气温偏差'])}）。
        从日际变率看，预报的最低气温标准差只有观测的 **{minmax_overall['min_spread_ratio']:.0%}**，
        而最高气温达到 {minmax_overall['max_spread_ratio']:.0%}。

        **结论**：被压缩的是**最低气温**这一侧。
        换句话说，模式抓不住夜间的强辐射降温——白天升温基本报得对，夜里降不下去。
        这一条同时解释了两个前面的现象：为什么冬季整体偏暖，以及为什么最冷的时刻偏得最厉害。
        """
    )

    st.divider()

    # ---------------------------------------------------------------- 四、反证
    st.subheader("四、排除一种最容易想到的假象")
    st.markdown(
        "“日较差被压小”有一个很自然的替代解释：**如果归档预报的时间分辨率比观测更粗，"
        "那么它自然会漏掉极值，看起来就像被压扁了。** 这跟模式好坏无关，纯属比较方式的问题。"
        "所以必须专门排除它。"
    )
    resolution = amp.range_vs_resolution(nj)
    steps = amp.series_step_fraction(nj)
    columns = st.columns(2)
    columns[0].markdown(
        card(f"{1 - steps['forecast_repeat']:.1%}", "预报相邻小时发生变化的比例"),
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        card(f"{1 - steps['observed_repeat']:.1%}", "观测相邻小时发生变化的比例"),
        unsafe_allow_html=True,
    )
    st.write("")
    st.dataframe(resolution.round(3), width="stretch", hide_index=True)
    st.markdown(
        f"""
        **两步检查的结果**：

        1. 预报序列里 **{1 - steps['forecast_repeat']:.1%}** 的相邻小时取值不同，
           说明归档预报确实是逐小时序列，不存在“三小时一次”的阶梯；
        2. 把观测也降到 3 小时、6 小时甚至 12 小时分辨率后，压缩比值只从 0.836
           变化到 0.819—0.852，**基本不动**。

        **结论**：如果压缩来自采样，降到同样分辨率后应该明显回升，但它没有。
        因此“模式压扁温度变幅”是数据本身的性质，不是比较方式的假象。
        （观测侧有 {steps['observed_repeat']:.0%} 的相邻小时取值相同，是因为 METAR 气温按整摄氏度报出，属正常量化。）
        """
    )

    st.divider()

    # ---------------------------------------------------------------- 五、小结
    st.subheader("五、把逻辑串起来")
    st.markdown(
        """
        ```
        观测到的事实：平均偏差只有 −0.21 ℃，但 MAE 高达 1.58 ℃
                              ↓  说明误差不是整体偏移，而是结构
        提出解释：模式对温度变化的“响应不足”（振幅阻尼）
                              ↓  由它推出四条可检验的预测
        四组独立证据：日变化被压小 ✓  距平响应不足 ✓  极端偏差最大 ✓  压缩落在最低气温一侧 ✓
                              ↓  再排除“采样造成的假象”
        排除反证：预报确为逐小时，降采样后压缩依旧 ✓
                              ↓
        结论：GFS 在南京的 2 米气温误差，主要形态是振幅阻尼，
              其落点是冬季夜间的最低气温。
        ```
        """
    )
    st.success(
        "**为什么这个结论比“模式偏暖 1 ℃”有用得多**：\n\n"
        "整体偏移无法告诉你“什么时候最不准”，而振幅阻尼给出了明确的**位置**"
        "（夜间最低气温）、**时段**（冷季）、**量级**（约 2.8 ℃）和**方向**（偏暖）。"
        "更重要的是，它可以被检验，也已经被检验过了。"
    )

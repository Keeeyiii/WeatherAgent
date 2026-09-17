"""核心发现页：模式压扁了温度的变幅（本项目的聚焦点）。"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tools import amplitude as amp
from views.common import (
    caption,
    C_BLUE,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    card,
    hero,
    load_nanjing,
    signed,
)


def render() -> None:
    hero(
        "核心发现：模式压扁了温度的变幅",
        "本项目聚焦的“细微点”——不追问模式准不准，而是追问它错成了什么形状",
    )
    nj = load_nanjing()

    st.markdown(
        """
        <div class="qbox">
        <b>假设</b><br>
        如果误差主要来自“模式对温度变化的响应不够充分”，那么它应当同时表现为：
        日较差偏小、季节振幅偏小、距平越大误差越大、极端时刻偏得最厉害。
        下面用四组<b>相互独立</b>的检验来支持或推翻这个假设。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    rng = amp.range_summary(nj)
    regression = amp.anomaly_regression(nj)
    extremes = amp.extreme_error(nj)

    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{rng['obs_range']:.2f} → {rng['fcst_range']:.2f} ℃", "观测日较差 → 预报日较差"),
            (f"−{rng['damping_pct']:.0f}%", "日变化振幅被压缩"),
            (f"{regression['slope']:+.3f}", "误差对观测距平的回归斜率"),
            (f"{extremes.loc[0, '平均误差 (℃)']:+.2f} ℃", "最冷 1% 时刻的模式偏暖"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    st.subheader("检验一：日较差的逐月结构")
    month_range = amp.range_by_month(nj)
    fig = go.Figure()
    fig.add_scatter(
        x=month_range["month"],
        y=month_range["obs_range"],
        name="观测",
        mode="lines+markers",
        marker_color=C_ORANGE,
        line=dict(width=3),
    )
    fig.add_scatter(
        x=month_range["month"],
        y=month_range["fcst_range"],
        name="预报",
        mode="lines+markers",
        marker_color=C_BLUE,
        line=dict(width=3),
    )
    fig.update_layout(title="逐月平均日较差（日最高气温 − 日最低气温）")
    fig.update_xaxes(dtick=1, title="月份")
    fig.update_yaxes(title="日较差 (℃)")
    st.plotly_chart(base_layout(fig, 350), use_container_width=True)
    st.markdown(
        f"""
        冬季（11—1 月）模式只报出观测日较差的 **66%—69%**，相当于把一天的温度起伏砍掉三分之一；
        而夏季（6—9 月）比值接近 **1.0**，几乎没有偏差。

        **压缩不是均匀的，而是集中在冷季节。** 这一点很关键：
        冬季夜间辐射降温被模式报得不够冷，白天升温又被报得不够暖，
        净效应就是季节平均偏暖 —— 季节偏差和日变化偏差，其实是同一个问题的两种表现。
        """
    )

    st.subheader("检验二：误差与观测距平成反比")
    bins = amp.anomaly_binned(nj)
    fig = go.Figure()
    fig.add_bar(
        x=bins["mean_anomaly"],
        y=bins["mean_error"],
        marker_color=[C_RED if value > 0 else C_BLUE for value in bins["mean_error"]],
        text=[f"{value:+.2f}" for value in bins["mean_error"]],
        textposition="outside",
        cliponaxis=False,
        name="分箱平均误差",
    )
    fit_x = [bins["mean_anomaly"].min(), bins["mean_anomaly"].max()]
    fig.add_scatter(
        x=fit_x,
        y=[
            regression["intercept"] + regression["slope"] * fit_x[0],
            regression["intercept"] + regression["slope"] * fit_x[1],
        ],
        mode="lines",
        name=f"线性拟合（斜率 {regression['slope']:+.3f}）",
        line=dict(color=C_NAVY, width=3, dash="dash"),
    )
    fig.update_layout(title="观测距平 vs 平均预报误差")
    fig.update_xaxes(title="观测距平（相对月份 × 北京时气候态，℃）")
    low = min(0.0, float(bins["mean_error"].min()))
    high = max(0.0, float(bins["mean_error"].max()))
    span = high - low
    fig.update_yaxes(title="平均误差 (℃)", range=[low - 0.22 * span, high + 0.26 * span])
    st.plotly_chart(base_layout(fig, 370), use_container_width=True)
    retained = 100 * (1 - abs(regression["slope"]))
    st.markdown(
        f"""
        关系几乎单调：观测比气候态冷 9 ℃ 时，模式平均偏暖 +0.93 ℃；
        观测比气候态暖 10 ℃ 时，模式平均偏冷 −0.93 ℃。
        回归结果为 **误差 = {signed(regression['intercept'])} − {abs(regression['slope']):.2f} × 观测距平**
        （r = {regression['correlation']:.2f}，n = {regression['n']:,}）。

        斜率的意义是：**观测距平每变化 1 ℃，误差就朝相反方向变化
        {abs(regression['slope']):.2f} ℃**。也就是说，模式只还原了约
        **{retained:.0f}%** 的距平信号 —— 这就是“振幅压缩”的定量表达。
        """
    )

    st.subheader("检验三：极端时刻最明显")
    fig = go.Figure()
    fig.add_bar(
        x=extremes["分组"],
        y=extremes["平均误差 (℃)"],
        marker_color=[
            C_RED if value > 0 else C_BLUE for value in extremes["平均误差 (℃)"]
        ],
        text=[f"{value:+.2f} ℃" for value in extremes["平均误差 (℃)"]],
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(title="最冷 / 全样本 / 最暖时刻的平均误差")
    low = min(0.0, float(extremes["平均误差 (℃)"].min()))
    high = max(0.0, float(extremes["平均误差 (℃)"].max()))
    span = high - low
    fig.update_yaxes(title="平均误差 (℃)", range=[low - 0.18 * span, high + 0.22 * span])
    st.plotly_chart(base_layout(fig, 330), use_container_width=True)
    st.dataframe(extremes.round(2), use_container_width=True, hide_index=True)
    caption(
        "全样本平均偏差只有 −0.21 ℃，但在最冷 1% 的时刻平均偏暖 +4.35 ℃，"
        "在最暖 1% 的时刻平均偏冷 −2.19 ℃。"
        "**平均指标掩盖了最值得关注的部分。** 这也解释了为什么订正模型必须随天气状态变化，"
        "而不能只减掉一个常数。"
    )

    st.subheader("检验四：剔除季节循环后，变率本身也被压缩")
    variability = amp.variability_ratio(nj)
    columns = st.columns(3)
    for column, (number, label) in zip(
        columns,
        [
            (f"{variability['obs_std']:.2f} ℃", "观测距平标准差"),
            (f"{variability['fcst_std']:.2f} ℃", "预报距平标准差"),
            (f"−{variability['damping_pct']:.1f}%", "天气尺度变率被压缩"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")
    caption(
        "把季节循环与日变化都剔除之后，预报的距平标准差仍然比观测小 5.6%。"
        "四个检验方向一致，因此“模式压扁温度变幅”不是某一个指标造成的假象。"
    )

    st.divider()
    st.subheader("稳健性检验：会不会只是时间分辨率的假象？")
    st.markdown(
        "上面所有结论都建立在“预报与观测的时间分辨率相同”这一前提上。"
        "如果归档预报实际上比逐小时观测更粗，那么日较差偏小就可能只是采样造成的假象。"
        "**这是本结论最需要排除的一种可能，因此单独检验。**"
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
    st.dataframe(resolution.round(3), use_container_width=True, hide_index=True)
    st.success(
        "**排除。** 预报序列中 93% 的相邻小时取值不同，说明归档预报确实是逐小时的；"
        "把观测也降到 3 小时、6 小时甚至 12 小时分辨率后，"
        "压缩比值仅从 0.836 变化到 0.819—0.852，"
        "说明“模式压扁温度变幅”不是采样造成的假象，而是数据本身的性质。"
        "（观测有 43% 的相邻小时取值相同，这是因为 METAR 气温按整摄氏度报出，"
        "属于正常的量化特征，不会系统性地改变日较差。）"
    )

    st.divider()
    st.subheader("检验五：压缩究竟落在哪一侧？")
    st.markdown(
        "如果“振幅压缩”只是笼统的说法，它就不应该给出具体的落点。"
        "把误差拆成**日最低气温偏差**和**日最高气温偏差**两部分，"
        "就能看出模式到底在哪一侧失真。"
    )
    minmax = amp.minmax_bias(nj)
    summary = amp.minmax_summary(nj)
    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{summary['winter_min']:+.2f} ℃", "冬季日最低气温偏差"),
            (f"{summary['winter_max']:+.2f} ℃", "冬季日最高气温偏差"),
            (f"{summary['summer_min']:+.2f} ℃", "夏季日最低气温偏差"),
            (f"{summary['summer_max']:+.2f} ℃", "夏季日最高气温偏差"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")
    st.dataframe(minmax.round(2), use_container_width=True, hide_index=True)
    st.success(
        f"**误差有非常明确的落点：冬季夜间的最低气温。**\n\n"
        f"12 月与 1 月模式把日最低气温分别报高了 "
        f"{minmax.loc[11, '最低气温偏差']:+.2f} ℃ 和 {minmax.loc[0, '最低气温偏差']:+.2f} ℃，"
        f"而同期日最高气温几乎没有偏差"
        f"（{minmax.loc[11, '最高气温偏差']:+.2f} ℃ / {minmax.loc[0, '最高气温偏差']:+.2f} ℃）。"
        f"从日际变化幅度看，预报的日最低气温标准差只有观测的 "
        f"{summary['min_spread_ratio']:.0%}，而日最高气温达到 "
        f"{summary['max_spread_ratio']:.0%}。"
        f"**也就是说：被压缩的是“最低气温”这一侧——模式抓不住夜间强辐射降温。**\n\n"
        f"这一条与前面的证据完全自洽：它解释了为什么最冷的 1% 时刻偏差最大"
        f"（+4.35 ℃），也解释了为什么冬季整体偏暖（{summary['winter_min']:+.2f} ℃ 与 "
        f"{summary['winter_max']:+.2f} ℃ 的净效应）。"
    )

    st.divider()
    st.subheader("这个发现为什么重要")
    st.markdown(
        """
        1. **它把“模型准不准”变成了“模型如何错”。** 后者才是可以据以行动的诊断信息。
        2. **它直接解释了订正实验的结果**：平均偏差≈0 ⇒ 常数去偏无效；
           偏差随季节变号 ⇒ 单独按小时去偏也无效。
           （第 5 页给出了这个推论的定量验证。）
        3. **它给出了可检验的物理线索。** 振幅压缩通常与地气耦合、
           边界层参数化、近地层通量交换的响应偏弱有关，
           这是后续可以借助模式诊断量继续追问的方向。
        4. **它做出了一个可以被推翻的预测**：如果压缩与站点和下垫面有关，
           那么在一个没有这种压缩的站点上，南京训练出的订正模型就应当失效。
           第 5 页用上海站检验了这一点，结果与预测一致。
        """
    )

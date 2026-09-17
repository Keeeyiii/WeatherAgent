"""真实案例页：把统计诊断放回具体的天气过程里检验。"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from views.common import (
    caption,
    C_GREEN,
    C_NAVY,
    C_ORANGE,
    base_layout,
    card,
    hero,
    run_case,
)

CASES = {
    "2024 年 1 月寒潮过程": ("2024-01-19", "2024-01-27"),
    "2024 年 8 月持续高温过程": ("2024-08-01", "2024-08-13"),
}


def _render_case(name: str, start: str, end: str) -> None:
    series, summary = run_case(start, end)
    raw = summary.iloc[0]
    deh = summary.iloc[1]
    forest = summary.iloc[2]

    obs_min = series["observed_temperature"].min()
    obs_max = series["observed_temperature"].max()
    swing = series["observed_temperature"].diff().abs().max()
    daily = series.groupby(series["time"].dt.normalize()).agg(
        obs_max=("observed_temperature", "max"),
        obs_min=("observed_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
    )
    obs_range = (daily["obs_max"] - daily["obs_min"]).mean()
    fcst_range = (daily["fcst_max"] - daily["fcst_min"]).mean()
    range_ratio = fcst_range / obs_range
    if range_ratio < 0.85:
        range_sentence = (
            f"过程期间平均日较差：观测 **{obs_range:.2f} ℃**，预报只有 "
            f"**{fcst_range:.2f} ℃**（比值 {range_ratio:.2f}）——"
            "**振幅压缩在真实天气过程中是肉眼可见的**。"
        )
        correction_sentence = (
            "订正后的曲线更贴合观测的**峰谷位置**，说明订正不只是整体平移，"
            "而在一定程度上恢复了变幅"
        )
    else:
        range_sentence = (
            f"这个过程里日较差的变化幅度本身不算大（观测 {obs_range:.2f} ℃ / "
            f"预报 {fcst_range:.2f} ℃，比值 {range_ratio:.2f}），"
            "因此误差主要不表现为“起伏被压扁”，而表现为**整体的系统性偏移**"
            "——这提示振幅压缩的强弱是随季节变化的，需要分开讨论。"
        )
        correction_sentence = (
            "订正后的曲线整体上移贴近观测，说明订正在这个过程里的主要作用是"
            "**消除系统性偏移**，而不是恢复变幅"
        )
    is_cold = "寒潮" in name

    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{len(series)} 小时", "过程时长"),
            (f"{obs_min:.1f} ~ {obs_max:.1f} ℃", "观测温度范围"),
            (f"{swing:.1f} ℃", "最大逐小时变化"),
            (f"{raw['MAE']:.2f} ℃", "原始预报 MAE"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    fig = go.Figure()
    fig.add_scatter(
        x=series["time"],
        y=series["observed_temperature"],
        name="观测（真值）",
        mode="lines",
        line=dict(color=C_NAVY, width=2.6),
    )
    fig.add_scatter(
        x=series["time"],
        y=series["forecast_temperature"],
        name="GFS 原始预报",
        mode="lines",
        line=dict(color=C_ORANGE, width=1.8, dash="dot"),
    )
    fig.add_scatter(
        x=series["time"],
        y=series["corrected_rf"],
        name="随机森林订正后",
        mode="lines",
        line=dict(color=C_GREEN, width=1.8),
    )
    fig.update_layout(title=f"{name}：观测 / 预报 / 订正后")
    fig.update_yaxes(title="2 米气温 (℃)")
    st.plotly_chart(base_layout(fig, 420), use_container_width=True)
    caption(
        f"订正模型只使用 {start} 之前的数据训练，因此这是严格的样本外检验——"
        "模型在“看到”这次天气过程之前就已经定好了参数。"
    )

    st.subheader("误差的时间演变")
    fig = go.Figure()
    fig.add_scatter(
        x=series["time"],
        y=series["raw_error"],
        name="原始预报误差",
        mode="lines",
        line=dict(color=C_ORANGE, width=2),
    )
    fig.add_scatter(
        x=series["time"],
        y=series["rf_error"],
        name="订正后误差",
        mode="lines",
        line=dict(color=C_GREEN, width=2),
    )
    fig.add_hline(y=0, line=dict(color=C_NAVY, width=1))
    fig.update_layout(title="误差演变（正 = 模式偏暖）")
    fig.update_yaxes(title="误差 (℃)")
    st.plotly_chart(base_layout(fig, 320), use_container_width=True)

    display = summary.copy()
    display.columns = ["方法", "bias (℃)", "MAE (℃)", "RMSE (℃)", "最大绝对误差 (℃)"]
    st.dataframe(display.round(3), use_container_width=True, hide_index=True)

    direction = (
        f"寒潮降温阶段，模式明显偏暖（bias {raw['bias']:+.2f} ℃）"
        "——这正是“振幅压缩”在真实过程中的直接体现：**该冷的时候不够冷**。"
        if is_cold
        else f"高温时段，模式明显偏冷（bias {raw['bias']:+.2f} ℃）"
        "——与夏季系统性负偏差一致：**该热的时候不够热**。"
    )
    st.markdown(
        f"""
        #### 从这个过程里能读出什么

        - 过程期间观测温度在 **{obs_min:.1f} ℃ 到 {obs_max:.1f} ℃** 之间变化。
          {range_sentence}
        - {direction}
        - {correction_sentence}
          （MAE 从 {raw['MAE']:.2f} ℃ 降到 {forest['MAE']:.2f} ℃，
          最大绝对误差从 {raw['max_abs_error']:.1f} ℃ 降到 {forest['max_abs_error']:.1f} ℃）。
        - 但**订正并非万能**：订正后仍有 {forest['max_abs_error']:.1f} ℃ 的瞬时误差，
          说明过程中最剧烈的时段超出了这套统计订正的能力范围。
          简单查表法（月×小时）在这个案例里几乎没有改善
          （MAE {raw['MAE']:.2f} → {deh['MAE']:.2f} ℃），
          因为它只针对气候态平均状态，不针对具体过程。

        #### 为什么这个案例值得单独拿出来

        平均指标（MAE {raw['MAE']:.2f} ℃）会让人以为“模式表现还可以”，
        但在一次真实过程中，关键时刻的误差可以达到 **{raw['max_abs_error']:.1f} ℃**。
        对预报员和公众而言，关键时刻的误差远比平均误差更有意义。
        **这正是本项目从“平均误差”走向“误差结构”的直接动机。**
        """
    )


def render() -> None:
    hero(
        "真实案例",
        "把统计诊断放回具体天气过程里检验：模式在极端过程中是否错得更厉害？"
        "订正方法在过程中还可靠吗？",
    )
    caption(
        "两个案例的选取不是随意的：一个是冷过程，一个是暖过程。"
        "如果“振幅压缩”的解释成立，模式在两者中的偏差方向应当**相反**。"
    )
    try:
        default_case = int(os.environ.get("WEATHERAGENT_CASE_INDEX", "0"))
    except ValueError:
        default_case = 0
    default_case = max(0, min(default_case, len(CASES) - 1))
    name = st.radio(
        "选择过程",
        list(CASES),
        index=default_case,
        horizontal=True,
        label_visibility="collapsed",
    )
    start, end = CASES[name]
    _render_case(name, start, end)

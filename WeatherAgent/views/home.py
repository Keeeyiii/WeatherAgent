"""首页：一个问题、一个结论，然后告诉你怎么用、去哪看细节。

顺序刻意做成：问题 → 结论 → 数字 → 用法 → 成果速览。
上半部分让人知道这是什么、结论是什么、可以怎么用；
完整的推导和图表都放在各分析页里，底部「成果速览」只给一句话版本。
"""

from __future__ import annotations

import streamlit as st

from tools import amplitude as amp
from views.common import (
    card,
    headline_numbers,
    hero,
    load_nanjing,
    run_ablation,
    signed,
)


def render() -> None:
    head = headline_numbers()
    _, summary = run_ablation()
    constant = summary.loc[summary["method"].str.contains("常数去偏", regex=False)].iloc[0]
    both_box = summary.loc[summary["method"].str.contains("月×小时", regex=False)].iloc[0]
    minmax = amp.minmax_summary(load_nanjing())

    # ---------------------------------------------------------------- 问题
    hero(
        "误差会变吗？",
        "同一个预报模式，误差是一个固定的数字，还是一个随条件变化的量？"
        "如果是变化的，它随什么变、怎么变、有没有规律可循？"
        "<br>这个网站用真实观测数据回答这个问题，并把分析方法开放出来。",
        large=True,
    )
    st.markdown(
        """
        <div class="qbox">
        <b>下面要回答的问题</b>　GFS 对南京 2 米气温的预报误差，是随机噪声，还是有结构的系统性偏差？
        如果是结构，它长什么样、能不能被订正？
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # ---------------------------------------------------------------- 核心结论
    st.markdown(
        """
        <div style="border:1px solid #dde7f0;border-left:5px solid #2f7fb5;
                    border-radius:8px;padding:.85rem 1.05rem;background:#f8fbfd;">
          <div style="font-size:.74rem;color:#8a95a0;letter-spacing:.14em;">核 心 结 论</div>
          <div style="font-size:1.16rem;font-weight:700;color:#123a5f;margin:.28rem 0 .3rem 0;">
            误差会变。最主要的形态是「振幅阻尼」——模式把温度的起伏压平了
          </div>
          <div style="font-size:.93rem;color:#33475b;line-height:1.72;">
            不是整体偏暖或偏冷，而是<b>该冷的时候不够冷、该热的时候不够热</b>，越极端越明显；
            偏差有明确的落点：<b>冬季夜间的最低气温</b>。
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

    # ---------------------------------------------------------------- 开始分析
    st.subheader("开始分析：选一份数据")
    uploaded = st.file_uploader(
        "上传我的数据（CSV）",
        type=["csv"],
        key="user_csv",
        help="必需列：time、forecast_temperature、observed_temperature；"
        "可选列：humidity、pressure、wind_speed。"
        "格式模板与完整说明见「🔬 误差分析（可换数据）」页。",
    )
    if uploaded is None:
        st.markdown(
            "- **不上传**：直接用内置的南京站 2.4 年真实数据，从左侧导航进入各分析页。\n"
            "- **上传后**：进入「🔬 误差分析（可换数据）」页，"
            "同一条分析链会自动跑在你的数据上。"
        )
    else:
        st.success(
            f"已收到 {uploaded.name}。"
            "请从左侧进入「🔬 误差分析（可换数据）」页查看完整分析。"
        )
    st.write("")

    # ---------------------------------------------------------------- 成果速览
    st.divider()
    st.subheader("成果速览")
    st.caption("每条只给一句话结论，完整推导与图表见对应页面。")
    st.markdown(
        f"""
- **日较差被压缩 {head['damping']:.0f}%**（观测 9.18 → 预报 7.68 ℃，冬季比值 0.66、夏季≈1.0）→ 详见「🔍 核心发现：振幅阻尼」页
- **误差随强度变**：天越冷越偏暖、天越热越偏冷（r = {head['reg_r']:.2f}），最冷 1% 偏暖 {signed(head['cold_error'], 2, '℃')}，而全样本平均偏差只有 {signed(head['bias'])} → 详见「📊 误差检验」页
- **订正要用对地方**：「减去平均偏差」只改善 {constant['MAE_improvement_pct']:.1f}%，季节×小时耦合订正可拿到 {both_box['MAE_improvement_pct']:.1f}%，但同一套模型迁移到上海站反而变差 11.2%（高度依赖站点）→ 详见「🧪 订正实验」页
- **邻站比历史更有用**：长三角 6 站最优插值把 RMSE 1.77 → 1.44 ℃（改善 18.6%），远好于本站历史偏差订正 → 详见「🛰 资料同化实验（OI）」页
"""
    )
    st.caption("结论的边界、被数据推翻的假设与下一步，见「🧭 局限与展望」页。")

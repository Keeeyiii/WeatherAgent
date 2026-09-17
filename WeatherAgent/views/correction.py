"""订正实验页：嵌套消融，回答“订正的收益来自哪一部分误差”。"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from tools.verification import error_autocorrelation
from views.common import (
    caption,
    C_BLUE,
    C_GREEN,
    C_GREY,
    C_NAVY,
    C_ORANGE,
    base_layout,
    headline_numbers,
    hero,
    load_nanjing,
    run_ablation,
    run_cross_station,
    run_importance,
)


def _pick(summary: pd.DataFrame, keyword: str) -> pd.Series:
    return summary.loc[summary["method"].str.contains(keyword, regex=False)].iloc[0]


def render() -> None:
    hero(
        "订正实验",
        "六个嵌套方法，回答一个反直觉的问题：误差订正的收益，究竟来自哪一部分误差？",
    )
    nj = load_nanjing()
    head = headline_numbers()
    results, summary = run_ablation()

    data_box = _pick(summary, "常数去偏")
    month_box = _pick(summary, "按月去偏")
    hour_box = _pick(summary, "按小时去偏")
    both_box = _pick(summary, "月×小时")
    forest = _pick(summary, "随机森林")

    st.markdown(
        """
        <div class="qbox">
        <b>为什么不做“我的模型提升 X%”这种叙事</b><br>
        本项目更关心的是：机器学习订正带来的改善，是因为它真的学到了复杂的非线性关系，
        还是仅仅因为它<b>顺带消除了一部分系统性结构</b>？
        把结构一层层加进基线，这个问题就可以被拆开回答。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.subheader("结果总览：每一层结构值多少")
    colors = [C_GREY, C_GREY, C_BLUE, C_GREY, C_BLUE, C_ORANGE]
    fig = go.Figure()
    fig.add_bar(
        x=summary["method"],
        y=summary["MAE_mean"],
        marker_color=colors,
        text=[f"{value:.3f}" for value in summary["MAE_mean"]],
        # 数值标签放在柱体内，避免与误差棒重叠
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        cliponaxis=False,
        error_y=dict(type="data", array=summary["MAE_std"], color="#99a3ad"),
    )
    fig.update_layout(title="各方法 MAE（误差棒为三折标准差）")
    fig.update_yaxes(title="MAE (℃)", range=[0, summary["MAE_mean"].max() * 1.16])
    fig.update_xaxes(tickangle=-15)
    st.plotly_chart(base_layout(fig, 400), width="stretch")

    table = summary[
        [
            "method",
            "MAE_mean",
            "MAE_std",
            "bias_mean",
            "MAE_improvement_pct",
            "step_gain_pct",
            "note",
        ]
    ].copy()
    table.columns = [
        "方法",
        "MAE 均值",
        "MAE 折间标准差",
        "bias 均值",
        "相对原始改善 %",
        "本层新增改善 %",
        "这一层检验什么",
    ]
    table = table.round(3)
    table["本层新增改善 %"] = table["本层新增改善 %"].apply(
        lambda value: "—" if pd.isna(value) else f"{value:+.1f}"
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown(
        f"""
        #### 三个反直觉的读法

        **1）常数去偏几乎无效（仅 {data_box['MAE_improvement_pct']:.1f}%）。**
        因为全时段的平均偏差本来就只有 {head['bias']:+.2f} ℃ ——
        需要消除的偏移本身并不存在。
        这一条直接否定了“后处理 = 消除系统偏差”的直觉。

        **2）单独按小时去偏也几乎无效（仅 {hour_box['MAE_improvement_pct']:.1f}%），
        而按月去偏却有 {month_box['MAE_improvement_pct']:.1f}%。**
        原因是日变化偏差在冬季与夏季**符号相反**，在长时段平均后互相抵消；
        季节才是主导的调制因子。但一旦把两者组合起来，
        改善立刻跳到 {both_box['MAE_improvement_pct']:.1f}%（本层新增
        {both_box['step_gain_pct']:.1f} 个百分点）——
        说明误差的真实结构是**季节与日变化的耦合**，而不是两者各自独立叠加。

        **3）随机森林在最强基线上仍有增量（额外 +{forest['step_gain_pct']:.1f} 个百分点）。**
        增量不算大，但方向明确：非线性多变量订正确实捕捉到了查表法拿不到的部分。
        同时必须指出它的折间标准差最大（±{forest['MAE_std']:.3f} ℃），
        **稳定性不如简单方法** —— 这是真实的权衡，不应该被藏在平均值里。
        """
    )

    st.subheader("误差订正模型在看什么")
    importance = run_importance()
    fig = go.Figure()
    fig.add_bar(
        x=importance["importance"],
        y=importance["feature"],
        orientation="h",
        marker_color=C_NAVY,
        text=[f"{value:.3f}" for value in importance["importance"]],
        textposition="outside",
    )
    fig.update_layout(title="随机森林特征重要性")
    fig.update_xaxes(title="重要性")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(base_layout(fig, 320), width="stretch")
    caption(
        "预报温度本身最重要（0.29），其次是风速（0.20）和湿度（0.16）——"
        "这两个量都与边界层湍流交换和地面能量收支密切相关，"
        "与“振幅压缩”的物理解释方向一致。"
        "时间变量的重要性合计约 0.26，说明订正量中确实有很强的气候态成分。"
    )

    st.subheader("逐折明细：结论是否只由某一折撑着？")
    pivot = results.pivot_table(index="method", columns="fold", values="MAE").round(3)
    pivot.columns = [f"第 {int(column)} 折" for column in pivot.columns]
    st.dataframe(pivot, width="stretch")
    caption(
        "三折中，(月×小时)去偏与随机森林都稳定优于原始预报；"
        "第 1 折增益最小，第 2、3 折增益最大 —— 这与“压缩集中在冷季节”的诊断一致。"
    )

    st.divider()
    st.subheader("一个可被推翻的预测：跨站点迁移应当失效")
    st.markdown(
        "如果订正模型本质上学到的是**南京特有的季节—日变化偏差结构**，"
        "那么把它原样搬到另一个站点，性能就应当下降。"
        "这是一个明确的、可以推翻上面整套解释的检验。"
    )
    cross = run_cross_station()
    display = cross[
        ["method", "MAE", "bias", "MAE_improvement_pct", "RMSE_improvement_pct"]
    ].copy()
    display.columns = ["方法", "上海 MAE", "上海 bias", "相对上海原始改善 %", "RMSE 改善 %"]
    st.dataframe(display.round(3), width="stretch", hide_index=True)

    forest_cross = cross.loc[cross["method"].str.contains("随机森林", regex=False)].iloc[0]
    st.error(
        f"**检验结果：预测成立，而且比预想更彻底。** "
        f"南京训练出的所有订正方法在上海都失效，随机森林甚至让 MAE 变差 "
        f"{abs(forest_cross['MAE_improvement_pct']):.1f}%。"
        f"原因有两层：其一，上海的原始预报误差本来就小得多"
        f"（MAE {cross['MAE'].iloc[0]:.2f} ℃，南京为 {head['mae']:.2f} ℃）；"
        f"其二，诊断显示上海几乎不存在振幅压缩问题（日较差比值 0.98，南京为 0.84）。"
        f"**结论：这类统计订正模型高度依赖站点，不能简单推广到新地点。**"
    )

    st.subheader("补充线索：误差高度持续，不是噪声")
    autocorr = error_autocorrelation(nj)
    fig = go.Figure()
    fig.add_bar(x=autocorr["lag_hour"], y=autocorr["autocorr"], marker_color=C_BLUE)
    fig.update_layout(title="预报误差的时间自相关")
    fig.update_xaxes(title="滞后（小时）", dtick=1)
    fig.update_yaxes(title="自相关系数")
    st.plotly_chart(base_layout(fig, 300), width="stretch")
    caption(
        f"滞后 1 小时的自相关高达 {autocorr['autocorr'].iloc[0]:.2f}，"
        f"12 小时后仍有 {autocorr['autocorr'].iloc[-1]:.2f}。"
        "误差不是白噪声，而是与天气过程同步演变的慢变量。"
        "这既解释了为什么订正必须依赖天气状态特征，"
        "也提示“把前一天的观测误差作为预报因子”是一个自然的下一步。"
    )

    with st.expander("方法学说明：为什么用查表法做基线"):
        st.markdown(
            """
            基线方法采用**分组建模 + 查表**（group-mean lookup）：
            在训练集上计算每个分组（月、小时或两者组合）的平均误差，
            然后在测试集上按组取值作为订正量。当某个组合在训练集中没有样本时，
            依次回退到上一级分组，最后回退到全局平均。

            这类基线看似朴素，但在气象后处理中是**很难打败的强基线**：
            它没有方差膨胀问题，也不需要调参，并且在样本足够时能无偏地估计系统性偏差。
            本项目用随机森林与它对比，正是为了避免“只赢弱基线”的常见陷阱。
            """
        )

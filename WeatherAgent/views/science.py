"""发现总览页：把两条核心发现放在一页，作为「科学发现」分区的入口。

只做汇总和导引，不重复计算——每条发现的完整证据在各自的详情页。
"""

from __future__ import annotations

import streamlit as st

from views.common import caption, card, hero


def _goto(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def render() -> None:
    hero(
        "发现总览",
        "两条核心发现：误差的形态（振幅阻尼），和误差的位置（季节 × 昼夜）",
    )

    # ---------------------------------------------------------------- Finding 01
    st.subheader("Finding 01 ｜ 振幅阻尼：模式把温度的起伏压平了")
    st.markdown(
        """
        全时段平均偏差只有 −0.21 ℃、看起来“很准”，但 MAE 高达 1.58 ℃——
        说明误差不是整体偏移，而是**结构**：预报曲线大体跟着观测走，
        但起伏总是偏小，峰值报低了、谷值报高了。这种现象叫**振幅阻尼**。
        """
    )
    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            ("66%", "冷季模式只报出观测日较差的比例（暖季接近 100%）"),
            ("−0.14", "误差对观测距平的回归斜率（模式只还原约 86% 的信号）"),
            ("+2.8 ℃", "冬季夜间最低气温的暖偏差（最高气温几乎无偏差）"),
            ("0.836", "把观测降到 12 小时分辨率后依然成立的压缩比值"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "**四组独立证据全部成立**：日较差被压小 ✓　距平响应不足 ✓　"
        "极端时刻偏差最大 ✓　压缩落在最低气温一侧 ✓"
    )
    if st.button("查看 Finding 01 的完整证据 →", key="goto_finding", width="stretch"):
        _goto("finding")

    st.divider()

    # ---------------------------------------------------------------- Finding 02
    st.subheader("Finding 02 ｜ 误差的位置：季节 × 昼夜结构")
    st.markdown(
        """
        振幅阻尼回答了“误差是什么形态”，下一个问题是“误差在哪里出现”——
        把误差按季节和一天中的时刻分解，偏差符号会完全翻转。
        """
    )
    columns = st.columns(3)
    for column, (number, label) in zip(
        columns,
        [
            ("冬暖 / 夏冷", "季节偏差符号完全相反，全年平均因此接近 0"),
            ("夜暖 / 日冷", "日变化内同样存在符号翻转"),
            ("偏差 ≈ 0", "随预报时效增长的是 RMSE，不是 bias"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "这两条发现是嵌套的：**振幅阻尼是形态，季节 × 昼夜结构是落点**——"
        "冬季夜间正是模式跟不上强辐射降温、阻尼最明显的地方。"
    )
    if st.button("查看误差检验的完整分解 →", key="goto_verification", width="stretch"):
        _goto("verification")

    st.divider()

    st.markdown("#### 这些发现是怎么得出来的？")
    st.markdown(
        "数据来源、质检标准、指标定义和全部方法细节见 **数据与方法** 页；"
        "每条发现的稳健性检验见 **🧪 验证与实验** 分区。"
    )
    if st.button("前往数据与方法 →", key="goto_data_method", width="stretch"):
        _goto("data-method")

    caption("本页所有数字均摘自示例数据的现有计算结果，本页不做任何重新计算。")

"""WeatherAgent · 真实数据版

一个关于“预报误差长什么样”的小型研究：
用南京禄口站 2.4 年真实逐小时观测检验 GFS 2 米气温预报，
诊断误差的结构，并检验误差订正的收益究竟来自哪里。

    运行：  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from views import (
    agent,
    cases,
    correction,
    da,
    data_method,
    finding,
    home,
    limits,
    verification,
)
from views.common import inject_css

st.set_page_config(
    page_title="WeatherAgent · GFS 气温预报误差结构诊断",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

PAGES = {
    "🏠 首页 · 成果总览": home.render,
    "📐 数据与方法": data_method.render,
    "📊 误差检验": verification.render,
    "🔍 核心发现：振幅阻尼": finding.render,
    "🧪 订正实验": correction.render,
    "🛰 资料同化实验（OI）": da.render,
    "🌀 真实案例": cases.render,
    "🧭 局限与展望": limits.render,
    "🤖 研究助手": agent.render,
}


def main() -> None:
    # 便于自动化检查：设置环境变量即可直接打开指定页面
    forced = os.environ.get("WEATHERAGENT_PAGE")
    if forced in PAGES:
        index = list(PAGES).index(forced)
    else:
        try:
            index = int(os.environ.get("WEATHERAGENT_PAGE_INDEX", "0"))
        except ValueError:
            index = 0
        index = max(0, min(index, len(PAGES) - 1))
    with st.sidebar:
        st.markdown("### 🌦 WeatherAgent")
        st.caption("GFS 2 米气温预报误差的结构诊断")
        choice = st.radio(
            "导航", list(PAGES), index=index, label_visibility="collapsed"
        )
        st.divider()
        st.markdown(
            "**一句话摘要**\n\n"
            "误差的形态是**振幅阻尼**——模式把温度的起伏压平了："
            "该冷时不够冷、该热时不够热。"
            "压缩落在**冬季夜间最低气温**（偏高约 2.8 ℃），日最高气温几乎无偏差。"
        )
        st.divider()
        st.caption(
            "数据：南京禄口 ZSNJ 真实逐小时观测 + GFS 历史预报归档\n\n"
            "样本：23,543 小时（2024-01 至 2026-09）"
        )
    PAGES[choice]()


if __name__ == "__main__":
    main()

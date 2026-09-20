"""WeatherAgent · 真实数据版

一个关于“预报误差长什么样”的小型研究：
用南京禄口站 2.4 年真实逐小时观测检验 GFS 2 米气温预报，
诊断误差的结构，并检验误差订正的收益究竟来自哪里。

信息架构（2026-09 重构）：
    🏠 研究问题 → 🔬 分析工作流 → 🧠 科学发现 → 🧪 验证与实验 → 🧭 局限与下一步
每个分区的第一页是该分区的导引页，原有功能页全部保留、只做归类。

    运行：  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from views import (
    agent,
    cases,
    correction,
    da,
    data_method,
    experiments,
    finding,
    home,
    limits,
    science,
    tool,
    verification,
    workflow,
)
from views.common import inject_css

st.set_page_config(
    page_title="WeatherAgent · GFS 气温预报误差结构诊断",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# 页面按信息架构分组。自管导航状态而不依赖虚拟页路由，保证页面内 CTA
# 在用户提交表单、调整研究计划后仍停留在同一条研究流程中。
PAGES = {
    # 注意：分组键必须是字符串；None 会令 st.navigation 的 protobuf 序列化报错。
    "🏠 研究问题": [
        ("home", "主页 · 研究问题", home.render),
    ],
    "🔬 分析工作流": [
        ("workflow", "分析工作流", workflow.render),
        ("workflow-run", "误差分析流水线（可换数据）", tool.render),
        ("agent", "研究助手", agent.render),
    ],
    "🧠 科学发现": [
        ("science", "发现总览", science.render),
        ("finding", "核心发现：振幅阻尼", finding.render),
        ("data-method", "数据与方法", data_method.render),
    ],
    "🧪 验证与实验": [
        ("experiments", "实验总览", experiments.render),
        ("verification", "误差检验", verification.render),
        ("correction", "订正实验", correction.render),
        ("da", "资料同化实验（OI）", da.render),
        ("cases", "真实案例", cases.render),
    ],
    "🧭 局限与下一步": [
        ("limits", "局限与展望", limits.render),
    ],
}


def main() -> None:
    page_by_key = {
        key: (label, render)
        for entries in PAGES.values()
        for key, label, render in entries
    }
    st.session_state.setdefault("active_page", "home")

    # 侧边栏：品牌区 → 五层研究导航 → 一句话摘要。
    with st.sidebar:
        st.markdown("### 🌦 WeatherAgent")
        st.caption("研究误差怎么变：用真实观测诊断 GFS 气温预报的误差结构")
        for group, entries in PAGES.items():
            st.markdown(f"**{group}**")
            for key, label, _ in entries:
                if st.button(
                    label,
                    key=f"nav_{key}",
                    width="stretch",
                    type="primary" if st.session_state["active_page"] == key else "secondary",
                ):
                    st.session_state["active_page"] = key
                    st.rerun()
        st.divider()
        st.markdown(
            "**一句话摘要**\n\n"
            "误差的形态是**振幅阻尼**——模式把温度的起伏压平了："
            "该冷时不够冷、该热时不够热。"
            "偏差集中在**冬季夜间最低气温**（预报偏高约 2.8 ℃），日最高气温几乎无偏差。"
        )
        st.divider()
        st.caption(
            "数据：南京禄口 ZSNJ 真实逐小时观测 + GFS 历史预报归档\n\n"
            "样本：23,543 小时（2024-01 至 2026-09）"
        )
    _, render = page_by_key[st.session_state["active_page"]]
    render()


if __name__ == "__main__":
    main()


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

# 页面按信息架构分组；url_path 用于深链接（?page=workflow）与页面间跳转。
PAGES = {
    # 注意：分组键必须是字符串；None 会令 st.navigation 的 protobuf 序列化报错。
    "🏠 研究问题": [
        st.Page(home.render, title="主页 · 研究问题", url_path="home", default=True),
    ],
    "🔬 分析工作流": [
        st.Page(workflow.render, title="分析工作流", url_path="workflow"),
        st.Page(tool.render, title="误差分析流水线（可换数据）", url_path="workflow-run"),
    ],
    "🧠 科学发现": [
        st.Page(science.render, title="发现总览", url_path="science"),
        st.Page(finding.render, title="核心发现：振幅阻尼", url_path="finding"),
        st.Page(data_method.render, title="数据与方法", url_path="data-method"),
    ],
    "🧪 验证与实验": [
        st.Page(experiments.render, title="实验总览", url_path="experiments"),
        st.Page(verification.render, title="误差检验", url_path="verification"),
        st.Page(correction.render, title="订正实验", url_path="correction"),
        st.Page(da.render, title="资料同化实验（OI）", url_path="da"),
        st.Page(cases.render, title="真实案例", url_path="cases"),
    ],
    "🧭 局限与下一步": [
        st.Page(limits.render, title="局限与展望", url_path="limits"),
    ],
    "🤖 辅助工具": [
        st.Page(agent.render, title="研究助手", url_path="agent"),
    ],
}


def main() -> None:
    # 侧边栏：品牌区 → 分组导航 → 一句话摘要
    with st.sidebar:
        st.markdown("### 🌦 WeatherAgent")
        st.caption("研究误差怎么变：用真实观测诊断 GFS 气温预报的误差结构")
    nav = st.navigation(PAGES, expanded=True)
    with st.sidebar:
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
    nav.run()


if __name__ == "__main__":
    main()

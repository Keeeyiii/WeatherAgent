"""研究问题入口页：整个应用的起点。

首页只做一件事——把访客的问题接住，交给分析工作流。
结论、图表、目录全部移到「科学发现」「验证与实验」分区，首页保持极简。
"""

from __future__ import annotations

import streamlit as st

from views.common import hero

EXAMPLE_QUESTIONS = [
    "为什么天气预报有时差好几度？",
    "预报误差是随机的，还是有规律的？",
    "为什么冬天的夜间最低气温总是报不准？",
    "用机器学习能订正多少预报误差？",
    "换一个站点，订正方法还好用吗？",
]


def _goto_workflow(question: str) -> None:
    st.session_state["user_question"] = question.strip()
    st.query_params["page"] = "workflow"
    st.rerun()


def render() -> None:
    hero(
        "WeatherAgent",
        "Interactive Atmospheric Research Assistant",
        large=True,
    )

    st.markdown("## 你想研究什么？")
    st.caption("点一个示例问题，或者直接输入你的问题——应用会围绕它展开分析。")

    for row in range(3):
        columns = st.columns(2, gap="small")
        for column, index in zip(columns, (2 * row, 2 * row + 1)):
            if index < len(EXAMPLE_QUESTIONS):
                if column.button(
                    EXAMPLE_QUESTIONS[index],
                    key=f"example_q_{index}",
                    width="stretch",
                ):
                    _goto_workflow(EXAMPLE_QUESTIONS[index])

    st.write("")
    question = st.text_input(
        "或者直接描述你的研究问题……",
        key="research_question",
        placeholder="例如：GFS 预报的 2 米气温误差，随季节和一天中的时刻怎么变化？",
    )
    if st.button("开始研究 →", key="start_research", type="primary", width="stretch"):
        if question.strip():
            _goto_workflow(question)
        else:
            st.warning("请先点一个示例问题，或在输入框里写下你的问题。")

    st.divider()
    st.caption(
        "这个应用用南京禄口机场 23,543 小时的真实观测，诊断 GFS 气温预报的误差结构。\n\n"
        "如果你想直接看结论，请前往左侧 **🧠 科学发现** 分区。"
    )

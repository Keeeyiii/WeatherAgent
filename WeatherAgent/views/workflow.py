"""分析工作流：把一个自然语言问题转为可复核的研究计划。

本页不假装已经用 LLM 理解了用户的问题。它先提供一套可调整、可复核的
研究计划；真正的指标和图表仍由下一页既有的分析流水线计算。
"""

from __future__ import annotations

import streamlit as st

from views.common import caption, go_to, hero

STEPS = {
    "数据可信度": "① 数据质量检查：行数、必要字段、缺失值、重复、时间有效性与单位合理性。",
    "误差画像": "② 总体误差检验 + ③ 季节与日变化分解：先给出总体指标，再检查误差是否有条件结构。",
    "极端与振幅": "④ 振幅阻尼判据 + ⑤ 最低/最高气温分解：用多组证据判断是否为模式响应不足。",
    "稳健性检验": "⑥ 排除采样假象 + ⑦ 误差持续性：排除比较方式造成的假象，并判断误差是否为慢变量。",
    "订正实验": "⑧ 订正方法消融 + ⑨ 机器学习订正：检验改善来自平均偏差还是误差结构。",
    "验证与泛化": "⑩ 跨站点迁移、邻站同化与真实个例：明确结论和方法可以推广到哪里。",
}

FOCUS_OPTIONS = {
    "诊断预报误差的结构": "先判断误差是否系统性，再定位它在季节、昼夜和极端条件下的表现。",
    "理解极端天气中的预报误差": "重点查看冷、热极端和日最低/最高气温；不把相关性表述成物理因果。",
    "检验预报订正是否有效": "用严格的时间划分比较订正方法，并把跨站点结果作为泛化边界。",
    "用自己的预报—观测数据复现分析": "先做数据质量检查，再用与示例数据相同的指标和诊断流程。",
}


def render() -> None:
    hero(
        "分析工作流",
        "从一个气象问题出发，逐步完成数据检查、误差诊断、结构分析和订正实验",
    )

    question = (st.session_state.get("user_question") or "").strip()
    if question:
        st.info(f"**当前研究问题**　{question}")
    else:
        st.info(
            "还没有设定研究问题——下面按默认问题展开。"
            "你可以回到 **🏠 研究问题** 页选择一个示例问题。"
        )

    st.subheader("研究问题 → 可检验的计划")
    st.markdown(
        f"**你正在研究**　{question or '为什么天气预报会出错？'}\n\n"
        "这里的“理解”是一个由你确认的研究计划，不是对任意问题自动生成的因果结论。"
        "所有结论都必须回到数据、图表和验证实验。"
    )

    with st.form("research_plan_form"):
        focus = st.selectbox("本次研究重点", list(FOCUS_OPTIONS), key="focus_choice")
        st.caption(FOCUS_OPTIONS[focus])
        selected_steps = st.multiselect(
            "选择本次要走的分析环节",
            list(STEPS),
            default=list(STEPS),
            help="可以缩小探索范围；未选中的环节仍会保留在左侧导航，方便后续补做。",
        )
        plan_confirmed = st.form_submit_button("确认研究计划", type="primary", width="stretch")

    if plan_confirmed:
        st.session_state["research_plan"] = {"focus": focus, "steps": selected_steps}
        st.success("研究计划已确认。你可以修改后再次确认，或直接开始执行。")

    plan = st.session_state.get("research_plan")
    if plan:
        st.subheader("当前研究计划")
        st.markdown(f"**研究重点**　{plan['focus']}")
        for name in plan["steps"]:
            st.markdown(f"- **{name}**　{STEPS[name]}")
    else:
        st.subheader("建议分析路径")
        for name, description in STEPS.items():
            st.markdown(f"- **{name}**　{description}")

    st.info(
        "**证据标准**：描述误差结构时使用现有指标与图表；解释可能机制时保持“可能相关”，"
        "不把单站统计关系写成已证明的物理因果。"
    )
    if st.button("执行这项研究 →", key="run_analysis", type="primary", width="stretch"):
        go_to("workflow-run")

    caption(
        "下一页会用现有的同一条分析流水线执行数据检查、误差诊断与订正对比；"
        "示例数据的证据和稳健性检验分别位于 **🧠 科学发现**、**🧪 验证与实验**。"
    )


"""研究助手页：用自然语言查询本项目的计算结果。"""

from __future__ import annotations

import os

import streamlit as st

from tools.research_agent import SUGGESTED, answer
from views.common import caption, hero


def _llm_available() -> bool:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    try:
        if not key and "DEEPSEEK_API_KEY" in st.secrets:
            key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:  # noqa: BLE001  (secrets 文件不存在时会抛异常)
        pass
    return bool(key)


def render() -> None:
    hero(
        "研究助手",
        "用自然语言查询本项目的计算结果；不配置任何 API Key 也可以使用",
    )

    caption(
        "助手只引用本应用已经算出的真实结果，不自行编造数字。"
        "配置 DeepSeek / OpenAI API Key 后可升级为 LLM 版本（见页面底部）。"
    )

    question = st.text_input(
        "问一个关于本项目的问题",
        placeholder="例如：为什么常数去偏没有效果？",
    )
    st.markdown("**可以试试：** " + " ｜ ".join(f"`{item}`" for item in SUGGESTED))

    if question:
        reply, evidence = answer(question)
        st.markdown(reply)
        if evidence:
            with st.expander("这条回答引用的计算结果"):
                for key, value in evidence.items():
                    st.markdown(f"- **{key}**：{value}")

    st.divider()
    st.subheader("设计说明")
    st.markdown(
        """
        本页默认使用**离线规则版助手**：所有回答都取自应用已计算出的结果表，
        因此没有网络、没有 API Key 也能工作，更不会编造数字。

        这是一个有意的取舍。气象数值产品最怕的就是"看起来合理但来源不明的数字"，
        所以本项目把分工写死在架构里：

        - **Python 负责**：数据读取、MAE / RMSE / bias 计算、消融实验、显著性对比等全部数值工作；
        - **语言模型只负责**：理解问题、选择合适的计算结果、组织成自然语言。
        """
    )

    if _llm_available():
        st.success("检测到 API Key：已具备启用 LLM 对话版本的条件（需要额外实现调用逻辑）。")
    else:
        st.info(
            "未检测到 API Key，当前运行的是离线规则版。"
            "如需启用 LLM 版本，可在 `Settings → Secrets` 中添加："
            "`DEEPSEEK_API_KEY = \"你的Key\"`。"
        )

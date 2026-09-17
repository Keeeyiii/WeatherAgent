"""数据与方法页：讲清楚数据从哪来、指标怎么算、实验为什么这样设计。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views.common import caption, headline_numbers, hero, load_nanjing


def render() -> None:
    hero(
        "数据与方法",
        "为什么用真实数据、为什么采用严格时序划分、指标如何定义",
    )
    head = headline_numbers()
    nj = load_nanjing()

    st.markdown(
        """
        <div class="qbox">
        <b>一条原则</b><br>
        所有结论都必须能被<b>真实观测</b>检验，所有指标都必须给出可复现的计算方式。
        本项目<b>不含任何合成数据</b>，也不使用“演示用随机数”。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. 数据来源")
    st.markdown(
        """
| 用途 | 数据内容 | 来源 | 处理说明 |
|---|---|---|---|
| 观测（真值） | 南京禄口机场 ZSNJ 逐小时地面观测 | Iowa Environmental Mesonet（METAR 归档） | 气温由 ℉ 换算为 ℃，风速由 kt 换算为 m/s |
| 观测（对照站） | 上海虹桥 ZSSS 逐小时地面观测 | 同上 | 用于跨站点迁移检验 |
| 预报 | GFS 2 米气温历史预报 | Open-Meteo 历史预报归档 | 与观测按同一时刻配对 |
| 预报（时效） | GFS 1/2/3/5/7 天预报 | Open-Meteo previous-runs 归档 | 用于误差随预报时效增长的分析 |
        """
    )
    caption(
        f"当前载入：南京 {len(nj):,} 个有效配对小时，"
        f"{nj['time'].min():%Y-%m-%d} 至 {nj['time'].max():%Y-%m-%d}；"
        "所有时间统一为 UTC 存储，展示时换算为北京时（UTC+8）。"
    )

    st.subheader("2. 指标定义")
    st.latex(r"\text{error} = F - O")
    st.latex(
        r"\text{bias} = \overline{F - O}, \qquad "
        r"\text{MAE} = \overline{|F - O|}, \qquad "
        r"\text{RMSE} = \sqrt{\overline{(F-O)^2}}"
    )
    st.markdown(
        f"""
        - 符号约定：**error > 0 表示预报偏高（偏暖），error < 0 表示预报偏低（偏冷）**。
        - **bias** 是带符号的平均误差，正负可以相互抵消；**MAE / RMSE** 反映误差量级，不抵消。
        - 正因为两种指标的性质不同，本项目第一步发现的落差就很有信息量：
          **平均偏差只有 {head['bias']:+.2f} ℃，MAE 却高达 {head['mae']:.2f} ℃。**
          这说明误差的主体不是“整体偏移”，而是某种**结构**。后续所有分析都在追问这个结构是什么。
        """
    )

    st.subheader("3. 为什么必须用严格时序划分")
    st.markdown(
        """
        本项目采用**扩张窗口 + 严格时序**的三折验证，而不是常见的随机划分：

        ```
        第 1 折   训练 ──────────► ｜ 测试 ──────►
        第 2 折   训练 ────────────────────► ｜ 测试 ──────►
        第 3 折   训练 ──────────────────────────────► ｜ 测试 ──────►
                 训练数据永远早于测试数据，三折测试段互不重叠
        ```

        - **不用随机划分的理由**：气象时间序列高度自相关。
          本项目实测误差在滞后 1 小时的自相关达到 **0.90**，
          随机划分会把同一次天气过程同时放进训练集和测试集，指标虚高，
          看起来很好，实际上不能反映真正的预报能力。
        - **三折测试段跨越不同季节**，顺带完成了“跨季节验证”。
        - 所有方法在**完全相同**的训练/测试样本上比较，因此指标差异只来自方法本身。
        """
    )

    st.subheader("4. 订正实验的设计：嵌套消融，而不是模型竞赛")
    st.markdown(
        """
        六个方法并非并列竞争的“选手”，而是**层层嵌套**的关系：
        后一个方法 = 前一个方法 + 一块新的误差结构。
        因此每一步的 MAE 下降量，直接度量了那块结构值多少。
        """
    )
    design = pd.DataFrame(
        [
            ("① 原始预报", "无", "建立基准"),
            ("② 常数去偏", "全时段平均偏差", "检验“整体偏移”是否是主要误差"),
            ("③ 按月去偏", "季节调制", "检验“季节结构”的价值"),
            ("④ 按小时去偏", "日变化", "单独检验“日变化”的价值（与③对照）"),
            ("⑤ 按(月×小时)去偏", "季节 × 日变化耦合", "检验两者是否必须同时考虑"),
            ("⑥ 随机森林", "六个要素的非线性组合", "检验非线性订正还能带来多少增量"),
        ],
        columns=["方法", "新引入的误差结构", "这一层要回答什么问题"],
    )
    st.dataframe(design, width="stretch", hide_index=True)

    st.info(
        "这个设计的价值在于：如果随机森林的改善完全来自它“顺带”消除的系统性结构，"
        "那么当基线已经包含这些结构时，随机森林的增量就应该很小。"
        "这是一个可以被数据推翻的判断——「🧪 订正实验」页给出结果。"
    )

    st.subheader("5. 复现方式")
    st.code(
        "# 重新抓取真实数据（仓库已内置数据文件，通常无需重跑）\n"
        "python tools/real_data.py\n\n"
        "# 跑完整分析并导出全部结果表\n"
        "python analysis/run_verification.py\n\n"
        "# 启动应用\n"
        "streamlit run app.py",
        language="bash",
    )
    caption(
        "依赖：Python 3.10+、pandas、numpy、scikit-learn、plotly、streamlit、requests。"
        "数据抓取需要联网；应用本身读取本地 CSV，离线也能运行。"
    )

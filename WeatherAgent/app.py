"""
WeatherAgent —— AI气象预报智能分析与误差订正系统

入口文件：负责搭建 Streamlit 网页界面。

本系统为科研实践/实验性原型，用于演示：
气象数据质量检查 -> 预报误差评价 -> 机器学习误差订正 -> 订正效果检验 -> DeepSeek AI Agent
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.weather_agent import WeatherAgent
from tools.data_checker import check_data
from tools.forecast_evaluator import evaluate_forecast
from tools.ml_correction import machine_learning_correction

# Streamlit Cloud 部署兼容：若在 Secrets 中配置了 Key，则写入环境变量供 Agent 读取。
# 注意：本地未配置 secrets.toml 时，访问 st.secrets 会抛异常，需捕获后忽略
#（此时改由 .env 中的 Key 生效，见 agent/weather_agent.py）。
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# 内置示例数据的路径（基于本文件所在目录，避免工作目录变化导致找不到）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_weather.csv")


@st.cache_data
def load_sample_data(path):
    """读取内置示例数据（带缓存，避免每次刷新都重复读盘）。"""
    return pd.read_csv(path)


# ---------------- 页面基础设置 ----------------
st.set_page_config(
    page_title="WeatherAgent",
    layout="wide",
)

st.title("WeatherAgent")
st.markdown("### AI气象预报智能分析与误差订正系统")

st.markdown("---")

# ---------------- ① 项目概览 ----------------
st.header("① 项目概览")
st.markdown(
    """
    本项目尝试将**气象数据分析、机器学习和 LLM Agent** 相结合，
    用于气象预报误差分析与后处理实验。

    **主要功能：**
    1. 气象数据质量检查
    2. 预报误差评价
    3. MAE、RMSE、Bias 计算
    4. Random Forest 误差订正
    5. 订正前后效果对比
    6. 特征重要性分析
    7. DeepSeek AI 气象分析 Agent
    """
)

# 方法说明（可展开）
with st.expander("方法说明"):
    st.markdown(
        """
        **1. 预报误差定义**
        `error = forecast - observed`（预报值减观测值）
        - error > 0：预报偏高（高估）；error < 0：预报偏低（低估）

        **2. 预报评价指标**
        - MAE = mean(|error|)
        - RMSE = sqrt(mean(error²))
        - Bias = mean(error)

        **3. 机器学习误差订正**
        使用 Random Forest 学习「订正量」`correction = observed - forecast`：
        `predicted_correction = f(forecast, humidity, pressure, wind_speed, hour, month)`
        然后 `corrected_forecast = forecast + predicted_correction`

        **4. 验证方式：滚动窗口交叉验证**
        训练窗口逐折扩大、测试段依次后移且互不重叠（共 3 折，每折测试约 20% 数据）；
        每一折的训练数据全部位于测试数据之前，无数据泄漏；
        最终汇总各折指标为「均值 ± 标准差」，反映订正效果的稳定性。
        """
    )

st.markdown("---")

# ---------------- ② 数据基本情况 ----------------
st.header("② 数据基本情况")
st.markdown("上传你自己的 CSV 数据文件；不上传则自动使用内置示例数据。")

uploaded_file = st.file_uploader("选择 CSV 文件（可选）", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source = "用户上传文件"
else:
    df = load_sample_data(SAMPLE_PATH)
    data_source = "内置示例数据 sample_weather.csv"

st.success(f"当前数据来源：{data_source}")

col1, col2 = st.columns(2)
col1.metric("数据行数", len(df))
col2.metric("数据列数", len(df.columns))

st.subheader("数据预览")
st.dataframe(df.head(10))

# 开始基础分析按钮
if st.button("开始气象数据分析", type="primary"):
    st.session_state["analyzed"] = True

if st.session_state.get("analyzed", False):
    st.markdown("---")

    # ---------------- ③ 数据质量检查 ----------------
    st.header("③ 数据质量检查")

    report = check_data(df)

    if report["ok"]:
        st.success("数据基础检查通过：必要字段齐全，时间字段可正常解析。")
    else:
        st.error("数据存在问题，请查看下方明细。")

    col1, col2, col3 = st.columns(3)
    col1.metric("重复行数", report["duplicate_rows"])
    col2.metric("缺失值总数", sum(report["missing_values"].values()))
    col3.metric("异常值总数", sum(report["outliers"].values()))

    st.subheader("必要字段")
    if report["missing_fields"]:
        st.error("缺失字段：" + "、".join(report["missing_fields"]))
    else:
        st.success("所有必要字段均存在。")

    st.subheader("时间字段")
    if report["time_ok"]:
        st.success("time 字段可正常解析为日期时间。")
    else:
        st.error(report["time_issue"])

    st.subheader("字段类型")
    st.write(report["dtypes"])
    for issue in report["type_issues"]:
        st.error(issue)

    st.subheader("缺失值统计")
    st.write(report["missing_values"])

    st.subheader("异常值统计（依据物理合理范围）")
    st.write(report["outliers"])

    st.markdown("---")

    # ---------------- ④ 原始预报检验 ----------------
    st.header("④ 原始预报检验")
    st.markdown(
        """
        由 Python 真实计算三个误差指标（`error = 预报 - 实况`）：
        - **MAE**（平均绝对误差）：`mean(|error|)`
        - **RMSE**（均方根误差）：`sqrt(mean(error²))`
        - **Bias**（平均偏差）：`mean(error)`，正值表示预报系统性偏高
        """
    )

    if "observed_temperature" not in df.columns or "forecast_temperature" not in df.columns:
        st.warning("缺少观测温度或预报温度字段，无法进行误差检验。")
    else:
        metrics = evaluate_forecast(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE 平均绝对误差", f"{metrics['MAE']:.3f} °C")
        col2.metric("RMSE 均方根误差", f"{metrics['RMSE']:.3f} °C")
        col3.metric("Bias 平均偏差", f"{metrics['Bias']:.3f} °C")

        # 观测 vs 原始预报曲线
        st.subheader("观测温度 vs 原始预报温度")
        time = pd.to_datetime(df["time"], errors="coerce")
        observed = pd.to_numeric(df["observed_temperature"], errors="coerce")
        forecast = pd.to_numeric(df["forecast_temperature"], errors="coerce")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=observed, mode="lines", name="Observed"))
        fig.add_trace(go.Scatter(x=time, y=forecast, mode="lines", name="Raw Forecast"))
        fig.update_layout(
            title="观测温度 vs 原始预报温度",
            xaxis_title="时间",
            yaxis_title="Temperature (°C)",
            legend_title="图例",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

# ---------------- 机器学习订正按钮 ----------------
st.markdown("---")
if st.button("开始机器学习订正", type="primary"):
    st.session_state["ml_done"] = True

if st.session_state.get("ml_done", False):
    # ---------------- ⑤ 机器学习误差订正 ----------------
    st.header("⑤ 机器学习误差订正（滚动窗口交叉验证）")
    st.markdown(
        """
        使用 **Random Forest** 学习预报的「订正量」`correction = observed - forecast`，
        再用 `corrected = forecast + predicted_correction` 完成订正。

        特征：预报温度、湿度、气压、风速、小时、月份。

        验证方式：**滚动窗口交叉验证**（rolling window cross-validation）——
        训练窗口逐折扩大、测试段依次后移且互不重叠，共 3 折；
        每折独立训练并检验，最终汇总为**均值 ± 标准差**，
        以体现订正效果在不同时间段的稳定性，而非单次划分的偶然结果。
        每一折的训练数据全部位于测试数据之前，不存在数据泄漏。
        """
    )

    ml_required = [
        "observed_temperature",
        "forecast_temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    missing_ml = [c for c in ml_required if c not in df.columns]

    if missing_ml:
        st.warning("缺少必要字段，无法进行机器学习订正：" + "、".join(missing_ml))
        st.session_state.pop("ml_result", None)
    else:
        with st.spinner("正在进行滚动窗口交叉验证（逐折训练随机森林）……"):
            ml_result = machine_learning_correction(df)

        st.session_state["ml_result"] = ml_result

        st.success(
            f"验证完成：共 {ml_result['n_splits']} 折，"
            f"总有效样本 {ml_result['total_size']} 条，"
            f"每折测试 {ml_result['test_len_per_fold']} 条"
            f"（第 1 折训练 {ml_result['initial_train_size']} 条，训练窗口逐折扩大）。"
        )

        folds = ml_result["folds"]
        raw_mean = ml_result["raw_mean"]
        cor_mean = ml_result["corrected_mean"]
        cor_std = ml_result["corrected_std"]

        # 每一折的详细结果
        st.subheader(f"每一折的检验结果（共 {ml_result['n_splits']} 折）")
        folds_df = pd.DataFrame(
            [
                {
                    "折": f["fold"],
                    "训练样本": f["train_size"],
                    "测试样本": f["test_size"],
                    "测试时段": f"{f['test_start']} ~ {f['test_end']}",
                    "原始 MAE": round(f["raw"]["MAE"], 3),
                    "订正后 MAE": round(f["corrected"]["MAE"], 3),
                    "原始 RMSE": round(f["raw"]["RMSE"], 3),
                    "订正后 RMSE": round(f["corrected"]["RMSE"], 3),
                    "原始 Bias": round(f["raw"]["Bias"], 3),
                    "订正后 Bias": round(f["corrected"]["Bias"], 3),
                }
                for f in folds
            ]
        )
        st.dataframe(folds_df.set_index("折"))

        # 每折 MAE 对比柱状图
        fig_folds = go.Figure()
        fig_folds.add_trace(
            go.Bar(
                x=folds_df["折"].astype(str),
                y=folds_df["原始 MAE"],
                name="原始预报 MAE",
            )
        )
        fig_folds.add_trace(
            go.Bar(
                x=folds_df["折"].astype(str),
                y=folds_df["订正后 MAE"],
                name="订正后 MAE",
            )
        )
        fig_folds.update_layout(
            title="各折 MAE 对比（滚动窗口交叉验证）",
            xaxis_title="折",
            yaxis_title="MAE (°C)",
            barmode="group",
            legend_title="图例",
            height=350,
        )
        st.plotly_chart(fig_folds, width="stretch")

        # 整体指标：各折均值 ± 标准差
        st.subheader("整体表现（各折均值 ± 标准差）")
        c1, c2 = st.columns(2)
        c1.metric("原始预报 MAE（均值）", f"{raw_mean['MAE']:.3f} °C")
        c2.metric(
            "订正后 MAE（均值 ± 标准差）",
            f"{cor_mean['MAE']:.3f} ± {cor_std['MAE']:.3f} °C",
        )

        c1, c2 = st.columns(2)
        c1.metric("原始预报 RMSE（均值）", f"{raw_mean['RMSE']:.3f} °C")
        c2.metric(
            "订正后 RMSE（均值 ± 标准差）",
            f"{cor_mean['RMSE']:.3f} ± {cor_std['RMSE']:.3f} °C",
        )

        c1, c2 = st.columns(2)
        c1.metric("原始预报 Bias（均值）", f"{raw_mean['Bias']:.3f} °C")
        c2.metric(
            "订正后 Bias（均值 ± 标准差）",
            f"{cor_mean['Bias']:.3f} ± {cor_std['Bias']:.3f} °C",
        )

        # 改善率（各折均值，如实显示，可能为负）
        c1, c2 = st.columns(2)
        c1.metric(
            "MAE 改善率（各折均值）",
            f"{ml_result['mae_improve_mean']:.2f} %",
        )
        c2.metric(
            "RMSE 改善率（各折均值）",
            f"{ml_result['rmse_improve_mean']:.2f} %",
        )

        st.caption(
            "标准差越小，说明订正效果在不同时间段越稳定；"
            "「均值 ± 标准差」共同反映模型效果的典型水平与波动范围。"
        )

        # 特征重要性（各折平均，表格 + 柱状图）
        st.subheader("Random Forest 特征重要性（各折平均）")
        importance_df = pd.DataFrame(
            list(ml_result["importance"].items()),
            columns=["特征", "重要性"],
        )
        st.dataframe(importance_df.set_index("特征"))

        fig_imp = go.Figure(
            go.Bar(x=importance_df["特征"], y=importance_df["重要性"])
        )
        fig_imp.update_layout(
            title="Random Forest 特征重要性",
            xaxis_title="特征",
            yaxis_title="重要性",
            height=350,
        )
        st.plotly_chart(fig_imp, width="stretch")

    st.markdown("---")

    # ---------------- ⑥ 可视化分析 ----------------
    st.header("⑥ 可视化分析")

    ml_result = st.session_state.get("ml_result")
    if ml_result is None:
        st.info("请先完成机器学习订正。")
    else:
        test_df = ml_result["test_results"]
        time = pd.to_datetime(test_df["time"], errors="coerce")

        st.subheader("观测 vs 原始预报 vs 订正后预报（滚动验证各折测试段）")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=time, y=test_df["observed_temperature"], mode="lines", name="Observed")
        )
        fig.add_trace(
            go.Scatter(x=time, y=test_df["forecast_temperature"], mode="lines", name="Raw Forecast")
        )
        fig.add_trace(
            go.Scatter(x=time, y=test_df["corrected_forecast"], mode="lines", name="Corrected Forecast")
        )
        fig.update_layout(
            title="订正前后温度对比（滚动验证各折测试段）",
            xaxis_title="时间",
            yaxis_title="Temperature (°C)",
            legend_title="图例",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("原始误差 vs 订正后误差")
        raw_error = test_df["forecast_temperature"] - test_df["observed_temperature"]
        corrected_error = test_df["corrected_forecast"] - test_df["observed_temperature"]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=time, y=raw_error, mode="lines", name="Raw Forecast Error")
        )
        fig2.add_trace(
            go.Scatter(x=time, y=corrected_error, mode="lines", name="Corrected Forecast Error")
        )
        fig2.update_layout(
            title="预报误差对比（滚动验证各折测试段）",
            xaxis_title="时间",
            yaxis_title="Error (°C)",
            legend_title="图例",
            height=400,
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    # ---------------- 实验结果总结（自动生成） ----------------
    st.header("实验结果总结")

    ml_result = st.session_state.get("ml_result")
    if ml_result is None:
        st.info("请先完成机器学习订正。")
    else:
        raw_mean = ml_result["raw_mean"]
        cor_mean = ml_result["corrected_mean"]
        cor_std = ml_result["corrected_std"]

        summary_df = pd.DataFrame(
            {
                "指标": ["MAE", "RMSE", "Bias"],
                "原始预报（各折均值）": [
                    round(raw_mean["MAE"], 4),
                    round(raw_mean["RMSE"], 4),
                    round(raw_mean["Bias"], 4),
                ],
                "订正后（各折均值）": [
                    round(cor_mean["MAE"], 4),
                    round(cor_mean["RMSE"], 4),
                    round(cor_mean["Bias"], 4),
                ],
                "订正后（各折标准差）": [
                    round(cor_std["MAE"], 4),
                    round(cor_std["RMSE"], 4),
                    round(cor_std["Bias"], 4),
                ],
            }
        ).set_index("指标")
        st.dataframe(summary_df)

        c1, c2 = st.columns(2)
        c1.metric("MAE 改善率（各折均值）", f"{ml_result['mae_improve_mean']:.2f} %")
        c2.metric("RMSE 改善率（各折均值）", f"{ml_result['rmse_improve_mean']:.2f} %")

        if (
            cor_mean["MAE"] < raw_mean["MAE"]
            and cor_mean["RMSE"] < raw_mean["RMSE"]
        ):
            st.success(
                f"滚动窗口交叉验证（{ml_result['n_splits']} 折）结果显示："
                "订正后各折平均误差低于原始预报，"
                "说明该模型在当前演示数据的不同时间段上均具有一定的误差订正效果；"
                "各折结果的具体波动请见上方标准差。"
            )
        else:
            st.warning(
                "本次机器学习订正未稳定降低各折测试段误差，"
                "说明当前特征和模型设置仍有进一步优化空间。"
            )

        st.info(
            "以上结果基于滚动窗口交叉验证（各折测试段互不重叠、"
            "训练数据全部位于测试数据之前），"
            "仅代表当前演示数据上的实验结果，"
            "不能直接代表模型在其他时间、地点或天气条件下的实际预报能力。"
        )

# ---------------- ⑦ AI 气象预报分析 Agent ----------------
st.markdown("---")
st.header("⑦ AI 气象预报分析 Agent")

st.markdown(
    """
    AI Agent 负责理解用户自然语言问题，并调用 Python 气象分析工具获取真实计算结果，
    再由 DeepSeek 对结果进行解释。

    **分工：**
    - Python 负责：数据读取、质量检查、MAE/RMSE/Bias 计算、随机森林订正等真实计算。
    - DeepSeek 负责：理解问题、选择工具、解释结果（不自己计算数值）。
    """
)

st.markdown(
    """
    **推荐问题：**
    1. 原始预报效果怎么样？
    2. 机器学习订正后改善了多少？
    3. 哪个特征对误差订正最重要？
    4. 为什么订正后 Bias 会发生变化？
    5. 滚动窗口交叉验证各折的订正效果稳定吗？
    6. 请全面分析当前实验结果及其局限性。
    """
)

# 初始化聊天历史
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 显示历史消息
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框
if prompt := st.chat_input("请输入你的气象分析问题..."):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在分析……"):
            agent = WeatherAgent(df)
            reply = agent.chat(prompt)
        st.markdown(reply)

    st.session_state["chat_history"].append({"role": "assistant", "content": reply})

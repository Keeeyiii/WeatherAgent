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
from tools.ml_correction import baseline_comparison, machine_learning_correction
from tools.oi_correction import optimal_interpolation_correction

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
    6. 基线对比实验（常数去偏 / 按小时去偏）
    7. 特征重要性分析
    8. 简化最优插值（OI）实验
    9. DeepSeek AI 气象分析 Agent
    """
)

# 方法说明（可展开）
with st.expander("方法说明"):
    st.markdown(
        """
        **1. 本项目在数值预报业务链中的定位**

        数值预报业务的典型流程：

```
观测资料
   ↓
资料同化（用观测资料订正模式初始场）
   ↓
数值预报模式积分运行
   ↓
原始预报输出
   ↓
统计后处理 / 误差订正   ← 本项目所在环节
   ↓
最终预报产品
```

        **后处理与资料同化的区别**：资料同化作用于预报运行**之前**——将观测资料
        融入模式初始场；统计后处理作用于预报运行**之后**——对模式输出的预报
        结果做误差订正。两者目标一致：都是让预报更接近真实大气状态。

        本项目是从**统计学习角度**切入这一目标的一次实践，只完成后处理环节的
        实验性实现，不涉及资料同化，也不涉及数值预报模式本身的实现。

        **2. 预报误差定义**
        `error = forecast - observed`（预报值减观测值）
        - error > 0：预报偏高（高估）；error < 0：预报偏低（低估）

        **3. 预报评价指标**
        - MAE = mean(|error|)
        - RMSE = sqrt(mean(error²))
        - Bias = mean(error)

        **4. 机器学习误差订正**
        使用 Random Forest 学习「订正量」`correction = observed - forecast`：
        `predicted_correction = f(forecast, humidity, pressure, wind_speed, hour, month)`
        然后 `corrected_forecast = forecast + predicted_correction`

        **5. 验证方式：滚动窗口交叉验证**
        训练窗口逐折扩大、测试段依次后移且互不重叠（共 3 折，每折测试约 20% 数据）；
        每一折的训练数据全部位于测试数据之前，无数据泄漏；
        最终汇总各折指标为「均值 ± 标准差」，反映订正效果的稳定性。

        **6. 基线对比实验**
        在随机森林之外，使用**完全相同的滚动窗口划分**对比两种简单基线：
        - 基线A（常数去偏）：`corrected = forecast - 训练集平均偏差`
        - 基线B（按小时去偏）：按 24 个小时分别用训练集各小时的平均偏差订正
        用于判断随机森林的收益中，有多少来自系统性偏差和日变化偏差的消除。

        **7. 实验局限性**
        当前使用的是**模拟生成的示例数据**，日变化信号较强，因此 `hour` 特征重要性偏高；
        该结果不代表在真实观测数据上的泛化能力，后续需接入真实气象站数据验证。
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

        # ---------------- ⑥ 基线对比实验 ----------------
        st.markdown("---")
        st.header("⑥ 基线对比实验")
        st.markdown(
            """
            在随机森林之外，引入两种**简单基线方法**，使用与随机森林
            **完全相同的滚动窗口划分和有效样本**，在同一测试集上对比：

            - **基线A（常数去偏）**：用训练集整体平均偏差做固定订正
            - **基线B（按小时去偏）**：按 24 个小时分别用训练集各小时的平均偏差订正

            若随机森林相对基线B 改善有限，说明订正收益主要来自
            系统性偏差和日变化偏差的消除，而非复杂的非线性关系。
            """
        )

        with st.spinner("正在进行基线对比实验（与随机森林相同的划分）……"):
            baseline_result = baseline_comparison(df)

        # 存入 session_state，供 ⑧ OI 实验板块并列对比使用
        st.session_state["baseline_result"] = baseline_result

        # 四种方法在同一测试集上的对比表（各折均值）
        methods_summary = [
            ("原始预报", baseline_result["raw_mean"]),
            ("基线A（常数去偏）", baseline_result["baseline_a_mean"]),
            ("基线B（按小时去偏）", baseline_result["baseline_b_mean"]),
            ("Random Forest 订正", ml_result["corrected_mean"]),
        ]
        compare_df = pd.DataFrame(
            [
                {
                    "方法": name,
                    "MAE (°C)": round(m["MAE"], 3),
                    "RMSE (°C)": round(m["RMSE"], 3),
                    "Bias (°C)": round(m["Bias"], 3),
                }
                for name, m in methods_summary
            ]
        ).set_index("方法")
        st.dataframe(compare_df)
        st.caption(
            "各数值为滚动窗口交叉验证各折测试段的均值；"
            "四种方法使用完全相同的训练/测试划分，可直接对比。"
        )

        # 分组柱状图：四种方法的 MAE 与 RMSE 对比
        method_names = [name for name, _ in methods_summary]
        fig_base = go.Figure()
        fig_base.add_trace(
            go.Bar(
                x=method_names,
                y=[m["MAE"] for _, m in methods_summary],
                name="MAE",
            )
        )
        fig_base.add_trace(
            go.Bar(
                x=method_names,
                y=[m["RMSE"] for _, m in methods_summary],
                name="RMSE",
            )
        )
        fig_base.update_layout(
            title="四种方法误差对比（各折均值）",
            xaxis_title="方法",
            yaxis_title="误差 (°C)",
            barmode="group",
            legend_title="指标",
            height=400,
        )
        st.plotly_chart(fig_base, width="stretch")

        # 结果讨论（依据真实计算结果自动生成）
        rf_mae = ml_result["corrected_mean"]["MAE"]
        base_a_mae = baseline_result["baseline_a_mean"]["MAE"]
        base_b_mae = baseline_result["baseline_b_mean"]["MAE"]
        improve_vs_b = (base_b_mae - rf_mae) / base_b_mae * 100

        # 相对基线B 的 MAE 改善幅度小于 5% 视为「改善有限」
        if improve_vs_b < 5.0:
            if improve_vs_b <= 0:
                rf_vs_b_text = (
                    f"Random Forest 并未超过基线B"
                    f"（MAE 反而高出 {-improve_vs_b:.1f}%）"
                )
            else:
                rf_vs_b_text = (
                    f"相对基线B 仅进一步降低 {improve_vs_b:.1f}%，改善有限"
                )
            st.info(
                f"**结果讨论**：Random Forest 的各折平均 MAE 为 {rf_mae:.3f} °C，"
                f"基线B（按小时去偏）为 {base_b_mae:.3f} °C，{rf_vs_b_text}。\n\n"
                f"结合上表：基线A（常数去偏）已将 Bias 从 "
                f"{baseline_result['raw_mean']['Bias']:+.3f} °C 大幅降至 "
                f"{baseline_result['baseline_a_mean']['Bias']:+.3f} °C；"
                f"基线B 在此基础上把 MAE 从 {base_a_mae:.3f} °C "
                f"进一步降到 {base_b_mae:.3f} °C。\n\n"
                "这说明当前示例数据上的订正收益主要来自**系统性偏差**和"
                "**日变化偏差**的消除，而非随机森林捕捉到的复杂非线性关系——"
                "对以这两类偏差为主的数据，简单的去偏方法已能取得大部分收益，"
                "效果甚至与随机森林相当。"
            )
        else:
            st.info(
                f"**结果讨论**：Random Forest 的各折平均 MAE 为 {rf_mae:.3f} °C，"
                f"基线B（按小时去偏）为 {base_b_mae:.3f} °C，"
                f"相对基线B 进一步降低 {improve_vs_b:.1f}%。\n\n"
                "这说明随机森林在系统性偏差和日变化偏差之外，"
                "还捕捉到了更复杂的非线性关系，带来了额外收益。"
            )

    st.markdown("---")

    # ---------------- ⑦ 可视化分析 ----------------
    st.header("⑦ 可视化分析")

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

    st.markdown("---")

    # ---------------- ⑧ 简化最优插值（OI）实验 ----------------
    st.header("⑧ 简化最优插值（OI）实验")
    st.markdown(
        """
        **最优插值（Optimal Interpolation, OI）是资料同化中最经典的分析方法**：
        给定背景场（预报值）与观测，按两者误差方差之比分配权重，
        使分析值（订正结果）的误差方差最小。

        本模块是其**单变量、标量权重**的简化演示：

        - 背景场误差方差 `sigma_b²`：训练集上 `forecast - observed` 的方差
        - 观测误差方差 `sigma_o²`：固定为 0.3，代表观测自身的不确定性
        - OI 权重 `w = sigma_b² / (sigma_b² + sigma_o²)`
        - 虚拟观测新息 = 训练集上 `observed − forecast` 的均值
          （等于平均偏差 Bias 取反；新息为负表示预报系统性偏高）
        - 分析值 `analysis = forecast + w × 新息`
          （以训练集的平均系统偏差作为「虚拟观测新息」做订正）

        后处理与资料同化在方法论上**同根同源**——都是用历史统计信息对预报
        做最优加权订正；区别在于真实资料同化（3D-Var、EnKF 等）处理的是
        **高维空间场**的误差协方差结构，比这里的标量演示复杂得多。

        验证方式与随机森林、基线实验完全相同（同一滚动窗口划分、同一测试集）。
        """
    )

    ml_result = st.session_state.get("ml_result")
    baseline_result = st.session_state.get("baseline_result")
    if ml_result is None or baseline_result is None:
        st.info("请先完成机器学习订正。")
    else:
        with st.spinner("正在进行简化最优插值（OI）实验……"):
            oi_result = optimal_interpolation_correction(df)

        st.session_state["oi_result"] = oi_result

        st.success(
            f"验证完成：共 {oi_result['n_splits']} 折，"
            f"每折测试 {oi_result['test_len_per_fold']} 条"
            f"（观测误差方差 sigma_o² = {oi_result['sigma_o_squared']}）。"
        )

        # 每一折的 OI 权重与检验结果
        st.subheader(f"每一折的 OI 权重与检验结果（共 {oi_result['n_splits']} 折）")
        oi_folds_df = pd.DataFrame(
            [
                {
                    "折": f["fold"],
                    "训练样本": f["train_size"],
                    "测试样本": f["test_size"],
                    "sigma_b²": round(f["sigma_b_squared"], 3),
                    "权重 w": round(f["weight"], 3),
                    "新息 (观测−预报)": round(f["innovation"], 3),
                    "原始 MAE": round(f["raw"]["MAE"], 3),
                    "OI MAE": round(f["oi"]["MAE"], 3),
                    "原始 Bias": round(f["raw"]["Bias"], 3),
                    "OI Bias": round(f["oi"]["Bias"], 3),
                }
                for f in oi_result["folds"]
            ]
        )
        st.dataframe(oi_folds_df.set_index("折"))

        # 五种方法在同一测试集上的对比表（各折均值）
        st.subheader("与 Random Forest、基线方法并列对比（各折均值，同一测试集）")
        methods_summary = [
            ("原始预报", baseline_result["raw_mean"]),
            ("基线A（常数去偏）", baseline_result["baseline_a_mean"]),
            ("基线B（按小时去偏）", baseline_result["baseline_b_mean"]),
            ("Random Forest 订正", ml_result["corrected_mean"]),
            ("简化 OI 订正", oi_result["oi_mean"]),
        ]
        compare_df = pd.DataFrame(
            [
                {
                    "方法": name,
                    "MAE (°C)": round(m["MAE"], 3),
                    "RMSE (°C)": round(m["RMSE"], 3),
                    "Bias (°C)": round(m["Bias"], 3),
                }
                for name, m in methods_summary
            ]
        ).set_index("方法")
        st.dataframe(compare_df)
        st.caption(
            "各数值为滚动窗口交叉验证各折测试段的均值；"
            "五种方法使用完全相同的训练/测试划分，可直接对比。"
        )

        # 分组柱状图：五种方法的 MAE 与 RMSE 对比
        method_names = [name for name, _ in methods_summary]
        fig_oi = go.Figure()
        fig_oi.add_trace(
            go.Bar(
                x=method_names,
                y=[m["MAE"] for _, m in methods_summary],
                name="MAE",
            )
        )
        fig_oi.add_trace(
            go.Bar(
                x=method_names,
                y=[m["RMSE"] for _, m in methods_summary],
                name="RMSE",
            )
        )
        fig_oi.update_layout(
            title="五种方法误差对比（各折均值）",
            xaxis_title="方法",
            yaxis_title="误差 (°C)",
            barmode="group",
            legend_title="指标",
            height=400,
        )
        st.plotly_chart(fig_oi, width="stretch")

        # 结果讨论（依据真实计算结果自动生成）
        oi_mae = oi_result["oi_mean"]["MAE"]
        raw_mae = baseline_result["raw_mean"]["MAE"]
        base_a_mae = baseline_result["baseline_a_mean"]["MAE"]
        base_b_mae = baseline_result["baseline_b_mean"]["MAE"]
        rf_mae = ml_result["corrected_mean"]["MAE"]
        w_mean = sum(f["weight"] for f in oi_result["folds"]) / len(
            oi_result["folds"]
        )

        st.info(
            f"**结果讨论**：简化 OI 的各折平均 MAE 为 {oi_mae:.3f} °C"
            f"（原始预报 {raw_mae:.3f}、基线A {base_a_mae:.3f}、"
            f"基线B {base_b_mae:.3f}、Random Forest {rf_mae:.3f}）。\n\n"
            f"简化 OI 只用**单一标量权重**（各折平均 w ≈ {w_mean:.2f}）"
            f"订正整体系统偏差，等价于按方差比「打折扣」的常数去偏："
            f"它通常与基线A 效果相当，但无法刻画日变化等结构化偏差，"
            f"因此不及能利用小时信息的基线B 与随机森林。\n\n"
            f"**方法论意义**：本实验展示了统计后处理与资料同化的共同根源——"
            f"都是基于误差统计对预报做最优加权订正；"
            f"真实的资料同化（3D-Var、EnKF 等）把这一思想推广到高维空间场，"
            f"用误差协方差矩阵描述背景场与观测的不确定性，"
            f"远比此处的标量演示复杂。"
        )

# ---------------- ⑨ AI 气象预报分析 Agent ----------------
st.markdown("---")
st.header("⑨ AI 气象预报分析 Agent")

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

"""订正工具页：把方法用在**用户自己的数据**上，并导出订正结果。

这一页与前面几页的区别：
前面几页是本项目对南京这一个数据集做的研究；
这一页是一个可以换数据、可以拿走结果的工具。
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools.pipeline import (
    OPTIONAL_COLUMNS,
    available_methods,
    diagnose,
    evaluate,
    fit_best,
    prepare,
    quality_report,
    template_csv,
)
from views.common import (
    C_BLUE,
    C_GREEN,
    C_GREY,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    caption,
    card,
    hero,
    signed,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_csv(uploaded) -> pd.DataFrame:
    """中文 CSV 常见 GBK 编码，依次尝试。"""
    raw = uploaded.getvalue()
    for encoding in ("utf-8-sig", "gbk", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"无法解析这个 CSV：{exc}") from exc
    raise ValueError("无法识别文件编码，请另存为 UTF-8 或 GBK 的 CSV。")


@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA, "sample_weather_real.csv"), encoding="utf-8-sig"
    )[["time", "forecast_temperature", "observed_temperature", "humidity", "pressure", "wind_speed"]]


def _bias_chart(clean: pd.DataFrame) -> go.Figure:
    by_month = clean.groupby("month")["error"].mean()
    by_hour = clean.groupby("bjt")["error"].mean()
    figure = go.Figure()
    figure.add_bar(
        x=by_month.index,
        y=by_month.values,
        name="逐月偏差",
        marker_color=[C_RED if v > 0 else C_BLUE for v in by_month.values],
    )
    figure.add_scatter(
        x=by_hour.index,
        y=by_hour.values,
        name="逐时偏差（北京时）",
        mode="lines+markers",
        marker_color=C_ORANGE,
        line=dict(width=2),
        xaxis="x2",
    )
    figure.update_layout(
        title="你的数据：偏差随季节与时刻的变化",
        xaxis=dict(domain=[0, 0.55], dtick=1, title="月份"),
        xaxis2=dict(domain=[0.62, 1.0], dtick=3, title="北京时"),
        yaxis=dict(title="偏差 (℃)"),
    )
    return base_layout(figure, 360)


def render() -> None:
    hero(
        "订正工具：把方法用在你的数据上",
        "前面几页是本项目对南京这一个数据集做的研究；"
        "这一页是一个可以换数据、可以拿走结果的工具。"
        "上传你自己的“预报—观测”数据，它会自动质检、诊断误差结构、"
        "比较几种订正方法在你数据上的效果，并把订正后的结果导出成 CSV。",
    )

    # ------------------------------------------------------------ 数据来源
    st.subheader("第一步：给数据")
    columns = st.columns([1.25, 1.0], gap="large")
    with columns[0]:
        uploaded = st.file_uploader(
            "上传 CSV",
            type=["csv"],
            help="必须包含 time、forecast_temperature、observed_temperature 三列；"
            "humidity、pressure、wind_speed 可选（有的话随机森林会更准）。",
        )
    with columns[1]:
        st.markdown("**必需列**　`time`、`forecast_temperature`、`observed_temperature`")
        st.markdown("**可选列**　" + "、".join(f"`{c}`" for c in OPTIONAL_COLUMNS))
        st.download_button(
            "⬇ 下载模板 CSV",
            data=template_csv().encode("utf-8-sig"),
            file_name="weather_template.csv",
            mime="text/csv",
        )
        st.caption("时间可以是任意可解析的格式，逐小时为佳；温度按摄氏度填写。")

    if uploaded is not None:
        try:
            raw = _read_csv(uploaded)
            source_label = f"已上传文件：{uploaded.name}"
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
    else:
        raw = sample_data()
        source_label = "正在使用内置示例数据（南京禄口 ZSNJ + GFS 预报）"
    st.info(source_label + f"　共 {len(raw):,} 行。")

    # ------------------------------------------------------------ 质检
    try:
        clean, features = prepare(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据处理失败：{exc}")
        return

    st.subheader("第二步：数据质检")
    report = quality_report(raw, clean)
    st.dataframe(report, width="stretch", hide_index=True)
    warnings = report[report["判断"].str.contains("异常|请确认|疑似", na=False)]
    if len(warnings):
        st.warning(
            "请先确认上面标出的问题（尤其是单位）。"
            "如果温度列不是摄氏度，后面的订正结果没有意义。"
        )

    # ------------------------------------------------------------ 诊断
    st.subheader("第三步：你的数据是什么误差结构")
    info = diagnose(clean)
    columns = st.columns(4)
    for column, (number, label) in zip(
        columns,
        [
            (f"{info['bias']:+.2f} ℃", "平均偏差 bias"),
            (f"{info['MAE']:.2f} ℃", "MAE"),
            (f"{info['RMSE']:.2f} ℃", "RMSE"),
            (f"{info['corr']:.3f}", "预报与观测相关系数"),
        ],
    ):
        column.markdown(card(number, label), unsafe_allow_html=True)
    st.write("")

    st.plotly_chart(_bias_chart(clean), width="stretch")

    damping_ratio = info["range_ratio"]
    damping_slope = info["anomaly_slope"]
    has_damping = (damping_ratio < 0.95) and (damping_slope < -0.05)
    if has_damping:
        st.success(
            f"**这份数据也存在“振幅阻尼”。** 预报日较差只有观测的 "
            f"{damping_ratio:.2f} 倍（<1 表示起伏被压小），"
            f"且误差与观测距平的回归斜率为 {damping_slope:+.3f}（负值表示"
            f"“该冷时偏暖、该热时偏冷”）。\n\n"
            "这意味着**简单的平均去偏不会有效**——因为偏差会随季节、时刻和天气状态改变符号。"
            "下面的订正对比可以直接验证这一点。"
        )
    else:
        st.info(
            f"**这份数据没有明显的振幅阻尼特征**（日较差比值 {damping_ratio:.2f}，"
            f"距平回归斜率 {damping_slope:+.3f}）。"
            "这类数据里，误差可能更接近整体偏移或随机噪声，简单方法的性价比会更高。"
        )

    # ------------------------------------------------------------ 对比
    st.subheader("第四步：在你的数据上比较订正方法")
    st.caption(
        "每个方法都在**完全相同的时序划分**上评估：训练段永远早于测试段，"
        "不会出现“用未来的数据预测过去”的作弊。"
    )
    try:
        with st.spinner("正在评估…"):
            results, meta = evaluate(clean, features)
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法完成评估：{exc}")
        return

    summary = meta["summary"]
    best = meta["best"]
    best_baseline = meta["best_baseline"]
    raw_mae = float(summary.iloc[0]["MAE"])

    figure = go.Figure()
    colors = [C_GREY if i == 0 else (C_ORANGE if name == best["method"] else C_BLUE)
              for i, name in enumerate(summary["method"])]
    figure.add_bar(
        x=summary["method"],
        y=summary["MAE"],
        marker_color=colors,
        text=[f"{v:.3f}" for v in summary["MAE"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        cliponaxis=False,
        error_y=dict(type="data", array=summary["MAE_std"].fillna(0), color="#99a3ad"),
        name="MAE",
    )
    figure.update_layout(title="各方法在你数据上的 MAE（越低越好，橙色为最佳）")
    figure.update_yaxes(title="MAE (℃)", range=[0, max(raw_mae * 1.15, summary["MAE"].max() * 1.15)])
    figure.update_xaxes(tickangle=-12)
    st.plotly_chart(base_layout(figure, 400), width="stretch")

    table = summary.copy()
    table.columns = ["方法", "MAE", "MAE 折间标准差", "RMSE", "bias", "相对不订正改善 %"]
    st.dataframe(table.round(3), width="stretch", hide_index=True)

    if best["method"].startswith("①"):
        st.warning(
            f"**结论：在你的数据上，订正没有带来改善。** "
            f"最佳方法仍是“不订正”（MAE {raw_mae:.3f} ℃）。"
            "常见原因是样本太短、误差以随机噪声为主，或训练段与测试段的偏差结构不同。"
            "这本身就是一个有用的结论——**说明这份预报不需要、也不适合做统计订正。**"
        )
    else:
        gain_over_baseline = best_baseline["MAE"] - best["MAE"]
        st.success(
            f"**结论：在你的数据上，{best['method']} 效果最好。**\n\n"
            f"MAE 从 {raw_mae:.3f} ℃ 降到 **{best['MAE']:.3f} ℃**"
            f"（改善 **{best['MAE_improvement_pct']:.1f}%**）。\n\n"
            + (
                f"它比最强的简单基线（{best_baseline['method']}，MAE "
                f"{best_baseline['MAE']:.3f} ℃）还低 {gain_over_baseline:.3f} ℃——"
                "说明机器学习确实抓到了查表法拿不到的东西。"
                if not best["method"].startswith(("②", "③", "④", "⑤"))
                else "在这个数据集上，**简单查表法已经够用**，"
                "复杂的非线性模型没有额外优势——这同样是值得知道的结论。"
            )
        )

    with st.expander("逐折明细（检查结论是否只由某一折撑着）"):
        pivot = results.pivot_table(index="method", columns="fold", values="MAE").round(3)
        pivot.columns = [f"第 {int(c)} 折" for c in pivot.columns]
        st.dataframe(pivot, width="stretch")

    # ------------------------------------------------------------ 输出
    st.subheader("第五步：拿走结果")
    method_choice = st.selectbox(
        "用哪个方法做最终订正？",
        list(summary["method"]),
        index=int(summary["MAE"].idxmin()),
        help="默认选中在时序检验里表现最好的那个。",
    )
    correct = fit_best(clean, features, method_choice)

    tab_history, tab_future = st.tabs(["① 订正这份数据（有观测）", "② 订正未来的预报（无观测）"])

    with tab_history:
        st.caption(
            "把选定的方法用全部数据重新训练，再逐行输出订正值。"
            "注意：这里的指标是有偏乐观的（模型见过这些样本），"
            "**诚实的精度以上面的时序检验为准。**"
        )
        corrected = clean.copy()
        corrected["corrected_temperature"] = correct(clean)
        corrected["correction"] = (
            corrected["corrected_temperature"] - corrected["forecast_temperature"]
        )
        preview = corrected[
            [
                "time",
                "observed_temperature",
                "forecast_temperature",
                "corrected_temperature",
                "correction",
            ]
        ].head(50)
        st.dataframe(preview.round(2), width="stretch", hide_index=True)
        st.download_button(
            "⬇ 下载订正后的数据（CSV）",
            data=corrected.to_csv(index=False).encode("utf-8-sig"),
            file_name="corrected_forecast.csv",
            mime="text/csv",
        )

    with tab_future:
        st.caption(
            "这才是实际用法：拿一份**只有预报、没有观测**的新数据（比如明天的预报），"
            "用上面训练好的订正关系处理它。"
        )
        future_file = st.file_uploader(
            "上传待订正的预报数据（需要 time 与 forecast_temperature"
            + ("，以及" + "、".join(c for c in features if c not in ('hour', 'month')) if len(features) > 3 else "")
            + "）",
            type=["csv"],
            key="future",
        )
        if future_file is not None:
            try:
                future = _read_csv(future_file)
                if "time" not in future.columns or "forecast_temperature" not in future.columns:
                    st.error("这份数据缺少 time 或 forecast_temperature 列。")
                else:
                    future = future.copy()
                    future["time"] = pd.to_datetime(future["time"], errors="coerce")
                    future["hour"] = future["time"].dt.hour
                    future["month"] = future["time"].dt.month
                    future["bjt"] = (future["hour"] + 8) % 24
                    future["forecast_temperature"] = pd.to_numeric(
                        future["forecast_temperature"], errors="coerce"
                    )
                    usable = future.dropna(subset=["time", "forecast_temperature"])
                    missing_features = [c for c in features if c not in usable.columns]
                    if missing_features:
                        st.error(
                            "这份数据缺少训练时用到的特征列："
                            + "、".join(missing_features)
                            + "。请补上这些列，或在上面改选一个只用温度与时间的简单方法（如按(月×小时)去偏）。"
                        )
                    else:
                        output = usable.copy()
                        output["corrected_temperature"] = correct(usable)
                        output["correction"] = (
                            output["corrected_temperature"] - output["forecast_temperature"]
                        )
                        st.dataframe(
                            output[["time", "forecast_temperature", "corrected_temperature", "correction"]]
                            .head(50)
                            .round(2),
                            width="stretch",
                            hide_index=True,
                        )
                        st.download_button(
                            "⬇ 下载订正后的预报（CSV）",
                            data=output.to_csv(index=False).encode("utf-8-sig"),
                            file_name="corrected_new_forecast.csv",
                            mime="text/csv",
                        )
            except Exception as exc:  # noqa: BLE001
                st.error(f"处理失败：{exc}")

    # ------------------------------------------------------------ 免责说明
    st.divider()
    st.subheader("适用条件与边界")
    st.markdown(
        """
        这个工具能做什么、不能做什么，写清楚比较好：

        | 项目 | 说明 |
        |---|---|
        | 需要多少数据 | 建议至少一年逐小时数据（约 8,760 行）。样本越短，订正关系越不可靠 |
        | 数据要求 | 必须同时有**同一时刻**的预报值和观测值，用于学习偏差 |
        | 订正原理 | 纯粹统计——它学习的是"历史上模式在什么情况下容易偏"，不改变模式的物理过程 |
        | 不能做的 | 无法修正单次极端天气预报（那是模式的物理问题）；不能跨站点套用；预测范围外推不可靠 |
        | 精度如何相信 | 以上面**严格时序检验**的指标为准，不要用"全部数据重新训练"的指标 |
        """
    )
    caption(
        "本工具做的是**统计后处理**：在预报产出之后修正它的系统性偏差。"
        "它不涉及资料同化（那是在模式运行之前修正初始场），两者目标相同但环节不同。"
    )

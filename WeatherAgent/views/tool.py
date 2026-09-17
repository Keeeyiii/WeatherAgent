"""误差分析页：换一份数据，跑同一套分析。

这一页与后面几页的关系：
后面几页是用**南京这一份示例数据**做出来的完整分析结果；
这一页把同一条流水线开放出来，换成任何一份"预报—观测"数据都能跑，
并且把订正后的结果导出。
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
    damping_evidence,
    diagnose,
    evaluate,
    fit_best,
    prepare,
    quality_report,
    template_csv,
)
from views.common import (
    C_BLUE,
    C_GREY,
    C_NAVY,
    C_ORANGE,
    C_RED,
    base_layout,
    caption,
    card,
    hero,
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
    )[
        [
            "time",
            "forecast_temperature",
            "observed_temperature",
            "humidity",
            "pressure",
            "wind_speed",
        ]
    ]


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
        title="误差随时间怎么变",
        xaxis=dict(domain=[0, 0.55], dtick=1, title="月份"),
        xaxis2=dict(domain=[0.62, 1.0], dtick=3, title="北京时"),
        yaxis=dict(title="偏差 (℃)"),
    )
    return base_layout(figure, 340)


def _damping_chart(clean: pd.DataFrame) -> go.Figure:
    daily = (
        clean.groupby(clean["time"].dt.normalize())
        .apply(
            lambda g: pd.Series(
                {
                    "obs": g["observed_temperature"].max() - g["observed_temperature"].min(),
                    "fcst": g["forecast_temperature"].max() - g["forecast_temperature"].min(),
                }
            ),
            include_groups=False,
        )
        .dropna()
    )
    limit = float(max(daily["obs"].max(), daily["fcst"].max())) * 1.05
    figure = go.Figure()
    figure.add_scatter(
        x=daily["obs"],
        y=daily["fcst"],
        mode="markers",
        marker=dict(size=5, color=C_BLUE, opacity=0.4),
        name="每天一个点",
    )
    figure.add_scatter(
        x=[0, limit],
        y=[0, limit],
        mode="lines",
        line=dict(color=C_GREY, dash="dash", width=1.6),
        name="预报 = 观测",
    )
    figure.update_layout(title="每天的日较差：点落在斜线下方说明起伏被压小")
    figure.update_xaxes(title="观测日较差 (℃)", range=[0, limit])
    figure.update_yaxes(title="预报日较差 (℃)", range=[0, limit])
    return base_layout(figure, 340)


def render() -> None:
    hero(
        "误差分析：换一份数据，跑同一套分析",
        "这一页和后面的页面用的是**同一条分析流水线**。"
        "后面几页跑的是内置的示例数据（南京 2.4 年）；"
        "这一页把入口开放出来——上传你自己的“预报—观测”数据，得到同样的诊断，并导出订正结果。",
    )

    # ------------------------------------------------------------ 选数据
    st.subheader("第一步：选一份数据")
    source = st.radio(
        "数据来源",
        ["用内置示例数据（南京禄口 ZSNJ + GFS，23,543 小时）", "上传我自己的数据"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source.startswith("上传"):
        columns = st.columns([1.3, 1.0], gap="large")
        with columns[0]:
            uploaded = st.file_uploader(
                "上传 CSV",
                type=["csv"],
                help="必须包含 time、forecast_temperature、observed_temperature；"
                "humidity、pressure、wind_speed 可选。",
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
        if uploaded is None:
            st.info("请上传一个 CSV 文件；文件格式见右侧模板。想看效果的话，可以先切回上面的示例数据。")
            return
        try:
            raw = _read_csv(uploaded)
            label = f"已上传：{uploaded.name}"
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
    else:
        raw = sample_data()
        label = "内置示例数据：南京禄口 ZSNJ 观测 + GFS 预报归档"

    st.info(f"{label}　共 {len(raw):,} 行。")

    try:
        clean, features = prepare(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据处理失败：{exc}")
        return

    # ------------------------------------------------------------ 质检
    st.subheader("第二步：数据质检")
    report = quality_report(raw, clean)
    st.dataframe(report, width="stretch", hide_index=True)
    if len(report[report["判断"].str.contains("异常|请确认|疑似", na=False)]):
        st.warning("请先确认上面标出的问题（尤其是单位）。温度不是摄氏度的话，后面的分析没有意义。")

    # ------------------------------------------------------------ 画像
    st.subheader("第三步：误差长什么样")
    info = diagnose(clean)
    columns = st.columns(4)
    for column, (number, text) in zip(
        columns,
        [
            (f"{info['bias']:+.2f} ℃", "平均偏差 bias"),
            (f"{info['MAE']:.2f} ℃", "平均绝对误差 MAE"),
            (f"{info['RMSE']:.2f} ℃", "均方根误差 RMSE"),
            (f"{info['corr']:.3f}", "预报与观测相关系数"),
        ],
    ):
        column.markdown(card(number, text), unsafe_allow_html=True)
    st.write("")
    st.plotly_chart(_bias_chart(clean), width="stretch")
    caption(
        "左图看误差随**季节**怎么变，右图看误差随**一天中的时刻**怎么变。"
        "如果这两张图都不是平的，就说明误差不是一个固定的数。"
    )

    # ------------------------------------------------------------ 振幅阻尼
    st.subheader("第四步：误差随“起伏幅度”怎么变")
    st.markdown(
        "这是本项目在示例数据上的核心发现，现在用**你的数据**再检验一遍："
        "如果模式对温度变化的响应不足（振幅阻尼），应该同时看到下面四件事。"
    )
    evidence = damping_evidence(clean)
    st.dataframe(evidence, width="stretch", hide_index=True)

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.plotly_chart(_damping_chart(clean), width="stretch")
    with right:
        fired = int((evidence["判断"] == "符合").sum())
        if fired >= 3:
            st.success(
                f"**你的数据也有振幅阻尼（{fired}/4 项符合）。**\n\n"
                "预报把温度的起伏压小了：该冷的时候不够冷、该热的时候不够热。"
                "这意味着**简单地减掉一个平均偏差不会有效**——"
                "因为偏差会随季节、时刻和天气状态改变符号。"
            )
        elif fired >= 1:
            st.warning(
                f"**部分符合（{fired}/4 项）。** 你的数据里能看到一些振幅压缩的迹象，"
                "但不完整。这通常意味着数据时段较短，或者误差里有别的成分占主导。"
            )
        else:
            st.info(
                "**四项都不符合。** 你的数据里没有明显的振幅阻尼特征——"
                "误差更接近整体偏移或随机噪声。这种情况下，简单方法的性价比会更高。"
            )
        st.caption(
            "判断标准写在表格第二列：先写下“如果存在振幅阻尼应该看到什么”，"
            "再看数据是否支持。这是本项目检验一切结论的方式。"
        )

    # ------------------------------------------------------------ 订正
    st.subheader("第五步：用什么方法订正最好")
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
    best, best_baseline = meta["best"], meta["best_baseline"]
    raw_mae = float(summary.iloc[0]["MAE"])

    figure = go.Figure()
    figure.add_bar(
        x=summary["method"],
        y=summary["MAE"],
        marker_color=[
            C_GREY if i == 0 else (C_ORANGE if name == best["method"] else C_BLUE)
            for i, name in enumerate(summary["method"])
        ],
        text=[f"{v:.3f}" for v in summary["MAE"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        cliponaxis=False,
        error_y=dict(type="data", array=summary["MAE_std"].fillna(0), color="#99a3ad"),
        name="MAE",
    )
    figure.update_layout(title="各方法的 MAE（越低越好，橙色为最佳）")
    figure.update_yaxes(
        title="MAE (℃)", range=[0, max(raw_mae, summary["MAE"].max()) * 1.15]
    )
    figure.update_xaxes(tickangle=-12)
    st.plotly_chart(base_layout(figure, 380), width="stretch")

    table = summary.copy()
    table.columns = ["方法", "MAE", "MAE 折间标准差", "RMSE", "bias", "相对不订正改善 %"]
    st.dataframe(table.round(3), width="stretch", hide_index=True)

    if best["method"].startswith("①"):
        st.warning(
            f"**结论：在你的数据上，订正没有带来改善。** 最佳方法仍是“不订正”"
            f"（MAE {raw_mae:.3f} ℃）。常见原因是样本太短、误差以随机噪声为主，"
            "或训练段与测试段的偏差结构不同。"
            "这本身就是有用的结论——**说明这份预报不需要、也不适合做统计订正。**"
        )
    else:
        gain = best_baseline["MAE"] - best["MAE"]
        st.success(
            f"**结论：{best['method']} 效果最好。** MAE 从 {raw_mae:.3f} ℃ 降到 "
            f"**{best['MAE']:.3f} ℃**（改善 **{best['MAE_improvement_pct']:.1f}%**）。"
            + (
                f"它比最强的简单基线（{best_baseline['method']}，MAE {best_baseline['MAE']:.3f} ℃）"
                f"还低 {gain:.3f} ℃，说明机器学习确实抓到了查表法拿不到的东西。"
                if not best["method"].startswith(("②", "③", "④", "⑤"))
                else "在这个数据集上，**简单查表法已经够用**，"
                "复杂的非线性模型没有额外优势——这同样是值得知道的结论。"
            )
        )

    with st.expander("逐折明细（检查结论是否只由某一折撑着）"):
        pivot = results.pivot_table(index="method", columns="fold", values="MAE").round(3)
        pivot.columns = [f"第 {int(c)} 折" for c in pivot.columns]
        st.dataframe(pivot, width="stretch")

    # ------------------------------------------------------------ 导出
    st.subheader("第六步：拿走订正结果")
    method_choice = st.selectbox(
        "用哪个方法做最终订正？",
        list(summary["method"]),
        index=int(summary["MAE"].idxmin()),
    )
    correct = fit_best(clean, features, method_choice)

    tab_history, tab_future = st.tabs(["① 订正这份数据", "② 订正未来的预报（只有预报值）"])
    with tab_history:
        st.caption(
            "用全部数据重新训练后逐行输出订正值。注意这里的精度是**偏乐观**的"
            "（模型见过这些样本），诚实的精度以第五步的时序检验为准。"
        )
        corrected = clean.copy()
        corrected["corrected_temperature"] = correct(clean)
        corrected["correction"] = (
            corrected["corrected_temperature"] - corrected["forecast_temperature"]
        )
        st.dataframe(
            corrected[
                [
                    "time",
                    "observed_temperature",
                    "forecast_temperature",
                    "corrected_temperature",
                    "correction",
                ]
            ]
            .head(50)
            .round(2),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇ 下载订正后的数据（CSV）",
            data=corrected.to_csv(index=False).encode("utf-8-sig"),
            file_name="corrected_forecast.csv",
            mime="text/csv",
        )

    with tab_future:
        st.caption("实际用法：拿一份只有预报、没有观测的新数据（比如明天的预报），用上面训练好的关系处理它。")
        future_file = st.file_uploader(
            "上传待订正的预报数据", type=["csv"], key="future"
        )
        if future_file is not None:
            try:
                future = _read_csv(future_file)
                if "time" not in future.columns or "forecast_temperature" not in future.columns:
                    st.error("这份数据缺少 time 或 forecast_temperature 列。")
                else:
                    future["time"] = pd.to_datetime(future["time"], errors="coerce")
                    future["hour"] = future["time"].dt.hour
                    future["month"] = future["time"].dt.month
                    future["bjt"] = (future["hour"] + 8) % 24
                    future["forecast_temperature"] = pd.to_numeric(
                        future["forecast_temperature"], errors="coerce"
                    )
                    usable = future.dropna(subset=["time", "forecast_temperature"])
                    missing = [c for c in features if c not in usable.columns]
                    if missing:
                        st.error(
                            "缺少训练时用到的特征列："
                            + "、".join(missing)
                            + "。请补上，或改选一个只用温度与时间的简单方法（如按(月×小时)去偏）。"
                        )
                    else:
                        output = usable.copy()
                        output["corrected_temperature"] = correct(usable)
                        output["correction"] = (
                            output["corrected_temperature"] - output["forecast_temperature"]
                        )
                        st.dataframe(
                            output[
                                [
                                    "time",
                                    "forecast_temperature",
                                    "corrected_temperature",
                                    "correction",
                                ]
                            ]
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

    # ------------------------------------------------------------ 边界
    st.divider()
    st.subheader("适用条件与边界")
    st.markdown(
        """
        | 项目 | 说明 |
        |---|---|
        | 需要多少数据 | 建议至少一年逐小时数据（约 8,760 行）；样本越短，订正关系越不可靠 |
        | 数据要求 | 必须同时有**同一时刻**的预报值和观测值 |
        | 订正原理 | 纯统计——学习"历史上模式在什么条件下容易偏"，不改变模式的物理过程 |
        | 不能做的 | 无法修正单次极端天气预报（那是模式的物理问题）；不能跨站点套用 |
        | 精度怎么信 | 以第五步**严格时序检验**的指标为准，不要用"全部数据重新训练"的指标 |
        """
    )
    caption(
        "本页做的是**统计后处理**：在预报产出之后修正它的系统性偏差。"
        "它不涉及资料同化（那是在模式运行之前修正初始场），两者目标相同但环节不同。"
    )

"""可复用的订正管线：给一份"预报—观测"数据，做质检、诊断、订正并输出结果。

与前面几个模块的区别
--------------------
`tools/verification.py` 和 `tools/da.py` 是**针对本项目那一份数据**写的分析脚本；
本模块则要求对**任意用户的 CSV** 都能跑：列可以缺（湿度/气压/风速是可选的）、
样本可多可少、时间格式不统一，都必须给出诚实的结论而不是直接报错。

因此这里的每个函数都遵守两个原则：
1. 先检查再计算——数据不合格就明确说出来，不硬算；
2. 不掩盖结果——如果随机森林没有跑赢简单基线，就如实报告。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REQUIRED_COLUMNS = ["time", "forecast_temperature", "observed_temperature"]
OPTIONAL_COLUMNS = ["humidity", "pressure", "wind_speed"]

# 训练随机森林时需要同时存在的特征（按重要性顺序尝试）
FEATURE_PRIORITY = [
    "forecast_temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "hour",
    "month",
]


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """把用户数据整理成内部格式，并返回 (数据, 可用特征列表)。"""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(
            "缺少必需的列：" + "、".join(missing)
            + "。至少需要 time、forecast_temperature、observed_temperature 三列。"
        )

    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    for column in ["forecast_temperature", "observed_temperature", *OPTIONAL_COLUMNS]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(subset=REQUIRED_COLUMNS)
    work = work.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    work = work.reset_index(drop=True)
    if work.empty:
        raise ValueError("清洗后没有可用的样本，请检查时间列和温度列是否为数值。")

    work["hour"] = work["time"].dt.hour
    work["month"] = work["time"].dt.month
    work["bjt"] = (work["hour"] + 8) % 24
    work["error"] = work["forecast_temperature"] - work["observed_temperature"]
    work["season"] = work["month"].map(
        {
            12: "冬季 DJF", 1: "冬季 DJF", 2: "冬季 DJF",
            3: "春季 MAM", 4: "春季 MAM", 5: "春季 MAM",
            6: "夏季 JJA", 7: "夏季 JJA", 8: "夏季 JJA",
            9: "秋季 SON", 10: "秋季 SON", 11: "秋季 SON",
        }
    )

    features = [c for c in FEATURE_PRIORITY if c in work.columns]
    return work, features


def quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """数据质检报告：逐项给出检查结果与是否通过。"""
    rows = []

    rows.append(("样本量", f"{len(clean):,} 行（原始 {len(raw):,} 行）",
                 "通过" if len(clean) >= 500 else "偏少"))

    span = clean["time"].max() - clean["time"].min()
    rows.append(("时间跨度", f"{clean['time'].min():%Y-%m-%d} 至 {clean['time'].max():%Y-%m-%d}"
                 f"（{span.days} 天）", "通过" if span.days >= 60 else "偏短"))

    gaps = clean["time"].diff().dt.total_seconds().div(3600)
    typical = gaps.mode()
    step = float(typical.iloc[0]) if not typical.empty else float("nan")
    rows.append(("时间间隔", f"最常见 {step:.1f} 小时", "通过"))

    missing_count = int(raw[REQUIRED_COLUMNS].isna().sum().sum()) if set(REQUIRED_COLUMNS) <= set(raw.columns) else -1
    rows.append(("必需列缺失值", f"{missing_count} 个" if missing_count >= 0 else "无法判断",
                 "通过" if missing_count == 0 else "已剔除"))

    duplicate = int(raw.duplicated(subset=["time"]).sum()) if "time" in raw.columns else 0
    rows.append(("重复时间戳", f"{duplicate} 个", "通过" if duplicate == 0 else "已去重"))

    for column, label in (
        ("forecast_temperature", "预报温度"),
        ("observed_temperature", "观测温度"),
    ):
        low, high = clean[column].min(), clean[column].max()
        ok = -70 <= low and high <= 60
        rows.append((f"{label}取值范围", f"{low:.1f} ~ {high:.1f}", "通过" if ok else "异常，请确认单位"))

    median = clean["observed_temperature"].median()
    rows.append(
        (
            "单位判断",
            f"观测温度中位数 {median:.1f}——{'看起来是摄氏度' if -40 < median < 45 else '疑似不是摄氏度'}",
            "通过" if -40 < median < 45 else "请确认是否为 ℃",
        )
    )

    optional_present = [c for c in OPTIONAL_COLUMNS if c in clean.columns]
    rows.append(
        (
            "附加特征",
            "、".join(optional_present) if optional_present else "无（随机森林将只用温度与时间特征）",
            "通过",
        )
    )
    return pd.DataFrame(rows, columns=["检查项", "结果", "判断"])


def diagnose(clean: pd.DataFrame) -> dict:
    """误差画像：总体指标 + 是否也存在振幅阻尼。"""
    error = clean["error"]
    daily = clean.groupby(clean["time"].dt.normalize()).agg(
        obs_max=("observed_temperature", "max"),
        obs_min=("observed_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
    )
    obs_range = (daily["obs_max"] - daily["obs_min"]).mean()
    fcst_range = (daily["fcst_max"] - daily["fcst_min"]).mean()

    climatology = clean.groupby(["month", "bjt"])["observed_temperature"].transform("mean")
    anomaly = clean["observed_temperature"] - climatology
    slope = float(np.polyfit(anomaly, error, 1)[0]) if anomaly.std() > 0 else float("nan")

    return {
        "n": int(len(clean)),
        "bias": float(error.mean()),
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "corr": float(clean["forecast_temperature"].corr(clean["observed_temperature"])),
        "obs_range": float(obs_range),
        "fcst_range": float(fcst_range),
        "range_ratio": float(fcst_range / obs_range) if obs_range else float("nan"),
        "anomaly_slope": slope,
        "anomaly_corr": float(anomaly.corr(error)),
    }


def damping_evidence(clean: pd.DataFrame) -> pd.DataFrame:
    """逐条给出"振幅阻尼"的四项证据，适用于任意数据集。

    这四项与示例数据的分析完全一致，因此上传自己的数据也能得到同样的诊断：
      ① 日变化尺度：预报的日较差是否小于观测
      ② 强度响应：误差是否与观测距平成反比
      ③ 极端时刻：最冷/最暖 1% 的偏差是否被放大
      ④ 落点：压缩是否集中在最低气温那一侧
    """
    info = diagnose(clean)
    error = clean["error"]
    daily = clean.groupby(clean["time"].dt.normalize()).agg(
        obs_max=("observed_temperature", "max"),
        obs_min=("observed_temperature", "min"),
        fcst_max=("forecast_temperature", "max"),
        fcst_min=("forecast_temperature", "min"),
    )
    bias_min = float((daily["fcst_min"] - daily["obs_min"]).mean())
    bias_max = float((daily["fcst_max"] - daily["obs_max"]).mean())

    low = clean["observed_temperature"].quantile(0.01)
    high = clean["observed_temperature"].quantile(0.99)
    cold_error = float(error[clean["observed_temperature"] <= low].mean())
    warm_error = float(error[clean["observed_temperature"] >= high].mean())

    def verdict(ok: bool) -> str:
        return "符合" if ok else "不符合"

    return pd.DataFrame(
        [
            (
                "① 日变化尺度",
                "预报的日较差应小于观测",
                f"{info['obs_range']:.2f} → {info['fcst_range']:.2f} ℃（比值 {info['range_ratio']:.2f}）",
                verdict(info["range_ratio"] < 0.97),
            ),
            (
                "② 强度响应",
                "误差应与观测距平成反比",
                f"斜率 {info['anomaly_slope']:+.3f}（r = {info['anomaly_corr']:+.2f}）",
                verdict(info["anomaly_slope"] < -0.05),
            ),
            (
                "③ 极端时刻",
                "越极端偏差越大，且符号相反",
                f"最冷 1% {cold_error:+.2f} ℃／最暖 1% {warm_error:+.2f} ℃",
                verdict(cold_error > 0 and warm_error < 0),
            ),
            (
                "④ 落在哪一侧",
                "日最低气温的偏差应比日最高气温更偏正（两头向中间收）",
                f"日最低气温偏差 {bias_min:+.2f} ℃／日最高 {bias_max:+.2f} ℃",
                verdict((bias_min - bias_max) > 0.3),
            ),
        ],
        columns=["检验角度", "如果存在振幅阻尼，应该看到", "数据里的结果", "判断"],
    )


# --------------------------------------------------------------------------- #
# 订正方法：先 fit，再 apply
# --------------------------------------------------------------------------- #
def _fit_constant(train: pd.DataFrame):
    value = float(train["error"].mean())
    return lambda frame: np.full(len(frame), value)


def _fit_group(keys: list[str]):
    def fit(train: pd.DataFrame):
        table = train.groupby(keys)["error"].mean()
        fallback = float(train["error"].mean())

        def apply(frame: pd.DataFrame):
            if len(keys) == 1:
                mapped = frame[keys[0]].map(table)
            else:
                index = frame.set_index(keys).index
                mapped = pd.Series(index.map(table), index=frame.index, dtype="float64")
                # 组合缺失时回退到第一个键，再回退到全局平均
                mapped = mapped.fillna(frame[keys[0]].map(train.groupby(keys[0])["error"].mean()))
            return mapped.fillna(fallback).to_numpy()

        return apply

    return fit


def _fit_forest(features: list[str], seed: int = 42, n_estimators: int = 200):
    def fit(train: pd.DataFrame):
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=5
        )
        usable = train.dropna(subset=features)
        model.fit(usable[features], usable["error"])
        return lambda frame: model.predict(frame[features])

    return fit


def available_methods(features: list[str]) -> dict:
    """方法名 → (构造函数, 说明, 需要的特征数)。"""
    methods = {
        "① 不订正": (None, "作为基准", 0),
        "② 常数去偏": (_fit_constant, "只需减去全时段平均偏差", 0),
        "③ 按月去偏": (_fit_group(["month"]), "减去各月平均偏差", 0),
        "④ 按小时去偏": (_fit_group(["hour"]), "减去各小时平均偏差", 0),
        "⑤ 按(月×小时)去偏": (_fit_group(["month", "hour"]), "减去月份与小时组合的平均偏差", 0),
    }
    extra = [c for c in features if c not in ("hour", "month")]
    if extra:
        methods["⑥ 随机森林"] = (
            _fit_forest(features),
            "用" + "、".join(extra) + "等特征构建非线性订正",
            len(features),
        )
    return methods


def _folds(n: int, k: int = 3, min_train: int = 500) -> list[tuple[int, int]]:
    """扩张窗口：返回 (训练结束位置, 测试结束位置)。

    把样本等分成 k+1 段：第 1 段作为第 1 折的训练集，之后每折训练段扩张一段、
    测试段后移一段。这样每折的训练量都不会小得离谱——否则会出现
    "用 300 个冬季样本训练、去订正 7700 个跨季节样本"这种荒谬结果。
    """
    if n < min_train + 100:
        return []
    k = max(1, min(k, n // min_train - 1))
    block = n // (k + 1)
    if block < 100:
        return []
    folds = []
    for index in range(k):
        train_end = block * (index + 1)
        test_end = block * (index + 2) if index < k - 1 else n
        folds.append((train_end, test_end))
    return folds


def evaluate(clean: pd.DataFrame, features: list[str], k: int = 3) -> tuple[pd.DataFrame, dict]:
    """严格时序的三折对比；返回 (逐折结果, 汇总)。"""
    methods = available_methods(features)
    rows = []
    folds = _folds(len(clean), k=k)
    if not folds:
        raise ValueError("样本量不足以做时序划分（建议至少 500 行）。")

    for fold_index, (train_end, test_end) in enumerate(folds, start=1):
        train = clean.iloc[:train_end]
        test = clean.iloc[train_end:test_end]
        raw_mae = float(test["error"].abs().mean())
        raw_rmse = float(np.sqrt((test["error"] ** 2).mean()))

        for name, (factory, _note, _n) in methods.items():
            if factory is None:
                error = test["error"].to_numpy()
            else:
                apply = factory(train)
                error = test["forecast_temperature"].to_numpy() - apply(test) - test[
                    "observed_temperature"
                ].to_numpy()
            rows.append(
                {
                    "fold": fold_index,
                    "train_n": len(train),
                    "test_n": len(test),
                    "method": name,
                    "bias": float(error.mean()),
                    "MAE": float(np.abs(error).mean()),
                    "RMSE": float(np.sqrt((error**2).mean())),
                    "MAE_improvement_pct": 100 * (raw_mae - np.abs(error).mean()) / raw_mae,
                    "RMSE_improvement_pct": 100
                    * (raw_rmse - float(np.sqrt((error**2).mean())))
                    / raw_rmse,
                }
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method")
        .agg(
            MAE=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE=("RMSE", "mean"),
            bias=("bias", "mean"),
            MAE_improvement_pct=("MAE_improvement_pct", "mean"),
        )
        .reset_index()
    )
    order = {name: i for i, name in enumerate(methods)}
    summary["__o"] = summary["method"].map(order)
    summary = summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    best = summary.loc[summary["MAE"].idxmin()]
    # 除了"不订正"以外最好的简单基线
    baselines = summary[summary["method"].str.startswith(("②", "③", "④", "⑤"))]
    best_baseline = baselines.loc[baselines["MAE"].idxmin()] if len(baselines) else best
    return results, {
        "summary": summary,
        "best": best,
        "best_baseline": best_baseline,
        "folds": folds,
    }


def fit_best(clean: pd.DataFrame, features: list[str], method_name: str):
    """用全部数据训练选定方法，返回一个可对新数据做订正的函数。"""
    methods = available_methods(features)
    if method_name not in methods:
        raise ValueError(f"未知的订正方法：{method_name}")
    factory = methods[method_name][0]
    if factory is None:
        return lambda frame: frame["forecast_temperature"].to_numpy()
    apply = factory(clean)
    return lambda frame: frame["forecast_temperature"].to_numpy() - apply(frame)


def template_csv() -> str:
    """给出上传模板的示例内容。"""
    return (
        "time,forecast_temperature,observed_temperature,humidity,pressure,wind_speed\n"
        "2024-01-01 00:00:00,1.2,-1.0,84,1030.9,2.4\n"
        "2024-01-01 01:00:00,2.6,2.0,76,1031.0,3.4\n"
    )

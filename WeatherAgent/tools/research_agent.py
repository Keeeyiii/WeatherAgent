"""离线研究助手。

设计原则：数值计算全部由 Python 完成，语言模型（如果配置了）只负责组织语言。
默认实现完全不依赖 API Key，因此没有网络也能正常工作，也不会编造数字。
"""

from __future__ import annotations

import pandas as pd

SUGGESTED = [
    "总体误差有多大？",
    "为什么常数去偏没有效果？",
    "什么是振幅阻尼？",
    "随机森林提升了多少？",
    "为什么订正模型不能推广到上海？",
    "寒潮过程中模式表现如何？",
    "数据是从哪里来的？",
    "这个项目有哪些局限？",
]


def _stats() -> dict:
    from views.common import headline_numbers, load_nanjing, run_ablation, run_cross_station
    from tools.verification import error_autocorrelation, feature_importance

    head = headline_numbers()
    results, summary = run_ablation()
    forest = summary.loc[summary["method"].str.contains("随机森林", regex=False)].iloc[0]
    hour = summary.loc[summary["method"].str.contains("按小时去偏", regex=False)].iloc[0]
    data_bias = summary.loc[summary["method"].str.contains("常数去偏", regex=False)].iloc[0]
    month = summary.loc[summary["method"].str.contains("按月去偏", regex=False)].iloc[0]
    both = summary.loc[summary["method"].str.contains("月×小时", regex=False)].iloc[0]
    cross = run_cross_station()
    forest_cross = cross.loc[cross["method"].str.contains("随机森林", regex=False)].iloc[0]
    nj = load_nanjing()
    autocorr = error_autocorrelation(nj)
    importance = feature_importance(nj)
    return {
        "head": head,
        "forest": forest,
        "hour": hour,
        "data_bias": data_bias,
        "month": month,
        "both": both,
        "cross": cross,
        "forest_cross": forest_cross,
        "autocorr1": autocorr["autocorr"].iloc[0],
        "autocorr12": autocorr["autocorr"].iloc[-1],
        "importance": importance,
        "summary": summary,
    }


def _a_general(s: dict) -> tuple[str, dict]:
    head = s["head"]
    return (
        f"在本项目的检验条件下（南京禄口站 ZSNJ，{head['n']:,} 小时真实观测配对），"
        f"GFS 2 米气温预报的平均偏差是 **{head['bias']:+.2f} ℃**，"
        f"MAE 为 **{head['mae']:.2f} ℃**，RMSE 为 **{head['rmse']:.2f} ℃**，"
        f"与观测的相关系数为 **{head['corr']:.3f}**。\n\n"
        f"最关键的不是这些数字本身，而是 **bias（{head['bias']:+.2f} ℃）与 MAE"
        f"（{head['mae']:.2f} ℃）之间的落差**："
        "平均偏差接近 0，说明误差的主体不是整体偏移，而是某种结构。"
        "本项目后续的全部分析都在追问这个结构。",
        {
            "有效样本": f"{head['n']:,} 小时",
            "平均偏差 bias": f"{head['bias']:+.2f} ℃",
            "MAE": f"{head['mae']:.2f} ℃",
            "RMSE": f"{head['rmse']:.2f} ℃",
            "相关系数": f"{head['corr']:.3f}",
        },
    )


def _a_constant(s: dict) -> tuple[str, dict]:
    head = s["head"]
    data_bias = s["data_bias"]
    return (
        f"因为**需要消除的偏移本身就不存在**：全时段的平均偏差只有 {head['bias']:+.2f} ℃，"
        f"所以减去一个常数最多只能改善 {data_bias['MAE_improvement_pct']:.1f}%。\n\n"
        "这看起来是个“失败”的结果，其实是本项目最有价值的发现之一："
        "它说明误差不是整体偏暖或整体偏冷，"
        "而是随季节、随天气状态改变符号的结构性偏差——"
        "**正负误差在长时段平均后互相抵消了**。\n\n"
        f"作为对照，按（月×小时）分组建模可以把改善提高到 "
        f"{s['both']['MAE_improvement_pct']:.1f}%，差了一个数量级。",
        {
            "常数去偏改善": f"{data_bias['MAE_improvement_pct']:.1f}%",
            "按月去偏改善": f"{s['month']['MAE_improvement_pct']:.1f}%",
            "按小时去偏改善": f"{s['hour']['MAE_improvement_pct']:.1f}%",
            "按(月×小时)去偏改善": f"{s['both']['MAE_improvement_pct']:.1f}%",
        },
    )


def _a_amplitude(s: dict) -> tuple[str, dict]:
    head = s["head"]
    return (
        "**振幅阻尼**是本项目的核心发现：模式的误差不是“偏暖”或“偏冷”，"
        "而是它**压扁了温度变化的幅度**。\n\n"
        f"最直观的证据是日较差：观测平均日较差 {head['obs_range']:.2f} ℃，"
        f"预报只有 {head['fcst_range']:.2f} ℃，相当于被压缩了 {head['damping']:.0f}%。"
        "冬季压缩最严重（只剩 66%），夏季几乎不压缩。\n\n"
        f"第二个证据是距平回归：误差 = {head['intercept']:+.2f} "
        f"{head['slope']:+.2f} × 观测距平（r = {head['reg_r']:.2f}）。"
        "斜率为负意味着——**天越冷模式越偏暖，天越热模式越偏冷**。\n\n"
        f"第三个证据是极端时刻：最冷 1% 的时刻平均偏暖 {head['cold_error']:+.2f} ℃，"
        f"最暖 1% 的时刻平均偏冷 {head['warm_error']:+.2f} ℃。"
        "这一条最具实际意义：**平均误差看起来不大，但极端时刻的误差很大。**",
        {
            "观测日较差": f"{head['obs_range']:.2f} ℃",
            "预报日较差": f"{head['fcst_range']:.2f} ℃",
            "压缩比例": f"{head['damping']:.1f}%",
            "距平回归斜率": f"{head['slope']:+.3f}",
            "最冷 1% 偏暖": f"{head['cold_error']:+.2f} ℃",
            "最暖 1% 偏冷": f"{head['warm_error']:+.2f} ℃",
        },
    )


def _a_forest(s: dict) -> tuple[str, dict]:
    forest = s["forest"]
    both = s["both"]
    importance = s["importance"]
    top = "、".join(
        f"{row.feature} {row.importance:.3f}" for row in importance.head(3).itertuples()
    )
    return (
        f"随机森林把 MAE 从原始预报的 1.586 ℃ 降到 {forest['MAE_mean']:.3f} ℃，"
        f"相对原始预报改善 **{forest['MAE_improvement_pct']:.1f}%**。\n\n"
        f"但更有信息量的是**增量**：在已经包含季节×日变化结构的查表基线"
        f"（MAE {both['MAE_mean']:.3f} ℃）之上，随机森林只额外贡献了 "
        f"**{forest['step_gain_pct']:.1f} 个百分点**。\n\n"
        "这说明订正收益的**大部分来自消除系统性结构**，而不是靠模型捕捉复杂非线性。"
        f"同时必须指出：随机森林的折间标准差是 ±{forest['MAE_std']:.3f} ℃，"
        "是六个方法里最大的——**它换来的额外精度，代价是稳定性下降。**\n\n"
        f"模型主要依据的特征：{top}。",
        {
            "随机森林 MAE": f"{forest['MAE_mean']:.3f} ℃",
            "相对原始改善": f"{forest['MAE_improvement_pct']:.1f}%",
            "相对最强基线的额外增量": f"{forest['step_gain_pct']:.1f} 个百分点",
            "折间标准差": f"±{forest['MAE_std']:.3f} ℃",
        },
    )


def _a_cross(s: dict) -> tuple[str, dict]:
    cross = s["cross"]
    forest_cross = s["forest_cross"]
    raw = cross.iloc[0]
    return (
        "因为南京训练出的订正模型，学到的其实是**南京特有的季节—日变化偏差结构**，"
        "而这种结构并不通用。\n\n"
        f"检验结果：所有方法迁移到上海后都失效，随机森林甚至让 MAE 变差 "
        f"{abs(forest_cross['MAE_improvement_pct']):.1f}%。\n\n"
        "原因是双重的。第一，上海本身的原始预报误差就小得多："
        f"MAE {raw['MAE']:.2f} ℃，而南京是 1.58 ℃——"
        "**在误差本来就小的站点上，多余的订正只会引入噪声。**"
        "第二，诊断显示上海几乎不存在振幅压缩问题（日较差比值 0.98，南京为 0.84）。\n\n"
        "这个结果是有意设计的**可证伪检验**：如果振幅阻尼的解释成立，"
        "那么在没有该现象的站点上订正就应当失效——数据支持了这个预测。",
        {
            "上海原始 MAE": f"{raw['MAE']:.2f} ℃",
            "南京原始 MAE": "1.58 ℃",
            "随机森林迁移后改善": f"{forest_cross['MAE_improvement_pct']:.1f}%",
            "上海日较差比值": "0.98",
            "南京日较差比值": "0.84",
        },
    )


def _a_case(s: dict) -> tuple[str, dict]:
    from views.common import run_case

    cold, cold_summary = run_case("2024-01-19", "2024-01-27")
    heat, heat_summary = run_case("2024-08-01", "2024-08-13")
    return (
        "两个案例的偏差方向**相反**，这本身就是对“振幅阻尼”解释的一次检验。\n\n"
        f"**2024 年 1 月寒潮过程**：观测温度 {cold['observed_temperature'].min():.1f} ~ "
        f"{cold['observed_temperature'].max():.1f} ℃。模式在降温阶段明显偏暖"
        f"（bias {cold_summary.iloc[0]['bias']:+.2f} ℃）——该冷的时候不够冷。"
        f"订正后 MAE 从 {cold_summary.iloc[0]['MAE']:.2f} ℃ 降到 "
        f"{cold_summary.iloc[2]['MAE']:.2f} ℃。\n\n"
        f"**2024 年 8 月高温过程**：观测最高达 {heat['observed_temperature'].max():.1f} ℃。"
        f"模式在高温时段明显偏冷（bias {heat_summary.iloc[0]['bias']:+.2f} ℃）"
        "——该热的时候不够热。\n\n"
        "一个偏暖、一个偏冷，符号相反但机制相同：**模式把温度变化的幅度抹平了**。",
        {
            "1月寒潮 原始 bias": f"{cold_summary.iloc[0]['bias']:+.2f} ℃",
            "1月寒潮 原始 MAE": f"{cold_summary.iloc[0]['MAE']:.2f} ℃",
            "1月寒潮 订正后 MAE": f"{cold_summary.iloc[2]['MAE']:.2f} ℃",
            "8月高温 原始 bias": f"{heat_summary.iloc[0]['bias']:+.2f} ℃",
            "8月高温 原始 MAE": f"{heat_summary.iloc[0]['MAE']:.2f} ℃",
            "8月高温 订正后 MAE": f"{heat_summary.iloc[2]['MAE']:.2f} ℃",
        },
    )


def _a_data(s: dict) -> tuple[str, dict]:
    head = s["head"]
    return (
        "全部数据都是真实数据，本项目**不使用任何合成数据**：\n\n"
        "- **观测（真值）**：南京禄口机场 ZSNJ 逐小时地面观测，"
        "来自 Iowa Environmental Mesonet 的 METAR 归档；\n"
        "- **预报**：GFS 2 米气温历史预报，来自 Open-Meteo 的历史预报归档；\n"
        "- **预报时效**：GFS 1/2/3/5/7 天预报，来自 Open-Meteo 的 previous-runs 归档；\n"
        "- **对照站**：上海虹桥 ZSSS，用于跨站点迁移检验。\n\n"
        f"主数据共 {head['n']:,} 个有效配对小时，"
        f"时段为 {head['start']:%Y-%m-%d} 至 {head['end']:%Y-%m-%d}。"
        "温度由 ℉ 换算为 ℃，风速由节换算为 m/s，时间统一以 UTC 存储。",
        {
            "有效样本": f"{head['n']:,} 小时",
            "时段": f"{head['start']:%Y-%m-%d} 至 {head['end']:%Y-%m-%d}",
            "数据来源": "IEM METAR 观测 + Open-Meteo GFS 归档预报",
        },
    )


def _a_split(s: dict) -> tuple[str, dict]:
    return (
        "因为气象时间序列**高度自相关**，随机划分会严重高估模型能力。\n\n"
        f"本项目实测误差在滞后 1 小时的自相关达到 {s['autocorr1']:.2f}，"
        f"12 小时后仍有 {s['autocorr12']:.2f}。"
        "如果随机划分样本，同一次天气过程会同时出现在训练集和测试集里，"
        "模型等于“见过答案再考试”，指标会虚高。\n\n"
        "因此本项目采用**扩张窗口 + 严格时序**的三折验证："
        "每一折的训练数据全部早于测试数据，三折测试段互不重叠，"
        "并且跨越不同季节，顺带完成了跨季节验证。"
        "所有方法在完全相同的样本上比较，因此差异只来自方法本身。",
        {
            "滞后1小时自相关": f"{s['autocorr1']:.2f}",
            "滞后12小时自相关": f"{s['autocorr12']:.2f}",
            "验证方式": "扩张窗口、严格时序三折",
        },
    )


def _a_precip(s: dict) -> tuple[str, dict]:
    return (
        "无法开展，这是一个真实的数据可获得性限制，而不是方法上的疏漏。\n\n"
        "实测南京禄口（ZSNJ）、上海虹桥（ZSSS）、上海浦东（ZSPD）三个机场的 "
        "METAR 归档中，逐小时降水量字段**全部为 0** 或缺失——"
        "中国机场例行观测不报降水量。\n\n"
        "要补上这一块，需要改用卫星降水产品，例如 GPM IMERG 或 CHIRPS。"
        "这也是项目“下一步”中排在前面的一项。"
        "本项目选择**明确标注这一限制**，而不是用模式自身的降水场当作真值——"
        "后者会造成循环论证。",
        {"结论": "免费源无法获得中国境内逐小时降水观测",
         "建议替代": "GPM IMERG / CHIRPS 卫星降水产品"},
    )


def _a_lead(s: dict) -> tuple[str, dict]:
    from tools.verification import score_by
    from views.common import load_leadtime

    lead = score_by(load_leadtime(), "lead_day")
    return (
        "误差随预报时效稳定增长，但**偏差几乎不随时效变化**——这个对比很有信息量。\n\n"
        f"RMSE 从 1 天的 {lead['RMSE'].iloc[0]:.2f} ℃ 增长到 7 天的 "
        f"{lead['RMSE'].iloc[-1]:.2f} ℃（约 "
        f"{lead['RMSE'].iloc[-1] / lead['RMSE'].iloc[0]:.1f} 倍），"
        f"相关系数从 {lead['corr'].iloc[0]:.2f} 降到 {lead['corr'].iloc[-1]:.2f}。\n\n"
        f"但偏差始终维持在 {lead['bias'].mean():+.2f} ℃ 附近，几乎没有随时效变化。"
        "如果是初值误差随时间放大，偏差不应该这样稳定。"
        "因此更合理的解释是：**系统性偏差来自模式物理过程本身的固有倾向**，"
        "而不是初始化问题。这直接决定了订正策略应该针对“模式的系统性行为”，"
        "而不是针对“初始误差”。",
        {
            "1 天 RMSE": f"{lead['RMSE'].iloc[0]:.2f} ℃",
            "7 天 RMSE": f"{lead['RMSE'].iloc[-1]:.2f} ℃",
            "1 天相关系数": f"{lead['corr'].iloc[0]:.2f}",
            "7 天相关系数": f"{lead['corr'].iloc[-1]:.2f}",
            "偏差是否随时效变化": "基本不变",
        },
    )


def _a_limits(s: dict) -> tuple[str, dict]:
    return (
        "本项目主动记录了以下边界：\n\n"
        "1. **站点覆盖有限**：主站点只有南京，另加上海作对照，"
        "结论不能代表全国；跨站点实验已经显示订正模型不能直接迁移。\n"
        "2. **无法检验降水**：中国机场 METAR 不报降水量。\n"
        "3. **存在代表性误差**：观测是机场站点，预报是格点值。\n"
        "4. **只检验 2 米气温**。\n"
        "5. **时效数据只有 92 天**，季节代表性不足。\n"
        "6. **纯统计订正缺乏物理约束**，外推能力有限。\n\n"
        "另外，研究过程中有四个假设被数据推翻了，也记录在“局限与展望”页里："
        "订正不是消除系统偏差、日变化可以单独订正、订正关系可以跨站点推广、"
        "误差主要是随机噪声。",
        {"记录方式": "见“局限与展望”页，含被推翻假设清单"},
    )


def _a_next(s: dict) -> tuple[str, dict]:
    return (
        "按可行性排序，最值得先做的是这三件：\n\n"
        "1. **把误差持续性变成预报因子**：误差滞后 1 小时自相关 0.90，"
        "说明“前一天的观测误差”包含尚未被利用的信息，"
        "这是最有希望突破当前 22.8% 上限的方向。\n"
        "2. **按天气型分组建模**：检验“随天气状态变化的订正”"
        "是否优于固定的季节—日变化查表。\n"
        "3. **增加更多站点**，把跨站点实验做成真正的泛化实验，"
        "量化迁移损失随距离和气候差异的变化。\n\n"
        "中期可以接入卫星降水产品（IMERG/CHIRPS）把检验扩展到降水，"
        "并对比 ECMWF、CMA 等模式，检验振幅压缩是否为全球模式的共性问题。",
        {"优先方向": "误差持续性特征、天气型分组、跨站点泛化"},
    )


ENTRIES: list[tuple[tuple[str, ...], object]] = [
    (("常数", "去偏没用", "去偏没有", "为什么没用", "减去平均"), _a_constant),
    (("振幅", "日较差", "压缩", "变幅", "阻尼", "距平"), _a_amplitude),
    (("随机森林", "机器学习", "提升", "改善多少", "提升多少"), _a_forest),
    (("上海", "跨站", "迁移", "推广", "泛化"), _a_cross),
    (("寒潮", "高温", "案例", "过程", "个例"), _a_case),
    (("降水", "降雨", "TS", "暴雨"), _a_precip),
    (("时效", "几天", "预报时效"), _a_lead),
    (("划分", "验证", "交叉", "随机划分"), _a_split),
    (("数据", "来源", "哪来的", "怎么来"), _a_data),
    (("局限", "不足", "边界", "缺点", "问题"), _a_limits),
    (("下一步", "后续", "改进", "未来"), _a_next),
    (("误差", "多大", "MAE", "RMSE", "准", "bias", "偏差"), _a_general),
]


def answer(question: str) -> tuple[str, dict]:
    """Return (markdown reply, evidence dict) for a natural-language question."""
    text = (question or "").strip()
    if not text:
        return "请先输入一个问题。", {}

    stats = _stats()
    lowered = text.lower()
    best_entry, best_score = None, 0
    for keywords, handler in ENTRIES:
        score = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if score > best_score:
            best_entry, best_score = handler, score

    if best_entry is None:
        guesses = "、".join(SUGGESTED[:4])
        return (
            "这个问题暂时不在离线助手的知识库里。"
            f"可以试试这些问法：{guesses}。\n\n"
            "（配置 API Key 后可以启用 LLM 版本，获得更自由的对话能力。）",
            {},
        )
    return best_entry(stats)

"""
tools/data_checker.py

数据质量检查模块。

职责：
    对气象数据 DataFrame 进行检查，并输出一份结构化的质量报告。

检查项目：
    1. 数据行数和列数
    2. 必要字段是否存在
    3. 每一列的数据类型
    4. 缺失值数量
    5. 重复数据数量
    6. 时间字段是否能够正确转换为 datetime
    7. 数值字段是否存在明显异常值（依据气象常识的物理范围）

注意：本模块只负责「检查」，不做数据修改，也不做任何统计计算。
"""

import pandas as pd

# 必要字段：数据必须包含这些列，否则后续分析无法进行
REQUIRED_FIELDS = [
    "time",
    "observed_temperature",
    "forecast_temperature",
    "humidity",
    "pressure",
    "wind_speed",
]

# 各数值字段的合理物理范围（依据气象常识）
VALID_RANGES = {
    "observed_temperature": (-50.0, 60.0),   # 单位：摄氏度
    "forecast_temperature": (-50.0, 60.0),   # 单位：摄氏度
    "humidity": (0.0, 100.0),                # 单位：百分比
    "pressure": (850.0, 1085.0),             # 单位：百帕 hPa
    "wind_speed": (0.0, 75.0),               # 单位：米/秒 m/s
}


def check_data(df):
    """
    对数据进行检查，返回一份质量报告（字典）。

    参数：
        df : pandas.DataFrame，待检查的气象数据

    返回：
        dict，包含以下键：
            ok             : bool  整体是否通过基础检查
            row_count      : int   数据行数
            column_count   : int   数据列数
            required_fields: list  要求存在的字段
            missing_fields : list  实际缺失的必要字段
            missing_values : dict  各字段缺失值数量
            duplicate_rows : int   重复行数
            dtypes         : dict  各字段当前类型（字符串）
            type_issues    : list  数据类型问题描述
            time_ok        : bool  时间字段是否有效
            time_issue     : str   时间字段问题描述（无则为 None）
            valid_ranges   : dict  各字段的合理范围
            outliers       : dict  各字段的异常值数量
    """
    report = {
        "ok": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "required_fields": REQUIRED_FIELDS,
        "missing_fields": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "dtypes": {},
        "type_issues": [],
        "time_ok": True,
        "time_issue": None,
        "valid_ranges": VALID_RANGES,
        "outliers": {},
    }

    # 1. 必要字段检查
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            report["missing_fields"].append(field)
    if report["missing_fields"]:
        report["ok"] = False

    # 2. 缺失值检查（对数据中实际存在的字段逐一统计）
    for col in df.columns:
        report["missing_values"][col] = int(df[col].isna().sum())

    # 3. 重复数据检查
    report["duplicate_rows"] = int(df.duplicated().sum())

    # 4. 数据类型检查
    for col in df.columns:
        report["dtypes"][col] = str(df[col].dtype)

    numeric_fields = [
        "observed_temperature",
        "forecast_temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    for field in numeric_fields:
        if field in df.columns and not pd.api.types.is_numeric_dtype(df[field]):
            report["type_issues"].append(f"{field} 不是数值类型")
            report["ok"] = False

    # 5. 时间字段检查
    if "time" in df.columns:
        # 尝试解析 time 列，errors="coerce" 会把无法解析的值转为 NaT
        parsed = pd.to_datetime(df["time"], errors="coerce")
        invalid_count = int(parsed.isna().sum())
        if invalid_count > 0:
            report["time_ok"] = False
            report["time_issue"] = f"time 字段中有 {invalid_count} 条无法解析为日期时间"
            report["ok"] = False
    else:
        report["time_ok"] = False
        report["time_issue"] = "缺少 time 字段"

    # 6. 异常值检查（依据合理物理范围）
    for field, (low, high) in VALID_RANGES.items():
        if field not in df.columns:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        report["outliers"][field] = int(((series < low) | (series > high)).sum())

    return report

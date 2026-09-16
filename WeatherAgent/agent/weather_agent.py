"""
agent/weather_agent.py

DeepSeek AI 气象预报分析 Agent。

职责分工：
    Python 负责：数据读取、质量检查、MAE/RMSE/Bias 计算、随机森林订正等真实计算。
    DeepSeek 负责：理解用户问题、选择合适的 Python 工具、对工具返回的真实结果做自然语言解释。

说明：
    本模块通过 OpenAI 兼容接口对接 DeepSeek（base_url = https://api.deepseek.com）。
    API Key 只从环境变量 OPENAI_API_KEY 读取，绝不硬编码。
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from tools.data_checker import check_data
from tools.forecast_evaluator import evaluate_forecast
from tools.ml_correction import machine_learning_correction

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """
你是一个气象数据分析助手。

你的任务是帮助用户分析气象预报数据和机器学习订正结果。

你必须遵守以下规则：
1. 所有 MAE、RMSE、Bias、改善率等数值必须来自 Python 工具返回的真实结果。
2. 不允许自行编造任何实验结果或数值。
3. 如果工具没有返回某个数值，不要猜测或编造。
4. 必须严格区分「原始预报」和「机器学习订正后的预报」。
5. 对实验结果进行客观、准确的解释。
6. 当前项目中的数据属于演示/实验数据，不能宣称具有真实业务预报能力。
7. 可以从气象学角度解释可能的误差来源（如日变化、天气系统、局地影响等）。
8. 不要把机器学习订正结果描述成已证明具有普适性。
9. 如果订正结果改善了误差，要说明这是「在当前测试数据上」的结果。
10. 如果订正结果没有改善，也必须如实说明，不要掩盖。
11. 机器学习订正结果来自滚动窗口交叉验证：报告整体水平时使用「均值 ± 标准差」格式，并可结合各折结果说明订正效果在不同时间段上的稳定性。

当你需要任何数值时，必须调用对应的 Python 工具获取真实结果，而不是自己计算。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_data",
            "description": "检查当前气象数据的质量，返回数据行数/列数、字段列表、时间范围、缺失值、重复数据、字段类型和异常值统计。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_forecast",
            "description": "计算原始预报的误差检验指标 MAE、RMSE、Bias（由 Python 真实计算）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_correction",
            "description": "运行随机森林预报订正（滚动窗口交叉验证），返回每一折的订正前后 MAE/RMSE/Bias、整体均值与标准差、MAE/RMSE 改善率（各折均值）和平均特征重要性。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class WeatherAgent:
    def __init__(self, df):
        self.df = df
        self.error = None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.client = None
            self.error = "未检测到 API Key。请在项目根目录创建 `.env` 文件，并写入：`OPENAI_API_KEY=你的DeepSeek Key` 后重新启动。"
        else:
            self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def _tool_check_data(self):
        report = check_data(self.df)
        time_start, time_end = None, None
        if "time" in self.df.columns:
            parsed = pd.to_datetime(self.df["time"], errors="coerce")
            if parsed.notna().any():
                time_start = str(parsed.min())
                time_end = str(parsed.max())
        return {
            "row_count": int(report["row_count"]),
            "column_count": int(report["column_count"]),
            "columns": list(self.df.columns),
            "time_range": {"start": time_start, "end": time_end},
            "missing_values": report["missing_values"],
            "duplicate_rows": int(report["duplicate_rows"]),
            "dtypes": report["dtypes"],
            "outliers": report["outliers"],
        }

    def _tool_evaluate_forecast(self):
        result = evaluate_forecast(self.df)
        return {"MAE": float(result["MAE"]), "RMSE": float(result["RMSE"]), "Bias": float(result["Bias"])}

    def _tool_ml_correction(self):
        result = machine_learning_correction(self.df)
        folds = [{"fold": f["fold"], "test_period": f"{f['test_start']} ~ {f['test_end']}", "train_samples": f["train_size"], "test_samples": f["test_size"], "raw_mae": round(f["raw"]["MAE"], 4), "raw_rmse": round(f["raw"]["RMSE"], 4), "raw_bias": round(f["raw"]["Bias"], 4), "corrected_mae": round(f["corrected"]["MAE"], 4), "corrected_rmse": round(f["corrected"]["RMSE"], 4), "corrected_bias": round(f["corrected"]["Bias"], 4), "mae_improvement_percent": round(f["mae_improve_percent"], 2)} for f in result["folds"]]
        return {"method": "rolling_window_cv（滚动窗口交叉验证，训练窗口逐折扩大）", "n_splits": result["n_splits"], "folds": folds, "raw_mean": {k: round(v, 4) for k, v in result["raw_mean"].items()}, "raw_std": {k: round(v, 4) for k, v in result["raw_std"].items()}, "corrected_mean": {k: round(v, 4) for k, v in result["corrected_mean"].items()}, "corrected_std": {k: round(v, 4) for k, v in result["corrected_std"].items()}, "mae_improvement_mean_percent": round(result["mae_improve_mean"], 2), "rmse_improvement_mean_percent": round(result["rmse_improve_mean"], 2), "feature_importance_avg": {k: round(float(v), 4) for k, v in result["importance"].items()}, "note": "corrected_mean / corrected_std 为各折订正后指标的均值与标准差，标准差越小表示订正效果在不同时间段越稳定。"}

    def _run_tool(self, name):
        try:
            if name == "check_data": return self._tool_check_data()
            if name == "evaluate_forecast": return self._tool_evaluate_forecast()
            if name == "ml_correction": return self._tool_ml_correction()
            return {"error": f"未知工具：{name}"}
        except Exception as exc:
            return {"error": f"工具 {name} 执行失败：{exc}"}

    def chat(self, user_message, max_steps=5):
        if self.error: return self.error
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
        for _ in range(max_steps):
            try:
                response = self.client.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2)
            except Exception as exc:
                return f"调用 DeepSeek API 失败：{exc}"
            message = response.choices[0].message
            if message.tool_calls:
                messages.append({"role": "assistant", "content": message.content, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]})
                for tc in message.tool_calls:
                    result = self._run_tool(tc.function.name)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
            else:
                return message.content or "（模型未返回内容）"
        return "分析步骤过多，已中止。请尝试提出更具体的问题。"

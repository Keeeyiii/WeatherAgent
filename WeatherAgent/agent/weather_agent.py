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

# 加载项目根目录下的 .env（若存在）
load_dotenv()

# DeepSeek 兼容 OpenAI 接口的地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 系统提示词：规定 DeepSeek 的行为边界
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

当你需要任何数值时，必须调用对应的 Python 工具获取真实结果，而不是自己计算。
"""

# 工具定义（OpenAI function calling 格式，DeepSeek 兼容）
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
            "description": "运行随机森林预报订正，返回训练/测试样本数、订正前后 MAE/RMSE/Bias、MAE/RMSE 改善率和特征重要性。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class WeatherAgent:
    """封装 DeepSeek 客户端与工具调用逻辑。"""

    def __init__(self, df):
        self.df = df
        self.error = None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.client = None
            self.error = (
                "未检测到 API Key。请在项目根目录创建 `.env` 文件，"
                "并写入：`OPENAI_API_KEY=你的DeepSeek Key` 后重新启动。"
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
            )

    # ---------------- 工具实现（由 Python 真实计算） ----------------

    def _tool_check_data(self):
        """检查数据质量，返回对模型友好的精简结构。"""
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
        """计算原始预报的 MAE / RMSE / Bias。"""
        result = evaluate_forecast(self.df)
        return {
            "MAE": float(result["MAE"]),
            "RMSE": float(result["RMSE"]),
            "Bias": float(result["Bias"]),
        }

    def _tool_ml_correction(self):
        """运行随机森林订正，返回订正前后指标与改善率。"""
        result = machine_learning_correction(self.df)

        raw = result["raw"]
        corrected = result["corrected"]

        mae_improve = (raw["MAE"] - corrected["MAE"]) / raw["MAE"] * 100
        rmse_improve = (raw["RMSE"] - corrected["RMSE"]) / raw["RMSE"] * 100

        return {
            "train_samples": int(result["train_size"]),
            "test_samples": int(result["test_size"]),
            "raw_mae": float(raw["MAE"]),
            "raw_rmse": float(raw["RMSE"]),
            "raw_bias": float(raw["Bias"]),
            "corrected_mae": float(corrected["MAE"]),
            "corrected_rmse": float(corrected["RMSE"]),
            "corrected_bias": float(corrected["Bias"]),
            "mae_improvement_percent": float(mae_improve),
            "rmse_improvement_percent": float(rmse_improve),
            "feature_importance": {
                k: float(v) for k, v in result["importance"].items()
            },
        }

    def _run_tool(self, name):
        """根据工具名执行对应 Python 工具，并捕获异常。"""
        try:
            if name == "check_data":
                return self._tool_check_data()
            if name == "evaluate_forecast":
                return self._tool_evaluate_forecast()
            if name == "ml_correction":
                return self._tool_ml_correction()
            return {"error": f"未知工具：{name}"}
        except Exception as exc:  # 工具失败也返回结构化错误，不让页面崩溃
            return {"error": f"工具 {name} 执行失败：{exc}"}

    # ---------------- 对话入口 ----------------

    def chat(self, user_message, max_steps=5):
        """
        处理一次用户提问，返回 DeepSeek 的自然语言回答。

        流程：发送问题 -> 模型选择工具 -> Python 执行工具 ->
              把真实结果回传 -> 模型生成最终解释。
        """
        if self.error:
            return self.error

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_steps):
            try:
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as exc:
                return f"调用 DeepSeek API 失败：{exc}"

            message = response.choices[0].message

            # 如果模型要求调用工具
            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                for tc in message.tool_calls:
                    result = self._run_tool(tc.function.name)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            else:
                # 模型给出最终回答
                return message.content or "（模型未返回内容）"

        return "分析步骤过多，已中止。请尝试提出更具体的问题。"

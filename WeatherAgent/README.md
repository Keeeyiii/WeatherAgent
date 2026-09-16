# WeatherAgent——AI气象预报智能分析与误差订正系统

## 1. 项目简介

WeatherAgent 是一个面向气象预报数据分析的智能辅助系统（科研实践/实验性原型），将**气象数据分析、机器学习和大语言模型 Agent** 相结合，用于气象预报误差分析与后处理实验。

系统实现的核心流程：

```
气象数据质量检查 → 预报误差评价 → 机器学习误差订正 → 订正效果检验 → DeepSeek AI Agent → 自然语言气象分析
```

> 说明：本项目定位为科研实践/实验性原型系统，用于展示方法流程，不是可实际业务化的天气预报系统。

## 2. 项目背景

数值天气预报与气象观测之间通常存在一定的**预报误差**，其来源包括模式初值误差、物理过程参数化误差、局地地形影响等。

- **机器学习**可以用于预报后处理与误差订正：通过历史数据学习预报误差的规律，对原始预报进行订正，从而降低系统误差。
- **大语言模型 Agent** 可以帮助用户以自然语言调用分析工具，无需编写代码即可完成数据检查、误差评价、订正分析等操作。

本项目通过一个完整的 Web 应用，把上述流程串起来，形成可交互、可展示的科研实践作品。

## 3. 项目功能

- 数据质量检查
- 预报误差评价
- MAE / RMSE / Bias 计算
- Random Forest 误差订正
- 订正效果评价
- 特征重要性分析
- 可视化（温度对比、误差对比、特征重要性）
- DeepSeek AI Agent（自然语言气象分析）

## 4. 技术路线

```
数据（CSV）
   ↓
数据质量检查（缺失值/重复/类型/异常值/时间解析）
   ↓
原始预报检验（MAE / RMSE / Bias）
   ↓
Random Forest 误差建模（滚动窗口交叉验证，多折评估稳定性）
   ↓
误差订正（corrected = forecast + predicted_correction）
   ↓
测试集评价（订正前后 MAE / RMSE / Bias 对比）
   ↓
DeepSeek Agent 自然语言解释
```

## 5. 技术栈

- Python
- Streamlit（Web 界面）
- Pandas / NumPy（数据处理）
- Scikit-learn（Random Forest）
- Matplotlib / Plotly（可视化）
- DeepSeek API（LLM Agent）

## 6. 核心方法

### 6.1 预报误差定义

```
error = forecast - observed
```

- error > 0：预报偏高（高估）
- error < 0：预报偏低（低估）

### 6.2 预报评价指标

- **MAE（平均绝对误差）**：`MAE = mean(|error|)`，反映平均误差大小。
- **RMSE（均方根误差）**：`RMSE = sqrt(mean(error²))`，对大误差更敏感。
- **Bias（平均偏差）**：`Bias = mean(error)`，正值表示预报系统性偏高。

### 6.3 Random Forest 误差订正

使用 Random Forest Regression 学习「订正量」（观测减预报）：

```
correction = observed - forecast
predicted_correction = f(forecast_temperature, humidity, pressure, wind_speed, hour, month)
corrected_forecast = forecast_temperature + predicted_correction
```

模型参数：`RandomForestRegressor(n_estimators=200, random_state=42)`。

### 6.4 验证方式：滚动窗口交叉验证

采用**滚动窗口交叉验证**（rolling window cross-validation）：

- 训练窗口逐折扩大、测试段依次后移且互不重叠（共 3 折，每折测试约 20% 数据）
- 每一折的训练数据全部位于测试数据之前，无数据泄漏
- 每折独立训练并计算 MAE / RMSE / Bias，最终汇总为「均值 ± 标准差」

这样可以反映订正效果在不同时间段上的稳定性，避免单次划分结果的偶然性。

## 7. Agent 架构

```
用户自然语言问题
   ↓
DeepSeek Agent（理解问题、选择工具）
   ↓
Python 分析工具（check_data / evaluate_forecast / ml_correction）
   ↓
返回真实计算结果（MAE / RMSE / Bias / 改善率 / 特征重要性 / 各折稳定性）
   ↓
DeepSeek 生成自然语言解释
```

**分工原则：**

- **Python 负责**：数据读取、质量检查、MAE/RMSE/Bias 计算、随机森林订正等所有真实数值计算。
- **DeepSeek 负责**：理解问题、选择工具、解释结果，**不自行计算任何统计指标**。

## 8. 实验结果

> 以下结果为当前演示数据（`data/sample_weather.csv`，240 条小时数据）上的真实计算结果。

### 8.1 原始预报检验（全量 240 条数据）

| 指标 | 数值 |
|------|------|
| MAE | 1.3943 ℃ |
| RMSE | 1.7327 ℃ |
| Bias | +1.2212 ℃ |

### 8.2 机器学习订正（滚动窗口交叉验证，3 折）

每一折的检验结果（测试段互不重叠，各 48 条）：

| 折 | 训练/测试样本 | 原始 MAE | 订正后 MAE | 原始 RMSE | 订正后 RMSE | 原始 Bias | 订正后 Bias |
|----|--------------|---------|-----------|----------|------------|----------|------------|
| 1 | 96 / 48 | 1.3092 | 0.4787 | 1.7263 | 0.6046 | +1.1925 | -0.1622 |
| 2 | 144 / 48 | 1.4587 | 0.5943 | 1.7669 | 0.7213 | +1.2646 | +0.1022 |
| 3 | 192 / 48 | 1.3802 | 0.6029 | 1.6902 | 0.7277 | +1.1660 | -0.1569 |

**整体表现（各折均值 ± 标准差）：**

| 指标 | 原始预报（均值） | 订正后（均值 ± 标准差） |
|------|----------------|----------------------|
| MAE | 1.3827 ℃ | 0.5586 ± 0.0694 ℃ |
| RMSE | 1.7278 ℃ | 0.6845 ± 0.0693 ℃ |
| Bias | +1.2077 ℃ | -0.0723 ± 0.1511 ℃ |

- **MAE 改善率（各折均值）：59.67%**
- **RMSE 改善率（各折均值）：60.37%**

### 8.3 特征重要性（各折平均）

| 特征 | 重要性 |
|------|--------|
| hour | 0.770 |
| forecast_temperature | 0.101 |
| humidity | 0.052 |
| wind_speed | 0.043 |
| pressure | 0.035 |
| month | 0.000 |

### 8.4 结果解读

- 三折订正后 MAE 均稳定低于原始预报（单折改善率 56% ~ 63%），整体为 0.5586 ± 0.0694 ℃，标准差较小，说明订正效果在不同时间段上稳定，并非单次划分的偶然结果。
- Bias 从 +1.208 降至 -0.072（接近 0），说明随机森林成功学习并基本消除了示例数据中的系统性偏差。
- 特征重要性中 `hour` 占绝对主导（各折平均 0.770），说明预报偏差存在明显的日变化规律，模型抓住了这一规律。

## 9. 实验局限性

- 当前数据规模较小（240 条小时数据）。
- 数据属于演示/实验数据，非真实业务观测与预报数据。
- 时间范围有限（仅 10 天）。
- 样本代表性不足（单一地点、单一季节的合成数据）。
- 特征数量有限（仅 6 个气象要素）。
- 尚未进行跨时间、跨地点验证。
- 实验结果不能直接用于业务天气预报。

## 10. 后续工作

- 增加更长时间序列数据。
- 增加更多气象要素（如辐射、云量、降水等）。
- 尝试 XGBoost、LightGBM 等模型。
- 进行跨季节验证。
- 进行跨站点验证。
- 尝试将模型应用于真实数值预报产品。
- 探索机器学习与数值天气预报后处理的结合。

## 11. 项目运行方法

### 11.1 环境安装

```bash
pip install -r requirements.txt
```

### 11.2 配置 DeepSeek API Key

1. 复制 `.env.example` 为 `.env`：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填入你的 DeepSeek API Key：

   ```
   OPENAI_API_KEY=你的DeepSeekKey
   ```

> 注意：`.env` 已被 `.gitignore` 忽略，不会被提交到 Git，请勿将真实 Key 写入代码或 README。

### 11.3 启动 Streamlit

```bash
streamlit run app.py
```

然后在浏览器打开 http://localhost:8501 即可使用。

### 11.4 部署到 Streamlit Community Cloud（可选）

如需获得一个可公开访问、长期稳定的链接，可将项目部署到 Streamlit Community Cloud：

1. 将项目代码上传到 GitHub 仓库（注意：`.env` 已被 `.gitignore` 忽略，不会被上传）。
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号登录，点击「New app」，选择你的仓库和入口文件 `app.py`。
3. 在应用的 **Settings → Secrets** 中配置 API Key（TOML 格式）：

   ```toml
   OPENAI_API_KEY = "你的DeepSeekKey"
   ```

   > 本项目代码已兼容 Streamlit Secrets：`app.py` 会把 Secrets 中的 Key 写入环境变量供 Agent 读取，无需修改代码。

4. 点击 **Deploy**，稍等片刻即可获得 `https://xxx.streamlit.app` 的公开链接。

## 项目结构

```
WeatherAgent/
├── app.py                    # Streamlit 主页面
├── agent/
│   └── weather_agent.py      # DeepSeek AI Agent
├── tools/
│   ├── data_checker.py       # 数据质量检查
│   ├── forecast_evaluator.py # 预报误差检验（MAE/RMSE/Bias）
│   └── ml_correction.py      # 随机森林误差订正
├── data/
│   └── sample_weather.csv    # 演示数据（240 条小时数据）
├── models/                   # 模型保存目录（预留）
├── outputs/                  # 图表输出目录（预留）
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> 说明：可视化图表（温度对比、误差对比、特征重要性）由 `app.py` 内置 Plotly 直接生成。

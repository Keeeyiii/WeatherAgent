# WeatherAgent

气象预报误差诊断与订正实验：一个用**真实观测数据**回答具体问题的 Streamlit 应用。

## 内容

| 路径 | 说明 |
|---|---|
| `WeatherAgent/` | 应用本体（入口 `WeatherAgent/app.py`） |
| `WeatherAgent/legacy/` | 早期版本代码快照，不参与运行，仅作过程记录 |
| `.devcontainer/` | 开发容器配置 |

## 运行

```bash
pip install -r WeatherAgent/requirements.txt
streamlit run WeatherAgent/app.py
```

应用读取仓库内已内置的 CSV 数据，**离线即可运行，不需要任何 API Key**。

## 文档

| 文件 | 用途 |
|---|---|
| `WeatherAgent/成果速览.md` | 半页版成果摘要 |
| `WeatherAgent/研究报告.md` | 完整研究报告（含图表） |
| `WeatherAgent/研究报告.html` | 报告的可打印版本，浏览器打开即可打印成 PDF |
| `WeatherAgent/整合说明.md` | 说明这一版与早期版本的关系、改了什么、为什么改 |
| `WeatherAgent/部署指南.md` | 部署到 Streamlit Community Cloud 的步骤 |
| `WeatherAgent/README.md` | 项目详细说明（数据结构、模块划分、结果） |

## 部署

Streamlit Community Cloud 的 **Main file path** 填 `WeatherAgent/app.py`。

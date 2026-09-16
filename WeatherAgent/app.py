"""
WeatherAgent â€”â€” AIæ°”è±¡é¢„æŠ¥æ™ºèƒ½åˆ†æžä¸Žè¯¯å·®è®¢æ­£ç³»ç»Ÿ

å…¥å£æ–‡ä»¶ï¼šè´Ÿè´£æ­å»º Streamlit ç½‘é¡µç•Œé¢ã€‚

æœ¬ç³»ç»Ÿä¸ºç§‘ç ”å®žè·µ/å®žéªŒæ€§åŽŸåž‹ï¼Œç”¨äºŽæ¼”ç¤ºï¼š
æ°”è±¡æ•°æ®è´¨é‡æ£€æŸ¥ -> é¢„æŠ¥è¯¯å·®è¯„ä»· -> æœºå™¨å­¦ä¹ è¯¯å·®è®¢æ­£ -> è®¢æ­£æ•ˆæžœæ£€éªŒ -> DeepSeek AI Agent
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.weather_agent import WeatherAgent
from tools.data_checker import check_data
from tools.forecast_evaluator import evaluate_forecast
from tools.ml_correction import machine_learning_correction

# Streamlit Cloud éƒ¨ç½²å…¼å®¹ï¼šè‹¥åœ¨ Secrets ä¸­é…ç½®äº† Keyï¼Œåˆ™å†™å…¥çŽ¯å¢ƒå˜é‡ä¾› Agent è¯»å–ã€‚
# æ³¨æ„ï¼šæœ¬åœ°æœªé…ç½® secrets.toml æ—¶ï¼Œè®¿é—® st.secrets ä¼šæŠ›å¼‚å¸¸ï¼Œéœ€æ•èŽ·åŽå¿½ç•¥
#ï¼ˆæ­¤æ—¶æ”¹ç”± .env ä¸­çš„ Key ç”Ÿæ•ˆï¼Œè§ agent/weather_agent.pyï¼‰ã€‚
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# å†…ç½®ç¤ºä¾‹æ•°æ®çš„è·¯å¾„ï¼ˆåŸºäºŽæœ¬æ–‡ä»¶æ‰€åœ¨ç›®å½•ï¼Œé¿å…å·¥ä½œç›®å½•å˜åŒ–å¯¼è‡´æ‰¾ä¸åˆ°ï¼‰
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_weather.csv")


@st.cache_data
def load_sample_data(path):
    """è¯»å–å†…ç½®ç¤ºä¾‹æ•°æ®ï¼ˆå¸¦ç¼“å­˜ï¼Œé¿å…æ¯æ¬¡åˆ·æ–°éƒ½é‡å¤è¯»ç›˜ï¼‰ã€‚"""
    return pd.read_csv(path)


# ---------------- é¡µé¢åŸºç¡€è®¾ç½® ----------------
st.set_page_config(
    page_title="WeatherAgent",
    layout="wide",
)

st.title("WeatherAgent")
st.markdown("### AIæ°”è±¡é¢„æŠ¥æ™ºèƒ½åˆ†æžä¸Žè¯¯å·®è®¢æ­£ç³»ç»Ÿ")

st.markdown("---")

# ---------------- â‘  é¡¹ç›®æ¦‚è§ˆ ----------------
st.header("â‘  é¡¹ç›®æ¦‚è§ˆ")
st.markdown(
    """
    æœ¬é¡¹ç›®å°è¯•å°†**æ°”è±¡æ•°æ®åˆ†æžã€æœºå™¨å­¦ä¹ å’Œ LLM Agent** ç›¸ç»“åˆï¼Œ
    ç”¨äºŽæ°”è±¡é¢„æŠ¥è¯¯å·®åˆ†æžä¸ŽåŽå¤„ç†å®žéªŒã€‚

    **ä¸»è¦åŠŸèƒ½ï¼š**
    1. æ°”è±¡æ•°æ®è´¨é‡æ£€æŸ¥
    2. é¢„æŠ¥è¯¯å·®è¯„ä»·
    3. MAEã€RMSEã€Bias è®¡ç®—
    4. Random Forest è¯¯å·®è®¢æ­£
    5. è®¢æ­£å‰åŽæ•ˆæžœå¯¹æ¯”
    6. ç‰¹å¾é‡è¦æ€§åˆ†æž
    7. DeepSeek AI æ°”è±¡åˆ†æž Agent
    """
)

# æ–¹æ³•è¯´æ˜Žï¼ˆå¯å±•å¼€ï¼‰
with st.expander("æ–¹æ³•è¯´æ˜Ž"):
    st.markdown(
        """
        **1. é¢„æŠ¥è¯¯å·®å®šä¹‰**
        `error = forecast - observed`ï¼ˆé¢„æŠ¥å€¼å‡è§‚æµ‹å€¼ï¼‰
        - error > 0ï¼šé¢„æŠ¥åé«˜ï¼ˆé«˜ä¼°ï¼‰ï¼›error < 0ï¼šé¢„æŠ¥åä½Žï¼ˆä½Žä¼°ï¼‰

        **2. é¢„æŠ¥è¯„ä»·æŒ‡æ ‡**
        - MAE = mean(|error|)
        - RMSE = sqrt(mean(errorÂ²))
        - Bias = mean(error)

        **3. æœºå™¨å­¦ä¹ è¯¯å·®è®¢æ­£**
        ä½¿ç”¨ Random Forest å­¦ä¹ ã€Œè®¢æ­£é‡ã€`correction = observed - forecast`ï¼š
        `predicted_correction = f(forecast, humidity, pressure, wind_speed, hour, month)`
        ç„¶åŽ `corrected_forecast = forecast + predicted_correction`

        **4. éªŒè¯æ–¹å¼ï¼šæ»šåŠ¨çª—å£äº¤å‰éªŒè¯**
        è®­ç»ƒçª—å£é€æŠ˜æ‰©å¤§ã€æµ‹è¯•æ®µä¾æ¬¡åŽç§»ä¸”äº’ä¸é‡å ï¼ˆå…± 3 æŠ˜ï¼Œæ¯æŠ˜æµ‹è¯•çº¦ 20% æ•°æ®ï¼‰ï¼›
        æ¯ä¸€æŠ˜çš„è®­ç»ƒæ•°æ®å…¨éƒ¨ä½äºŽæµ‹è¯•æ•°æ®ä¹‹å‰ï¼Œæ— æ•°æ®æ³„æ¼ï¼›
        æœ€ç»ˆæ±‡æ€»å„æŠ˜æŒ‡æ ‡ä¸ºã€Œå‡å€¼ Â± æ ‡å‡†å·®ã€ï¼Œåæ˜ è®¢æ­£æ•ˆæžœçš„ç¨³å®šæ€§ã€‚
        """
    )

st.markdown("---")

# ---------------- â‘¡ æ•°æ®åŸºæœ¬æƒ…å†µ ----------------
st.header("â‘¡ æ•°æ®åŸºæœ¬æƒ…å†µ")
st.markdown("ä¸Šä¼ ä½ è‡ªå·±çš„ CSV æ•°æ®æ–‡ä»¶ï¼›ä¸ä¸Šä¼ åˆ™è‡ªåŠ¨ä½¿ç”¨å†…ç½®ç¤ºä¾‹æ•°æ®ã€‚")

uploaded_file = st.file_uploader("é€‰æ‹© CSV æ–‡ä»¶ï¼ˆå¯é€‰ï¼‰", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source = "ç”¨æˆ·ä¸Šä¼ æ–‡ä»¶"
else:
    df = load_sample_data(SAMPLE_PATH)
    data_source = "å†…ç½®ç¤ºä¾‹æ•°æ® sample_weather.csv"

st.success(f"å½“å‰æ•°æ®æ¥æºï¼š{data_source}")

col1, col2 = st.columns(2)
col1.metric("æ•°æ®è¡Œæ•°", len(df))
col2.metric("æ•°æ®åˆ—æ•°", len(df.columns))

st.subheader("æ•°æ®é¢„è§ˆ")
st.dataframe(df.head(10))

# å¼€å§‹åŸºç¡€åˆ†æžæŒ‰é’®
if st.button("å¼€å§‹æ°”è±¡æ•°æ®åˆ†æž", type="primary"):
    st.session_state["analyzed"] = True

if st.session_state.get("analyzed", False):
    st.markdown("---")

    # ---------------- â‘¢ æ•°æ®è´¨é‡æ£€æŸ¥ ----------------
    st.header("â‘¢ æ•°æ®è´¨é‡æ£€æŸ¥")

    report = check_data(df)

    if report["ok"]:
        st.success("æ•°æ®åŸºç¡€æ£€æŸ¥é€šè¿‡ï¼šå¿…è¦å­—æ®µé½å…¨ï¼Œæ—¶é—´å­—æ®µå¯æ­£å¸¸è§£æžã€‚")
    else:
        st.error("æ•°æ®å­˜åœ¨é—®é¢˜ï¼Œè¯·æŸ¥çœ‹ä¸‹æ–¹æ˜Žç»†ã€‚")

    col1, col2, col3 = st.columns(3)
    col1.metric("é‡å¤è¡Œæ•°", report["duplicate_rows"])
    col2.metric("ç¼ºå¤±å€¼æ€»æ•°", sum(report["missing_values"].values()))
    col3.metric("å¼‚å¸¸å€¼æ€»æ•°", sum(report["outliers"].values()))

    st.subheader("å¿…è¦å­—æ®µ")
    if report["missing_fields"]:
        st.error("ç¼ºå¤±å­—æ®µï¼š" + "ã€".join(report["missing_fields"]))
    else:
        st.success("æ‰€æœ‰å¿…è¦å­—æ®µå‡å­˜åœ¨ã€‚")

    st.subheader("æ—¶é—´å­—æ®µ")
    if report["time_ok"]:
        st.success("time å­—æ®µå¯æ­£å¸¸è§£æžä¸ºæ—¥æœŸæ—¶é—´ã€‚")
    else:
        st.error(report["time_issue"])

    st.subheader("å­—æ®µç±»åž‹")
    st.write(report["dtypes"])
    for issue in report["type_issues"]:
        st.error(issue)

    st.subheader("ç¼ºå¤±å€¼ç»Ÿè®¡")
    st.write(report["missing_values"])

    st.subheader("å¼‚å¸¸å€¼ç»Ÿè®¡ï¼ˆä¾æ®ç‰©ç†åˆç†èŒƒå›´ï¼‰")
    st.write(report["outliers"])

    st.markdown("---")

    # ---------------- â‘£ åŽŸå§‹é¢„æŠ¥æ£€éªŒ ----------------
    st.header("â‘£ åŽŸå§‹é¢„æŠ¥æ£€éªŒ")
    st.markdown(
        """
        ç”± Python çœŸå®žè®¡ç®—ä¸‰ä¸ªè¯¯å·®æŒ‡æ ‡ï¼ˆ`error = é¢„æŠ¥ - å®žå†µ`ï¼‰ï¼š
        - **MAE**ï¼ˆå¹³å‡ç»å¯¹è¯¯å·®ï¼‰ï¼š`mean(|error|)`
        - **RMSE**ï¼ˆå‡æ–¹æ ¹è¯¯å·®ï¼‰ï¼š`sqrt(mean(errorÂ²))`
        - **Bias**ï¼ˆå¹³å‡åå·®ï¼‰ï¼š`mean(error)`ï¼Œæ­£å€¼è¡¨ç¤ºé¢„æŠ¥ç³»ç»Ÿæ€§åé«˜
        """
    )

    if "observed_temperature" not in df.columns or "forecast_temperature" not in df.columns:
        st.warning("ç¼ºå°‘è§‚æµ‹æ¸©åº¦æˆ–é¢„æŠ¥æ¸©åº¦å­—æ®µï¼Œæ— æ³•è¿›è¡Œè¯¯å·®æ£€éªŒã€‚")
    else:
        metrics = evaluate_forecast(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE å¹³å‡ç»å¯¹è¯¯å·®", f"{metrics['MAE']:.3f} Â°C")
        col2.metric("RMSE å‡æ–¹æ ¹è¯¯å·®", f"{metrics['RMSE']:.3f} Â°C")
        col3.metric("Bias å¹³å‡åå·®", f"{metrics['Bias']:.3f} Â°C")

        # è§‚æµ‹ vs åŽŸå§‹é¢„æŠ¥æ›²çº¿
        st.subheader("è§‚æµ‹æ¸©åº¦ vs åŽŸå§‹é¢„æŠ¥æ¸©åº¦")
        time = pd.to_datetime(df["time"], errors="coerce")
        observed = pd.to_numeric(df["observed_temperature"], errors="coerce")
        forecast = pd.to_numeric(df["forecast_temperature"], errors="coerce")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=observed, mode="lines", name="Observed"))
        fig.add_trace(go.Scatter(x=time, y=forecast, mode="lines", name="Raw Forecast"))
        fig.update_layout(
            title="è§‚æµ‹æ¸©åº¦ vs åŽŸå§‹é¢„æŠ¥æ¸©åº¦",
            xaxis_title="æ—¶é—´",
            yaxis_title="Temperature (Â°C)",
            legend_title="å›¾ä¾‹",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

# ---------------- æœºå™¨å­¦ä¹ è®¢æ­£æŒ‰é’® ----------------
st.markdown("---")
if st.button("å¼€å§‹æœºå™¨å­¦ä¹ è®¢æ­£", type="primary"):
    st.session_state["ml_done"] = True

if st.session_state.get("ml_done", False):
    # ---------------- â‘¤ æœºå™¨å­¦ä¹ è¯¯å·®è®¢æ­£ ----------------
    st.header("â‘¤ æœºå™¨å­¦ä¹ è¯¯å·®è®¢æ­£ï¼ˆæ»šåŠ¨çª—å£äº¤å‰éªŒè¯ï¼‰")
    st.markdown(
        """
        ä½¿ç”¨ **Random Forest** å­¦ä¹ é¢„æŠ¥çš„ã€Œè®¢æ­£é‡ã€`correction = observed - forecast`ï¼Œ
        å†ç”¨ `corrected = forecast + predicted_correction` å®Œæˆè®¢æ­£ã€‚

        ç‰¹å¾ï¼šé¢„æŠ¥æ¸©åº¦ã€æ¹¿åº¦ã€æ°”åŽ‹ã€é£Žé€Ÿã€å°æ—¶ã€æœˆä»½ã€‚

        éªŒè¯æ–¹å¼ï¼š**æ»šåŠ¨çª—å£äº¤å‰éªŒè¯**ï¼ˆrolling window cross-validationï¼‰â€”â€”
        è®­ç»ƒçª—å£é€æŠ˜æ‰©å¤§ã€æµ‹è¯•æ®µä¾æ¬¡åŽç§»ä¸”äº’ä¸é‡å ï¼Œå…± 3 æŠ˜ï¼›
        æ¯æŠ˜ç‹¬ç«‹è®­ç»ƒå¹¶æ£€éªŒï¼Œæœ€ç»ˆæ±‡æ€»ä¸º**å‡å€¼ Â± æ ‡å‡†å·®**ï¼Œ
        ä»¥ä½“çŽ°è®¢æ­£æ•ˆæžœåœ¨ä¸åŒæ—¶é—´æ®µçš„ç¨³å®šæ€§ï¼Œè€Œéžå•æ¬¡åˆ’åˆ†çš„å¶ç„¶ç»“æžœã€‚
        æ¯ä¸€æŠ˜çš„è®­ç»ƒæ•°æ®å…¨éƒ¨ä½äºŽæµ‹è¯•æ•°æ®ä¹‹å‰ï¼Œä¸å­˜åœ¨æ•°æ®æ³„æ¼ã€‚
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
        st.warning("ç¼ºå°‘å¿…è¦å­—æ®µï¼Œæ— æ³•è¿›è¡Œæœºå™¨å­¦ä¹ è®¢æ­£ï¼š" + "ã€".join(missing_ml))
        st.session_state.pop("ml_result", None)
    else:
        with st.spinner("æ­£åœ¨è¿›è¡Œæ»šåŠ¨çª—å£äº¤å‰éªŒè¯ï¼ˆé€æŠ˜è®­ç»ƒéšæœºæ£®æž—ï¼‰â€¦â€¦"):
            ml_result = machine_learning_correction(df)

        st.session_state["ml_result"] = ml_result

        st.success(
            f"éªŒè¯å®Œæˆï¼šå…± {ml_result['n_splits']} æŠ˜ï¼Œ"
            f"æ€»æœ‰æ•ˆæ ·æœ¬ {ml_result['total_size']} æ¡ï¼Œ"
            f"æ¯æŠ˜æµ‹è¯• {ml_result['test_len_per_fold']} æ¡"
            f"ï¼ˆç¬¬ 1 æŠ˜è®­ç»ƒ {ml_result['initial_train_size']} æ¡ï¼Œè®­ç»ƒçª—å£é€æŠ˜æ‰©å¤§ï¼‰ã€‚"
        )

        folds = ml_result["folds"]
        raw_mean = ml_result["raw_mean"]
        cor_mean = ml_result["corrected_mean"]
        cor_std = ml_result["corrected_std"]

        # æ¯ä¸€æŠ˜çš„è¯¦ç»†ç»“æžœ
        st.subheader(f"æ¯ä¸€æŠ˜çš„æ£€éªŒç»“æžœï¼ˆå…± {ml_result['n_splits']} æŠ˜ï¼‰")
        folds_df = pd.DataFrame(
            [
                {
                    "æŠ˜": f["fold"],
                    "è®­ç»ƒæ ·æœ¬": f["train_size"],
                    "æµ‹è¯•æ ·æœ¬": f["test_size"],
                    "æµ‹è¯•æ—¶æ®µ": f"{f['test_start']} ~ {f['test_end']}",
                    "åŽŸå§‹ MAE": round(f["raw"]["MAE"], 3),
                    "è®¢æ­£åŽ MAE": round(f["corrected"]["MAE"], 3),
                    "åŽŸå§‹ RMSE": round(f["raw"]["RMSE"], 3),
                    "è®¢æ­£åŽ RMSE": round(f["corrected"]["RMSE"], 3),
                    "åŽŸå§‹ Bias": round(f["raw"]["Bias"], 3),
                    "è®¢æ­£åŽ Bias": round(f["corrected"]["Bias"], 3),
                }
                for f in folds
            ]
        )
        st.dataframe(folds_df.set_index("æŠ˜"))

        # æ¯æŠ˜ MAE å¯¹æ¯”æŸ±çŠ¶å›¾
        fig_folds = go.Figure()
        fig_folds.add_trace(
            go.Bar(
                x=folds_df["æŠ˜"].astype(str),
                y=folds_df["åŽŸå§‹ MAE"],
                name="åŽŸå§‹é¢„æŠ¥ MAE",
            )
        )
        fig_folds.add_trace(
            go.Bar(
                x=folds_df["æŠ˜"].astype(str),
                y=folds_df["è®¢æ­£åŽ MAE"],
                name="è®¢æ­£åŽ MAE",
            )
        )
        fig_folds.update_layout(
            title="å„æŠ˜ MAE å¯¹æ¯”ï¼ˆæ»šåŠ¨çª—å£äº¤å‰éªŒè¯ï¼‰",
            xaxis_title="æŠ˜",
            yaxis_title="MAE (Â°C)",
            barmode="group",
            legend_title="å›¾ä¾‹",
            height=350,
        )
        st.plotly_chart(fig_folds, width="stretch")

        # æ•´ä½“æŒ‡æ ‡ï¼šå„æŠ˜å‡å€¼ Â± æ ‡å‡†å·®
        st.subheader("æ•´ä½“è¡¨çŽ°ï¼ˆå„æŠ˜å‡å€¼ Â± æ ‡å‡†å·®ï¼‰")
        c1, c2 = st.columns(2)
        c1.metric("åŽŸå§‹é¢„æŠ¥ MAEï¼ˆå‡å€¼ï¼‰", f"{raw_mean['MAE']:.3f} Â°C")
        c2.metric(
            "è®¢æ­£åŽ MAEï¼ˆå‡å€¼ Â± æ ‡å‡†å·®ï¼‰",
            f"{cor_mean['MAE']:.3f} Â± {cor_std['MAE']:.3f} Â°C",
        )

        c1, c2 = st.columns(2)
        c1.metric("åŽŸå§‹é¢„æŠ¥ RMSEï¼ˆå‡å€¼ï¼‰", f"{raw_mean['RMSE']:.3f} Â°C")
        c2.metric(
            "è®¢æ­£åŽ RMSEï¼ˆå‡å€¼ Â± æ ‡å‡†å·®ï¼‰",
            f"{cor_mean['RMSE']:.3f} Â± {cor_std['RMSE']:.3f} Â°C",
        )

        c1, c2 = st.columns(2)
        c1.metric("åŽŸå§‹é¢„æŠ¥ Biasï¼ˆå‡å€¼ï¼‰", f"{raw_mean['Bias']:.3f} Â°C")
        c2.metric(
            "è®¢æ­£åŽ Biasï¼ˆå‡å€¼ Â± æ ‡å‡†å·®ï¼‰",
            f"{cor_mean['Bias']:.3f} Â± {cor_std['Bias']:.3f} Â°C",
        )

        # æ”¹å–„çŽ‡ï¼ˆå„æŠ˜å‡å€¼ï¼Œå¦‚å®žæ˜¾ç¤ºï¼Œå¯èƒ½ä¸ºè´Ÿï¼‰
        c1, c2 = st.columns(2)
        c1.metric(
            "MAE æ”¹å–„çŽ‡ï¼ˆå„æŠ˜å‡å€¼ï¼‰",
            f"{ml_result['mae_improve_mean']:.2f} %",
        )
        c2.metric(
            "RMSE æ”¹å–„çŽ‡ï¼ˆå„æŠ˜å‡å€¼ï¼‰",
            f"{ml_result['rmse_improve_mean']:.2f} %",
        )

        st.caption(
            "æ ‡å‡†å·®è¶Šå°ï¼Œè¯´æ˜Žè®¢æ­£æ•ˆæžœåœ¨ä¸åŒæ—¶é—´æ®µè¶Šç¨³å®šï¼›"
            "ã€Œå‡å€¼ Â± æ ‡å‡†å·®ã€å…±åŒåæ˜ æ¨¡åž‹æ•ˆæžœçš„å…¸åž‹æ°´å¹³ä¸Žæ³¢åŠ¨èŒƒå›´ã€‚"
        )

        # ç‰¹å¾é‡è¦æ€§ï¼ˆå„æŠ˜å¹³å‡ï¼Œè¡¨æ ¼ + æŸ±çŠ¶å›¾ï¼‰
        st.subheader("Random Forest ç‰¹å¾é‡è¦æ€§ï¼ˆå„æŠ˜å¹³å‡ï¼‰")
        importance_df = pd.DataFrame(
            list(ml_result["importance"].items()),
            columns=["ç‰¹å¾", "é‡è¦æ€§"],
        )
        st.dataframe(importance_df.set_index("ç‰¹å¾"))

        fig_imp = go.Figure(
            go.Bar(x=importance_df["ç‰¹å¾"], y=importance_df["é‡è¦æ€§"])
        )
        fig_imp.update_layout(
            title="Random Forest ç‰¹å¾é‡è¦æ€§",
            xaxis_title="ç‰¹å¾",
            yaxis_title="é‡è¦æ€§",
            height=350,
        )
        st.plotly_chart(fig_imp, width="stretch")

    st.markdown("---")

    # ---------------- â‘¥ å¯è§†åŒ–åˆ†æž ----------------
    st.header("â‘¥ å¯è§†åŒ–åˆ†æž")

    ml_result = st.session_state.get("ml_result")
    if ml_result is None:
        st.info("è¯·å…ˆå®Œæˆæœºå™¨å­¦ä¹ è®¢æ­£ã€‚")
    else:
        test_df = ml_result["test_results"]
        time = pd.to_datetime(test_df["time"], errors="coerce")

        st.subheader("è§‚æµ‹ vs åŽŸå§‹é¢„æŠ¥ vs è®¢æ­£åŽé¢„æŠ¥ï¼ˆæ»šåŠ¨éªŒè¯å„æŠ˜æµ‹è¯•æ®µï¼‰")
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
            title="è®¢æ­£å‰åŽæ¸©åº¦å¯¹æ¯”ï¼ˆæ»šåŠ¨éªŒè¯å„æŠ˜æµ‹è¯•æ®µï¼‰",
            xaxis_title="æ—¶é—´",
            yaxis_title="Temperature (Â°C)",
            legend_title="å›¾ä¾‹",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("åŽŸå§‹è¯¯å·® vs è®¢æ­£åŽè¯¯å·®")
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
            title="é¢„æŠ¥è¯¯å·®å¯¹æ¯”ï¼ˆæ»šåŠ¨éªŒè¯å„æŠ˜æµ‹è¯•æ®µï¼‰",
            xaxis_title="æ—¶é—´",
            yaxis_title="Error (Â°C)",
            legend_title="å›¾ä¾‹",
            height=400,
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    # ---------------- å®žéªŒç»“æžœæ€»ç»“ï¼ˆè‡ªåŠ¨ç”Ÿæˆï¼‰ ----------------
    st.header("å®žéªŒç»“æžœæ€»ç»“")

    ml_result = st.session_state.get("ml_result")
    if ml_result is None:
        st.info("è¯·å…ˆå®Œæˆæœºå™¨å­¦ä¹ è®¢æ­£ã€‚")
    else:
        raw_mean = ml_result["raw_mean"]
        cor_mean = ml_result["corrected_mean"]
        cor_std = ml_result["corrected_std"]

        summary_df = pd.DataFrame(
            {
                "æŒ‡æ ‡": ["MAE", "RMSE", "Bias"],
                "åŽŸå§‹é¢„æŠ¥ï¼ˆå„æŠ˜å‡å€¼ï¼‰": [
                    round(raw_mean["MAE"], 4),
                    round(raw_mean["RMSE"], 4),
                    round(raw_mean["Bias"], 4),
                ],
                "è®¢æ­£åŽï¼ˆå„æŠ˜å‡å€¼ï¼‰": [
                    round(cor_mean["MAE"], 4),
                    round(cor_mean["RMSE"], 4),
                    round(cor_mean["Bias"], 4),
                ],
                "è®¢æ­£åŽï¼ˆå„æŠ˜æ ‡å‡†å·®ï¼‰": [
                    round(cor_std["MAE"], 4),
                    round(cor_std["RMSE"], 4),
                    round(cor_std["Bias"], 4),
                ],
            }
        ).set_index("æŒ‡æ ‡")
        st.dataframe(summary_df)

        c1, c2 = st.columns(2)
        c1.metric("MAE æ”¹å–„çŽ‡ï¼ˆå„æŠ˜å‡å€¼ï¼‰", f"{ml_result['mae_improve_mean']:.2f} %")
        c2.metric("RMSE æ”¹å–„çŽ‡ï¼ˆå„æŠ˜å‡å€¼ï¼‰", f"{ml_result['rmse_improve_mean']:.2f} %")

        if (
            cor_mean["MAE"] < raw_mean["MAE"]
            and cor_mean["RMSE"] < raw_mean["RMSE"]
        ):
            st.success(
                f"æ»šåŠ¨çª—å£äº¤å‰éªŒè¯ï¼ˆ{ml_result['n_splits']} æŠ˜ï¼‰ç»“æžœæ˜¾ç¤ºï¼š"
                "è®¢æ­£åŽå„æŠ˜å¹³å‡è¯¯å·®ä½ŽäºŽåŽŸå§‹é¢„æŠ¥ï¼Œ"
                "è¯´æ˜Žè¯¥æ¨¡åž‹åœ¨å½“å‰æ¼”ç¤ºæ•°æ®çš„ä¸åŒæ—¶é—´æ®µä¸Šå‡å…·æœ‰ä¸€å®šçš„è¯¯å·®è®¢æ­£æ•ˆæžœï¼›"
                "å„æŠ˜ç»“æžœçš„å…·ä½“æ³¢åŠ¨è¯·è§ä¸Šæ–¹æ ‡å‡†å·®ã€‚"
            )
        else:
            st.warning(
                "æœ¬æ¬¡æœºå™¨å­¦ä¹ è®¢æ­£æœªç¨³å®šé™ä½Žå„æŠ˜æµ‹è¯•æ®µè¯¯å·®ï¼Œ"
                "è¯´æ˜Žå½“å‰ç‰¹å¾å’Œæ¨¡åž‹è®¾ç½®ä»æœ‰è¿›ä¸€æ­¥ä¼˜åŒ–ç©ºé—´ã€‚"
            )

        st.info(
            "ä»¥ä¸Šç»“æžœåŸºäºŽæ»šåŠ¨çª—å£äº¤å‰éªŒè¯ï¼ˆå„æŠ˜æµ‹è¯•æ®µäº’ä¸é‡å ã€"
            "è®­ç»ƒæ•°æ®å…¨éƒ¨ä½äºŽæµ‹è¯•æ•°æ®ä¹‹å‰ï¼‰ï¼Œ"
            "ä»…ä»£è¡¨å½“å‰æ¼”ç¤ºæ•°æ®ä¸Šçš„å®žéªŒç»“æžœï¼Œ"
            "ä¸èƒ½ç›´æŽ¥ä»£è¡¨æ¨¡åž‹åœ¨å…¶ä»–æ—¶é—´ã€åœ°ç‚¹æˆ–å¤©æ°”æ¡ä»¶ä¸‹çš„å®žé™…é¢„æŠ¥èƒ½åŠ›ã€‚"
        )

# ---------------- â‘¦ AI æ°”è±¡é¢„æŠ¥åˆ†æž Agent ----------------
st.markdown("---")
st.header("â‘¦ AI æ°”è±¡é¢„æŠ¥åˆ†æž Agent")

st.markdown(
    """
    AI Agent è´Ÿè´£ç†è§£ç”¨æˆ·è‡ªç„¶è¯­è¨€é—®é¢˜ï¼Œå¹¶è°ƒç”¨ Python æ°”è±¡åˆ†æžå·¥å…·èŽ·å–çœŸå®žè®¡ç®—ç»“æžœï¼Œ
    å†ç”± DeepSeek å¯¹ç»“æžœè¿›è¡Œè§£é‡Šã€‚

    **åˆ†å·¥ï¼š**
    - Python è´Ÿè´£ï¼šæ•°æ®è¯»å–ã€è´¨é‡æ£€æŸ¥ã€MAE/RMSE/Bias è®¡ç®—ã€éšæœºæ£®æž—è®¢æ­£ç­‰çœŸå®žè®¡ç®—ã€‚
    - DeepSeek è´Ÿè´£ï¼šç†è§£é—®é¢˜ã€é€‰æ‹©å·¥å…·ã€è§£é‡Šç»“æžœï¼ˆä¸è‡ªå·±è®¡ç®—æ•°å€¼ï¼‰ã€‚
    """
)

st.markdown(
    """
    **æŽ¨èé—®é¢˜ï¼š**
    1. åŽŸå§‹é¢„æŠ¥æ•ˆæžœæ€Žä¹ˆæ ·ï¼Ÿ
    2. æœºå™¨å­¦ä¹ è®¢æ­£åŽæ”¹å–„äº†å¤šå°‘ï¼Ÿ
    3. å“ªä¸ªç‰¹å¾å¯¹è¯¯å·®è®¢æ­£æœ€é‡è¦ï¼Ÿ
    4. ä¸ºä»€ä¹ˆè®¢æ­£åŽ Bias ä¼šå‘ç”Ÿå˜åŒ–ï¼Ÿ
    5. æ»šåŠ¨çª—å£äº¤å‰éªŒè¯å„æŠ˜çš„è®¢æ­£æ•ˆæžœç¨³å®šå—ï¼Ÿ
    6. è¯·å…¨é¢åˆ†æžå½“å‰å®žéªŒç»“æžœåŠå…¶å±€é™æ€§ã€‚
    """
)

# åˆå§‹åŒ–èŠå¤©åŽ†å²
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# æ˜¾ç¤ºåŽ†å²æ¶ˆæ¯
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# èŠå¤©è¾“å…¥æ¡†
if prompt := st.chat_input("è¯·è¾“å…¥ä½ çš„æ°”è±¡åˆ†æžé—®é¢˜..."):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("æ­£åœ¨åˆ†æžâ€¦â€¦"):
            agent = WeatherAgent(df)
            reply = agent.chat(prompt)
        st.markdown(reply)

    st.session_state["chat_history"].append({"role": "assistant", "content": reply})

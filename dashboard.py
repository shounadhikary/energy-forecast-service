import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict import predict_one
from config import PROCESSED_DATA_PATH, DEFAULT_DAYOFYEAR
from live_data import fetch_recent_demand

st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")


@st.cache_data(ttl=900)  # cache for 15 minutes, then refetch
def load_history():
    """Fetch recent live demand from EIA."""
    df = fetch_recent_demand(hours=400)
    return df


def build_features_for_time(dt, recent):
    """Build a feature dict for a given datetime using recent history."""
    return {
        "hour": dt.hour,
        "dayofweek": dt.weekday(),
        "month": dt.month,
        "dayofyear": dt.timetuple().tm_yday,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "lag_1h": recent["consumption"].iloc[-1],
        "lag_24h": recent["consumption"].iloc[-24],
        "lag_168h": recent["consumption"].iloc[-168],
        "roll_mean_24h": recent["consumption"].iloc[-24:].mean(),
        "roll_std_24h": recent["consumption"].iloc[-24:].std(),
    }


@st.cache_data(ttl=900)
def compute_forecast(recent):
    """Roll the model forward 24 hours. Cached so it only recomputes when the
    underlying history changes, not on every slider/widget interaction."""
    last_time = recent["Datetime"].iloc[-1]
    working = recent.copy()
    future_times, future_preds = [], []
    for i in range(1, 25):
        next_time = last_time + pd.Timedelta(hours=i)
        feats = build_features_for_time(next_time, working)
        pred = predict_one(feats)
        future_times.append(next_time)
        future_preds.append(pred)
        # Append the prediction so the next step can use it as a lag
        working = pd.concat([working, pd.DataFrame({
            "Datetime": [next_time], "consumption": [pred]
        })], ignore_index=True)
    return future_times, future_preds




df = load_history()
recent = df.tail(500).reset_index(drop=True)

# ---------- Header + live clock ----------
header_col, btn_col = st.columns([4, 1])
with header_col:
    st.title("⚡ Energy Consumption Forecasting")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with btn_col:
    if st.button("🔄 Refresh"):
        st.rerun()

# ---------- KPI cards ----------
latest = recent["consumption"].iloc[-1]
peak = recent["consumption"].max()
avg = recent["consumption"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Current Load (MW)", f"{latest:,.0f}")
c2.metric("Recent Peak (MW)", f"{peak:,.0f}")
c3.metric("Recent Average (MW)", f"{avg:,.0f}")

# ---------- Recent trend chart ----------
st.subheader("Recent Consumption Trend")
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=recent["Datetime"], y=recent["consumption"],
    mode="lines", name="Consumption", line=dict(color="#2563eb"),
))
fig_trend.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_trend, width='stretch')

# ---------- Next 24-hour forecast ----------
st.subheader("Next 24-Hour Forecast")
future_times, future_preds = compute_forecast(recent)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=recent["Datetime"].tail(48), y=recent["consumption"].tail(48),
    mode="lines", name="History", line=dict(color="#9ca3af"),
))
fig_fc.add_trace(go.Scatter(
    x=future_times, y=future_preds,
    mode="lines+markers", name="Forecast", line=dict(color="#16a34a"),
))
fig_fc.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_fc, width='stretch')

# ---------- Interactive prediction ----------
st.subheader("Try Your Own Prediction")
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour of day", 0, 23, 14)
    dayofweek = st.slider("Day of week (0=Mon)", 0, 6, 2)
    month = st.slider("Month", 1, 12, 6)
with col2:
    lag_1h = st.number_input("Consumption 1h ago (MW)", value=float(recent["consumption"].iloc[-1]))
    lag_24h = st.number_input("Consumption 24h ago (MW)", value=float(recent["consumption"].iloc[-24]))
    roll_mean = st.number_input("24h rolling mean (MW)", value=float(recent["consumption"].iloc[-24:].mean()))

manual_feats = {
    "hour": hour, "dayofweek": dayofweek, "month": month,
    "dayofyear": DEFAULT_DAYOFYEAR, "is_weekend": 1 if dayofweek >= 5 else 0,
    "lag_1h": lag_1h, "lag_24h": lag_24h,
    "lag_168h": recent["consumption"].iloc[-168],
    "roll_mean_24h": roll_mean,
    "roll_std_24h": recent["consumption"].iloc[-24:].std(),
}
manual_pred = predict_one(manual_feats)
st.metric("Predicted Consumption (MW)", f"{manual_pred:,.2f}")
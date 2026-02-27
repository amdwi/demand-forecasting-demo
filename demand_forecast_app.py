import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="Demand Forecasting Demo", layout="wide")
st.title("📦 Product Demand Forecasting")
st.markdown("An interactive ML demo using **Meta Prophet** + **Plotly**")

# ── Sidebar Controls ─────────────────────────────────────────
st.sidebar.header("🎛️ Controls")

product = st.sidebar.selectbox(
    "Select Product Category",
    ["Electronics", "Clothing", "Groceries", "Furniture"]
)

horizon = st.sidebar.slider("Forecast Horizon (weeks)", 4, 52, 12)
seasonality = st.sidebar.select_slider(
    "Seasonality Strength", options=["Low", "Medium", "High"], value="Medium"
)
show_components = st.sidebar.checkbox("Show Trend & Seasonality Breakdown", value=True)
show_metrics = st.sidebar.checkbox("Show Model Accuracy Metrics", value=True)

# ── Data Generation ───────────────────────────────────────────
@st.cache_data
def generate_data(product, seasonality):
    np.random.seed({"Electronics": 1, "Clothing": 2, "Groceries": 3, "Furniture": 4}[product])
    periods = 104
    dates = pd.date_range(start="2022-01-01", periods=periods, freq="W")
    trend = {"Electronics": 100, "Clothing": 60, "Groceries": 30, "Furniture": 40}[product]
    base = {"Electronics": 300, "Clothing": 200, "Groceries": 500, "Furniture": 150}[product]
    s_amp = {"Low": 20, "Medium": 60, "High": 120}[seasonality]
    noise = {"Low": 5, "Medium": 20, "High": 40}[seasonality]
    sales = (
        base
        + np.sin(np.linspace(0, 4 * np.pi, periods)) * s_amp
        + np.linspace(0, trend, periods)
        + np.random.normal(0, noise, periods)
    )
    return pd.DataFrame({"ds": dates, "y": np.maximum(sales, 0)})

df = generate_data(product, seasonality)

# ── Train / Test Split ────────────────────────────────────────
test_size = 12
train_df = df[:-test_size]
test_df = df[-test_size:]

# ── Model Training ────────────────────────────────────────────
with st.spinner("Training forecast model..."):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=test_size + horizon, freq="W")
    forecast = model.predict(future)

# ── Main Forecast Chart ───────────────────────────────────────
st.subheader(f"📈 Forecast for {product} — Next {horizon} Weeks")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["ds"], y=df["y"],
    mode="lines+markers", name="Actual Sales",
    line=dict(color="#4A90D9", width=2), marker=dict(size=4)
))
fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines", name="Forecast",
    line=dict(color="#E87040", width=2, dash="dash")
))
fig.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(232,112,64,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Interval"
))
forecast_start = df["ds"].max()
fig.add_vline(x=forecast_start, line_dash="dot", line_color="gray",
              annotation_text="Forecast Start", annotation_position="top right")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Units Sold",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified", height=450,
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="white")
)
st.plotly_chart(fig, use_container_width=True)

# ── Metrics ───────────────────────────────────────────────────
if show_metrics:
    st.subheader("📊 Model Accuracy (on held-out test set)")
    preds_test = forecast[forecast["ds"].isin(test_df["ds"])]
    mae = mean_absolute_error(test_df["y"].values, preds_test["yhat"].values)
    mape = mean_absolute_percentage_error(test_df["y"].values, preds_test["yhat"].values) * 100
    rmse = np.sqrt(np.mean((test_df["y"].values - preds_test["yhat"].values) ** 2))
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.1f} units")
    col2.metric("MAPE", f"{mape:.1f}%")
    col3.metric("RMSE", f"{rmse:.1f} units")

# ── Components Chart ──────────────────────────────────────────
if show_components:
    st.subheader("🔍 Trend & Seasonality Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        fig_trend = px.line(forecast, x="ds", y="trend", title="Overall Trend",
                            color_discrete_sequence=["#4A90D9"])
        fig_trend.update_layout(height=300, plot_bgcolor="#0e1117",
                                paper_bgcolor="#0e1117", font=dict(color="white"))
        st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        fig_season = px.line(forecast, x="ds", y="yearly", title="Yearly Seasonality",
                             color_discrete_sequence=["#E87040"])
        fig_season.update_layout(height=300, plot_bgcolor="#0e1117",
                                 paper_bgcolor="#0e1117", font=dict(color="white"))
        st.plotly_chart(fig_season, use_container_width=True)

# ── Raw Data Preview ──────────────────────────────────────────
with st.expander("🗂️ View Raw Data"):
    st.dataframe(df.rename(columns={"ds": "Date", "y": "Units Sold"}), use_container_width=True)

st.caption("Demo built with Prophet · Plotly · Streamlit — open source & free")



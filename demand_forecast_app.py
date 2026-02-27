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
    return pd.DataFrame({"ds": dates, "y": np.maximum(sales, 0).round(0).astype(int)})

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
    line=dict(color="#4A90D9", width=2),
    marker=dict(size=4)
))

fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines", name="Forecast",
    line=dict(color="#E87040", width=2, dash="dash")
))

fig.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(0,196,140,0.20)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Interval"
))

# ── Vertical line using add_shape (compatible with all Plotly versions) ──
forecast_start = df["ds"].max().strftime("%Y-%m-%d")

fig.add_shape(
    type="line",
    x0=forecast_start,
    x1=forecast_start,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="gray", dash="dot", width=1.5)
)

fig.add_annotation(
    x=forecast_start,
    y=0.95,
    xref="x", yref="paper",
    text="Forecast Start",
    showarrow=False,
    font=dict(color="gray", size=12),
    bgcolor="rgba(0,0,0,0)"
)

fig.update_layout(
    xaxis_title="Date", yaxis_title="Units Sold",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified", height=450,
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics ───────────────────────────────────────────────────
# Move preds_test OUTSIDE the if block so it's always available
preds_test = forecast[forecast["ds"].isin(test_df["ds"])]

if show_metrics:
    st.subheader("📊 Model Accuracy (on held-out test set)")
    mae = mean_absolute_error(test_df["y"].values, preds_test["yhat"].values)
    mape = mean_absolute_percentage_error(test_df["y"].values, preds_test["yhat"].values) * 100
    rmse = np.sqrt(np.mean((test_df["y"].values - preds_test["yhat"].values) ** 2))
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.1f} units")
    col2.metric("MAPE", f"{mape:.1f}%")
    col3.metric("RMSE", f"{rmse:.1f} units")
    mae = mean_absolute_error(test_df["y"].values, preds_test["yhat"].values)
    mape = mean_absolute_percentage_error(test_df["y"].values, preds_test["yhat"].values) * 100
    rmse = np.sqrt(np.mean((test_df["y"].values - preds_test["yhat"].values) ** 2))
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.1f} units")
    col2.metric("MAPE", f"{mape:.1f}%")
    col3.metric("RMSE", f"{rmse:.1f} units")

st.subheader("🔍 Actual vs Predicted — Last 12 Weeks")

fig_eval = go.Figure()

# Actual real values
fig_eval.add_trace(go.Scatter(
    x=test_df["ds"],
    y=test_df["y"],
    mode="lines+markers",
    name="Actual (Real)",
    line=dict(color="#4A90D9", width=2),
    marker=dict(size=6)
))

# What model predicted
fig_eval.add_trace(go.Scatter(
    x=preds_test["ds"],
    y=preds_test["yhat"],
    mode="lines+markers",
    name="Predicted (Model)",
    line=dict(color="#E87040", width=2, dash="dash"),
    marker=dict(size=6)
))

# Confidence interval for test period
fig_eval.add_trace(go.Scatter(
    x=pd.concat([preds_test["ds"], preds_test["ds"][::-1]]),
    y=pd.concat([preds_test["yhat_upper"], preds_test["yhat_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(0,196,140,0.20)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Interval"
))

fig_eval.update_layout(
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode="x unified",
    height=400,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white")
)

st.plotly_chart(fig_eval, use_container_width=True)

# Show a table comparing actual vs predicted week by week
st.subheader("📋 Week by Week Comparison")
comparison_df = pd.DataFrame({
    "Week": test_df["ds"].dt.strftime("%Y-%m-%d").values,
    "Actual Units": test_df["y"].values,
    "Predicted Units": preds_test["yhat"].round(0).astype(int).values,
    "Error (units)": (test_df["y"].values - preds_test["yhat"].values).round(0).astype(int),
    "Error (%)": ((abs(test_df["y"].values - preds_test["yhat"].values) / test_df["y"].values) * 100).round(1)
})
st.dataframe(comparison_df, use_container_width=True)

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

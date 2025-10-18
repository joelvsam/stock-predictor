import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# ------------------------
# Streamlit App Title
# ------------------------
st.title("Stock Price Prediction Dashboard")
st.write("Predict next-day closing prices using Linear Regression & XGBoost")

# ------------------------
# User Input: Ticker
# ------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# ------------------------
# Fetch Stock Data
# ------------------------
data = yf.download(ticker, period="5y", interval="1d")
df = data.copy()
df.dropna(inplace=True)

# ------------------------
# Feature Engineering
# ------------------------
close_series = df["Close"].squeeze()
df["SMA_20"] = ta.trend.sma_indicator(close_series, window=20)
df["SMA_50"] = ta.trend.sma_indicator(close_series, window=50)
df["EMA_20"] = ta.trend.ema_indicator(close_series, window=20)
df["EMA_50"] = ta.trend.ema_indicator(close_series, window=50)
df["RSI_14"] = ta.momentum.rsi(close_series, window=14)
df["MACD"] = ta.trend.macd(close_series)
df["MACD_Signal"] = ta.trend.macd_signal(close_series)
bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
df["BB_High"] = bollinger.bollinger_hband()
df["BB_Low"] = bollinger.bollinger_lband()
df["BB_Width"] = df["BB_High"] - df["BB_Low"]
df["Return_1D"] = df["Close"].pct_change()
df["Return_5D"] = df["Close"].pct_change(5)
df["Volume_Change"] = df["Volume"].pct_change()
df.dropna(inplace=True)

# ------------------------
# Prepare Features & Target
# ------------------------
features = [
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_High", "BB_Low", "BB_Width",
    "Return_1D", "Return_5D", "Volume_Change"
]
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df["Target"]

# ------------------------
# Train/Test Split
# ------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ------------------------
# Train Models
# ------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ------------------------
# Model Metrics
# ------------------------
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

rmse_lr, r2_lr = calculate_metrics(y_test, y_pred_lr)
rmse_xgb, r2_xgb = calculate_metrics(y_test, y_pred_xgb)

st.subheader("Model Performance")
st.write(f"**Linear Regression** → RMSE = {rmse_lr:.2f}, R² = {r2_lr:.3f}")
st.write(f"**XGBoost** → RMSE = {rmse_xgb:.2f}, R² = {r2_xgb:.3f}")

# ------------------------
# Interactive Plot with Plotly
# ------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test.index, y=y_test.values,
    mode="lines", name="Actual Close", line=dict(color="yellow")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=y_pred_lr,
    mode="lines", name="Linear Regression", line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=y_pred_xgb,
    mode="lines", name="XGBoost", line=dict(color="green")
))

fig.update_layout(
    title=f"{ticker} - Next-Day Close Prediction",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Next-Day Trend Suggestion
# ------------------------
last_close = df["Close"].iloc[-1].item()
pred_next_close_lr = float(y_pred_lr[-1])
pred_next_close_xgb = float(y_pred_xgb[-1])

if pred_next_close_lr > last_close:
    trend_msg = "Price expected to rise."
elif pred_next_close_lr < last_close:
    trend_msg = "Price expected to fall."
else:
    trend_msg = "Price expected to remain stable."

st.subheader("Next-Day Price Trend (Based on Linear Regression)")
st.write(f"Last Close: {last_close:.2f}")
st.write(f"Linear Regression Predicted Next Close: {pred_next_close_lr:.2f}")
st.write(f"XGBoost Predicted Next Close: {pred_next_close_xgb:.2f}")
st.write(trend_msg)

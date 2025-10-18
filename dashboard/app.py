import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Stock Price Prediction Dashboard")
st.write("Predict next-day closing prices using Linear Regression & XGBoost")

# ------------------------
# Ticker input
# ------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# ------------------------
# Fetch data
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

features = [
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_High", "BB_Low", "BB_Width",
    "Return_1D", "Return_5D", "Volume_Change"
]

# Target for next-day price
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df["Target"]

# ------------------------
# Train/Test Split
# ------------------------
split_index = int(len(df)*0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ------------------------
# Train models
# ------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ------------------------
# Metrics
# ------------------------
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

st.subheader("Model Performance")
st.write(f"**Linear Regression** → RMSE = {rmse_lr:.2f}, R² = {r2_lr:.3f}")
st.write(f"**XGBoost** → RMSE = {rmse_xgb:.2f}, R² = {r2_xgb:.3f}")

# ------------------------
# Plot predictions
# ------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual Close", color="black")
plt.plot(y_pred_lr, label="Linear Regression", color="blue")
plt.plot(y_pred_xgb, label="XGBoost", color="green")
plt.title(f"{ticker} - Next-Day Close Prediction")
plt.xlabel("Test Data Points")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)

# ------------------------
# Next-Day Trend Suggestion
# ------------------------
last_close = df["Close"].iloc[-1].item()
pred_next_close_lr = y_pred_lr[-1].item()
pred_next_close_xgb = y_pred_xgb[-1].item()

# Use Linear Regression as main signal
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

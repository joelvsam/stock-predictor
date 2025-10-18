import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# TensorFlow / Keras for LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------
# Streamlit App Title
# ------------------------
st.title("Stock Price Prediction Dashboard")
st.write("Predict next-day closing prices using Linear Regression, XGBoost, and LSTM")

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
# Feature Engineering (for LR & XGB)
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
# Prepare Features & Target (for LR & XGB)
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
# Train/Test Split (shared split index)
# ------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ------------------------
# Train Models: Linear Regression & XGBoost
# ------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ------------------------
# LSTM Pipeline (using Close-only sequences)
# ------------------------
# Scale close prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df[['Close']].values)

# Create sequences of seq_length days to predict the next day
seq_length = 60

def create_sequences(data, seq_len=60):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(data)):
        X_seq.append(data[i - seq_len:i, 0])
        y_seq.append(data[i, 0])  # next day's scaled close
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(scaled_close, seq_length)
# Corresponding dates for each y_seq point
seq_dates = df.index[seq_length:]

# Align split to sequences: subtract seq_length from split_index
seq_split_index = split_index - seq_length
if seq_split_index < 1:
    seq_split_index = int(len(X_seq) * 0.8)

X_seq_train, X_seq_test = X_seq[:seq_split_index], X_seq[seq_split_index:]
y_seq_train, y_seq_test = y_seq[:seq_split_index], y_seq[seq_split_index:]
seq_dates_train, seq_dates_test = seq_dates[:seq_split_index], seq_dates[seq_split_index:]

# Reshape to [samples, timesteps, features]
X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], X_seq_train.shape[1], 1))
X_seq_test = X_seq_test.reshape((X_seq_test.shape[0], X_seq_test.shape[1], 1))

# Build and train LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, verbose=0)

# Predict and inverse transform
y_pred_lstm_scaled = lstm_model.predict(X_seq_test, verbose=0)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_lstm = scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()

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
rmse_lstm, r2_lstm = calculate_metrics(y_test_lstm, y_pred_lstm)

st.subheader("Model Performance")
st.write(f"Linear Regression → RMSE = {rmse_lr:.2f}, R² = {r2_lr:.3f}")
st.write(f"XGBoost → RMSE = {rmse_xgb:.2f}, R² = {r2_xgb:.3f}")
st.write(f"LSTM → RMSE = {rmse_lstm:.2f}, R² = {r2_lstm:.3f}")

# ------------------------
# Interactive Plot with Plotly
# ------------------------
fig = go.Figure()

# Actual (aligned to LR/XGB test)
fig.add_trace(go.Scatter(
    x=y_test.index, y=y_test.values,
    mode="lines", name="Actual Close", line=dict(color="yellow", width=3)
))

# Linear Regression
fig.add_trace(go.Scatter(
    x=y_test.index, y=y_pred_lr,
    mode="lines", name="Linear Regression", line=dict(color="blue", width=3)
))

# XGBoost
fig.add_trace(go.Scatter(
    x=y_test.index, y=y_pred_xgb,
    mode="lines", name="XGBoost", line=dict(color="green", width=3)
))

# LSTM (uses its own test date index)
fig.add_trace(go.Scatter(
    x=seq_dates_test, y=y_pred_lstm,
    mode="lines", name="LSTM", line=dict(color="red", width=3)
))

fig.update_layout(
    title=f"{ticker} - Next-Day Close Prediction",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Next-Day Trend (Neutral, based on Linear Regression)
# ------------------------
last_close = df["Close"].iloc[-1].item()
pred_next_close_lr = float(y_pred_lr[-1])
pred_next_close_xgb = float(y_pred_xgb[-1])

# For LSTM next-day prediction:
# Build the last sequence from the final seq_length closes
last_seq = scaled_close[-seq_length:]
last_seq = last_seq.reshape((1, seq_length, 1))
next_pred_lstm_scaled = lstm_model.predict(last_seq, verbose=0)
pred_next_close_lstm = float(scaler.inverse_transform(next_pred_lstm_scaled.reshape(-1, 1)).flatten()[0])

if pred_next_close_lr > last_close:
    trend_msg = "Predicted trend: upward movement."
elif pred_next_close_lr < last_close:
    trend_msg = "Predicted trend: downward movement."
else:
    trend_msg = "Predicted trend: stable."

st.subheader("Next-Day Price Trend (Based on Linear Regression)")
st.write(f"Last Close: {last_close:.2f}")
st.write(f"Linear Regression Predicted Next Close: {pred_next_close_lr:.2f}")
st.write(f"XGBoost Predicted Next Close: {pred_next_close_xgb:.2f}")
st.write(f"LSTM Predicted Next Close: {pred_next_close_lstm:.2f}")
st.write(trend_msg)

# Stock Price Prediction Dashboard

This project is a web application built with Streamlit that predicts the next-day closing price of a stock using three different models: Linear Regression, XGBoost, and LSTM (Long Short-Term Memory neural network). The app fetches historical stock data using Yahoo Finance, performs feature engineering, trains the models, and provides interactive visualizations and trend predictions.

---

## Features

- Fetch historical stock price data for any ticker symbol.
- Compute technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, and returns.
- Train three predictive models:
  - Linear Regression
  - XGBoost Regression
  - LSTM Neural Network
- Display model performance metrics (RMSE and R²).
- Interactive Plotly charts comparing actual vs predicted prices.
- Next-day price trend prediction using a majority vote of all models.
- User-friendly interface with sidebar glossary and options to select models for visualization.

---

## Requirements

- Python 3.8 or higher
- An internet connection (to fetch stock data and download dependencies)

---

## Installation and Setup Guide

Follow these steps to run the application locally on your machine:

### 1. Install Python

Make sure Python is installed on your system. You can download it from:

https://www.python.org/downloads/

Verify installation by running in your terminal or command prompt:

python --version

### 2. Clone or Download the Repository

Clone this repository or download the source code as a ZIP file and extract it:

git clone <repository-url>
cd <repository-folder>

(Replace <repository-url> with the URL if applicable.)

### 3. Create a Virtual Environment (Optional but Recommended)

It’s good practice to create a virtual environment to manage dependencies:

python -m venv venv

Activate the virtual environment:

- On Windows:

venv\Scripts\activate

- On macOS/Linux:

source venv/bin/activate

### 4. Install Required Python Packages

All required Python libraries and their specific versions are listed in the `requirements.txt` file.

To install the dependencies, run the following command in your terminal or command prompt:

pip install -r requirements.txt

This will install all necessary packages, including:

- streamlit: For creating the web app
- yfinance: To download stock data
- pandas, numpy: Data manipulation
- ta: Technical analysis indicators
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting model
- plotly: Interactive plotting
- tensorflow: Deep learning framework for LSTM
- And other supporting libraries

### 5. Run the Streamlit Application

Start the app by running:

streamlit run app.py

(Replace app.py with your script filename if different.)

This will open a new tab in your web browser showing the Stock Price Prediction Dashboard.

---

## How to Use the App

1. Enter Stock Ticker: In the input box, enter the stock symbol you want to analyze (e.g., AAPL for Apple).

2. Select Start Date: Choose the start date for historical data (default is 5 years ago).

3. View Predictions and Metrics: The app will fetch data, train models, and display performance metrics.

4. Select Models for Visualization: Use the checkboxes to choose which models to display on the interactive plot.

5. Analyze Next-Day Trend: See the majority vote prediction for the next day's stock price movement.

---

<img width="390" height="590" alt="image" src="https://github.com/user-attachments/assets/240ac084-1801-4b59-bcae-cae6901c3aae" />


## Code Structure Overview

- Data Fetching: Uses yfinance to get daily stock price data from Yahoo Finance.
- Feature Engineering: Calculates technical indicators using the ta library.
- Model Training:
  - Linear Regression and XGBoost use engineered features.
  - LSTM uses scaled closing prices and sequences for deep learning.
- Model Evaluation: Computes RMSE and R² scores to evaluate prediction accuracy.
- Visualization: Plotly graphs showing actual and predicted prices.
- Prediction Voting: Combines model outputs to suggest overall trend direction.

---

## Troubleshooting

- If you get an error fetching stock data, verify the ticker symbol and date range.
- Ensure all dependencies are installed with compatible versions.
- If TensorFlow installation is slow or problematic, consult the official TensorFlow installation guide at https://www.tensorflow.org/install.
- On first run, training the LSTM model may take some time depending on your machine.




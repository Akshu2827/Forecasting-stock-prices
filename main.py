import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load and prepare data
# ------------------------------
ticker = "ASHOKLEY.NS"
df_raw = yf.download(ticker, period="5y", auto_adjust=False)[["Adj Close"]]

# Fix: Flatten multi-index columns to single level
df_raw.columns = df_raw.columns.get_level_values(0)

# Now df_raw['Adj Close'] is a Series
df = df_raw.copy()
for i in range(1, 10, 2):
    df[f'lag_{i}'] = df['Adj Close'].shift(i)

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_RSI(df['lag_1'])
df.dropna(inplace=True)

X = df.drop('Adj Close', axis=1)
y = df['Adj Close']

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    return {
    "mae" : mean_absolute_error(y_true, y_pred), 
    "rmse" : np.sqrt(mean_squared_error(y_true, y_pred)), 
    "mape" :(np.abs((y_true - y_pred) / y_true)).mean() * 100, 
    "r2" : r2_score(y_true, y_pred)
    }


split = int(len(df) * 0.5)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_metrics = evaluate(y_test, lr_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plotting the index (Dates) against the values
plt.plot(y_test.index, y_test.values, label='Actual Price', linewidth=2)
plt.plot(y_test.index, lr_pred, label='Predicted Price', linestyle='--')

plt.title(f'Stock {ticker}, Actual vs Predicted Closing Price (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout() 

plt.show()

print(lr_metrics)
print(X, "feature")
print(y,"traget column")

# ------------------------------
# 2. Recursive prediction for next 5 days
# ------------------------------
# Get full price history (original, all actual prices)
prices = df_raw['Adj Close'].tolist()   # Now it's a Series -> .tolist() works
last_date = df_raw.index[-1]

future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
predictions = []

for _ in range(5):
    # Extract required lags from the current price list
    lag_1 = prices[-1]
    lag_3 = prices[-3]
    lag_5 = prices[-5]
    lag_7 = prices[-7]
    lag_9 = prices[-9]

    # Compute RSI on the whole series of lag_1 values (price history)
    lag1_series = pd.Series(prices)
    rsi_val = compute_RSI(lag1_series).iloc[-1]

    # Feature vector
    features = np.array([[lag_1, lag_3, lag_5, lag_7, lag_9, rsi_val]])

    # Predict next price
    next_price = lr.predict(features)[0]
    predictions.append(next_price)

    # Append predicted price to history for next iteration
    prices.append(next_price)

# Display results
print("Next 5 trading day predictions:")
for date, price in zip(future_dates, predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")

# Plot actual + predicted prices
plt.figure(figsize=(10,5))
plt.plot(df_raw.index[-30:], df_raw['Adj Close'].iloc[-30:], label='Actual (last 30 days)', color='blue')
plt.plot(future_dates, predictions, 'ro--', label='Predicted (next 5 days)', markersize=8)
plt.title(f'{ticker} - Actual and Forecasted Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

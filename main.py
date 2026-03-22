import numpy as np
import yfinance as yf 
import pandas as pd

ticker = "ASHOKLEY.NS"

df = yf.download(ticker, period="5y", auto_adjust=False)[["Adj Close"]]

# Create lag features
for i in range(1, 10, 2):
    df[f'lag_{i}'] = df['Adj Close'].shift(i)

# RSI function
def compute_RSI(df, column='lag_1', window=14):
    delta = df[column].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# ✅ Add RSI column
df['RSI_on_lag_1'] = compute_RSI(df, column='lag_1')

# Drop NaNs AFTER all features are created
df.dropna(inplace=True)

# Features and target
X = df.drop("Adj Close", axis=1)
y = df["Adj Close"]

print(X)
print(y)

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

# Rotate the date labels so they don't overlap
plt.xticks(rotation=45)

# Automatically adjusts subplot params so the dates fit in the figure area
plt.tight_layout() 

plt.show()

print(lr_metrics)
print(X, "feature")
print(y,"traget column")

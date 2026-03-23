import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load and prepare data
# ------------------------------
ticker = "ASHOKLEY.NS"
df_raw = yf.download(ticker, period="1y", auto_adjust=False)[["Adj Close"]]

# Flatten multi-index columns
df_raw.columns = df_raw.columns.get_level_values(0)


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

# ------------------------------
# 2. Define evaluation metrics
# ------------------------------
def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": (np.abs((y_true - y_pred) / y_true)).mean() * 100,
        "r2": r2_score(y_true, y_pred)
    }

# ------------------------------
# 3. Walk‑forward validation
# ------------------------------
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

fold_metrics = []
fold_predictions = []  # store (y_test, y_pred) for later plots

print("Walk‑forward validation results:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Pipeline: scaling + linear regression
    pipeline = Pipeline([
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = evaluate(y_test, y_pred)
    fold_metrics.append(metrics)
    fold_predictions.append((y_test, y_pred))
    
    print(f"Fold {fold+1}: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")

# Average metrics across folds
avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
print("\nAverage metrics across folds:")
print(f"MAE: {avg_metrics['mae']:.4f}")
print(f"RMSE: {avg_metrics['rmse']:.4f}")
print(f"MAPE: {avg_metrics['mape']:.2f}%")
print(f"R²: {avg_metrics['r2']:.4f}")

# ------------------------------
# 4. Visualise predictions for each fold
# ------------------------------
fig, axes = plt.subplots(n_splits, 1, figsize=(12, 4*n_splits), sharex=True)
if n_splits == 1:
    axes = [axes]

for i, (y_test, y_pred) in enumerate(fold_predictions):
    ax = axes[i]
    ax.plot(y_test.index, y_test.values, label='Actual', linewidth=2, color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted', linestyle='--', linewidth=2, color='orange')
    ax.set_title(f'Fold {i+1} – Actual vs Predicted (Test period)')
    ax.set_ylabel('Adjusted Close Price')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Optional: summary bar chart of RMSE per fold
plt.figure(figsize=(10, 5))
rmse_vals = [m['rmse'] for m in fold_metrics]
plt.bar(range(1, n_splits+1), rmse_vals, color='skyblue', edgecolor='black')
plt.axhline(y=np.mean(rmse_vals), color='red', linestyle='--', label=f'Mean RMSE = {np.mean(rmse_vals):.4f}')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE per Walk‑forward Fold')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ------------------------------
# 5. Train final model on all data for recursive forecasting
# ------------------------------
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
final_pipeline.fit(X, y)

# ------------------------------
# 6. Recursive prediction for next 5 days
# ------------------------------
prices = df_raw['Adj Close'].tolist()
last_date = df_raw.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

predictions = []
feature_columns = X.columns.tolist()

for _ in range(5):
    if len(prices) < 9:
        print("Error: Not enough historical prices for features.")
        break

    lag_1 = prices[-1]
    lag_3 = prices[-3]
    lag_5 = prices[-5]
    lag_7 = prices[-7]
    lag_9 = prices[-9]

    price_series = pd.Series(prices)
    rsi_val = compute_RSI(price_series).iloc[-1]

    feature_dict = {
        'lag_1': lag_1,
        'lag_3': lag_3,
        'lag_5': lag_5,
        'lag_7': lag_7,
        'lag_9': lag_9,
        'RSI': rsi_val
    }
    features = np.array([[feature_dict[col] for col in feature_columns]])

    next_price = final_pipeline.predict(features)[0]
    predictions.append(next_price)
    prices.append(next_price)

print("\nNext 5 trading day predictions:")
for date, price in zip(future_dates, predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")

# ------------------------------
# 7. Plot last 30 actual days + 5‑day forecast with uncertainty
# ------------------------------
rmse = avg_metrics['rmse']  # Use average RMSE from walk‑forward as error estimate

plt.figure(figsize=(12, 6))
plt.plot(df_raw.index[-30:], df_raw['Adj Close'].iloc[-30:],
         label='Actual (last 30 days)', color='blue', linewidth=2)

# Plot predictions with error bars
plt.errorbar(future_dates, predictions, yerr=rmse, fmt='ro--',
             capsize=5, markersize=8, linewidth=2,
             label=f'Predicted ±1 RMSE ({rmse:.2f})')

# Optional: add shaded region for confidence band
plt.fill_between(future_dates,
                 [p - rmse for p in predictions],
                 [p + rmse for p in predictions],
                 color='red', alpha=0.2, label='±1 RMSE band')

plt.title(f'{ticker} – Actual and 5‑Day Forecast with Uncertainty (RMSE = {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
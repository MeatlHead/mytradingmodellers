import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def calculate_atr(high, low, close, period=10):
    tr = np.maximum.reduce([np.abs(high - low), np.abs(high - close), np.abs(low - close)])
    atr = pd.Series(tr).rolling(window=period).mean()
    return atr

def calculate_super_trend(high, low, close, atr, multiplier=3):
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    super_trend = np.where(close > basic_ub, basic_ub, basic_lb)
    return super_trend

def calculate_fvg(high, low, close):
    fvg = (high + low) / 2 - close
    return fvg

def trading_bot(data, atr_period=10, multiplier=3):
    # Calculate ATR
    atr = calculate_atr(data['High'], data['Low'], data['Close'], atr_period)

    # Calculate Super Trend
    super_trend = calculate_super_trend(data['High'], data['Low'], data['Close'], atr, multiplier)

    # Calculate FVG
    fvg = calculate_fvg(data['High'], data['Low'], data['Close'])

    # Initialize signals
    signal = np.zeros(len(data))
    position = np.zeros(len(data))

    # Trading logic
    for i in range(1, len(data)):
        if fvg[i] > 0 and fvg[i-1] < 0 and data['Close'][i] > data['Close'][i-1] and data['Close'][i-1] > data['Close'][i-2]:
            signal[i] = 1  # Buy signal
        elif super_trend[i] < data['Close'][i]:
            signal[i] = -1  # Sell signal

        # Position management
        if signal[i] == 1:
            position[i] = 1  # Long position
        elif signal[i] == -1:
            position[i] = -1  # Short position

        # Stop loss and take profit
        if position[i] == 1:
            stop_loss = data['Low'][i-1]  # Recent support level
            take_profit = data['High'][i-1] + (data['High'][i-1] - data['Low'][i-1])  # Take profit at next resistance level
        elif position[i] == -1:
            stop_loss = data['High'][i-1]  # Recent resistance level
            take_profit = data['Low'][i-1] - (data['High'][i-1] - data['Low'][i-1])  # Take profit at next support level

    return signal, position

def backtest(data, atr_period=10, multiplier=3):
    signal, position = trading_bot(data, atr_period, multiplier)
    equity = np.zeros(len(data))
    equity[0] = 10000  # Initial equity
    for i in range(1, len(data)):
        if position[i] == 1:
            equity[i] = equity[i-1] * (1 + (data['Close'][i] - data['Close'][i-1]) / data['Close'][i-1])
        elif position[i] == -1:
            equity[i] = equity[i-1] * (1 - (data['Close'][i] - data['Close'][i-1]) / data['Close'][i-1])
        else:
            equity[i] = equity[i-1]
    return equity

def walk_forward_optimization(data, atr_periods, multipliers):
    results = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        for atr_period in atr_periods:
            for multiplier in multipliers:
                equity = backtest(train, atr_period, multiplier)
                final_equity = equity[-1]
                results.append((atr_period, multiplier, final_equity))
    return results

def save_results_to_csv(results, filename):
    df = pd.DataFrame(results, columns=['ATR Period', 'Multiplier', 'Final Equity'])
    df.to_csv(filename, index=False)

def plot_equity_curve(equity, title='Equity Curve'):
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage
try:
    data = yf.download("EURUSD=X", start="2024-01-01", end="2024-05-17")
    if data.empty:
        raise ValueError("Downloaded data is empty.")
    data.reset_index(inplace=True)
except Exception as e:
    print("Error downloading data: ", e)
    # Use example data if download fails
    dates = pd.date_range(start="2024-01-01", end="2024-05-17", freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'High': np.random.random(len(dates)) * 1.1 + 1.1,
        'Low': np.random.random(len(dates)) * 0.9 + 1.0,
        'Close': np.random.random(len(dates)) * 1.0 + 1.05
    })
    data.set_index('Date', inplace=True)

atr_periods = [5, 10, 15, 20]
multipliers = [2, 3, 4, 5]

results = walk_forward_optimization(data, atr_periods, multipliers)
print(results)

# Select the best parameters
best_result = max(results, key=lambda x: x[2])
print("Best parameters: ATR period =", best_result[0], "Multiplier =", best_result[1], "Equity =", best_result[2])

# Save results to CSV
save_results_to_csv(results, 'optimization_results.csv')

# Perform backtest with the best parameters
equity = backtest(data, best_result[0], best_result[1])

# Plot equity curve
plot_equity_curve(equity, title='Equity Curve with Best Parameters')

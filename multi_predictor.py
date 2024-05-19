import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the currency pairs
currency_pairs = ['GBPUSD=X', 'EURUSD=X', 'NZDUSD=X', 'USDJPY=X', 'DX-Y.NYB']

# Download historical data for the past year
historical_data = yf.download(currency_pairs, start="2023-01-01", end="2024-05-14")
print(historical_data)

# Function to prepare data for training
def prepare_data(df):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Timestamp'] = df['Date'].map(pd.Timestamp.timestamp)
    X = df[['Timestamp']]
    y = df['Close']
    return X, y, df['Date']

# Function to train and evaluate model
def train_and_evaluate(X, y, pair, dates):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{pair} - MSE: {mse}, R2: {r2}")
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(dates[len(X_train):], y_test, label='Actual Prices')
    plt.plot(dates[len(X_train):], y_pred, label='Predicted Prices')
    plt.title(f'Price Prediction for {pair}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    
    return model, pd.DataFrame({'Date': dates[len(X_train):], f'{pair}_Actual': y_test, f'{pair}_Predicted': y_pred})

# Train model for each currency pair and collect predictions
models = {}
prediction_frames = []

for pair in currency_pairs:
    try:
        pair_data = historical_data['Close'][pair].dropna()
    except KeyError:
        print(f"No 'Close' data available for {pair}")
        continue
    
    if pair_data.empty:
        print(f"No data available for {pair}")
        continue
    
    X, y, dates = prepare_data(pair_data.to_frame(name='Close'))
    model, prediction_frame = train_and_evaluate(X, y, pair, dates)
    models[pair] = model
    prediction_frames.append(prediction_frame)

# Combine all prediction data into a single DataFrame
if prediction_frames:
    predictions_df = pd.concat(prediction_frames, axis=1)
    predictions_df = predictions_df.loc[:, ~predictions_df.columns.duplicated()]

    # Save prediction data to a CSV file
    predictions_df.to_csv("predictions_data.csv", index=False)

# Save historical data to a CSV file
historical_data.to_csv("forex_historical_data_2024.csv")

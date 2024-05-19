import yfinance as yf
import pandas as pd

# Define the currency pairs
currency_pairs = ['GBPUSD=X', 'EURUSD=X', 'NZDUSD=X', 'USDJPY=X','DX-Y.NYB']

# Download historical data for the past year
historical_data = yf.download(currency_pairs, start="2024-01-01", end="2024-05-14")
print(historical_data)

# Fetch detailed information and real-time data for a specific pair
gbpusd = yf.Ticker('GBPUSD=X')
info = gbpusd.info
real_time_data = gbpusd.history(period='1d', interval='15min')
print(info)
print(real_time_data)

# Fetch historical dividends (if any) and splits data
dividends = gbpusd.dividends
splits = gbpusd.splits
print(dividends)
print(splits)

# Fetch earnings data (note: not typically relevant for currency pairs)
# earnings = gbpusd.earnings
# print(earnings)

# Save historical data to a CSV file
historical_data.to_csv("historical_data_2024.csv")

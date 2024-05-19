import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
file_path = "predictions.csv"  # Change this to your actual file path
data = pd.read_csv(file_path)

# Ensure 'Date' column is in datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)

# Check if the required columns are present
required_columns = ['GBPUSD=X_Actual', 'GBPUSD=X_Predicted', 
                    'EURUSD=X_Actual', 'EURUSD=X_Predicted', 
                    'NZDUSD=X_Actual', 'NZDUSD=X_Predicted', 
                    'USDJPY=X_Actual', 'USDJPY=X_Predicted']

missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Error: Missing columns in the CSV file: {missing_columns}")
    exit()

# Plot each currency pair's actual and predicted prices
for currency_pair in ['GBPUSD=X', 'EURUSD=X', 'NZDUSD=X', 'USDJPY=X']:
    actual_column = f"{currency_pair}_Actual"
    predicted_column = f"{currency_pair}_Predicted"

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[actual_column], label='Actual Price', color='blue')
    plt.plot(data.index, data[predicted_column], label='Predicted Price', color='red')
    plt.title(f'Actual vs. Predicted Prices for {currency_pair}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

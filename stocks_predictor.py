import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

class StockPricePredictor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_historical_data(self, ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(os.path.join(self.data_dir, f"{ticker}_historical_data.csv"))

    def load_historical_data(self, ticker):
        file_path = os.path.join(self.data_dir, f"{ticker}_historical_data.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0)
        else:
            print("Historical data not found. Please fetch data first.")
            return None

    def store_prediction(self, ticker, prediction):
        with open(os.path.join(self.data_dir, f"{ticker}_predictions.txt"), "a") as f:
            f.write(prediction + "\n")

    def predict_price_movement(self, ticker, start_date, end_date, model_type="linear_regression"):
        # Fetch historical data if not already available
        if not os.path.exists(os.path.join(self.data_dir, f"{ticker}_historical_data.csv")):
            self.fetch_historical_data(ticker, start_date, end_date)

        # Load historical data
        data = self.load_historical_data(ticker)

        # Preprocess data
        data = data[["Close"]]  # Only use closing prices
        data = data.dropna()  # Drop missing values
        data["MA_50"] = data["Close"].rolling(window=50).mean()  # 50-day moving average
        data["MA_200"] = data["Close"].rolling(window=200).mean()  # 200-day moving average
        data["RSI"] = self.calculate_rsi(data["Close"])  # Relative Strength Index

        # Split data into features and target variable
        X = data[["MA_50", "MA_200", "RSI"]]
        y = data["Close"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100)
        else:
            raise ValueError("Invalid model type. Choose either 'linear_regression' or 'random_forest'.")

        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")

        # Predict future price movement
        last_day_data = data.iloc[-1]
        new_data = pd.DataFrame({"MA_50": [last_day_data["MA_50"]], "MA_200": [last_day_data["MA_200"]], "RSI": [last_day_data["RSI"]]})
        future_price = model.predict(new_data)[0]

        # Determine price movement and timeframe
        price_change = (future_price - last_day_data["Close"]) / last_day_data["Close"] * 100
        timeframe = "short-term" if abs(price_change) < 2 else "medium-term"

        # Generate prediction string
        if price_change > 0:
            prediction = f"The price of {ticker} is predicted to **increase** by {abs(price_change):.2f}% in the {timeframe}."
        else:
            prediction = f"The price of {ticker} is predicted to **decrease** by {abs(price_change):.2f}% in the {timeframe}."

        # Store prediction
        self.store_prediction(ticker, prediction)

        return prediction

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Example usage
if __name__ == "__main__":
    predictor = StockPricePredictor()
    ticker = "JPM"  # Replace with your desired ticker eg. MSFT, AAPL, GOOG, etc.
    start_date = "2024-01-01"
    end_date = "2024-05-18"
    prediction = predictor.predict_price_movement(ticker, start_date, end_date, model_type="random_forest")
    print(prediction)

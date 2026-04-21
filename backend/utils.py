import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_stock_data(stock):
    stock = stock.upper().strip()

    df = yf.download(
        tickers=stock,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        ticker = yf.Ticker(stock)
        df = ticker.history(
            period="2y",
            interval="1d",
            auto_adjust=False
        )

    if df.empty:
        raise ValueError(f"No data found for {stock}. Yahoo Finance may be blocking the request or the ticker is invalid.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def preprocess_data(df):
    data = df.filter(["Close"])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    return scaled_data, scaler

def create_input_sequence(scaled_data):
    last_60_days = scaled_data[-60:]

    X = []
    X.append(last_60_days)

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X

def predict_price(model, X, scaler):
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(prediction)

    return prediction[0][0]

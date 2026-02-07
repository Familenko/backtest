import yfinance as yf
import pandas as pd

from config import START_DATE, TODAY, stocks, crypto


def load_data(tickers, start_date=START_DATE, end_date=TODAY):
    print(f"Loading data for {tickers} from {start_date} to {end_date}...")
    if isinstance(tickers, dict):
        df = pd.DataFrame()
        for k, v in tickers.items():
            df[k] = yf.download(v, start=start_date, end=end_date, auto_adjust=False)["Close"]
    else:
        df = pd.DataFrame()
        for ticker in tickers:
            df[ticker] = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)["Close"]

    return df


if __name__ == "__main__":
    df_stocks = load_data(stocks)
    df_crypto = load_data(crypto)
    
    df_stocks.to_csv("data/stocks.csv")
    df_crypto.to_csv("data/crypto.csv")

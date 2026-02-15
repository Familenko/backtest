import yfinance as yf
import pandas as pd
import time

from config import STOCKS, CRYPTO


def load_data(tickers):
    df = pd.DataFrame()
    for k, v in tickers.items():
        print(f"Loading data for {k} from {v['start_date']} to {v['end_date']}...")
        df[k] = yf.download(v["tickers"], start=v["start_date"], end=v["end_date"], auto_adjust=False)["Close"]
        time.sleep(1)

    return df


if __name__ == "__main__":
    df_stocks = load_data(STOCKS)
    df_crypto = load_data(CRYPTO)
    
    df_stocks.to_csv("data/stocks.csv")
    df_crypto.to_csv("data/crypto.csv")

from datetime import datetime


START_DATE = "2025-02-13"
TODAY = datetime.now().strftime("%Y-%m-%d")

# crypto
CRYPTO = {
    "BTC": {"tickers": "BTC-USD", "start_date": START_DATE, "end_date": TODAY},
    "ETH": {"tickers": "ETH-USD", "start_date": START_DATE, "end_date": TODAY},
    "BNB": {"tickers": "BNB-USD", "start_date": START_DATE, "end_date": TODAY},
    "XRP": {"tickers": "XRP-USD", "start_date": START_DATE, "end_date": TODAY},
    "SOL": {"tickers": "SOL-USD", "start_date": START_DATE, "end_date": TODAY},
    "DOT": {"tickers": "DOT-USD", "start_date": START_DATE, "end_date": TODAY},
    "ADA": {"tickers": "ADA-USD", "start_date": START_DATE, "end_date": TODAY},
    "TON": {"tickers": "TON11419-USD", "start_date": START_DATE, "end_date": TODAY},
    "LINK": {"tickers": "LINK-USD", "start_date": START_DATE, "end_date": TODAY},
    "AVAX": {"tickers": "AVAX-USD", "start_date": START_DATE, "end_date": TODAY}
}
crypto_regular_amount = 10
crypto_freq = "W-MON"
crypto_fee = 0.001
crypto_profit_multiple = 2
crypto_cooldown_days = 180

# stocks
STOCKS = {
    'SPY': {"tickers": "SPY", "start_date": START_DATE, "end_date": TODAY}
}
stock_regular_amount = 300
stock_freq = "WOM-3THU"
stock_fee = 0.01
stock_profit_multiple = 10
stock_cooldown_days = 180

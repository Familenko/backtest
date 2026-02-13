from datetime import datetime


START_DATE = "2026-02-10"
TODAY = datetime.now().strftime("%Y-%m-%d")

# crypto
crypto = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "DOT": "DOT-USD",
    "ADA": "ADA-USD",
    "TON": "TON-USD",
    "LINK": "LINK-USD",
    "AVAX": "AVAX-USD"
}
crypto_regular_amount = 10
crypto_freq = "W-MON"
crypto_fee = 0.001
crypto_profit_multiple = 2
crypto_cooldown_days = 180

# stocks
stocks = [
    'SPY'
]
stock_regular_amount = 300
stock_freq = "WOM-3THU"
stock_fee = 0.01
stock_profit_multiple = 10
stock_cooldown_days = 180

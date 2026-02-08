from datetime import datetime


START_DATE = "2023-02-07"
TODAY = datetime.now().strftime("%Y-%m-%d")

# crypto
crypto = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "TRON": "TRX-USD",
    "ADA": "ADA-USD",
    "XMR": "XMR-USD",
    "LINK": "LINK-USD",
    "AVAX": "AVAX-USD"
}
crypto_regular_amount = 10
crypto_freq = "W-MON"
crypto_fee = 0.001
crypto_profit_multiple = 2

# stocks
stocks = [
    'SPY', 
    'ASML.AS'
]
stock_regular_amount = 200
stock_freq = "WOM-3THU"
stock_fee = 0.01
stock_profit_multiple = 10

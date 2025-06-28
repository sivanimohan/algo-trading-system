"""
Trading System Constants
Last Updated: 2025-06-26 15:46:30 UTC
Author: sivanimohan

"""
import os
from dotenv import load_dotenv

load_dotenv() 

# Trading symbols (10 popular Nifty 50 stocks, Yahoo Finance format)
SYMBOLS = [
    'RELIANCE.NS',   # Reliance Industries Ltd.
    'INFY.NS',       # Infosys Ltd.
    'TCS.NS',        # Tata Consultancy Services Ltd.
    'LT.NS',         # Larsen & Toubro Ltd.
    'SBIN.NS',       # State Bank of India
    'BHARTIARTL.NS'  # Bharti Airtel Ltd.
]

# Trading parameters
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_POSITION_SIZE_PCT = 0.02
DEFAULT_STOP_LOSS_PCT = 0.02
DEFAULT_TAKE_PROFIT_PCT = 0.04

# Technical indicators parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# Trading hours (Indian Market)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
MARKET_TIMEZONE = 'Asia/Kolkata'

# Google Sheets configuration
SPREADSHEET_ID = '1yhmCMKFpMsjvsEfRzzXZWXCaU7Mao8bYt1MSK7r-12A'
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Backtest parameters
DEFAULT_BACKTEST_DAYS = 365
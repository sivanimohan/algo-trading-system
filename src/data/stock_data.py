"""
Stock Data Fetching Module
Last Updated: 2025-06-26 15:02:11 UTC
Author: sivanimohan
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional
import logging
from functools import lru_cache

class StockDataFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=100)
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                self.logger.error(f"No data available for {symbol}")
                return None
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
"""
Stock Data Fetching Module
Last Updated: 2025-06-27 09:29:07 UTC
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
        """
        Fetch stock data from Yahoo Finance for the given symbol and date range.
        Ensures columns are standardized and resets the index.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if data is None or data.empty:
                self.logger.error(f"No data available for {symbol} ({start_date} to {end_date})")
                return None
            # Standardize columns and reset index for downstream compatibility
            data = data.reset_index()
            # Ensure required columns exist for technical indicators and strategies
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"Missing column '{col}' in data for {symbol}")
                    return None
            # For safety, fill any missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
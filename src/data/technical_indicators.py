"""
Technical Indicators Module
Last Updated: 2025-06-27 09:25:34 UTC
Author: sivanimohan
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from .constants import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, 
    MACD_SIGNAL, BOLLINGER_WINDOW, BOLLINGER_STD
)

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        """Calculate RSI"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = data['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = data['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.DataFrame, 
        window: int = BOLLINGER_WINDOW,
        std: int = BOLLINGER_STD
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = data['Close'].rolling(window=window).mean()
        std_dev = data['Close'].rolling(window=window).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        try:
            # Add RSI
            data['RSI'] = self.calculate_rsi(data)
            
            # Add MACD
            data['MACD'], data['Signal_Line'], data['MACD_Histogram'] = self.calculate_macd(data)
            
            # Add Bollinger Bands
            data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = self.calculate_bollinger_bands(data)
            
            # Add moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # Add ATR
            data['ATR'] = self.calculate_atr(data)
            
            # Add momentum indicators
            data['ROC'] = data['Close'].pct_change(periods=10) * 100
            data['MOM'] = data['Close'].diff(periods=10)

            # Optionally, fill NaN values for Streamlit/ML use
            data = data.fillna(method='bfill').fillna(method='ffill')

            return data
            
        except Exception as e:
            raise Exception(f"Error adding indicators: {str(e)}")

    def get_signal(self, data: pd.DataFrame) -> str:
        """
        Generate trading signals based on technical indicators
        Returns: 'buy', 'sell', or 'hold'
        """
        try:
            # Ensure enough data for indicators and to safely use iloc[-2]
            if data is None or len(data) < 50:
                return 'hold'
            if len(data) < 2:
                return 'hold'
            
            # Get latest data point
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Initialize signal counters
            buy_signals = 0
            sell_signals = 0
            
            # 1. RSI Signals
            if current['RSI'] < 30:
                buy_signals += 1
            elif current['RSI'] > 70:
                sell_signals += 1
                
            # 2. MACD Crossover
            if (current['MACD'] > current['Signal_Line'] and 
                previous['MACD'] <= previous['Signal_Line']):
                buy_signals += 1
            elif (current['MACD'] < current['Signal_Line'] and 
                  previous['MACD'] >= previous['Signal_Line']):
                sell_signals += 1
                
            # 3. Bollinger Bands
            if current['Close'] <= current['Lower_Band']:
                buy_signals += 1
            elif current['Close'] >= current['Upper_Band']:
                sell_signals += 1
                
            # 4. Moving Average Crossovers
            if (current['MA20'] > current['MA50'] and 
                previous['MA20'] <= previous['MA50']):
                buy_signals += 1
            elif (current['MA20'] < current['MA50'] and 
                  previous['MA20'] >= previous['MA50']):
                sell_signals += 1

            # 5. Volume Confirmation
            if len(data['Volume']) >= 20:  # Ensure enough for rolling mean
                vol_sma = data['Volume'].rolling(20).mean().iloc[-1]
                if not np.isnan(vol_sma):
                    if current['Volume'] > vol_sma * 1.5:
                        if current['Close'] > previous['Close']:
                            buy_signals += 1
                        else:
                            sell_signals += 1
                    
            # 6. Momentum
            if current['ROC'] > 0 and current['MOM'] > 0:
                buy_signals += 1
            elif current['ROC'] < 0 and current['MOM'] < 0:
                sell_signals += 1
            
            # Decision making
            if buy_signals >= 3:
                return 'buy'
            elif sell_signals >= 3:
                return 'sell'
            return 'hold'
            
        except Exception as e:
            raise Exception(f"Error generating signal: {str(e)}")

available_indicators: List[str] = [
    "RSI",
    "MACD",
    "MACD_Histogram",
    "Signal_Line",
    "Upper_Band",
    "Middle_Band",
    "Lower_Band",
    "MA20",
    "MA50",
    "MA200",
    "ATR",
    "ROC",
    "MOM",
    "Volume"
]
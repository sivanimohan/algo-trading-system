"""
Trading Strategies Module
Last Updated: 2025-06-27 09:28:32 UTC
Author: sivanimohan
"""

import pandas as pd
from typing import Dict
import numpy as np

class TradingStrategies:
    @staticmethod
    def rsi_strategy(data: pd.DataFrame) -> str:
        """RSI-based trading strategy"""
        if 'RSI' not in data.columns or len(data) < 14:
            return 'hold'
        last_rsi = data['RSI'].iloc[-1]
        if pd.isna(last_rsi):
            return 'hold'
        if last_rsi < 30:
            return 'buy'
        elif last_rsi > 70:
            return 'sell'
        return 'hold'

    @staticmethod
    def macd_strategy(data: pd.DataFrame) -> str:
        """MACD-based trading strategy"""
        if 'MACD' not in data.columns or 'Signal_Line' not in data.columns:
            return 'hold'
        if len(data) < 2:
            return 'hold'
        current = data.iloc[-1]
        previous = data.iloc[-2]
        if pd.isna(current['MACD']) or pd.isna(current['Signal_Line']) or pd.isna(previous['MACD']) or pd.isna(previous['Signal_Line']):
            return 'hold'
        if (current['MACD'] > current['Signal_Line'] and 
            previous['MACD'] <= previous['Signal_Line']):
            return 'buy'
        elif (current['MACD'] < current['Signal_Line'] and 
              previous['MACD'] >= previous['Signal_Line']):
            return 'sell'
        return 'hold'

    @staticmethod
    def bollinger_strategy(data: pd.DataFrame) -> str:
        """Bollinger Bands trading strategy"""
        if 'Upper_Band' not in data.columns or 'Lower_Band' not in data.columns:
            return 'hold'
        if len(data) < 1:
            return 'hold'
        current = data.iloc[-1]
        if pd.isna(current['Close']) or pd.isna(current['Upper_Band']) or pd.isna(current['Lower_Band']):
            return 'hold'
        if current['Close'] <= current['Lower_Band']:
            return 'buy'
        elif current['Close'] >= current['Upper_Band']:
            return 'sell'
        return 'hold'

    @staticmethod
    def moving_average_strategy(data: pd.DataFrame) -> str:
        """Moving Average Crossover strategy"""
        if 'MA20' not in data.columns or 'MA50' not in data.columns:
            return 'hold'
        if len(data) < 2:
            return 'hold'
        current = data.iloc[-1]
        previous = data.iloc[-2]
        if pd.isna(current['MA20']) or pd.isna(current['MA50']) or pd.isna(previous['MA20']) or pd.isna(previous['MA50']):
            return 'hold'
        if (current['MA20'] > current['MA50'] and 
            previous['MA20'] <= previous['MA50']):
            return 'buy'
        elif (current['MA20'] < current['MA50'] and 
              previous['MA20'] >= previous['MA50']):
            return 'sell'
        return 'hold'

    @staticmethod
    def combined_strategy(data: pd.DataFrame) -> str:
        """Combined strategy using multiple indicators"""
        signals = [
            TradingStrategies.rsi_strategy(data),
            TradingStrategies.macd_strategy(data),
            TradingStrategies.bollinger_strategy(data),
            TradingStrategies.moving_average_strategy(data)
        ]
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        if buy_count >= 2:
            return 'buy'
        elif sell_count >= 2:
            return 'sell'
        return 'hold'

# Add this dictionary to expose strategies for import in app.py
STRATEGIES = {
    "rsi": "Relative Strength Index (RSI) Strategy",
    "macd": "MACD Crossover Strategy",
    "bollinger": "Bollinger Bands Strategy",
    "moving_average": "Moving Average Crossover Strategy",
    "combined": "Combined Multi-indicator Strategy"
}
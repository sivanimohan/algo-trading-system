import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators (RSI, MACD, Bollinger Bands) as features to the DataFrame.
    Fills missing values forward and backward for robust downstream ML/trading use.
    """
    df = df.copy()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # Fill missing values for ML/analytics
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df
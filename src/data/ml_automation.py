"""
ML Automation Module
Last Updated: 2025-06-27
Author: sivanimohan
Description:
    Machine Learning automation for next-day movement prediction.
    - Uses Decision Tree or Logistic Regression (user selectable)
    - Features: RSI, MACD, Signal_Line, MACD_Histogram, Upper_Band, Lower_Band, MA20, MA50, MA200, ATR, ROC, MOM, Volume
    - Provides prediction accuracy and ML-based buy/sell signals.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class MLAutomation:
    def __init__(self, model_type='decision_tree'):
        """
        model_type: 'decision_tree' or 'logistic_regression'
        """
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=200)
        else:
            self.model = DecisionTreeClassifier()
        self.fitted = False

    @staticmethod
    def get_feature_columns():
        """Return the list of feature columns expected as input."""
        return [
            'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
            'Upper_Band', 'Lower_Band', 'MA20', 'MA50', 'MA200',
            'ATR', 'ROC', 'MOM', 'Volume'
        ]

    def prepare_features(self, df: pd.DataFrame):
        """
        Prepare ML features from dataframe.
        Assumes indicators are already calculated and present in df.
        """
        feature_cols = self.get_feature_columns()
        features = df[feature_cols].copy()
        features = features.fillna(method='ffill').fillna(method='bfill')
        return features

    def make_labels(self, df: pd.DataFrame):
        """Label: 1 if next day's Close > today's Close, else 0. Exclude last row for y."""
        labels = (df['Close'].shift(-1) > df['Close']).astype(int)
        return labels[:-1]  # last label is NaN

    def train(self, df: pd.DataFrame):
        """Train the ML model on historical data. Returns training accuracy."""
        X = self.prepare_features(df)[:-1]
        y = self.make_labels(df)
        # Drop any rows with missing values in features or labels
        valid = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid]
        y = y[valid]
        self.model.fit(X, y)
        self.fitted = True

        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        return acc

    def predict(self, df: pd.DataFrame):
        """Predict movement (1=up, 0=down) for each row in df (after training)."""
        if not self.fitted:
            raise Exception("Model is not trained yet.")
        X = self.prepare_features(df)
        # Drop rows with missing values
        X = X.dropna()
        preds = self.model.predict(X)
        return preds

    def generate_signal(self, df: pd.DataFrame):
        """
        Generate buy/sell signal for the most recent data row.
        Returns 'buy' if predicted up, 'sell' if predicted down.
        """
        if not self.fitted:
            raise Exception("Model is not trained yet.")
        X = self.prepare_features(df)
        latest_X = X.iloc[[-1]].dropna()
        if latest_X.empty:
            return "hold"
        pred = self.model.predict(latest_X)[0]
        return 'buy' if pred == 1 else 'sell'

    def backtest_signals(self, df: pd.DataFrame):
        """
        Backtest ML signals on the provided dataframe.
        Returns a DataFrame with actual vs predicted and accuracy.
        """
        X = self.prepare_features(df)[:-1]
        y = self.make_labels(df)
        # Only use rows where all features and label are present
        valid = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid]
        y = y[valid]
        preds = self.model.predict(X)
        result_df = df.iloc[:len(X)].copy()
        result_df['ML_Pred'] = preds
        result_df['Actual'] = y.values
        result_df['Correct'] = result_df['ML_Pred'] == result_df['Actual']
        accuracy = accuracy_score(y, preds)
        return result_df, accuracy
"""
Trading Visualization Module
Last Updated: 2025-06-27 09:39:04 UTC
Author: sivanimohan
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import os
from datetime import datetime, UTC

class TradingVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_theme()
        sns.set_palette("husl")

    def _find_date_column(self, df: pd.DataFrame):
        """Return the column name to use as date/time (case-insensitive search)."""
        for candidate in ['date', 'Date', 'timestamp', 'Timestamp', 'datetime', 'Datetime']:
            if candidate in df.columns:
                return candidate
        # If not found, try a case-insensitive match
        for col in df.columns:
            if col.lower() in ['date', 'timestamp', 'datetime']:
                return col
        raise KeyError(f"No date/timestamp column found in DataFrame. Columns: {df.columns.tolist()}")

    def plot_multi_symbol_performance(self, symbols_results: Dict[str, Dict]):
        """Bar chart comparing return (%) and win rate (%) for each stock"""
        try:
            plt.figure(figsize=(15, 10))
            symbols = list(symbols_results.keys())
            returns = [result['metrics']['return_pct'] for result in symbols_results.values()]
            win_rates = [result['metrics']['win_rate'] * 100 for result in symbols_results.values()]
            x = np.arange(len(symbols))
            width = 0.35
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            # Returns plot
            ax1.bar(x - width/2, returns, width, label='Return (%)')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Total Return by Stock')
            ax1.set_xticks(x)
            ax1.set_xticklabels(symbols)
            ax1.legend()
            # Win rates plot
            ax2.bar(x + width/2, win_rates, width, label='Win Rate (%)')
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_title('Winning Trade Percentage by Stock')
            ax2.set_xticks(x)
            ax2.set_xticklabels(symbols)
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'multi_symbol_performance.png'))
            plt.close()
        except Exception as e:
            raise Exception(f"Error plotting multi-symbol performance: {str(e)}")

    def plot_equity_curves(self, symbols_results: Dict[str, Dict]):
        """Plot equity curve (portfolio value over time) for each stock"""
        plt.figure(figsize=(15, 8))
        for symbol, results in symbols_results.items():
            equity_data = pd.DataFrame(results['equity_curve'])
            if equity_data.empty:
                continue
            date_col = self._find_date_column(equity_data)
            plt.plot(equity_data[date_col], equity_data['equity'], label=symbol)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'equity_curves.png'))
        plt.close()

    def plot_trade_distribution(self, symbols_results: Dict[str, Dict]):
        """Show the distribution of profit and loss per trade for all stocks"""
        plt.figure(figsize=(15, 8))
        for symbol, results in symbols_results.items():
            pnl_data = [t['pnl'] for t in results['trades']]
            if len(pnl_data) > 1:
                sns.kdeplot(data=pnl_data, label=symbol)
        plt.title('Trade P&L Distribution')
        plt.xlabel('Profit / Loss per Trade (₹)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'pnl_distribution.png'))
        plt.close()

    def plot_monthly_returns(self, symbols_results: Dict[str, Dict]):
        """Heatmap of monthly returns for each stock"""
        monthly_returns = {}
        for symbol, results in symbols_results.items():
            trades = results['trades']
            if trades:
                returns = pd.Series([t['pnl'] for t in trades])
                returns.index = pd.to_datetime([t['exit_date'] for t in trades])
                monthly_returns[symbol] = returns.resample('M').sum()
        returns_df = pd.DataFrame(monthly_returns)
        plt.figure(figsize=(15, 8))
        sns.heatmap(returns_df.T, cmap='RdYlGn', center=0, annot=True, fmt='.0f')
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Stock')
        plt.savefig(os.path.join(self.figures_dir, 'monthly_returns_heatmap.png'))
        plt.close()

    def plot_drawdowns(self, symbols_results: Dict[str, Dict]):
        """Plot drawdown curves for each stock (max drop from peak equity)"""
        plt.figure(figsize=(15, 8))
        for symbol, results in symbols_results.items():
            equity_data = pd.DataFrame(results['equity_curve'])
            if equity_data.empty:
                continue
            date_col = self._find_date_column(equity_data)
            equity_series = equity_data['equity']
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max * 100
            plt.plot(equity_data[date_col], drawdown, label=symbol)
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'drawdowns.png'))
        plt.close()

    def plot_all(self, symbols_results: dict):
        """
        Generate all standard charts for stock backtest results and return the path to the performance report HTML file.
        """
        self.plot_multi_symbol_performance(symbols_results)
        self.plot_equity_curves(symbols_results)
        self.plot_trade_distribution(symbols_results)
        self.plot_monthly_returns(symbols_results)
        self.plot_drawdowns(symbols_results)
        # Optionally: generate a report file if you have that method, or just return None or a summary string
        if hasattr(self, "generate_performance_report"):
            return self.generate_performance_report(symbols_results)
        return None
    

def plot_performance(symbols_results):
    """
    Show a simple equity curve (portfolio value over time) for each stock (for Streamlit).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, results in symbols_results.items():
        equity_data = pd.DataFrame(results['equity_curve'])
        if not equity_data.empty:
            # Robust date column detection
            date_col = None
            for candidate in ['date', 'Date', 'timestamp', 'Timestamp', 'datetime', 'Datetime']:
                if candidate in equity_data.columns:
                    date_col = candidate
                    break
            if not date_col:
                for col in equity_data.columns:
                    if col.lower() in ['date', 'timestamp', 'datetime']:
                        date_col = col
                        break
            if not date_col:
                raise KeyError(f"No date/timestamp column found in DataFrame. Columns: {equity_data.columns.tolist()}")
            ax.plot(equity_data[date_col], equity_data['equity'], label=symbol)
    ax.set_title('Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (₹)')
    ax.legend()
    ax.grid(True)
    return fig
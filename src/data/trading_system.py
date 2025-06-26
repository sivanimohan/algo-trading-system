"""
Main Trading System Module
Last Updated: 2025-06-26 15:05:23 UTC
Author: sivanimohan
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from src.data.stock_data import StockDataFetcher
from src.data.technical_indicators import TechnicalIndicators
from src.data.strategies import TradingStrategies
from src.data.visualization import TradingVisualizer
from src.data.constants import (
    DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE_PCT,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
    SYMBOLS
)
from src.data.google_sheets_logger import GoogleSheetsLogger

class TradingSystem:
    def __init__(self, 
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 position_size_pct: float = DEFAULT_POSITION_SIZE_PCT,
                 stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
                 take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
                 use_sheets: bool = True):
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize components
        self.stock_data = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.strategies = TradingStrategies()
        self.positions: Dict[str, List[Dict]] = {symbol: [] for symbol in SYMBOLS}
        self.trades_history: Dict[str, List[Dict]] = {symbol: [] for symbol in SYMBOLS}
        
        # Create results directory
        self.results_dir = os.path.join(os.getcwd(), 'trading_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize visualization
        self.visualizer = TradingVisualizer(self.results_dir)
        
        # Initialize Google Sheets logging
        self.sheets_logger = None
        if use_sheets:
            try:
                credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
                self.sheets_logger = GoogleSheetsLogger(credentials_path)
                self.logger.info("✅ Google Sheets logging enabled")
                self.logger.info(f"Connected to: {self.sheets_logger.get_spreadsheet_url()}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Sheets logging: {e}")
                self.logger.info("Continuing with local logging only")
        
        # Trading state
        self.is_trading = False
        self.active_positions: Dict[str, List[Dict]] = {symbol: [] for symbol in SYMBOLS}
        self.market_conditions: Dict[str, Any] = {}

    def start_trading(self) -> None:
        if self.is_trading:
            self.logger.warning("Trading system is already running.")
            return
        self.is_trading = True
        self._initialize_trading_session()
        if self.sheets_logger:
            self.sheets_logger.log_system_event("Trading Started", {
                "capital": self.current_capital,
                "timestamp": datetime.now(UTC)
            })
        self.logger.info("Trading system started.")

    def stop_trading(self) -> None:
        if not self.is_trading:
            self.logger.warning("Trading system is not running.")
            return
        self.is_trading = False
        self._close_all_positions()
        self.generate_daily_report()
        if self.sheets_logger:
            self.sheets_logger.log_system_event("Trading Stopped", {
                "final_capital": self.current_capital,
                "timestamp": datetime.now(UTC)
            })
        self.logger.info("Trading system stopped.")

    def _initialize_trading_session(self) -> None:
        self.update_market_conditions()
        self.logger.info("Initialized trading session.")

    def _close_all_positions(self) -> None:
        for symbol, positions in self.active_positions.items():
            for pos in positions:
                pos['closed'] = True
                pos['close_time'] = datetime.now(UTC)
        self.active_positions = {symbol: [] for symbol in SYMBOLS}
        self.logger.info("All positions closed.")

    def update_positions(self) -> None:
        for symbol, positions in self.active_positions.items():
            current_price = self.get_current_price(symbol)
            for pos in positions:
                pos['current_price'] = current_price

    def update_market_conditions(self) -> None:
        self.market_conditions = {"status": "stable", "timestamp": datetime.now(UTC)}
        self.logger.info("Market conditions updated.")

    def get_current_price(self, symbol: str) -> Optional[float]:
        data = self.stock_data.fetch_data(symbol, datetime.now(UTC)-timedelta(days=2), datetime.now(UTC))
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None

    @lru_cache(maxsize=100)
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            data = self.stock_data.fetch_data(symbol, start_date, end_date)
            if data is not None:
                return self.indicators.add_indicators(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def fetch_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_data, symbol, start_date, end_date): symbol 
                for symbol in SYMBOLS
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        data_dict[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {e}")
        return data_dict

    def calculate_position_size(self, price: float, symbol: str) -> float:
        available_capital = self.current_capital * self.position_size_pct
        return available_capital / price

    def get_signal(self, data: pd.DataFrame, strategy: str = 'combined') -> str:
        if strategy == 'rsi':
            return self.strategies.rsi_strategy(data)
        elif strategy == 'macd':
            return self.strategies.macd_strategy(data)
        elif strategy == 'bollinger':
            return self.strategies.bollinger_strategy(data)
        elif strategy == 'moving_average':
            return self.strategies.moving_average_strategy(data)
        else:
            return self.strategies.combined_strategy(data)

    def open_position(self, 
                     symbol: str, 
                     price: float, 
                     entry_date: datetime,
                     direction: str = 'long') -> Optional[Dict]:
        try:
            shares = self.calculate_position_size(price, symbol)
            position_size = shares * price
            
            if position_size > self.current_capital:
                self.logger.warning(f"Insufficient capital for {symbol} position")
                return None
            
            position = {
                'symbol': symbol,
                'direction': direction,
                'entry_date': entry_date,
                'entry_price': price,
                'shares': shares,
                'position_size': position_size,
                'stop_loss': price * (1 - self.stop_loss_pct if direction == 'long' else 1 + self.stop_loss_pct),
                'take_profit': price * (1 + self.take_profit_pct if direction == 'long' else 1 - self.take_profit_pct),
                'status': 'open'
            }
            
            self.current_capital -= position_size
            self.positions[symbol].append(position)
            
            self.logger.info(
                f"Opened {direction} position in {symbol} at ${price:.2f}, "
                f"Shares: {shares:.2f}, Size: ${position_size:.2f}"
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to open position for {symbol}: {e}")
            return None

    def close_position(self, 
                      position: Dict, 
                      exit_price: float,
                      exit_date: datetime,
                      reason: str = 'signal') -> Dict:
        try:
            symbol = position['symbol']
            position['status'] = 'closed'
            position['exit_date'] = exit_date
            position['exit_price'] = exit_price
            position['close_reason'] = reason

            if position['direction'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['shares']
            else:
                pnl = (position['entry_price'] - exit_price) * position['shares']

            position['pnl'] = pnl
            position['pnl_pct'] = (pnl / position['position_size']) * 100
            
            self.current_capital += position['position_size'] + pnl
            self.trades_history[symbol].append(position)
            self.positions[symbol].remove(position)
            
            self.logger.info(
                f"Closed {position['direction']} position in {symbol} at ${exit_price:.2f}, "
                f"P&L: ${pnl:.2f} ({position['pnl_pct']:.2f}%)"
            )
            
            # Log to Google Sheets if available
            if self.sheets_logger:
                try:
                    self.sheets_logger.log_trade(position)
                except Exception as e:
                    self.logger.warning(f"Failed to log to Google Sheets: {e}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            raise

    def run_strategy(self, 
                symbol: str, 
                data: pd.DataFrame,
                strategy: str = 'combined') -> List[Dict]:
     trades = []
    # --- FIX: Check for empty/short DataFrame ---
     if data is None or data.empty:
        self.logger.error(f"No data for {symbol} in run_strategy, skipping.")
        return trades
     if len(data) < 2:
        self.logger.error(f"Not enough data for {symbol} (rows: {len(data)}), skipping.")
        return trades
    # --------------------------------------
     for index, row in data.iterrows():
        for position in self.positions[symbol][:]:
            current_price = row['Close']
            if (position['direction'] == 'long' and current_price <= position['stop_loss']) or \
               (position['direction'] == 'short' and current_price >= position['stop_loss']):
                closed_position = self.close_position(
                    position, current_price, index, 'stop_loss')
                trades.append(closed_position)
            elif (position['direction'] == 'long' and current_price >= position['take_profit']) or \
                 (position['direction'] == 'short' and current_price <= position['take_profit']):
                closed_position = self.close_position(
                    position, current_price, index, 'take_profit')
                trades.append(closed_position)

        signal = self.get_signal(data.loc[:index], strategy)
        if signal == 'buy' and not self.positions[symbol]:
            position = self.open_position(symbol, row['Close'], index)
            if position:
                self.logger.info(f"Opened long position in {symbol} at {row['Close']}")
        elif signal == 'sell' and not self.positions[symbol]:
            position = self.open_position(symbol, row['Close'], index, 'short')
            if position:
                self.logger.info(f"Opened short position in {symbol} at {row['Close']}")
     return trades

    def run_backtest(self, 
                start_date: datetime,
                end_date: datetime,
                strategy: str = 'combined') -> Dict[str, Dict]:
      self.logger.info(f"\nStarting backtest for all symbols")
      self.logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
      try:
        data_dict = self.fetch_all_data(start_date, end_date)
        results = {}
        for symbol, data in data_dict.items():
            self.logger.info(f"\nProcessing {symbol}...")

            # --- FIX: Skip processing if DataFrame is empty or too short ---
            if data is None or data.empty:
                self.logger.error(f"No data for {symbol}, skipping.")
                continue
            if len(data) < 2:
                self.logger.error(f"Not enough data for {symbol} (rows: {len(data)}), skipping.")
                continue
            # ---

            symbol_results = {
                'trades': self.run_strategy(symbol, data, strategy),
                'metrics': {},
                'equity_curve': []
            }
            symbol_results['metrics'] = self.calculate_performance_metrics(
                symbol_results['trades'])
            symbol_results['equity_curve'] = self.calculate_equity_curve(
                symbol_results['trades'])
            results[symbol] = symbol_results
            self.logger.info(
                f"Completed {symbol}: "
                f"Trades: {len(symbol_results['trades'])}, "
                f"P&L: ${symbol_results['metrics']['total_pnl']:,.2f}"
            )
        report_file = self.visualizer.plot_all(results)
        self.logger.info(f"\nPerformance report generated: {report_file}")
        return results
      except Exception as e:
        self.logger.error(f"Error in backtest: {e}")
        raise

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'return_pct': 0
            }
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        return {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades if total_trades else 0,
            'max_drawdown': self.calculate_max_drawdown(trades),
            'sharpe_ratio': self.calculate_sharpe_ratio(trades),
            'return_pct': (total_pnl / self.initial_capital) * 100 if self.initial_capital else 0
        }

    def calculate_equity_curve(self, trades: List[Dict]) -> List[Dict]:
        equity_curve = []
        current_equity = self.initial_capital
        for trade in sorted(trades, key=lambda x: x['exit_date']):
            current_equity += trade['pnl']
            equity_curve.append({
                'date': trade['exit_date'],
                'equity': current_equity
            })
        return equity_curve

    @staticmethod
    def calculate_max_drawdown(trades: List[Dict]) -> float:
        if not trades:
            return 0
        equity_curve = []
        current_equity = 0
        max_drawdown = 0
        peak = 0
        for trade in sorted(trades, key=lambda x: x['exit_date']):
            current_equity += trade['pnl']
            if current_equity > peak:
                peak = current_equity
            drawdown = peak - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    @staticmethod
    def calculate_sharpe_ratio(trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        if not trades:
            return 0
        returns = [t['pnl_pct'] for t in trades]
        if not returns:
            return 0
        returns_series = pd.Series(returns)
        excess_returns = returns_series - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def generate_daily_report(self) -> Dict:
        report = {
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "capital": self.current_capital,
            "daily_pnl": self.current_capital - self.initial_capital,
            "total_trades": sum(len(trs) for trs in self.trades_history.values()),
            "open_positions": sum(len(pos) for pos in self.active_positions.values()),
            "market_conditions": self.market_conditions
        }
        self._save_report(report, "daily")
        if self.sheets_logger:
            self.sheets_logger.log_daily_report(report)
        return report

    def generate_weekly_report(self) -> Dict:
        report = {
            "week_ending": datetime.now(UTC).strftime("%Y-%m-%d"),
            "capital": self.current_capital,
            "weekly_pnl": self._calculate_period_pnl(days=7),
            "weekly_trades": self._calculate_period_trades(days=7),
            "performance_metrics": self._calculate_performance_metrics(days=7)
        }
        self._save_report(report, "weekly")
        if self.sheets_logger:
            self.sheets_logger.log_weekly_report(report)
        return report

    def generate_monthly_report(self) -> Dict:
        report = {
            "month_ending": datetime.now(UTC).strftime("%Y-%m-%d"),
            "capital": self.current_capital,
            "monthly_pnl": self._calculate_period_pnl(days=30),
            "monthly_trades": self._calculate_period_trades(days=30),
            "performance_metrics": self._calculate_performance_metrics(days=30)
        }
        self._save_report(report, "monthly")
        if self.sheets_logger:
            self.sheets_logger.log_monthly_report(report)
        return report

    def _save_report(self, report: Dict, report_type: str) -> None:
        filename = f"{report_type}_report_{datetime.now(UTC).strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        self.logger.info(f"{report_type.capitalize()} report saved to {filepath}")

    def _calculate_period_pnl(self, days: int) -> float:
        # Placeholder; real calculation would use trade dates
        return np.random.uniform(-1000, 1000)

    def _calculate_period_trades(self, days: int) -> int:
        # Placeholder; real calculation would use trade dates
        return np.random.randint(5, 15)

    def _calculate_performance_metrics(self, days: int) -> Dict:
        # Placeholder for metrics like Sharpe, Sortino, etc.
        return {"sharpe": np.random.uniform(0, 2), "sortino": np.random.uniform(0, 2)}
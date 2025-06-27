"""
Main Trading System Runner
Last Updated: 2025-06-27
Author: sivanimohan
ML Automation Extension
"""

import logging
from datetime import datetime, timedelta, UTC
import os
import pandas as pd
import sys
from typing import Dict, List
import argparse
import json
import getpass

from src.data.trading_system import TradingSystem
from src.data.automation_controller import AutomationController
from src.data.constants import (
    SYMBOLS, DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE_PCT,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT
)
from src.data.ml_automation import MLAutomation

def setup_logging() -> None:
    """Setup logging configuration"""
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    current_time = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_{current_time}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_config(config_file: str) -> Dict:
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {}

def save_results(results: Dict, output_dir: str) -> None:
    """Save trading results"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = os.path.join(output_dir, f'results_{current_time}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'summary_{current_time}.txt')
        with open(summary_file, 'w') as f:
            f.write("Trading System Results Summary\n")
            f.write("=============================\n\n")
            
            for symbol, result in results.items():
                f.write(f"\nResults for {symbol}:\n")
                metrics = result.get("metrics")
                if metrics is None:
                    f.write("No metrics found for this symbol!\n")
                    continue
                f.write(f"Total Trades: {metrics.get('total_trades', 'N/A')}\n")
                f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                f.write(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}\n")
                f.write(f"Return: {metrics.get('return_pct', 0):.2f}%\n")
                f.write("-----------------------------\n")
        
        logging.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def run_backtest(args: argparse.Namespace) -> None:
    """Run backtest mode"""
    try:
        # Initialize trading system
        trading_system = TradingSystem(
            initial_capital=args.capital,
            position_size_pct=args.position_size,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            use_sheets=args.use_sheets
        )
        
        # Set up backtest parameters
        start_date = datetime.now(UTC) - timedelta(days=args.days)
        end_date = datetime.now(UTC)
        
        # Run backtest
        results = trading_system.run_backtest(
            start_date=start_date,
            end_date=end_date,
            strategy=args.strategy
        )
        
        # Save results
        save_results(results, args.output_dir)
        
        # Print summary
        print("\nBacktest Results Summary:")
        print("=========================")
        
        total_pnl = 0
        total_trades = 0
        
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            metrics = result.get("metrics", {})
            print(f"Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"P&L: ${metrics.get('total_pnl', 0):,.2f}")
            print(f"Return: {metrics.get('return_pct', 0):.2f}%")
            
            total_pnl += metrics.get('total_pnl', 0)
            total_trades += metrics.get('total_trades', 0)
        
        print("\nOverall Performance:")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Return on Capital: {(total_pnl / args.capital) * 100:.2f}%")

        # ML Automation Section
        if args.ml:
            print("\n[ML Automation] Running ML-based analytics...")
            model_type = args.ml_model
            ml_results = {}
            for symbol, result in results.items():
                # The following assumes you have a 'history' key with a DataFrame or list of dicts in result.
                # This may need to be adapted based on your TradingSystem's output structure.
                df = None
                if isinstance(result.get("history"), pd.DataFrame):
                    df = result["history"]
                elif isinstance(result.get("history"), list):
                    df = pd.DataFrame(result["history"])
                if df is not None and not df.empty:
                    ml = MLAutomation(model_type=model_type)
                    try:
                        acc = ml.train(df)
                        signal = ml.generate_signal(df)
                        print(f"\nML ({model_type}) for {symbol}:")
                        print(f"  - Prediction Accuracy: {acc:.2%}")
                        print(f"  - Signal for next day: {signal.upper()}")
                        ml_results[symbol] = {
                            "accuracy": acc,
                            "signal": signal
                        }
                    except Exception as ml_e:
                        print(f"  [ML] Unable to run ML on {symbol}: {ml_e}")
                else:
                    print(f"  [ML] No historical data for {symbol} to run ML.")
            # Optionally, save or further process ml_results

    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        sys.exit(1)

def run_live_trading(args: argparse.Namespace) -> None:
    """Run live trading mode"""
    try:
        # Initialize trading system
        trading_system = TradingSystem(
            initial_capital=args.capital,
            position_size_pct=args.position_size,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            use_sheets=args.use_sheets
        )
        
        # Initialize automation controller
        controller = AutomationController(
            trading_system=trading_system,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        # Run automation
        controller.run()
        
    except Exception as e:
        logging.error(f"Error in live trading: {e}")
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System Runner')
    
    # Mode selection
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                      help='Trading mode (default: backtest)')
    
    # Trading parameters
    parser.add_argument('--capital', type=float, default=DEFAULT_INITIAL_CAPITAL,
                      help='Initial capital')
    parser.add_argument('--position-size', type=float, default=DEFAULT_POSITION_SIZE_PCT,
                      help='Position size as percentage of capital')
    parser.add_argument('--stop-loss', type=float, default=DEFAULT_STOP_LOSS_PCT,
                      help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=DEFAULT_TAKE_PROFIT_PCT,
                      help='Take profit percentage')
    
    # Backtest parameters
    parser.add_argument('--days', type=int, default=365,
                      help='Number of days to backtest')
    parser.add_argument('--strategy', choices=['rsi', 'macd', 'bollinger', 'combined'],
                      default='combined', help='Trading strategy to use')
    
    # Live trading parameters
    parser.add_argument('--max-retries', type=int, default=3,
                      help='Maximum number of retries for operations')
    parser.add_argument('--retry-delay', type=int, default=5,
                      help='Delay between retries in seconds')
    
    # Output options
    parser.add_argument('--output-dir', default='trading_results',
                      help='Output directory for results')
    parser.add_argument('--use-sheets', action='store_true',
                      help='Enable Google Sheets logging')
    
    # Configuration
    parser.add_argument('--config', help='Path to configuration file')

    # ML Automation
    parser.add_argument('--ml', action='store_true',
                      help='Enable ML automation and analytics')
    parser.add_argument('--ml-model', choices=['decision_tree', 'logistic_regression'],
                      default='decision_tree', help='ML model type for automation (default: decision_tree)')

    return parser.parse_args()

def main() -> None:
    """Main function"""
    try:
        # Print startup information
        current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Current Date and Time (UTC): {current_time}")
        
        # Use getpass.getuser() for username
        try:
            username = getpass.getuser()
        except Exception:
            username = "unknown"
        print(f"Current User: {username}")
        
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load configuration if provided
        if args.config:
            config = load_config(args.config)
            # Update args with config values
            for key, value in config.items():
                setattr(args, key, value)
        
        logger.info(f"Starting trading system in {args.mode} mode")
        logger.info(f"Initial capital: ${args.capital:,.2f}")
        
        # Run selected mode
        if args.mode == 'backtest':
            run_backtest(args)
        else:
            run_live_trading(args)
        
    except KeyboardInterrupt:
        print("\nTrading system stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
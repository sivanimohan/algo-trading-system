"""
Trading System Main Script
Last Updated: 2025-06-26 11:41:15 UTC
Author: sivanimohan
"""

from datetime import datetime, timedelta, UTC
import logging
from src.data.trading_system import TradingSystem

def rsi_strategy(data):
    """Simple RSI strategy"""
    if 'RSI' not in data.columns or len(data) < 14:
        return 'hold'
    
    last_rsi = data['RSI'].iloc[-1]
    
    if last_rsi < 30:
        return 'buy'
    elif last_rsi > 70:
        return 'sell'
    return 'hold'

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Current Date and Time (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: sivanimohan")
    
    try:
        # Initialize trading system
        trading_system = TradingSystem(
            initial_capital=100000.0,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            credentials_path='credentials.json'
        )
        
        # Run backtest
        symbol = 'AAPL'
        start_date = datetime.now(UTC) - timedelta(days=365)
        end_date = datetime.now(UTC)
        
        results = trading_system.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=rsi_strategy
        )
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Trades: {results['metrics']['total_trades']}")
        print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
        print(f"Total P&L: ${results['metrics']['total_pnl']:,.2f}")
        print(f"Final Capital: ${results['metrics']['final_capital']:,.2f}")
        print(f"Return: {results['metrics']['return_pct']:.2f}%")
        print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
        
        # Print trade history
        print("\nTrade History:")
        for i, trade in enumerate(results['trades'], 1):
            print(f"\nTrade {i}:")
            print(f"Entry Date: {trade['entry_date'].strftime('%Y-%m-%-d')}")
            print(f"Entry Price: ${trade['entry_price']:.2f}")
            print(f"Exit Date: {trade['exit_date'].strftime('%Y-%m-%d')}")
            print(f"Exit Price: ${trade['exit_price']:.2f}")
            print(f"P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"Reason: {trade['close_reason']}")
        
        # Print locations of results
        print("\nResults saved to:")
        print(f"1. Local CSV: {trading_system.results_logger.trades_file}")
        print(f"2. Local JSON: {trading_system.results_logger.summary_file}")
        print(f"3. Google Sheets: {trading_system.sheets_logger.spreadsheet_url}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
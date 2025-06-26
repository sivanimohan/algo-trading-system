"""
Main Trading System Streamlit Frontend
Last Updated: 2025-06-26 18:27:49 UTC
Author: sivanimohan
"""

import streamlit as st
import pandas as pd
import json
import os
import getpass
import logging
from datetime import datetime, timedelta, timezone


from src.data.trading_system import TradingSystem
from src.data.automation_controller import AutomationController
from src.data.constants import (
    SYMBOLS, DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE_PCT,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT
)
from src.data.visualization import plot_performance  # Optional: add your plotting here
from src.data.strategies import STRATEGIES  # If you expose a STRATEGIES dict/list
from src.data.technical_indicators import available_indicators  # If you expose available indicators

def setup_logging():
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_{current_time}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_config(config_file: str):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return {}

def save_results(results: dict, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'results_{current_time}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
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
        st.success(f"Results saved to {output_dir}")
    except Exception as e:
        st.error(f"Error saving results: {e}")

st.set_page_config(page_title="AlgoTradingSystem", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {color: white; background: linear-gradient(to right, #0f2027, #2c5364);}
    .stSidebar {background-color: #1b263b;}
    .css-1d391kg {background: #0f2027;}
    .stDataFrame {background: #f1f6f9;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🚀 AlgoTradingSystem Dashboard")
st.caption("Main Trading System Streamlit Frontend &bull; Last Updated: 2025-06-26 18:27:49 UTC &bull; Author: sivanimohan")

st.sidebar.header("⚙️ Configuration")

# Config file
config_file = st.sidebar.text_input("Config file (optional, JSON)", value="")
config = load_config(config_file) if config_file else {}

# Trading parameters
capital = st.sidebar.number_input(
    "💰 Initial Capital", min_value=1000.0, value=float(config.get('capital', DEFAULT_INITIAL_CAPITAL)), step=1000.0
)
position_size = st.sidebar.slider(
    "📊 Position Size (% of Capital)", min_value=0.01, max_value=1.0, value=float(config.get('position_size', DEFAULT_POSITION_SIZE_PCT)), step=0.01
)
stop_loss = st.sidebar.slider(
    "🛑 Stop Loss (%)", min_value=0.01, max_value=0.5, value=float(config.get('stop_loss', DEFAULT_STOP_LOSS_PCT)), step=0.01
)
take_profit = st.sidebar.slider(
    "🎯 Take Profit (%)", min_value=0.01, max_value=0.5, value=float(config.get('take_profit', DEFAULT_TAKE_PROFIT_PCT)), step=0.01
)
use_sheets = st.sidebar.checkbox(
    "🗒️ Enable Google Sheets Logging", value=bool(config.get('use_sheets', False))
)
output_dir = st.sidebar.text_input("📁 Output directory", value=config.get('output_dir', 'trading_results'))

# Mode selection
mode = st.sidebar.radio("Mode", ["Backtest", "Live"])
strategy = st.sidebar.selectbox("Strategy", ["rsi", "macd", "bollinger", "combined"], index=["rsi", "macd", "bollinger", "combined"].index(config.get("strategy", "combined")))

# Backtest parameters
days = st.sidebar.number_input("Days for Backtest", min_value=30, max_value=1825, value=int(config.get("days", 365)), step=1)
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=days)

# Live trading parameters
max_retries = st.sidebar.number_input("Max Retries (Live)", min_value=0, value=int(config.get("max_retries", 3)), step=1)
retry_delay = st.sidebar.number_input("Retry Delay (sec, Live)", min_value=1, value=int(config.get("retry_delay", 5)), step=1)

# Symbols to trade
symbols = st.multiselect(
    "Select Symbols",
    options=SYMBOLS,
    default=SYMBOLS if not config.get("symbols") else config.get("symbols")
)
if not symbols:
    st.warning("Please select at least one symbol to proceed.")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Current Date and Time (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.write(f"**Current User:** {getpass.getuser()}")

setup_logging()
logger = logging.getLogger(__name__)

with st.expander("ℹ️ About This App"):
    st.write(
        """
        This dashboard integrates all core modules in your trading system:
        - **TradingSystem** (Backtest engine)
        - **AutomationController** (Live trading automation)
        - **Strategies** (From `strategies.py`)
        - **Technical Indicators** (From `technical_indicators.py`)
        - **Visualization** (From `visualization.py`)
        - **Google Sheets Logging** (From `google_sheets_logger.py`)
        - **Constants and Configuration** (From `constants.py`)
        - **Stock Data Management** (From `stock_data.py`)
        """
    )

st.markdown("---")

if st.button(f"🚦 Run {mode}"):
    logger.info(f"Starting trading system in {mode} mode")
    logger.info(f"Initial capital: ${capital:,.2f}")

    trading_system = TradingSystem(
        initial_capital=capital,
        position_size_pct=position_size,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        use_sheets=use_sheets
    )
    if mode == 'Backtest':
        with st.spinner("Running backtest..."):
            results = {}
            for symbol in symbols:
                # If your TradingSystem does not support symbols param, run for each symbol
                results[symbol] = trading_system.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                      # Make sure your method supports this!
                )
        save_results(results, output_dir)
        st.subheader("📈 Backtest Results Summary")
        total_pnl = 0
        total_trades = 0
        summary_rows = []
        for symbol, result in results.items():
            metrics = result.get("metrics", {})
            summary_rows.append({
                "Symbol": symbol,
                "Total Trades": metrics.get("total_trades", 0),
                "Win Rate (%)": round(metrics.get("win_rate", 0) * 100, 2),
                "P&L ($)": round(metrics.get("total_pnl", 0), 2),
                "Return (%)": round(metrics.get("return_pct", 0), 2)
            })
            total_pnl += metrics.get("total_pnl", 0)
            total_trades += metrics.get("total_trades", 0)
        st.dataframe(pd.DataFrame(summary_rows))
        st.write(f"**Overall Total Trades:** {total_trades}")
        st.write(f"**Overall Total P&L:** ${total_pnl:,.2f}")
        if capital:
            st.write(f"**Return on Capital:** {((total_pnl/capital)*100):.2f}%")
        # Detailed trades (if available)
        for symbol, result in results.items():
            trades = result.get("trades")
            if trades:
                st.subheader(f"📄 Trade Log for {symbol}")
                st.dataframe(pd.DataFrame(trades))
                st.download_button(
                    f"⬇️ Download {symbol} Trades CSV",
                    pd.DataFrame(trades).to_csv(index=False),
                    file_name=f"{symbol}_trades.csv"
                )
            # Optional: plot performance
            if hasattr(result, "performance") and callable(plot_performance):
                st.subheader(f"📊 Performance Chart for {symbol}")
                st.pyplot(plot_performance(result["performance"]))
    else:
        st.warning("Live trading started! (Make sure this is intentional.)")
        controller = AutomationController(
            trading_system=trading_system,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        controller.run()
        st.success("Live trading session ended.")

# Option to preview Google Sheets log
if st.checkbox("🗒️ Preview Google Sheets Log (if enabled)"):
    try:
        from src.data.google_sheets_logger import GoogleSheetsLogger
        creds_path = "src/data/credentials.json"
        logger = GoogleSheetsLogger(creds_path)
        sheet_records = logger.sheet.get_all_records()
        st.write(pd.DataFrame(sheet_records))
    except Exception as e:
        st.warning(f"Unable to preview Google Sheets log: {e}")

with st.expander("📚 Strategies Available"):
    try:
        st.write(STRATEGIES)
    except Exception:
        st.info("STRATEGIES dict/list not exposed in strategies.py. Add for richer UI.")

with st.expander("🧮 Technical Indicators Available"):
    try:
        st.write(available_indicators)
    except Exception:
        st.info("available_indicators not exposed in technical_indicators.py. Add for richer UI.")

with st.expander("📦 Raw Constants & Config"):
    st.json({
        "SYMBOLS": SYMBOLS,
        "DEFAULT_INITIAL_CAPITAL": DEFAULT_INITIAL_CAPITAL,
        "DEFAULT_POSITION_SIZE_PCT": DEFAULT_POSITION_SIZE_PCT,
        "DEFAULT_STOP_LOSS_PCT": DEFAULT_STOP_LOSS_PCT,
        "DEFAULT_TAKE_PROFIT_PCT": DEFAULT_TAKE_PROFIT_PCT
    })

with st.expander("📃 View trading_results folder"):
    try:
        files = os.listdir(output_dir)
        st.write(files)
    except Exception:
        st.info("No results directory found yet. Run a backtest first.")

with st.expander("📝 View Code Modules in src/data"):
    st.write(os.listdir("src/data"))
"""
Main Trading System Streamlit Frontend
Last Updated: 2025-06-27
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
from src.data.visualization import plot_performance, TradingVisualizer
from src.data.strategies import STRATEGIES
from src.data.technical_indicators import available_indicators

from src.data.ml_automation import MLAutomation

import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Stylish UI Settings ----------
st.set_page_config(
    page_title="🚀 AlgoTradingSystem Dashboard",
    layout="wide",
    page_icon="📈"
)
st.markdown(
    """
    <style>
    body, .main {background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;}
    .stButton>button {
        color: white !important;
        background: linear-gradient(90deg, #1e3c72, #2a5298) !important;
        border-radius: 10px !important;
        border: 0px !important;
        padding: 0.6em 2em !important;
        font-weight: bold !important;
        font-size: 1.12em !important;
    }
    .stSidebar {
        background: #22223b !important;
    }
    .stDataFrame, .stTable {
        background: #f9fafb !important;
        border-radius: 8px;
        box-shadow: 0 2px 16px rgba(44,62,80,0.13);
    }
    .block-container {
        padding-top: 1.4rem !important;
    }
    .css-1d391kg {background: #0f2027;}
    .metric-label {font-weight: bold !important;}
    .stExpanderHeader {font-size: 1.08em !important;}
    </style>
    """, unsafe_allow_html=True
)
# -----------------------------------------

st.title("🚀 AlgoTradingSystem Dashboard")
st.caption("""
<span style='font-weight:bold; color:#365486;'>Main Trading System Streamlit Frontend &bull; Last Updated: 2025-06-27 &bull; Author: sivanimohan</span>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2721/2721288.png", width=100)
st.sidebar.header("⚙️ Configuration")

config_file = st.sidebar.text_input("Config file (optional, JSON)", value="")
def load_config(config_file: str):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return {}
config = load_config(config_file) if config_file else {}

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

mode = st.sidebar.radio("Mode", ["Backtest", "Live"])
strategy = st.sidebar.selectbox("Strategy", ["rsi", "macd", "bollinger", "combined"], index=["rsi", "macd", "bollinger", "combined"].index(config.get("strategy", "combined")))

days = st.sidebar.number_input("Days for Backtest", min_value=30, max_value=1825, value=int(config.get("days", 365)), step=1)
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=days)

max_retries = st.sidebar.number_input("Max Retries (Live)", min_value=0, value=int(config.get("max_retries", 3)), step=1)
retry_delay = st.sidebar.number_input("Retry Delay (sec, Live)", min_value=1, value=int(config.get("retry_delay", 5)), step=1)

symbols = st.sidebar.multiselect(
    "Select Symbols",
    options=SYMBOLS,
    default=SYMBOLS if not config.get("symbols") else config.get("symbols")
)
if not symbols:
    st.sidebar.warning("Please select at least one symbol to proceed.")

# ML Automation options
st.sidebar.header("🤖 ML Automation (Bonus)")
enable_ml = st.sidebar.checkbox("Enable ML Automation (predict next-day movement)", value=False)
ml_model_type = st.sidebar.selectbox(
    "ML Model",
    [
        "decision_tree", "logistic_regression", "random_forest",
        "svm", "xgboost", "lightgbm", "mlp"
    ],
    index=0
)

# ---------- Main Section ----------
st.markdown("---")

col1, col2, col3 = st.columns([2,1,2])
with col1:
    st.metric("Current Date (UTC)", f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
with col2:
    st.metric("Current Time (UTC)", f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}")
with col3:
    st.metric("User", f"{getpass.getuser()}")

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
setup_logging()
logger = logging.getLogger(__name__)

with st.expander("ℹ️ About This App", expanded=False):
    st.info(
        """
        This dashboard integrates all core modules of your trading system:
        - **TradingSystem** (Backtest engine)
        - **AutomationController** (Live trading)
        - **Strategies** (RSI, MACD, etc.)
        - **Technical Indicators**
        - **Google Sheets Logging**
        - **ML Automation & Analytics**
        - **Visualization & Metrics**
        """
    )

results = {}

if st.button(f"🚦 Run {mode}", help="Start backtest/live trading and ML analytics!"):
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
        with st.spinner("⏳ Running backtest..."):
            for symbol in symbols:
                results[symbol] = trading_system.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                )
        # --- Table of metrics only, no images ---
        st.subheader("📈 Backtest Results Summary", divider="rainbow")
        summary_rows = []
        total_pnl, total_trades = 0, 0
        for symbol in results:
            inner = results[symbol]
            if symbol in inner:
                result = inner[symbol]
            else:
                st.warning(f"Symbol {symbol} not found in inner dict.")
                continue
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            try:
                total_trades_val = int(metrics.get("total_trades", 0) or 0)
                win_rate_val = float(metrics.get("win_rate", 0) or 0)
                total_pnl_val = float(metrics.get("total_pnl", 0) or 0)
                return_pct_val = float(metrics.get("return_pct", 0) or 0)
                sharpe_val = float(metrics.get("sharpe_ratio", 0) or 0)
            except Exception as e:
                st.warning(f"Could not parse metrics for {symbol}: {e}")
                total_trades_val = win_rate_val = total_pnl_val = return_pct_val = sharpe_val = 0
            summary_rows.append({
                "Symbol": symbol,
                "Total Trades": total_trades_val,
                "Win Rate": f"{win_rate_val*100:.2f}%",
                "Total P&L": f"${total_pnl_val:.2f}",
                "Return %": f"{return_pct_val*100:.2f}%",
                "Sharpe Ratio": f"{sharpe_val:.2f}"
            })
            total_pnl += total_pnl_val
            total_trades += total_trades_val
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        st.markdown(f"""
            <div style='margin-top:12px; font-size:1.1em;'>
            <b>Overall Total Trades:</b> {total_trades} &nbsp; &nbsp;
            <b>Overall Total P&L:</b> ${total_pnl:,.2f} &nbsp; &nbsp;
            <b>Return on Capital:</b> {((total_pnl/capital)*100):.2f}%
            </div>
        """, unsafe_allow_html=True)

        # ---------- PNG Performance Visualizations ----------
        st.markdown("---")
        st.subheader("📊 Visualizations", divider="rainbow")
        visualizer = TradingVisualizer(output_dir)
        figures_dir = os.path.join(output_dir, "figures")
        try:
            st.pyplot(plot_performance({s: inner[s] for s, inner in results.items() if s in inner}))
            performance_report = visualizer.plot_all({s: inner[s] for s, inner in results.items() if s in inner})
            st.success(f"Performance report generated: {performance_report}")

            for img_file in [
                'multi_symbol_performance.png',
                'equity_curves.png',
                'pnl_distribution.png',
                'monthly_returns_heatmap.png',
                'drawdowns.png'
            ]:
                img_path = os.path.join(figures_dir, img_file)
                if os.path.exists(img_path):
                    st.image(img_path, caption=img_file.replace(".png", "").replace("_", " ").title())
                else:
                    st.warning(f"{img_file} not found.")

            with open(performance_report, 'r') as f:
                st.components.v1.html(f.read(), height=900, scrolling=True)
        except Exception as viz_e:
            st.warning(f"Unable to display visualizations: {viz_e}")

        # ---------- ML Section ----------
        if enable_ml:
            from src.data.ml_pipeline import train_ml_pipeline
            from src.data.ml_predict import predict_next

            st.markdown("---")
            st.header("🤖 ML Analytics: Next-Day Prediction, Probabilities & Analysis", divider="violet")
            ml_summary = []
            for symbol in results:
                inner = results[symbol]
                if symbol in inner:
                    result = inner[symbol]
                else:
                    continue
                df = result.get("history")
                if isinstance(df, list):
                    df = pd.DataFrame(df)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        model, scaler, acc, feature_cols, X_test, y_test, y_pred, y_proba, report, cm = train_ml_pipeline(df, model_type=ml_model_type)
                        signal = predict_next(model, scaler, df, feature_cols)
                        st.markdown(f"""
                        <div style='background:#f1f6f9;padding:1em 1em 0.8em 1em;border-radius:9px;margin:15px 0 16px 0;'>
                        <span style='font-size:1.12em;color:#1e3c72;font-weight:bold;'>{symbol}</span>
                        <ul style='font-size:0.99em;margin:0.5em 0 0 0.5em;'>
                            <li><b>ML Model:</b> <span style='color:#3c096c;'>{ml_model_type}</span></li>
                            <li><b>Test Set Accuracy:</b> <span style='color:#3c096c;'>{acc:.2%}</span></li>
                            <li><b>Next-Day Signal:</b> <span style='font-weight:bold;color:{'green' if signal else 'crimson'};'>{'UP' if signal else 'DOWN'}</span></li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        ml_summary.append({
                            "Symbol": symbol,
                            "ML Model": ml_model_type,
                            "Test Accuracy": f"{acc:.2%}",
                            "Last Prediction": "UP" if signal else "DOWN"
                        })

                        # Probabilities plot
                        if y_proba is not None:
                            st.subheader(f"Prediction Probabilities (Test Set)", divider="blue")
                            fig, ax = plt.subplots(figsize=(6,3))
                            sns.histplot(y_proba, bins=20, ax=ax, kde=True, color='#2a5298')
                            ax.set_title(f"Prediction Probabilities for {symbol} ({ml_model_type})")
                            ax.set_xlabel("Probability of UP Movement")
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)

                        # Confusion matrix
                        st.subheader("Confusion Matrix", divider="blue")
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title(f"Confusion Matrix ({symbol})")
                        st.pyplot(fig)

                        # Classification report
                        st.subheader("Classification Report", divider="blue")
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                    except Exception as ml_e:
                        st.warning(f"[ML] Unable to run ML on {symbol}: {ml_e}")
                else:
                    st.info(f"[ML] No historical data for {symbol} to run ML.")

            if ml_summary:
                st.subheader("🧠 ML Prediction Summary Table (All Symbols)", divider="rainbow")
                st.dataframe(pd.DataFrame(ml_summary), use_container_width=True)

        # ---------- Trade Log Section ----------
        for symbol in results:
            inner = results[symbol]
            if symbol in inner:
                result = inner[symbol]
            else:
                continue
            trades = result.get("trades")
            if trades:
                st.subheader(f"📄 Trade Log for {symbol}")
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
                st.download_button(
                    f"⬇️ Download {symbol} Trades CSV",
                    pd.DataFrame(trades).to_csv(index=False),
                    file_name=f"{symbol}_trades.csv"
                )
    elif mode == "Live":
        st.warning("Live trading started! (Make sure this is intentional.)")
        controller = AutomationController(
            trading_system=trading_system,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        controller.run()
        st.success("Live trading session ended.")

# ---------- Utility Expanders ----------
with st.expander("🗒️ Preview Google Sheets Log (if enabled)"):
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
    try:
        st.write(os.listdir("src/data"))
    except Exception:
        st.info("src/data directory not found.")
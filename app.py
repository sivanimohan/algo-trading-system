"""
Main Trading System Streamlit Frontend
Last Updated: 2025-06-27
Author: sivanimohan
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
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
st.sidebar.header("⚙️ Trading Configuration")

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
    "💰 Starting Capital", min_value=1000.0, value=float(config.get('capital', DEFAULT_INITIAL_CAPITAL)), step=1000.0
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

days = st.sidebar.number_input("Backtest Window (Days)", min_value=30, max_value=1825, value=int(config.get("days", 365)), step=1)
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=days)

max_retries = st.sidebar.number_input("Max Retries (Live)", min_value=0, value=int(config.get("max_retries", 3)), step=1)
retry_delay = st.sidebar.number_input("Retry Delay (sec, Live)", min_value=1, value=int(config.get("retry_delay", 5)), step=1)

symbols = st.sidebar.multiselect(
    "Stock Tickers",
    options=SYMBOLS,
    default=SYMBOLS if not config.get("symbols") else config.get("symbols")
)
if not symbols:
    st.sidebar.warning("Please select at least one stock to proceed.")

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
    st.metric("Logged-in User", f"{getpass.getuser()}")

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
        This dashboard brings together your core trading system modules:
        - **TradingSystem** (Backtest Engine)
        - **AutomationController** (Live Trading)
        - **Strategy Modules** (RSI, MACD, etc.)
        - **Technical Indicators**
        - **Google Sheets Logging**
        - **ML Analytics**
        - **Stock Performance Visualizations & Metrics**
        """
    )

results = {}

if st.button(f"🚦 Run {mode}", help="Run Backtest or Live Trading including ML analytics!"):
    logger.info(f"Starting trading system in {mode} mode")
    logger.info(f"Starting capital: ${capital:,.2f}")

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
                st.warning(f"Stock {symbol} not found in results.")
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
                "Stock": symbol,
                "Number of Trades": total_trades_val,
                "Win Rate (%)": f"{win_rate_val*100:.2f}",
                "Total P&L (₹)": f"{total_pnl_val:.2f}",
                "Return (%)": f"{return_pct_val:.2f}",
                "Sharpe Ratio": f"{sharpe_val:.2f}"
            })
            total_pnl += total_pnl_val
            total_trades += total_trades_val
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        st.markdown(f"""
            <div style='margin-top:12px; font-size:1.1em;'>
            <b>Total Trades:</b> {total_trades} &nbsp; &nbsp;
            <b>Total P&L:</b> ₹{total_pnl:,.2f} &nbsp; &nbsp;
            <b>Return on Capital (%):</b> {((total_pnl/capital)*100):.2f}
            </div>
        """, unsafe_allow_html=True)

        # ---------- Stock Market Visualizations ----------
        st.markdown("---")
        st.subheader("📊 Stock Performance Visualizations", divider="rainbow")
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
            st.warning(f"")

        # ---------- ML Section ----------
        if enable_ml:
            from src.data.ml_pipeline import train_ml_pipeline
            from src.data.ml_predict import predict_next

            st.markdown("---")
            st.header("🤖 ML Analytics: Next-Day Stock Prediction", divider="violet")
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

                        msg = (
                            f"**Stock:** {symbol}\n\n"
                            f"- Model: '{ml_model_type.replace('_',' ').title()}'\n"
                            f"- Backtest Accuracy: **{acc*100:.1f}%**\n"
                            f"- Model's prediction for next day: "
                            f"**{'UP 📈' if signal else 'DOWN 📉'}**"
                        )
                        st.success(msg)

                        ml_summary.append({
                            "Stock": symbol,
                            "Model": ml_model_type,
                            "Backtest Accuracy (%)": f"{acc*100:.1f}",
                            "Prediction (Next Day)": "UP" if signal else "DOWN"
                        })

                        if y_proba is not None:
                            st.info(
                                "Distribution of model probabilities for predicting 'UP'."
                            )
                            st.subheader(f"Prediction Probability Distribution", divider="blue")
                            fig, ax = plt.subplots(figsize=(6,3))
                            sns.histplot(y_proba, bins=20, ax=ax, kde=True, color='#2a5298')
                            ax.set_title(f"Probability (UP) for {symbol} ({ml_model_type})")
                            ax.set_xlabel("Probability (UP)")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                        st.info("Confusion matrix shows the number of correct/incorrect predictions.")
                        st.subheader("Confusion Matrix", divider="blue")
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title(f"Confusion Matrix ({symbol})")
                        st.pyplot(fig)

                        st.info("Classification report with precision, recall, f1-score, and support.")
                        st.subheader("Classification Report", divider="blue")
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                    except Exception as ml_e:
                        st.warning(f"[ML] Unable to run ML on {symbol}: {ml_e}")
                else:
                    st.info(f"[ML] No historical data for {symbol} to run ML.")

            if ml_summary:
                st.subheader("🧠 ML Stock Prediction Summary Table", divider="rainbow")
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
                    f"⬇️ Download {symbol} Trade Log (CSV)",
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
with st.expander("🗒️ Preview Google Sheets Trade Log (if enabled)"):
    try:
        from src.data.google_sheets_logger import GoogleSheetsLogger
        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        logger = GoogleSheetsLogger(creds_path)
        sheet_records = logger.sheet.get_all_records()
        st.write(pd.DataFrame(sheet_records))
    except Exception as e:
        st.warning(f"Unable to preview Google Sheets log: {e}")

with st.expander("📚 Available Trading Strategies"):
    try:
        st.write(STRATEGIES)
    except Exception:
        st.info("No trading strategies found.")

with st.expander("🧮 Available Technical Indicators"):
    try:
        st.write(available_indicators)
    except Exception:
        st.info("No technical indicators found.")

with st.expander("📦 Stock Market Constants & Config"):
    st.json({
        "SYMBOLS": SYMBOLS,
        "DEFAULT_INITIAL_CAPITAL": DEFAULT_INITIAL_CAPITAL,
        "DEFAULT_POSITION_SIZE_PCT": DEFAULT_POSITION_SIZE_PCT,
        "DEFAULT_STOP_LOSS_PCT": DEFAULT_STOP_LOSS_PCT,
        "DEFAULT_TAKE_PROFIT_PCT": DEFAULT_TAKE_PROFIT_PCT
    })

with st.expander("📃 View trading_results directory (Reports & Charts)"):
    try:
        files = os.listdir(output_dir)
        st.write(files)
    except Exception:
        st.info("No trading_results directory found yet. Run a backtest first.")

with st.expander("📝 Code Modules in src/data"):
    try:
        st.write(os.listdir("src/data"))
    except Exception:
        st.info("src/data directory not found.")
# Algo Trading System

An advanced, modular algorithmic trading platform with a rich Streamlit dashboard for backtesting, live trading, ML analytics, and visualizations.

## Features

- **Streamlit Dashboard**: Intuitive web-based UI for configuring and running the trading system.
- **Backtest & Live Trading**: Seamlessly switch between historical backtesting and live automated trading.
- **Modular Strategies**: Easily extendable with multiple built-in strategies (RSI, MACD, Bollinger Bands, Combined, and more).
- **Technical Indicators**: Wide range of technical indicators available.
- **ML Automation**: Optional ML pipeline for predicting next-day stock movements, supporting various models (Decision Tree, Random Forest, XGBoost, etc.).
- **Google Sheets Logging**: Optionally log trades and results to Google Sheets.
- **Data Visualization**: Rich visual performance reports, equity curves, heatmaps, and more.
- **Detailed Metrics**: Trade logs, performance metrics, and summary tables for each backtest.

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) Google Cloud credentials for Google Sheets logging

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sivanimohan/algo-trading-system.git
   cd algo-trading-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - (Optional) Create a `.env` file for secrets (e.g., Google credentials path).

4. **Run the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```

## Usage

- Use the sidebar to configure trading parameters, select strategies, and enable/disable ML or Google Sheets logging.
- Choose between "Backtest" and "Live" mode.
- Run the system to view results, visualizations, download logs, and explore metrics.

## Project Structure

```
algo-trading-system/
├── app.py                       # Main Streamlit frontend and entry point
├── src/
│   └── data/
│       ├── trading_system.py    # Core trading logic
│       ├── automation_controller.py
│       ├── strategies.py
│       ├── constants.py
│       ├── technical_indicators.py
│       ├── visualization.py
│       ├── ml_automation.py
│       ├── ml_pipeline.py
│       ├── ml_predict.py
│       └── google_sheets_logger.py
├── logs/                        # Trading and system logs
├── trading_results/             # Generated reports and figures
├── requirements.txt
└── README.md
```

## Example Strategies & Indicators

- **Strategies**: RSI, MACD, Bollinger Bands, Combined, and more (see the sidebar in the app for full list).
- **Indicators**: Full set available in `src/data/technical_indicators.py`.

## ML Analytics

Enable ML in the sidebar to run classification models (e.g., Random Forest, XGBoost) for next-day price movement predictions, with accuracy reports, confusion matrices, and probability distributions.

## Logging & Reports

- **Logs**: All trading activity and errors are stored in the `logs/` directory.
- **Reports**: Backtest results, trade logs, and performance charts are stored in `trading_results/`.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License.

---

*Happy Trading! 🚀*

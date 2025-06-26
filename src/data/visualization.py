"""
Trading Visualization Module
Last Updated: 2025-06-26 14:37:23 UTC
Author: sivanimohan
"""

"""
Trading Visualization Module
Last Updated: 2025-06-26 15:09:12 UTC
Author: sivanimohan
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from datetime import datetime, UTC

class TradingVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set style using a valid matplotlib style
        plt.style.use('default')  # Changed from 'seaborn' to 'default'
        
        # Configure seaborn
        sns.set_theme()  # Use seaborn's default theme
        sns.set_palette("husl")  # Set color palette

    def plot_multi_symbol_performance(self, symbols_results: Dict[str, Dict]):
        """Plot performance comparison across multiple symbols"""
        try:
            # Create performance comparison plots
            plt.figure(figsize=(15, 10))
            
            # Prepare data
            symbols = list(symbols_results.keys())
            returns = [result['metrics']['return_pct'] for result in symbols_results.values()]
            win_rates = [result['metrics']['win_rate'] * 100 for result in symbols_results.values()]
            
            # Plot returns and win rates
            x = np.arange(len(symbols))
            width = 0.35
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Returns plot
            bars1 = ax1.bar(x - width/2, returns, width, label='Return %')
            ax1.set_ylabel('Return %')
            ax1.set_title('Returns by Symbol')
            ax1.set_xticks(x)
            ax1.set_xticklabels(symbols)
            ax1.legend()
            
            # Win rates plot
            bars2 = ax2.bar(x + width/2, win_rates, width, label='Win Rate %')
            ax2.set_ylabel('Win Rate %')
            ax2.set_title('Win Rates by Symbol')
            ax2.set_xticks(x)
            ax2.set_xticklabels(symbols)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'multi_symbol_performance.png'))
            plt.close()
        except Exception as e:
            raise Exception(f"Error plotting multi-symbol performance: {str(e)}")

    # ... rest of the methods remain the same ...

    def plot_equity_curves(self, symbols_results: Dict[str, Dict]):
        """Plot equity curves for all symbols"""
        plt.figure(figsize=(15, 8))
        
        for symbol, results in symbols_results.items():
            equity_data = pd.DataFrame(results['equity_curve'])
            plt.plot(equity_data['date'], equity_data['equity'], label=symbol)
        
        plt.title('Portfolio Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'equity_curves.png'))
        plt.close()

    def plot_trade_distribution(self, symbols_results: Dict[str, Dict]):
        """Plot trade P&L distribution for all symbols"""
        plt.figure(figsize=(15, 8))
        
        for symbol, results in symbols_results.items():
            pnl_data = [t['pnl'] for t in results['trades']]
            sns.kdeplot(data=pnl_data, label=symbol)
        
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'pnl_distribution.png'))
        plt.close()

    def plot_monthly_returns(self, symbols_results: Dict[str, Dict]):
        """Plot monthly returns heatmap"""
        # Prepare monthly returns data
        monthly_returns = {}
        
        for symbol, results in symbols_results.items():
            trades = results['trades']
            returns = pd.Series([t['pnl'] for t in trades])
            returns.index = pd.to_datetime([t['exit_date'] for t in trades])
            monthly_returns[symbol] = returns.resample('M').sum()
        
        # Create heatmap
        returns_df = pd.DataFrame(monthly_returns)
        plt.figure(figsize=(15, 8))
        sns.heatmap(returns_df.T, cmap='RdYlGn', center=0, annot=True, fmt='.0f')
        plt.title('Monthly Returns Heatmap')
        plt.savefig(os.path.join(self.figures_dir, 'monthly_returns_heatmap.png'))
        plt.close()

    def plot_drawdowns(self, symbols_results: Dict[str, Dict]):
        """Plot drawdown analysis"""
        plt.figure(figsize=(15, 8))
        
        for symbol, results in symbols_results.items():
            equity_data = pd.DataFrame(results['equity_curve'])
            equity_series = equity_data['equity']
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max * 100
            plt.plot(equity_data['date'], drawdown, label=symbol)
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'drawdowns.png'))
        plt.close()

    def generate_performance_report(self, symbols_results: Dict[str, Dict]):
        """Generate comprehensive performance report"""
        report_time = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.figures_dir, f'performance_report_{report_time}.html')
        
        html_content = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "img { max-width: 100%; height: auto; margin: 20px 0; }",
            "</style>",
            "</head><body>",
            f"<h1>Trading Performance Report - {report_time}</h1>"
        ]
        
        # Add performance metrics table
        html_content.extend([
            "<h2>Performance Metrics</h2>",
            "<table>",
            "<tr><th>Symbol</th><th>Total Trades</th><th>Win Rate</th><th>Total P&L</th><th>Return %</th><th>Sharpe Ratio</th></tr>"
        ])
        
        for symbol, results in symbols_results.items():
            metrics = results['metrics']
            html_content.append(
                f"<tr><td>{symbol}</td>"
                f"<td>{metrics['total_trades']}</td>"
                f"<td>{metrics['win_rate']:.2%}</td>"
                f"<td>${metrics['total_pnl']:,.2f}</td>"
                f"<td>{metrics['return_pct']:.2f}%</td>"
                f"<td>{metrics['sharpe_ratio']:.2f}</td></tr>"
            )
        
        html_content.append("</table>")
        
        # Add charts
        charts = [
            'multi_symbol_performance.png',
            'equity_curves.png',
            'pnl_distribution.png',
            'monthly_returns_heatmap.png',
            'drawdowns.png'
        ]
        
        for chart in charts:
            html_content.extend([
                f"<h2>{chart.replace('.png', '').replace('_', ' ').title()}</h2>",
                f"<img src='{chart}' alt='{chart}'>"
            ])
        
        html_content.append("</body></html>")
        
        with open(report_file, 'w') as f:
            f.write("\n".join(html_content))
        
        return report_file

    def plot_all(self, symbols_results: Dict[str, Dict]):
        """Generate all plots and report"""
        self.plot_multi_symbol_performance(symbols_results)
        self.plot_equity_curves(symbols_results)
        self.plot_trade_distribution(symbols_results)
        self.plot_monthly_returns(symbols_results)
        self.plot_drawdowns(symbols_results)
        return self.generate_performance_report(symbols_results)
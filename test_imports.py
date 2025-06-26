"""
Test Imports
Last Updated: 2025-06-26 15:09:12 UTC
Author: sivanimohan
"""

import sys
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    try:
        logger.info("Testing imports...")
        
        # Test StockDataFetcher
        from src.data.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()
        logger.info("✅ StockDataFetcher imported successfully")
        
        # Test TradingSystem
        from src.data.trading_system import TradingSystem
        trading_system = TradingSystem(use_sheets=False)
        logger.info("✅ TradingSystem imported successfully")
        
        # Test TechnicalIndicators
        from src.data.technical_indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        logger.info("✅ TechnicalIndicators imported successfully")
        
        # Test Visualization
        from src.data.visualization import TradingVisualizer
        test_dir = os.path.join(os.getcwd(), 'test_results')
        visualizer = TradingVisualizer(test_dir)
        logger.info("✅ TradingVisualizer imported successfully")
        
        logger.info("All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing imports: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
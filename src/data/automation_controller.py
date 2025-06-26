"""
Automation Controller Module
Last Updated: 2025-06-26 15:25:33 UTC
Author: sivanimohan
"""

import logging
import time
from datetime import datetime, time as dt_time, UTC
import threading
import schedule
from typing import Optional

from .trading_system import TradingSystem

class AutomationController:
    def __init__(self, 
                 trading_system: TradingSystem,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """Initialize the automation controller"""
        self.logger = logging.getLogger(__name__)
        self.trading_system = trading_system
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.is_running = False
        self.market_open = dt_time(13, 30)  # 13:30 UTC (9:30 EST)
        self.market_close = dt_time(20, 0)   # 20:00 UTC (4:00 EST)
        
    def is_market_hours(self) -> bool:
        """Check if it's during market hours"""
        current_time = datetime.now(UTC).time()
        return self.market_open <= current_time <= self.market_close
    
    def execute_with_retry(self, func, *args, **kwargs) -> Optional[bool]:
        """Execute a function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed")
                    return None
    
    def check_positions(self) -> None:
        """Check and update all positions"""
        try:
            self.trading_system.update_positions()
            self.logger.info("Positions updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def generate_reports(self) -> None:
        """Generate trading reports"""
        try:
            # Generate daily performance report
            daily_report = self.trading_system.generate_daily_report()
            self.logger.info("Daily report generated")
            
            # Update performance metrics
            self.trading_system.update_metrics()
            self.logger.info("Performance metrics updated")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def generate_weekly_report(self) -> None:
        """Generate weekly trading summary"""
        try:
            weekly_report = self.trading_system.generate_weekly_report()
            self.logger.info("Weekly report generated")
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
    
    def generate_monthly_report(self) -> None:
        """Generate monthly trading summary"""
        try:
            monthly_report = self.trading_system.generate_monthly_report()
            self.logger.info("Monthly report generated")
        except Exception as e:
            self.logger.error(f"Error generating monthly report: {e}")
    
    def check_market_conditions(self) -> None:
        """Check market conditions and update trading parameters"""
        try:
            self.trading_system.update_market_conditions()
            self.logger.info("Market conditions updated")
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
    
    def schedule_jobs(self) -> None:
        """Schedule automated jobs"""
        # Market hours checks (every 5 minutes)
        schedule.every(5).minutes.do(self.check_positions)
        
        # Market condition updates (every 15 minutes)
        schedule.every(15).minutes.do(self.check_market_conditions)
        
        # Daily reports (after market close)
        schedule.every().day.at("20:05").do(self.generate_reports)
        
        # Weekly report (Friday after market close)
        schedule.every().friday.at("20:15").do(self.generate_weekly_report)
        
        # Monthly report (last day of month after market close)
        schedule.every().day.at("20:30").do(self.check_monthly_report)
    
    def check_monthly_report(self) -> None:
        """Check if it's the last day of month and generate monthly report"""
        today = datetime.now(UTC)
        tomorrow = today.replace(day=today.day + 1)
        if tomorrow.month != today.month:  # Last day of month
            self.generate_monthly_report()
    
    def run_scheduler(self) -> None:
        """Run the scheduler"""
        self.schedule_jobs()
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def start_trading(self) -> None:
        """Start trading operations"""
        try:
            self.logger.info("Starting trading...")
            self.trading_system.start_trading()
        except Exception as e:
            self.logger.error(f"Error starting trading: {e}")
    
    def stop_trading(self) -> None:
        """Stop trading operations"""
        try:
            self.logger.info("Stopping trading...")
            self.trading_system.stop_trading()
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    def run(self) -> None:
        """Main run method"""
        self.logger.info("Starting automation controller...")
        self.is_running = True
        
        # Start scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.start()
        
        self.logger.info("Starting trading loop...")
        
        try:
            while self.is_running:
                if self.is_market_hours():
                    self.logger.info("Market is open. Starting trading...")
                    self.start_trading()
                    
                    # Main trading loop
                    while self.is_market_hours() and self.is_running:
                        time.sleep(1)  # Prevent CPU overuse
                    
                    self.stop_trading()
                    self.logger.info("Market is closed. Stopped trading.")
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal...")
        finally:
            self.is_running = False
            scheduler_thread.join()
            self.logger.info("Automation controller stopped.")
    
    def stop(self) -> None:
        """Stop the automation controller"""
        self.logger.info("Stopping automation controller...")
        self.is_running = False
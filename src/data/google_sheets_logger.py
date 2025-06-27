from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import logging
from datetime import datetime, UTC
import json
import os
from typing import Dict, Optional
import gspread
import traceback

class GoogleSheetsLogger:
    def __init__(self, credentials_path: str, spreadsheet_name: str = "TradingSystemLog", worksheet_name: str = "Trade Log"):
        self.logger = logging.getLogger(__name__)
        self.credentials_path = self._find_credentials(credentials_path)
        self.spreadsheet_name = spreadsheet_name
        self.worksheet_name = worksheet_name

        # gspread initialization (for append_row)
        try:
            self.gc = gspread.service_account(filename=self.credentials_path)
            print(f"DEBUG: self.gc type: {type(self.gc)}")
            self.sheet = self.gc.open(self.spreadsheet_name).worksheet(self.worksheet_name)
            print(f"DEBUG: self.sheet type: {type(self.sheet)}")
            self.logger.info("gspread worksheet opened successfully.")
        except Exception as e:
            print(f"DEBUG: Exception type: {type(e)}, Exception value: {e}")
            traceback.print_exc()
            self.logger.error(f"gspread initialization failed: {e}")
            raise

        # Google Sheets API initialization (for batch operations etc)
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.service = build('sheets', 'v4', credentials=credentials)
            self.logger.info("Google Sheets API service initialized.")
        except Exception as e:
            print(f"DEBUG: Exception type: {type(e)}, Exception value: {e}")
            traceback.print_exc()
            self.logger.error(f"Google Sheets API initialization failed: {e}")
            raise

        self.spreadsheet_id = '1yhmCMKFpMsjvsEfRzzXZWXCaU7Mao8bYt1MSK7r-12A'

    def _find_credentials(self, provided_path: Optional[str] = None) -> str:
        possible_locations = [
            provided_path,
            os.path.join(os.getcwd(), 'src', 'data', 'credentials.json'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credentials.json'),
            os.path.join(os.getcwd(), 'credentials.json'),
            'credentials.json'
        ]
        for path in [p for p in possible_locations if p]:
            if os.path.exists(path):
                return os.path.abspath(path)
        raise FileNotFoundError("Credentials file not found.")

    def log_system_event(self, event_type: str, details: dict):
        try:
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                event_type,
                json.dumps(details, default=str)
            ]
            result = self.sheet.append_row(row)
            self.logger.info(f"System event logged: {row}")
        except Exception as e:
            print(f"DEBUG: Exception type: {type(e)}, Exception value: {e}")
            traceback.print_exc()
            self.logger.error(f"Failed to log system event to Google Sheets: {e}")

    def get_spreadsheet_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}"
    def log_trade(self, trade: dict):
        """
        Log a trade to Google Sheets.
        Converts all datetime/Timestamp fields to string before logging.
        """
        def safe_str(x):
            if isinstance(x, (pd.Timestamp, np.datetime64)):
                return str(x)
            elif isinstance(x, datetime):
                return x.strftime('%Y-%m-%d %H:%M:%S')
            return x

        row = [
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            safe_str(trade.get('symbol', '')),
            safe_str(trade.get('direction', '')),
            safe_str(trade.get('entry_date', '')),
            safe_str(trade.get('entry_price', '')),
            safe_str(trade.get('exit_date', '')),
            safe_str(trade.get('exit_price', '')),
            safe_str(trade.get('shares', '')),
            safe_str(trade.get('position_size', '')),
            safe_str(trade.get('pnl', '')),
            safe_str(trade.get('pnl_pct', '')),
            safe_str(trade.get('close_reason', ''))
            
        ]
        self.sheet.append_row(row, value_input_option="USER_ENTERED")
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("🔄 Starting Google Sheets access test...")
    print(f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger = GoogleSheetsLogger("credentials.json")
    logger.log_system_event("Test Start", {"info": "This is a test event"})
    print(f"Spreadsheet URL: {logger.get_spreadsheet_url()}")
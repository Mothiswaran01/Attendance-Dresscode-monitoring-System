import gspread
import threading
import queue
import time
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from utils.config import (
    SHEET_NAME, CREDENTIALS_FILE, COL_NAME, 
    COL_SESSIONS, COL_ID_CARD, COL_TUCKIN, COL_SHOES
)

class GoogleSheetsLogger:
    def __init__(self):
        # 1. Setup Authentication
        self.scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
        try:
            self.creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, self.scope)
            self.client = gspread.authorize(self.creds)
            self.spreadsheet = self.client.open(SHEET_NAME)
            
            self.sheet = self._get_or_create_daily_sheet()
            
            print(f"Google Sheets connected. Active tab: {self.sheet.title}")
        except Exception as e:
            print(f"Google Sheets connection failed: {e}")
            self.sheet = None

        self.student_rows = {}
        if self.sheet:
            self._refresh_row_map()

        self.log_queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _get_or_create_daily_sheet(self):
        """Creates a new tab for today or ensures headers exist in current one."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        required_headers = [
            "Name", "Session 1", "Session 2", "Session 3", "Session 4", 
            "Session 5", "Session 6", "Session 7", "ID Card", "Tuck-in", "Shoes"
        ]
        
        try:
            ws = self.spreadsheet.worksheet(today_str)
            
            first_row = ws.row_values(1)
            if not first_row:
                ws.append_row(required_headers)
                print(f"Restored missing headers in {today_str}")
            return ws

        except gspread.exceptions.WorksheetNotFound:
            new_sheet = self.spreadsheet.add_worksheet(title=today_str, rows="100", cols="20")
            new_sheet.append_row(required_headers)
            print(f"Created new daily sheet with headers: {today_str}")
            return new_sheet

    def _refresh_row_map(self):
        """Initial scan of the sheet to find where students are located."""
        try:
            names = self.sheet.col_values(COL_NAME)
            for i, name in enumerate(names):
                if name and i > 0:  # Skip header
                    self.student_rows[name] = i + 1
            print(f"Loaded {len(self.student_rows)} students from {self.sheet.title}")
        except Exception as e:
            print(f"Error refreshing row map: {e}")

    def _worker(self):
        """The 'Secretary' thread that processes logs one by one."""
        while True:
            name, col_index = self.log_queue.get()
            if self.sheet:
                try:
                    if name not in self.student_rows:
                        self.sheet.append_row([name])
                        new_row_index = len(self.sheet.col_values(COL_NAME))
                        self.student_rows[name] = new_row_index
                        print(f"Enrolled new student: {name}")

                    row = self.student_rows[name]
                    
                    current_val = self.sheet.cell(row, col_index).value or ""
                    
                    if current_val.endswith("2"):
                        new_val = "3"
                    elif current_val.endswith("1"):
                        new_val = "2"
                    else:
                        new_val = "1"

                    if current_val != "3":
                        self.sheet.update_cell(row, col_index, new_val)
                        print(f"Logged {name} in column {col_index} -> {new_val}")
                    
                except Exception as e:
                    print(f"Background worker error: {e}")
                    time.sleep(2) 
            
            self.log_queue.task_done()

    def submit_log(self, name, metric_type, session_id=None):
        """
        Public method to push logs.
        """
        col_index = None
        
        if metric_type == 'attendance' and session_id:
            col_index = COL_SESSIONS.get(session_id)
        elif metric_type == 'id_card':
            col_index = COL_ID_CARD
        elif metric_type == 'tuckin':
            col_index = COL_TUCKIN
        elif metric_type == 'shoes':
            col_index = COL_SHOES

        if col_index:
            self.log_queue.put((name, col_index))
# Google Sheets Setup

The system logs attendance and dress code compliance data to Google Sheets in real time.

## Prerequisites

- A Google Cloud Platform project
- Google Sheets API enabled
- A Google Service Account

## Step 1: Create a Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select or create a project
3. Navigate to **IAM & Admin > Service Accounts**
4. Click **Create Service Account**
5. Give it a name (e.g., `attendance-logger`)
6. Assign role **Editor** (or create a custom role with Sheets API access)
7. Click **Done**

## Step 2: Generate a JSON Key

1. In the Service Accounts list, click on your new account
2. Go to the **Keys** tab
3. Click **Add Key > Create New Key**
4. Choose **JSON** and download the file
5. Rename it to `credentials.json` and place it in the project root

## Step 3: Create a Google Sheet

1. Create a new Google Sheet at [sheets.google.com](https://sheets.google.com/)
2. Note the sheet name (default: `Attendance_Logs`)
3. Click **Share** and add your service account email (found in `credentials.json` under `client_email`) as an **Editor**

## Step 4: Configure Environment

Set these variables (or update `utils/config.py`):

```bash
GOOGLE_SHEET_NAME=Attendance_Logs
GOOGLE_CREDENTIALS_FILE=credentials.json
```

## Sheet Structure

Each day gets a new tab named `YYYY-MM-DD` with these columns:

| Column | Header | Purpose |
|---|---|---|
| A | Name | Student name |
| B–H | Session 1–7 | Attendance per session (1, 2, or 3) |
| I | ID Card | ID card compliance |
| J | Tuck-in | Tuck-in compliance |
| K | Shoes | Shoes compliance |

## How Logging Works

- Logs are queued via a background worker thread (non-blocking)
- Each session interval has a 200s audit window
- Students with >= 15 face detections within the window are approved
- Dress code checks (ID, tuck, shoes) require >= 10 detections
- Values increment: empty → `1` → `2` → `3` (3 is the maximum)

# Attendance & Dress Code Monitoring System

Real-time attendance tracking and dress code compliance using face recognition and computer vision.

## Features

- **Face Recognition** — InsightFace (SCRFD detection + ArcFace embeddings) with temporal voting buffer for stable identification
- **Attendance Logging** — Automatic audit bursts (200s windows) per session interval, logs to Google Sheets
- **Dress Code Compliance** — YOLO-based detection of ID cards, shoes, and tuck-in status
- **Live Streaming** — Flask MJPEG stream with real-time overlay (optional desktop GUI)
- **Google Sheets Integration** — Async worker queue for non-blocking sheet updates

## Project Structure

```
├── main.py                  # Desktop GUI version (OpenCV window)
├── main_stream.py           # Flask streaming server version
├── attendance/
│   ├── face_encoding.py     # Load embeddings, match faces (cosine similarity)
│   └── attendance_recognition.py  # Face analyzer, temporal buffer, frame processing
├── utils/
│   ├── config.py            # Paths, thresholds, session schedule, environment variables
│   ├── excel_utils.py       # Google Sheets async logger
│   └── video_to_frames.py   # Extract face crops from video files
├── models/
│   └── face_recognition/
│       ├── generate_database.py     # Build embedding database from face dataset
│       ├── test_arcface_embedding.py    # Test ArcFace embedding extraction
│       └── yolo_face_test.py            # Test YOLO face detection
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `rtsp://camera-host/stream` | RTSP camera URL |
| `GOOGLE_SHEET_NAME` | `Attendance_Logs` | Google Sheets document name |
| `GOOGLE_CREDENTIALS_FILE` | `credentials.json` | Service account JSON key path |
| `POSE_MODEL_NAME` | `yolov8n-pose.pt` | YOLO pose model filename |

### 3. Model weights

Place model weights in `models/dresscode_detection/saved_models/`:
- `tuck_best.pt`
- `shoe_v3_final.pt`
- `shoes_best.pt`
- `idcard_best.pt`

Face recognition uses InsightFace's `buffalo_l` model (auto-downloaded on first run).

### 4. Face embedding database

Generate a face embedding database using `generate_database.py` or place a pre-built `student_embeddings.pkl` in the `data/` directory.

### 5. Google Sheets credentials

Place a service account JSON key at `credentials.json` and share your Google Sheet with the service account email.

## Usage

### Desktop version

```bash
python main.py
```

Displays an OpenCV window with real-time annotations. Press `q` to quit.

### Flask stream

```bash
python main_stream.py
```

- Video feed: `http://localhost:5000/video_feed`
- Health check: `http://localhost:5000/health`

## How It Works

1. **Audit Bursts** — On startup and at each session interval change, the system runs a 200-second high-intensity audit window.
2. **Face Recognition** — Every frame is processed through InsightFace. Detected faces are matched against the embedding database using cosine similarity (threshold: 0.50).
3. **Temporal Buffer** — A 7-frame sliding window with majority voting (minimum 4 votes) prevents flickering identity changes.
4. **Dress Code Checks** — During audit windows, YOLO models detect ID cards and shoes; pose keypoints estimate tuck-in compliance.
5. **Logging** — Students meeting the hit threshold (15/300 frames) are logged to Google Sheets per session interval.

## Configuration

Edit `utils/config.py` to adjust:
- Session timings (default: 7 periods, 09:20–16:30)
- Audit duration and hit thresholds
- Compliance confidence thresholds
- Google Sheets column mappings

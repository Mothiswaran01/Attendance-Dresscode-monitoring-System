import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR                = os.path.join(BASE_DIR, "data")
MODELS_DIR              = os.path.join(BASE_DIR, "models")
EXCEL_DIR               = os.path.join(BASE_DIR, "excel")
LOGS_DIR                = os.path.join(BASE_DIR, "logs")


EMBEDDINGS_PATH         = os.path.join(DATA_DIR, "student_embeddings.pkl")  # Path to the embeddings database
FACE_RECOGNITION_THRESHOLD = 0.50
TEMPORAL_BUFFER_SIZE    = 7
TEMPORAL_MIN_VOTES      = 4


TUCK_MODEL_PATH         = os.path.join(MODELS_DIR, "dresscode_detection", "saved_models", "tuck_best.pt")
IDCARD_MODEL_PATH       = os.path.join(MODELS_DIR, "dresscode_detection", "saved_models", "idcard_best.pt")
SHOES_MODEL_PATH        = os.path.join(MODELS_DIR, "dresscode_detection", "saved_models", "shoes_best.pt")

COMPLIANCE_CONFIDENCE   = 0.5
COMPLIANCE_BUFFER_SIZE  = 7
COMPLIANCE_MIN_VOTES    = 4
POSE_MODEL_NAME = os.environ.get("POSE_MODEL_NAME", "yolov8n-pose.pt")

SHOE_CLASS_COMPLIANT    = [1]        # 'Shoe'
SHOE_CLASS_VIOLATION    = [0]        # 'Sandal'

TUCK_CLASS_COMPLIANT    = 0          # 'Shirt Tucked In'
TUCK_CLASS_VIOLATION    = 1          # 'Shirt Tucked Out'


SESSIONS = {
    1: ("09:20", "10:10"),
    2: ("10:10", "11:00"),
    3: ("11:15", "12:05"),
    4: ("12:05", "12:50"),
    5: ("13:45", "14:40"),
    6: ("14:40", "15:35"),
    7: ("15:35", "16:30"),
}


SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Attendance_Logs")
CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")

AUDIT_DURATION = 200  # High-intensity burst (seconds)
HIT_THRESHOLD = 15   # 15 detections out of 300 to approve

COL_NAME = 1
COL_SESSIONS = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
COL_ID_CARD = 9
COL_TUCKIN = 10
COL_SHOES = 11

def get_current_session_info():
    """Calculates active session and which 16-min interval we are in."""
    now_str = datetime.now().strftime("%H:%M")
    fmt = "%H:%M"
    
    for sid, (start, end) in SESSIONS.items():
        if start <= now_str <= end:
            d1, d2 = datetime.strptime(start, fmt), datetime.strptime(end, fmt)
            total_mins = (d2 - d1).seconds / 60
            interval_len = total_mins / 3
            
            elapsed = (datetime.strptime(now_str, fmt) - d1).seconds / 60
            interval_id = int(elapsed // interval_len) + 1
            return sid, min(interval_id, 3)
    return None, None


EXCEL_PATH              = os.path.join()  # Update with your actual Excel file name


CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "rtsp://camera-host/stream")


PROOF_DIR               = os.path.join(DATA_DIR, "processed", "dresscode_frames")


if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("EMBEDDINGS_PATH:", EMBEDDINGS_PATH)
    print("TUCK_MODEL:", TUCK_MODEL_PATH)
    print("IDCARD_MODEL:", IDCARD_MODEL_PATH)
    print("SHOES_MODEL:", SHOES_MODEL_PATH)
    
    for name, path in [("Tuck", TUCK_MODEL_PATH), ("ID Card", IDCARD_MODEL_PATH), ("Shoes", SHOES_MODEL_PATH), ("Embeddings", EMBEDDINGS_PATH)]:
        status = "Found" if os.path.exists(path) else "MISSING"
        print(f"{status} — {name}: {path}")
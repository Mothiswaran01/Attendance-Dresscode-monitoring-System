import cv2
import numpy as np
import warnings
import os
import time
import threading
from datetime import datetime
from flask import Flask, Response

# Suppress warnings
warnings.filterwarnings("ignore", message="`rcond` parameter will change to the default")

from ultralytics import YOLO
from attendance.attendance_recognition import load_face_analyzer, process_frame, TemporalBuffer
from attendance.face_encoding import load_embeddings

# Import updated custom modules
from utils.config import (
    CAMERA_SOURCE, IDCARD_MODEL_PATH, SHOES_MODEL_PATH, POSE_MODEL_NAME,
    get_current_session_info, AUDIT_DURATION, HIT_THRESHOLD
)
from utils.excel_utils import GoogleSheetsLogger

# ==========================================
# Flask App Initialization
# ==========================================
app = Flask(__name__)

# ==========================================
# 1. THREADED CAPTURE CLASS
# ==========================================
class LatestFrameReader:
    def __init__(self, rtsp_url):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;10240000|max_delay;0|fflags;nobuffer|flags;low_delay"
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = False, None
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(1)
                continue
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret, self.frame = ret, frame
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Camera thread warning: {e}")
                time.sleep(1)

    def get_latest(self):
        with self.lock:
            if self.frame is None: return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("Booting monitoring system...")

face_analyzer = load_face_analyzer()
student_embeddings = load_embeddings()
shoe_model = YOLO(SHOES_MODEL_PATH)
idcard_model = YOLO(IDCARD_MODEL_PATH)
pose_model = YOLO(POSE_MODEL_NAME)

sheet_logger = GoogleSheetsLogger()
stream = LatestFrameReader(CAMERA_SOURCE)

# State Management
face_buffers = {}
audit_data = {}  
current_interval_key = None 
last_audit_reset = 0
first_run_audit_complete = False  

process_interval = 0.1 
last_process_time = 0

print("Initialization complete. Monitoring...")

# ==========================================
# 3. FRAME GENERATOR FOR STREAMING
# ==========================================
def generate_frames():
    global face_buffers, audit_data, current_interval_key, last_audit_reset, first_run_audit_complete, last_process_time
    
    while True:
        current_time = time.time()
        
        if current_time - last_process_time >= process_interval:
            success, frame = stream.get_latest()
            if not success: continue
            last_process_time = current_time 

            session_id, interval_id = get_current_session_info()
            interval_key = (session_id, interval_id) if session_id else None

            # --- AUDIT TRIGGER ---
            is_auditing = False
            if not first_run_audit_complete:
                if last_audit_reset == 0: 
                    last_audit_reset = current_time
                    print("Startup audit: starting monitoring burst...")
                
                if current_time - last_audit_reset <= AUDIT_DURATION: 
                    is_auditing = True
                else: 
                    first_run_audit_complete = True
                    current_interval_key = interval_key
                    print("Startup audit finished.")
            
            elif interval_key != current_interval_key:
                current_interval_key = interval_key
                audit_data = {} 
                last_audit_reset = current_time
                print(f"Interval audit: session {session_id}, part {interval_id} started.")

            if interval_key and (current_time - last_audit_reset <= AUDIT_DURATION):
                is_auditing = True

            height, width, _ = frame.shape
            confirmed_names_this_frame = []

            # --- ENGINE 1: FACE ID ---
            if face_analyzer:
                face_results = process_frame(frame, face_analyzer, student_embeddings)
                for name, score, bbox in face_results:
                    # Terminal log: see each face detection attempt
                    print(f"Detection update ({score:.2f})", end='\r')

                    if name == "Unknown" or score < 0.50: 
                        continue 

                    if name not in face_buffers: face_buffers[name] = TemporalBuffer()
                    face_buffers[name].add(name)
                    confirmed_name = face_buffers[name].get_decision()
                    
                    if confirmed_name:
                        confirmed_names_this_frame.append(confirmed_name)
                        if is_auditing:
                            if confirmed_name not in audit_data:
                                audit_data[confirmed_name] = {"att": 0, "id": 0, "tuck": 0, "shoe": 0, "logged": False}
                            audit_data[confirmed_name]["att"] += 1
                    
                    # Visuals
                    x1, y1, x2, y2 = bbox
                    label = f"{confirmed_name}" if confirmed_name else "Identifying..."
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- ENGINE 2: ID CARD ---
            if is_auditing and confirmed_names_this_frame:
                id_res = idcard_model.predict(frame, conf=0.5, verbose=False)
                if any(len(r.boxes) > 0 for r in id_res):
                    for name in confirmed_names_this_frame:
                        if name in audit_data: audit_data[name]["id"] += 1

            # --- ENGINE 3: SHOES ---
            if is_auditing and confirmed_names_this_frame:
                shoe_res = shoe_model.predict(frame, conf=0.25, verbose=False)
                shoe_pass = any(shoe_model.names[int(box.cls[0])] == "shoe" for r in shoe_res for box in r.boxes)
                if shoe_pass:
                    for name in confirmed_names_this_frame:
                        if name in audit_data: audit_data[name]["shoe"] += 1

            # --- ENGINE 4: TUCK-IN ---
            if is_auditing and confirmed_names_this_frame:
                pose_res = pose_model.predict(frame, conf=0.5, verbose=False)
                for r in pose_res:
                    if r.keypoints is not None and len(r.keypoints.xy) > 0:
                        kp = r.keypoints.xy[0].cpu().numpy()
                        if len(kp) >= 13 and all(kp[i][0] > 0 for i in [5, 6, 11, 12]):
                            center_x = int((kp[5][0] + kp[6][0]) / 2)
                            c_y1, c_y2 = int(kp[5][1]) + 30, int((kp[5][1] + kp[11][1]) / 2)
                            b_y = int((kp[11][1] + kp[12][1]) / 2)
                            d_y1 = max(0, b_y - 25)
                            
                            if c_y1 < c_y2:
                                c_crop, d_crop = frame[c_y1:c_y2, center_x-20:center_x+20], frame[d_y1:b_y, center_x-20:center_x+20]
                                if c_crop.size > 0 and d_crop.size > 0:
                                    if abs(np.mean(c_crop) - np.mean(d_crop)) > 40:
                                        for name in confirmed_names_this_frame:
                                            if name in audit_data: audit_data[name]["tuck"] += 1

            # Final audit terminal log
            if current_time - last_audit_reset > AUDIT_DURATION and audit_data:
                print("\n--- Audit Results Summary ---")
                for name, counts in audit_data.items():
                    if not counts["logged"]:
                        status = "APPROVED" if counts["att"] >= HIT_THRESHOLD else "REJECTED (low hits)"
                        print(f"Result summary | Hits: {counts['att']}/{HIT_THRESHOLD} | Status: {status}")
                        print(f"Dress code summary | ID: {counts['id']} | Shoe: {counts['shoe']} | Tuck: {counts['tuck']}")
                        
                        if counts["att"] >= HIT_THRESHOLD:
                            sheet_logger.submit_log(name, 'attendance', session_id)
                            if counts["id"] >= 10: sheet_logger.submit_log(name, 'id_card')
                            if counts["shoe"] >= 10: sheet_logger.submit_log(name, 'shoes')
                            if counts["tuck"] >= 10: sheet_logger.submit_log(name, 'tuckin')
                        
                        counts["logged"] = True
                print("--------------------------------\n")

            # UI Visuals
            status_color = (0, 255, 0) if is_auditing else (0, 165, 255)
            text = f"BURST ACTIVE: {int(AUDIT_DURATION - (current_time - last_audit_reset))}s" if is_auditing else "IDLE - WAITING"
            cv2.putText(frame, text, (width-420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # ==========================================
            # MJPEG Frame Encoding
            # ==========================================
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 4. FLASK ROUTE
# ==========================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {"status": "online"}, 200

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stream.stop()

import cv2
import threading
import time
import signal
from flask import Flask, Response, request
import move_jetbot  # Import ไฟล์ที่คุณสร้างไว้

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
run_thread = True
new_frame_available = False

# ตัวแปรสำหรับระบบความปลอดภัย
last_command_time = time.time()

# --- 1. ส่วนของกล้อง (ดึงมาจาก stream_video.py ของคุณ) ---
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)816, height=(int)616, format=(string)NV12, framerate=(fraction)21/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True"
    )

def fetch_frames():
    global output_frame, run_thread, new_frame_available
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    try:
        while run_thread:
            success, frame = cap.read()
            if not success: continue
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with lock:
                    output_frame = buffer.tobytes()
                    new_frame_available = True
    finally:
        cap.release()

def generate():
    global output_frame, run_thread, new_frame_available
    while run_thread:
        if not new_frame_available:
            time.sleep(0.001)
            continue
        with lock:
            frame_to_send = output_frame
            new_frame_available = False
        if frame_to_send:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 2. ส่วนรับคำสั่งมอเตอร์ (เพิ่มเข้ามาใหม่) ---
@app.route('/motor', methods=['POST'])
def motor_control():
    global last_command_time
    data = request.get_json()
    
    left = float(data.get('left', 0))
    right = float(data.get('right', 0))
    
    # เรียกใช้ผ่าน move_jetbot ตามที่ต้องการ
    move_jetbot.set_motors(left, right)
    
    last_command_time = time.time()  # รีเซ็ตเวลา Watchdog
    return "OK", 200

# --- 3. ระบบความปลอดภัย (Watchdog) ---
def watchdog():
    while run_thread:
        # ถ้า PC เงียบหายไปเกิน 0.5 วินาที ให้หยุดรถทันที
        if time.time() - last_command_time > 0.5:
            move_jetbot.stop()
        time.sleep(0.1)

if __name__ == '__main__':
    # รัน Thread สำหรับดึงภาพ
    threading.Thread(target=fetch_frames, daemon=True).start()
    # รัน Thread สำหรับ Watchdog
    threading.Thread(target=watchdog, daemon=True).start()
    
    # รัน Flask Server
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
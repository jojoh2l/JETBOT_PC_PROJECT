import cv2
import threading
import time
import signal
from flask import Flask, Response

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
run_thread = True
new_frame_available = False # ตัวแปรเช็คว่ามีภาพใหม่มาหรือยัง

def gstreamer_pipeline():
    # ปรับเป็น 224x224 เพื่อความเร็วสูงสุด และใช้ nvvidconv จัดการ
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
            if not success:
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]) # ลด Quality ลงนิดเพื่อความเร็ว
            if ret:
                with lock:
                    output_frame = buffer.tobytes()
                    new_frame_available = True
    finally:
        cap.release()

def generate():
    global output_frame, run_thread, new_frame_available
    while run_thread:
        # รอจนกว่าจะมีภาพใหม่จริงๆ (Event-based logic)
        if not new_frame_available:
            time.sleep(0.001) # Sleep สั้นมากๆ เพื่อไม่ให้กิน CPU
            continue

        with lock:
            frame_to_send = output_frame
            new_frame_available = False # รีเซ็ตสถานะ

        if frame_to_send:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=fetch_frames)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
import cv2
import threading
from flask import Flask, Response

app = Flask(__name__)
output_frame = None
lock = threading.Lock() # ใช้ Lock เพื่อป้องกันการเขียน/อ่านภาพพร้อมกันจนพัง

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)20/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def fetch_frames():
    global output_frame
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # เข้ารหัสภาพเป็น JPG ครั้งเดียว
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            with lock:
                output_frame = buffer.tobytes()

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            frame_to_send = output_frame
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # สร้าง Background Thread เพื่ออ่านกล้องแยกต่างหาก
    t = threading.Thread(target=fetch_frames)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
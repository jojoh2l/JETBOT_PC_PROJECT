import cv2
import os
import sys
import socket
from datetime import datetime
from pathlib import Path

def check_jetbot_connection(ip, port, timeout=2):
    """ส่ง Socket ไปเช็คว่า Server เปิดอยู่ไหมก่อนเปิดกล้องจริง"""
    try:
        socket.create_connection((ip, port), timeout=timeout)
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

# --- 1. เลือกประเภทข้อมูลที่จะเก็บ (เพิ่มรายการที่ 3) ---
print("=== Jetbot Video Capture Tool (Socket Safety) ===")
print("1: Road Following (เก็บใน datasets/raw/video-road)")
print("2: Collision Avoidance (เก็บใน datasets/raw/video-collision)")
print("3: YOLO Object Detection (เก็บใน datasets/raw/YOLO)")

try:
    choice = input("กรุณาเลือกประเภท (1, 2 หรือ 3) หรือกด Ctrl+C เพื่อออก: ")
except KeyboardInterrupt:
    print("\nยกเลิกโดยผู้ใช้")
    sys.exit()

# ปรับ Map ตามชื่อโฟลเดอร์ที่คุณต้องการ
task_map = {
    "1": "video-road", 
    "2": "video-collision", 
    "3": "YOLO"
}
task_name = task_map.get(choice)

if not task_name:
    print("❌ ตัวเลือกไม่ถูกต้อง! ปิดโปรแกรม")
    sys.exit()

# ตั้งค่า IP และ Port (เหมือนเดิม)
JETBOT_IP = "172.20.10.2"
JETBOT_PORT = 5000
url = f"http://{JETBOT_IP}:{JETBOT_PORT}/video_feed"

# --- 2. เช็คดวงก่อนเปิดกล้อง ---
print(f"\n🔍 กำลังตรวจสอบการเชื่อมต่อกับ {JETBOT_IP}...")

if not check_jetbot_connection(JETBOT_IP, JETBOT_PORT):
    print(f"❌ ERROR: ไม่สามารถติดต่อ Jetbot ได้ที่ {JETBOT_IP}:{JETBOT_PORT}")
    print("ระบบตัดการทำงานทันที (ป้องกันอาการค้าง)")
    print("กรุณาเช็ค WiFi หรือรัน Server บน Jetbot ก่อนครับ")
    sys.exit()

print("✅ พบสัญญาณ Jetbot! กำลังเตรียมเปิดกล้อง...")

# สร้างโฟลเดอร์เก็บข้อมูล
session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path("datasets") / "raw" / task_name / session_name
save_dir.mkdir(parents=True, exist_ok=True)

# --- 3. เริ่มการทำงานวิดีโอ ---
cap = cv2.VideoCapture(url)
out = None # ตัวแปรสำหรับ VideoWriter

try:
    if not cap.isOpened():
        print("❌ OpenCV เปิดกล้องไม่ได้แม้ว่าจะ Ping เจอ (อาจเป็นปัญหาที่ URL)")
    else:
        # อ่านเฟรมแรกเพื่อเอาขนาดภาพ (Width, Height)
        ret, first_frame = cap.read()
        if ret:
            h, w, _ = first_frame.shape
            video_filename = str(save_dir / f"{session_name}.mp4")
            
            # ตั้งค่า Codec และ VideoWriter (ใช้ mp4v สำหรับไฟล์ .mp4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
            
            print(f"📹 กำลังบันทึกวิดีโอไปที่: {video_filename}")
            print("กด 'q' เพื่อหยุดการบันทึกและบันทึกไฟล์")
        else:
            print("⚠️ ไม่สามารถรับข้อมูลเฟรมจากกล้องได้")
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ สัญญาณภาพขาดหาย...")
                break

            # เขียนเฟรมลงไฟล์วิดีโอ
            out.write(frame)

            # แสดงผลหน้าจอ
            cv2.imshow(f"Recording Video [{task_name.upper()}]", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\n🛑 หยุดการบันทึกโดยผู้ใช้")

finally:
    # เคลียร์ทรัพยากร
    if cap.isOpened():
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"\n📦 เรียบร้อย! วิดีโอถูกบันทึกในโฟลเดอร์ {session_name}")
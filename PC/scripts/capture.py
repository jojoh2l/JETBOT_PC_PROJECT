import cv2
import os
import sys
import socket
from datetime import datetime
from pathlib import Path

def check_jetbot_connection(ip, port, timeout=2):
    """ส่ง Socket ไปเช็คว่า Server เปิดอยู่ไหมก่อนเปิดกล้องจริง"""
    try:
        # พยายามเชื่อมต่อแบบตั้งเวลาจำกัด (Timeout)
        socket.create_connection((ip, port), timeout=timeout)
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

# --- 1. เลือกประเภทข้อมูลที่จะเก็บ ---
print("=== Jetbot Data Capture Tool (Socket Safety) ===")
print("1: Road Following (เก็บใน datasets/raw/road)")
print("2: Collision Avoidance (เก็บใน datasets/raw/collision)")
print("3: Collision Avoidance (เก็บใน datasets/raw/YOLO)")

try:
    choice = input("กรุณาเลือกประเภท (1 หรือ 2 หรือ 3) หรือกด Ctrl+C เพื่อออก: ")
except KeyboardInterrupt:
    print("\nยกเลิกโดยผู้ใช้")
    sys.exit()

task_map = {"1": "road", "2": "collision", "3": "YOLO"}
task_name = task_map.get(choice)

if not task_name:
    print("❌ ตัวเลือกไม่ถูกต้อง! ปิดโปรแกรม")
    sys.exit()

# ตั้งค่า IP และ Port
JETBOT_IP = "172.20.10.2"
JETBOT_PORT = 5000
url = f"http://{JETBOT_IP}:{JETBOT_PORT}/video_feed"

# --- 2. จุดเด็ด: เช็คดวงก่อนเปิดกล้อง ---
print(f"\n🔍 กำลังตรวจสอบการเชื่อมต่อกับ {JETBOT_IP}...")

if not check_jetbot_connection(JETBOT_IP, JETBOT_IP_PORT := JETBOT_PORT):
    print(f"❌ ERROR: ไม่สามารถติดต่อ Jetbot ได้ที่ {JETBOT_IP}:{JETBOT_PORT}")
    print("ระบบตัดการทำงานทันที (ป้องกันอาการค้าง)")
    print("กรุณาเช็ค WiFi หรือรัน Server บน Jetbot ก่อนครับ")
    sys.exit()

print("✅ พบสัญญาณ Jetbot! กำลังเตรียมเปิดกล้อง...")

# สร้างโฟลเดอร์เก็บข้อมูล
session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path("datasets") / "raw" / task_name / session_name
save_dir.mkdir(parents=True, exist_ok=True)

# --- 3. เริ่มการทำงานปกติ ---
cap = cv2.VideoCapture(url)

frame_count = 0

try:
    if not cap.isOpened():
        print("❌ OpenCV เปิดกล้องไม่ได้แม้ว่าจะ Ping เจอ (อาจเป็นปัญหาที่ URL)")
    else:
        print(f"📸 บันทึกไปที่: {save_dir}")
        print("กด 's' เพื่อบันทึกภาพ | กด 'q' เพื่อออก")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ สัญญาณภาพขาดหาย...")
                break

            cv2.imshow(f"Capture [{task_name.upper()}]", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                frame_count += 1
                filename = f"{session_name}_{frame_count:06d}.jpg"
                filepath = save_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"📷 บันทึก: {filename}")

            elif key == ord("q"):
                break

except KeyboardInterrupt:
    print("\n🛑 หยุดการทำงานโดยผู้ใช้")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📦 เรียบร้อย! บันทึกทั้งหมด {frame_count} ภาพ")
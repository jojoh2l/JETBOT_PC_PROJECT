import cv2
import csv
import os
from pathlib import Path

# --- ตั้งค่า (Multiplier System) ---
RAW_DIR       = Path("datasets/raw/road")
OUT_DIR       = Path("datasets/road_following_80p")
IMAGES_DIR    = OUT_DIR / "images"
CSV_PATH      = OUT_DIR / "labels.csv"

CROP_RATIO    = 0.8        # เก็บเฉพาะ 80% ด้านล่าง
SAVE_SIZE     = (224, 224) # ขนาดที่ AI ใช้เทรน
DISPLAY_SCALE = 2          # ตัวคูณขยายหน้าจอ (ปรับเป็น 2, 3 ได้ตามชอบ)

# คำนวณขนาดหน้าต่าง UI อัตโนมัติ
DISPLAY_W = int(SAVE_SIZE[0] * DISPLAY_SCALE)
DISPLAY_H = int(SAVE_SIZE[1] * DISPLAY_SCALE)
WINDOW_NAME = f"Road Labeler | Scale x{DISPLAY_SCALE} | D=Next A=Prev X=Delete Q=Exit"

# สร้างโฟลเดอร์ถ้ายังไม่มี
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# โหลดข้อมูลเก่าจาก CSV (ถ้ามี)
labels = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = (float(row["x"]), float(row["y"]))

# ดึงรายชื่อภาพดิบทั้งหมด
all_images = sorted(list(RAW_DIR.glob("*/*.jpg")))
if not all_images:
    print(f"❌ ไม่พบภาพใน {RAW_DIR}"); exit()

current_idx = 0

def save_csv():
    """บันทึก labels ทั้งหมดลง CSV"""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x", "y"])
        writer.writeheader()
        for fname, (nx, ny) in sorted(labels.items()):
            writer.writerow({"filename": fname, "x": nx, "y": ny})

def process_frame(img_path):
    """ทำ Preprocessing: Crop -> Resize -> Scale for Display"""
    img = cv2.imread(str(img_path))
    if img is None: return None, None
    
    fname = img_path.name
    h, w = img.shape[:2]
    # Crop 80% ล่าง
    start_y = int(h * (1 - CROP_RATIO))
    cropped = img[start_y:, :]
    
    # Resize เป็นขนาดเทรน (224x224)
    resized = cv2.resize(cropped, SAVE_SIZE)
    
    # ขยายเพื่อโชว์บนหน้าจอ UI
    display = cv2.resize(resized, (DISPLAY_W, DISPLAY_H))
    
    if fname in labels:
        nx, ny = labels[fname]
        px, py = int(nx * DISPLAY_W), int(ny * DISPLAY_H)
        
        # 1. วาดเส้นจากพิกัดที่คลิก ไปยัง กึ่งกลางล่างของจอ (สีฟ้า)
        cv2.line(display, (px, py), (int(DISPLAY_W / 2), DISPLAY_H), (255, 255, 0), 2)

        cv2.line(display, (int(0), int(DISPLAY_H / 2)), (int(DISPLAY_W), int(DISPLAY_H / 2)), (255, 255, 0), 1)
        cv2.line(display, (int(DISPLAY_W / 2), int(0)), (int(DISPLAY_W / 2), int(DISPLAY_H)), (255, 255, 0), 1)
        
        cv2.circle(display, (px, py), 4, (0, 0, 225), -1)

        # --- ส่วนที่เพิ่มเข้ามา: แสดงพิกัด x, y ข้างจุด ---
        cv2.putText(display, f"{nx:.3f}, {ny:.3f}", (px - 50, py - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
        
        cv2.putText(display, "SAVED", (DISPLAY_W-80, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return display, resized

def on_mouse(event, x, y, flags, param):
    """คลิกเมาส์เพื่อ Label"""
    global current_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        fname = all_images[current_idx].name
        # Normalize พิกัดตามขนาดหน้าจอที่โชว์จริง
        labels[fname] = (x / DISPLAY_W, y / DISPLAY_H)
        
        # บันทึกภาพ 224x224
        _, resized = process_frame(all_images[current_idx])
        cv2.imwrite(str(IMAGES_DIR / fname), resized)
        save_csv()
        print(f"✅ Saved: {fname} at ({labels[fname][0]:.3f}, {labels[fname][1]:.3f})")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    p = all_images[current_idx]
    display, _ = process_frame(p)
    
    if display is not None:
        header = f"[{current_idx+1}/{len(all_images)}] {p.name}"
        cv2.putText(display, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('d'): current_idx = (current_idx + 1) % len(all_images)
    elif key == ord('a'): current_idx = (current_idx - 1) % len(all_images)
    elif key == ord('x'): # ลบข้อมูล
        fname = p.name
        if fname in labels:
            del labels[fname]
            target_file = IMAGES_DIR / fname
            if target_file.exists(): target_file.unlink()
            save_csv()
            print(f"🗑️ Deleted Label & Image: {fname}")

cv2.destroyAllWindows()
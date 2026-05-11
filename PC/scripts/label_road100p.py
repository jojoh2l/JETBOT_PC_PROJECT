"""
label_road.py
-------------
Label tool สำหรับ road following
- เปิดภาพจาก datasets/raw/ ทุก session
- คลิกภาพ = บันทึกจุด x,y (normalized 0.0-1.0)
- บันทึกลง datasets/road_following/labels.csv
- copy ภาพต้นฉบับลง datasets/road_following/images/

ปุ่ม:
  คลิกซ้าย  = กำหนด / เปลี่ยนจุด
  →  หรือ d  = ภาพถัดไป
  ←  หรือ a  = ภาพก่อนหน้า
  Delete     = ลบ label ภาพนี้
  q          = ออก
"""

import cv2
import csv
import os
import shutil
from pathlib import Path

# ─── การตั้งค่า ────────────────────────────────────────────────────────────────
RAW_DIR      = Path("datasets/raw")
OUT_DIR      = Path("datasets/road_following")
IMAGES_DIR   = OUT_DIR / "images"
CSV_PATH     = OUT_DIR / "labels.csv"
DISPLAY_SIZE = 448          # ขนาดที่แสดงบนจอ (ภาพจริง 224x224 → ขยาย 2x)
WINDOW_NAME  = "Label Tool  |  คลิก=วางจุด  |  D/→=ถัดไป  A/←=ย้อนกลับ  Backspace/X=ลบ  Q=ออก"

# ─── เตรียมโฟลเดอร์ ────────────────────────────────────────────────────────────
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ─── โหลด CSV ที่มีอยู่แล้ว ────────────────────────────────────────────────────
# labels = { "filename.jpg": (x, y) }
labels = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = (float(row["x"]), float(row["y"]))
    print(f"โหลด label เดิม {len(labels)} รายการ")

# ─── รวบรวมภาพทั้งหมดจากทุก session ────────────────────────────────────────────
all_images = sorted(RAW_DIR.glob("*/*.jpg"))
if not all_images:
    print(f"ไม่พบภาพใน {RAW_DIR}")
    exit()

print(f"พบภาพทั้งหมด {len(all_images)} ภาพ")

# ─── หาภาพแรกที่ยังไม่ได้ label ───────────────────────────────────────────────
labeled_names = set(labels.keys())
unlabeled = [p for p in all_images if p.name not in labeled_names]
if unlabeled:
    start_index = all_images.index(unlabeled[0])
    print(f"label แล้ว {len(labeled_names)} ภาพ — เริ่มที่ภาพ {start_index + 1}")
else:
    start_index = 0
    print("label ครบทุกภาพแล้ว — เปิดตั้งแต่ต้น")

# ─── บันทึก CSV ───────────────────────────────────────────────────────────────
def save_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x", "y"])
        writer.writeheader()
        for fname, (x, y) in sorted(labels.items()):
            writer.writerow({"filename": fname, "x": round(x, 4), "y": round(y, 4)})

# ─── วาดภาพพร้อมจุดและข้อความ ────────────────────────────────────────────────
def draw_frame(img_path, index):
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        return None

    # ขยายภาพสำหรับแสดงผล
    display = cv2.resize(img_orig, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

    fname = img_path.name
    total = len(all_images)
    labeled_count = len(labels)

    # แถบสถานะด้านบน
    bar = f"  [{index + 1}/{total}]  labeled: {labeled_count}  |  {fname}"
    cv2.putText(display, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ถ้ามี label วาดจุดและข้อความ
    if fname in labels:
        x_norm, y_norm = labels[fname]
        px = int(x_norm * DISPLAY_SIZE)
        py = int(y_norm * DISPLAY_SIZE)

        # วงกลมจุด
        cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(display, (px, py), 7, (255, 255, 255), 1)

        # ข้อความ x,y เหนือจุด
        text = f"({x_norm:.2f}, {y_norm:.2f})"
        tx = max(px - 45, 4)
        ty = max(py - 12, 16)
        cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return display

# ─── callback เมื่อคลิกเมาส์ ──────────────────────────────────────────────────
current_index = [start_index]

def on_mouse(event, mx, my, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    img_path = all_images[current_index[0]]
    fname    = img_path.name

    # แปลงพิกัดจอ → normalized (จากภาพจริง)
    x_norm = round(mx / DISPLAY_SIZE, 4)
    y_norm = round(my / DISPLAY_SIZE, 4)

    # คลิปให้อยู่ในช่วง 0-1
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    # บันทึก label
    labels[fname] = (x_norm, y_norm)
    save_csv()

    # copy ภาพต้นฉบับไปที่ images/ (ถ้ายังไม่มี)
    dest = IMAGES_DIR / fname
    if not dest.exists():
        shutil.copy2(str(img_path), str(dest))

    # วาดใหม่
    frame = draw_frame(img_path, current_index[0])
    if frame is not None:
        cv2.imshow(WINDOW_NAME, frame)

# ─── main loop ────────────────────────────────────────────────────────────────
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

# แสดงภาพแรก
frame = draw_frame(all_images[current_index[0]], current_index[0])
if frame is not None:
    cv2.imshow(WINDOW_NAME, frame)

while True:
    key = cv2.waitKey(20) & 0xFF
    changed = False

    # → หรือ d = ถัดไป
    if key in (ord("d"), 83):   # 83 = arrow right
        if current_index[0] < len(all_images) - 1:
            current_index[0] += 1
            changed = True
        else:
            print("ถึงภาพสุดท้ายแล้ว")

    # ← หรือ a = ย้อนกลับ
    elif key in (ord("a"), 81): # 81 = arrow left
        if current_index[0] > 0:
            current_index[0] -= 1
            changed = True

    # Delete = ลบ label
    elif key == 8 or key == ord("x"):  # 255=Delete, 8=Backspace
        fname = all_images[current_index[0]].name
        if fname in labels:
            del labels[fname]
            save_csv()
            # ลบไฟล์ที่ copy ไปด้วย
            dest = IMAGES_DIR / fname
            if dest.exists():
                dest.unlink()
            print(f"ลบ label: {fname}")
            changed = True

    # q = ออก
    elif key == ord("q"):
        break

    # วาดภาพใหม่ถ้ามีการเปลี่ยน
    if changed:
        frame = draw_frame(all_images[current_index[0]], current_index[0])
        if frame is not None:
            cv2.imshow(WINDOW_NAME, frame)

    # ถ้าปิดหน้าต่างด้วยปุ่ม X
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
print(f"\nจบ — label ทั้งหมด {len(labels)} ภาพ")
print(f"CSV อยู่ที่  : {CSV_PATH}")
print(f"ภาพอยู่ที่  : {IMAGES_DIR}")
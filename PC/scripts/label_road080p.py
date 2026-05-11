import cv2
import csv
import os
from pathlib import Path

# --- ตั้งค่า ---
RAW_DIR      = Path("datasets/raw/road")
OUT_DIR      = Path("datasets/road_following_80p")
IMAGES_DIR   = OUT_DIR / "images"
CSV_PATH     = OUT_DIR / "labels.csv"
CROP_BOTTOM  = 0.8
SAVE_SIZE    = (224, 224)
DISPLAY_SIZE = 448
WINDOW_NAME  = "Road Labeler | Click=Point | D=Next A=Prev X=Delete Q=Exit"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

labels = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = (float(row["x"]), float(row["y"]))

all_images = sorted(list(RAW_DIR.glob("*/*.jpg")))
if not all_images:
    print(f"ไม่พบภาพใน {RAW_DIR}"); exit()

current_idx = 0

def save_csv():
    """บันทึกข้อมูลลงไฟล์ CSV ทันที"""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x", "y"])
        writer.writeheader()
        for fname, (x, y) in sorted(labels.items()):
            writer.writerow({"filename": fname, "x": x, "y": y})

def process_and_draw(img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    cropped = img[int(h*(1-CROP_BOTTOM)):, :]
    resized = cv2.resize(cropped, SAVE_SIZE)
    display = cv2.resize(resized, (DISPLAY_SIZE, DISPLAY_SIZE))
    
    fname = img_path.name
    cv2.putText(display, f"{current_idx+1}/{len(all_images)}: {fname}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if fname in labels:
        x_n, y_n = labels[fname]
        px, py = int(x_n * DISPLAY_SIZE), int(y_n * DISPLAY_SIZE)
        cv2.circle(display, (px, py), 5, (0, 255, 0), -1) # เปลี่ยนเป็นสีเขียวเมื่อเซฟแล้ว
        cv2.putText(display, "SAVED", (DISPLAY_SIZE-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return display, resized

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        fname = all_images[current_idx].name
        labels[fname] = (x / DISPLAY_SIZE, y / DISPLAY_SIZE)
        _, resized = process_and_draw(all_images[current_idx])
        cv2.imwrite(str(IMAGES_DIR / fname), resized)
        save_csv() # เซฟทันที

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    display, _ = process_and_draw(all_images[current_idx])
    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q'): break
    elif key == ord('d'): current_idx = (current_idx + 1) % len(all_images)
    elif key == ord('a'): current_idx = (current_idx - 1) % len(all_images)
    elif key == ord('x'):
        fname = all_images[current_idx].name
        if fname in labels:
            del labels[fname]
            img_file = IMAGES_DIR / fname
            if img_file.exists(): img_file.unlink()
            save_csv() # เซฟทันทีที่ลบ
            print(f"ลบข้อมูล: {fname}")

cv2.destroyAllWindows()
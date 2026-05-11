import cv2
import csv
from pathlib import Path

# --- ตั้งค่า ---
RAW_DIR      = Path("datasets/raw/collision")
OUT_DIR      = Path("datasets/collision_avoidance_80p")
IMAGES_DIR   = OUT_DIR / "images"
CSV_PATH     = OUT_DIR / "labels.csv"
CROP_BOTTOM  = 0.8
SAVE_SIZE    = (224, 224)
DISPLAY_SIZE = 448
WINDOW_NAME  = "Collision Labeler | 0,1,2=Class | X=Delete | Q=Exit"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

labels = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = int(row["class_id"])

all_images = sorted(list(RAW_DIR.glob("*/*.jpg")))
if not all_images: print(f"ไม่พบภาพใน {RAW_DIR}"); exit()

current_idx = 0

def save_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class_id"])
        writer.writeheader()
        for fname, cid in sorted(labels.items()):
            writer.writerow({"filename": fname, "class_id": cid})

while True:
    p = all_images[current_idx]
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    cropped = img[int(h*(1-CROP_BOTTOM)):, :]
    resized = cv2.resize(cropped, SAVE_SIZE)
    display = cv2.resize(resized, (DISPLAY_SIZE, DISPLAY_SIZE))

    fname = p.name
    status = f"Class: {labels[fname]}" if fname in labels else "Unlabeled"
    color = (0, 255, 0) if fname in labels else (0, 0, 255)
    
    cv2.putText(display, f"{current_idx+1}/{len(all_images)} | {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(WINDOW_NAME, display)
    
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'): break
    elif key == ord('d'): current_idx = (current_idx + 1) % len(all_images)
    elif key == ord('a'): current_idx = (current_idx - 1) % len(all_images)
    elif ord('0') <= key <= ord('2'): # รองรับ 0, 1, 2
        class_id = int(chr(key))
        labels[fname] = class_id
        cv2.imwrite(str(IMAGES_DIR / fname), resized)
        save_csv() # เซฟทันที
        current_idx = (current_idx + 1) % len(all_images) # ไปรูปถัดไปทันทีเพื่อความเร็ว
    elif key == ord('x'):
        if fname in labels:
            del labels[fname]
            img_file = IMAGES_DIR / fname
            if img_file.exists(): img_file.unlink()
            save_csv() # เซฟทันที
            print(f"ลบ Label: {fname}")

cv2.destroyAllWindows()
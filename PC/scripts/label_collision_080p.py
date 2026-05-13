import cv2
import csv
from pathlib import Path

# --- ตั้งค่า ---
RAW_DIR       = Path("datasets/raw/collision")
OUT_DIR       = Path("datasets/collision_avoidance_80p")
IMAGES_DIR    = OUT_DIR / "images"
CSV_PATH      = OUT_DIR / "labels.csv"

CROP_RATIO    = 0.8
SAVE_SIZE     = (224, 224)
DISPLAY_SCALE = 2

DISPLAY_W = int(SAVE_SIZE[0] * DISPLAY_SCALE)
DISPLAY_H = int(SAVE_SIZE[1] * DISPLAY_SCALE)
WINDOW_NAME = "Collision Labeler | 0,1,2=Class | X=Delete | Q=Exit"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

labels = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = int(row["class_id"])

all_images = sorted(list(RAW_DIR.glob("**/*.jpg")))
if not all_images: 
    print(f"❌ ไม่พบภาพใน {RAW_DIR}"); exit()

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
    
    # Preprocessing
    h, w = img.shape[:2]
    start_y = int(h * (1 - CROP_RATIO))
    cropped = img[start_y:, :]
    resized = cv2.resize(cropped, SAVE_SIZE)
    display = cv2.resize(resized, (DISPLAY_W, DISPLAY_H))

    fname = p.name
    status = f"Class: {labels[fname]}" if fname in labels else "Unlabeled"
    color = (0, 255, 0) if fname in labels else (0, 0, 255)
    
    cv2.putText(display, f"[{current_idx+1}/{len(all_images)}] {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow(WINDOW_NAME, display)
    
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'): break
    elif key == ord('d'): current_idx = (current_idx + 1) % len(all_images)
    elif key == ord('a'): current_idx = (current_idx - 1) % len(all_images)
    elif ord('0') <= key <= ord('2'): # รับค่า 0, 1, 2
        class_id = int(chr(key))
        labels[fname] = class_id
        cv2.imwrite(str(IMAGES_DIR / fname), resized)
        save_csv()
        print(f"✅ Saved {fname} as Class {class_id}")
        current_idx = (current_idx + 1) % len(all_images) # ไปภาพถัดไปทันที
    elif key == ord('x'):
        if fname in labels:
            del labels[fname]
            if (IMAGES_DIR / fname).exists(): (IMAGES_DIR / fname).unlink()
            save_csv()
            print(f"🗑️ Deleted: {fname}")

cv2.destroyAllWindows()
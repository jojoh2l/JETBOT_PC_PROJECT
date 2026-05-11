import cv2
import sys
from ai_brain import RoadFollowingAI

def main():
    # 1. ตั้งค่า Path และโหลดโมเดล
    model_path = 'best_steering_model_xy.pth'
    image_path = r'datasets\raw\20260510_174756\20260510_174756_000358.jpg'
    
    try:
        ai_node = RoadFollowingAI(model_path=model_path)
        print("[AI Node] Load AI Complete")
    except Exception as e:
        print(f"[Error] ไม่สามารถโหลดโมเดลได้: {e}")
        return

    # 2. อ่านไฟล์ภาพ
    frame = cv2.imread(image_path)

    # 3. ตรวจสอบว่าอ่านภาพสำเร็จหรือไม่
    if frame is None:
        print(f"[Error] ไม่พบไฟล์ภาพที่: {image_path}")
        sys.exit()

    # 4. ประมวลผลผ่าน AI
    # สมมติว่า output x, y ที่ได้คือค่า Normalize 0-1
    x, y, display_frame = ai_node.predict(frame)
    print(f"Target -> X: {x:.2f}, Y: {y:.2f}")

    # 5. วาดจุดเป้าหมาย (แปลงจาก 0-1 เป็นพิกัด Pixel)
    # ใช้ขนาด 224x224 ตามที่กำหนดไว้ในขั้นตอนการเทรน
    x_pixel, y_pixel = int(x * 224), int(y * 224)
    cv2.circle(display_frame, (x_pixel, y_pixel), 6, (0, 255, 0), -1)

    # 6. แสดงผลและเปิดหน้าต่างค้างไว้
    cv2.imshow("AI Prediction Result", display_frame)
    
    print("กดปุ่มใดก็ได้บนคีย์บอร์ดเพื่อปิดหน้าต่าง...")
    cv2.waitKey(0)  # 0 คือการรอไปเรื่อยๆ จนกว่าจะมีปุ่มถูกกด
    cv2.destroyAllWindows() # คืนหน่วยความจำและปิดหน้าต่างทั้งหมด

if __name__ == "__main__":
    main()
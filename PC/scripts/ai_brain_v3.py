import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
# เพิ่มการ import YOLO
from ultralytics import YOLO 

class JetbotAI:
    """Base Class สำหรับจัดการการประมวลผลภาพ (Preprocessing)"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, frame):
        """จัดการภาพ 640x480 -> Crop 80% ล่าง -> Resize 224x224"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        start_y = int(h * 0.2)
        cropped = image[start_y:h, 0:w]
        resized = cv2.resize(cropped, (224, 224))
        image_pil = Image.fromarray(resized)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        return image_tensor

class RoadFollowingAI(JetbotAI):
    """Class สำหรับวิ่งตามเส้น (ใช้ ResNet-18)"""
    def __init__(self, model_path='best_steering_model_xy.pth'):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            print(f"[AI] Road Following Loaded: {model_path}")
        except Exception as e:
            print(f"[Error] Load Road Model Failed: {e}")

    def predict(self, frame):
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
        predictions = output.detach().cpu().numpy().flatten()
        return float(predictions[0]), float(predictions[1])

class CollisionAvoidanceAI(JetbotAI):
    """Class สำหรับหลบหลีกสิ่งกีดขวาง (ใช้ AlexNet)"""
    def __init__(self, model_path='best_model_collision.pth'):
        super().__init__()
        self.model = models.alexnet(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            print(f"[AI] Collision Avoidance Loaded: {model_path}")
        except Exception as e:
            print(f"[Error] Load Collision Model Failed: {e}")

    def predict(self, frame):
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
        prob = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        return {"blocked": float(prob[0]), "free": float(prob[1])}

class ObjectDetectionAI(JetbotAI):
    """Class สำหรับตรวจจับวัตถุ (ใช้ YOLOv8)"""
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__()
        try:
            # โหลดโมเดล YOLOv8
            self.model = YOLO(model_path)
            # ส่งโมเดลไปที่ device (CUDA/CPU)
            self.model.to(self.device)
            print(f"[AI] YOLOv8 Loaded: {model_path} on {self.device}")
        except Exception as e:
            print(f"[Error] Load YOLO Model Failed: {e}")

    def predict(self, frame, conf=0.5):
        """
        รับภาพดิบ 640x480 และประมวลผลโดยตรง
        conf: ค่าความเชื่อมั่นขั้นต่ำ (Confidence Threshold)
        """
        # YOLOv8 รับ numpy array (BGR จาก OpenCV) ได้โดยตรง 
        # และจัดการ resize/normalize ภายในเอง
        results = self.model.predict(source=frame, conf=conf, verbose=False)
        
        # คืนค่าผลลัพธ์แรก (เนื่องจากเราส่งไปทีละภาพ)
        return results[0]
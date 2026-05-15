import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

class JetbotAI:
    """Base Class สำหรับจัดการการประมวลผลภาพ (Preprocessing)"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # มาตรฐาน Normalization ตามที่ใช้ในไฟล์เทรน (ImageNet)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, frame):
        """
        จัดการภาพ 640x480 -> Crop 80% ล่าง -> Resize 224x224
        """
        # 1. แปลง BGR (OpenCV) เป็น RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Crop 80% ด้านล่าง (ตัด 20% บนทิ้ง)
        # 480 * 0.2 = 96 พิกเซลแรกจากด้านบนถูกตัดออก
        h, w = image.shape[:2]
        start_y = int(h * 0.2)
        cropped = image[start_y:h, 0:w]
        
        # 3. Resize เป็น 224x224 ตามที่โมเดลต้องการ
        resized = cv2.resize(cropped, (224, 224))
        
        # 4. แปลงเป็น Tensor และ Normalize
        image_pil = Image.fromarray(resized)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        return image_tensor

class RoadFollowingAI(JetbotAI):
    """Class สำหรับวิ่งตามเส้น (ใช้ ResNet-18)"""
    def __init__(self, model_path='best_steering_model_xy.pth'):
        super().__init__()
        # สร้างโครงสร้าง ResNet-18 ตามไฟล์เทรน
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2) # Output 2 ค่า (x, y)
        
        # โหลด Weight
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            print(f"[AI] Road Following (ResNet-18) Loaded: {model_path}")
        except Exception as e:
            print(f"[Error] Load Road Model Failed: {e}")

    def predict(self, frame):
        """รับภาพดิบ 640x480 -> คืนค่า (x, y) [-1.0 ถึง 1.0]"""
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
        
        predictions = output.detach().cpu().numpy().flatten()
        return float(predictions[0]), float(predictions[1])

class CollisionAvoidanceAI(JetbotAI):
    """Class สำหรับหลบหลีกสิ่งกีดขวาง (ใช้ AlexNet)"""
    def __init__(self, model_path='best_model_collision.pth'):
        super().__init__()
        # สร้างโครงสร้าง AlexNet ตามไฟล์เทรน
        self.model = models.alexnet(pretrained=False)
        # แก้ไข Classifier Layer สุดท้ายให้เป็น 2 Output (blocked, free)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
        
        # โหลด Weight
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            print(f"[AI] Collision Avoidance (AlexNet) Loaded: {model_path}")
        except Exception as e:
            print(f"[Error] Load Collision Model Failed: {e}")

    def predict(self, frame):
        """รับภาพดิบ 640x480 -> คืนค่า Probability ของแต่ละ Class"""
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
        
        # ใช้ Softmax เพื่อหาค่าความน่าจะเป็น (0.0 - 1.0)
        prob = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        
        # คืนค่าเป็น dict เพื่อให้เรียกใช้งานง่าย
        # [0] = blocked, [1] = free (อ้างอิงตามโฟลเดอร์ที่ใช้เทรน)
        return {
            "blocked": float(prob[0]),
            "free": float(prob[1])
        }
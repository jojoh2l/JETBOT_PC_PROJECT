import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class RoadFollowingAI:
    def __init__(self, model_path='best_steering_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AI Node] Preparing Device: {self.device}")

        # Load Model Structure
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(512, 2)

        # Load Weights Parameter
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transformation Preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])   
        print("[AI Node] Load AI Complete ")
    
    def predict(self, frame):
        # 1. Pre-process Images
        height, width = frame.shape[:2]
        start_y = int(height * 0.2)           # Crop Image 100-0
        cropped_frame = frame[start_y:height, 0:width]
        resized_frame = cv2.resize(cropped_frame, (224, 224))

        image_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # 2. Calculate (Inference)
        with torch.no_grad():
            output = self.model(image_tensor)
            x_norm, y_norm = output[0].cpu().numpy()

        # return X, Y and resize image
        return x_norm, y_norm, resized_frame
import cv2
import time

class CameraReceiverNode:
    def __init__(self, stream_url=0):
        self.stream_url = stream_url
        print(f"[Camera Node] Trying Conecting Camera: {self.stream_url}")

        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            print(f"[Camera Node] Not Contect Please check IP")
        else:
            print(f"[Camera Node] Connected Camera -> Ready")
        
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def release(self):
        self.cap.release()
        print("[Camera Node] Disconnect")
        
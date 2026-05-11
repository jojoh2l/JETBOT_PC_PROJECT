import cv2
import time

from stream_receiver import CameraReceiverNode
from ai_brain import RoadFollowingAI

JETBOT_STREAM_URL = 0

camera_node = CameraReceiverNode(stream_url=JETBOT_STREAM_URL)
ai_node = RoadFollowingAI(model_path='best_steering_xy.pth')

print("Start Auto Drive Pipeline... key 'q' stop program")


while True:
    frame = camera_node.get_frame()

    if frame is None:
        print("Image fail... wait new image")
        time.sleep(1)
        continue

    x, y, display_frame = ai_node.predict(frame)

    x_pixel, y_pixel, = int(x * 224), int(y * 224)
    cv2.circle(display_frame, (x_pixel, y_pixel), 6, (0, 255, 0), -1)

    cv2.imshow("Main PC Controller", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera_node.release()
cv2.destroyAllWindows
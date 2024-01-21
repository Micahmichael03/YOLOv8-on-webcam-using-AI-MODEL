
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2
import os



# Load YOLOv5 model
model_file = os.path.abspath("./runs/detect/train2/weights/best.pt")

model = YOLO(model_file) # You can use other YOLOv5 variants as well, like 'yolov5m' or 'yolov5l'

cap = cv2.VideoCapture("./videos1/pure3.mp4")
cap.set(3, 480)
cap.set(4, 580)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Perform object detection using YOLOv5
    results = model(frame)

    for result in results.pred[0]:
        x1, y1, x2, y2, conf, class_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

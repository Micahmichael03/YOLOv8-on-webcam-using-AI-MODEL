import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 580)

model = YOLO("../runs/detect/train2/weights/last.pt")
# model_file = os.path.abspath("./runs/detect/train2/weights/best.pt")

# model = YOLO(model_file)

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Draw standard rectangles
            
            w, h = x2 - x1, y2 - y1  
            # cvzone.cornerRect(img, (x1, y1, w, h))  # Comment out this line
            
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

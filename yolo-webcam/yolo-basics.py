from ultralytics import YOLO
import cv2

model = YOLO("../runs/detect/train2/weights/yolov8l.pt")  # build a new model from scratch
results = model("")  # predict on an image
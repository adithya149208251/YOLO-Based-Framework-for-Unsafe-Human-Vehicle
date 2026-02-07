from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
results = model("bus.jpg")

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(cls, conf, x1, y1, x2, y2)
persons = []
vehicles = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if cls == 0:
            persons.append((cx, cy))
        if cls in [2, 5, 7]:
            vehicles.append((cx, cy))
import math

RISK_THRESHOLD = 150  # pixels (temporary)

for p in persons:
    for v in vehicles:
        dist = math.sqrt((p[0] - v[0])**2 + (p[1] - v[1])**2)

        if dist < RISK_THRESHOLD:
            print("DANGER: Unsafe human–vehicle proximity", dist)
        else:
            print("SAFE", dist)
vehicle_width = abs(x2 - x1)
adaptive_threshold = vehicle_width * 1.2
from ultralytics import YOLO
import cv2
import math
import os

model = YOLO("yolov8n.pt")

image_names = [f"img{i}.jpg" for i in range(1, 13)] + ["img11.jpg", "img12.jpg", "img13.jpg"]

os.makedirs("output", exist_ok=True)

for img_name in image_names:
    results = model(img_name)
    img = cv2.imread(img_name)

    persons = []
    vehicles = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            width = x2 - x1

            if cls == 0:  # person
                persons.append((cx, cy))
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

            if cls in [2, 5, 7]:  # car, bus, truck
                vehicles.append((cx, cy, width))
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)

    for p in persons:
        for v in vehicles:
            dist = math.dist(p, v[:2])
            threshold = v[2] * 1.2  # adaptive threshold

            if dist < threshold:
                color = (0,0,255)
                label = "DANGER"
            else:
                color = (0,255,0)
                label = "SAFE"

            cv2.line(img, p, v[:2], color, 2)
            cv2.putText(img, label, p,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(f"output/out_{img_name}", img)

print("All images processed and saved.")

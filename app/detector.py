from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

INDIA_HAZARD_MAP = {
    "person": ("pedestrian", "high"),
    "cow": ("cattle on road", "critical"),
    "dog": ("animal on road", "medium"),
    "bicycle": ("slow vehicle", "medium"),
    "motorcycle": ("two-wheeler", "high"),
    "bus": ("heavy vehicle", "medium"),
    "truck": ("heavy vehicle", "medium"),
    "car": ("vehicle", "low"),
    "auto rickshaw": ("auto-rickshaw", "high"),
}

def detect_hazards(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    results = model(img)[0]
    
    hazards = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        if label in INDIA_HAZARD_MAP and confidence > 0.4:
            hazard_name, severity = INDIA_HAZARD_MAP[label]
            hazards.append({
                "object": label,
                "hazard": hazard_name,
                "severity": severity,
                "confidence": round(confidence, 2)
            })
    
    return hazards
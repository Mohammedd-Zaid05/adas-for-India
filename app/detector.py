from ultralytics import YOLO
import cv2
import numpy as np
import datetime

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

def get_india_context():
    hour = datetime.datetime.now().hour
    if 7 <= hour < 9:
        return {
            "time_context": "school_hours",
            "risk_multiplier": 1.5,
            "advisory": "School hours (7-9 AM): Exercise caution near school zones, children may be on the road."
        }
    elif 17 <= hour < 20:
        return {
            "time_context": "peak_traffic",
            "risk_multiplier": 1.4,
            "advisory": "Peak traffic (5-8 PM): High density flow, expect sudden braking and lane changes."
        }
    elif hour >= 22 or hour < 5:
        return {
            "time_context": "night_driving",
            "risk_multiplier": 1.6,
            "advisory": "Night driving (10 PM-5 AM): Reduced visibility, watch for stray animals and unlit vehicles."
        }
    else:
        return {
            "time_context": "normal",
            "risk_multiplier": 1.0,
            "advisory": "Normal driving conditions."
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
    
    context = get_india_context()
    return {"hazards": hazards, "context": context}
import cv2
import datetime
import logging

import numpy as np
from ultralytics import YOLO

from app.config import DETECTION_CONFIDENCE_THRESHOLD, INDIA_HAZARD_MAP

model = YOLO("yolov8n.pt")

logger = logging.getLogger(__name__)

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
    if img is None:
        logger.warning("Image decode failed; skipping hazard detection")
        return {"hazards": [], "context": get_india_context()}

    results = model(img)[0]
    
    hazards = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        if label in INDIA_HAZARD_MAP and confidence >= DETECTION_CONFIDENCE_THRESHOLD:
            hazard_name, severity = INDIA_HAZARD_MAP[label]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            hazards.append({
                "object": label,
                "hazard": hazard_name,
                "severity": severity,
                "confidence": round(confidence, 2)
                ,"bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                }
            })
    
    context = get_india_context()
    logger.info(
        "Hazard detection complete: hazards=%s threshold=%.2f time_context=%s",
        len(hazards),
        DETECTION_CONFIDENCE_THRESHOLD,
        context.get("time_context", "unknown"),
    )
    return {"hazards": hazards, "context": context}
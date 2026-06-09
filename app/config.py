import os


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("ADAS_DETECTION_THRESHOLD", "0.4"))

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

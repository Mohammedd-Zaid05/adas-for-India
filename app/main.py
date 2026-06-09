import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import LOG_LEVEL
from app.detector import detect_hazards
from app.scorer import calculate_driver_score
from app.voice import transcribe_audio

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ADAS India API")
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    logger.info("Root endpoint requested")
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health():
    logger.debug("Health check requested")
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    logger.info("Detect request received: filename=%s content_type=%s", file.filename, file.content_type)
    image_bytes = await file.read()
    result = detect_hazards(image_bytes)
    hazards = result["hazards"]
    context = result["context"]
    driver_score = calculate_driver_score(hazards, context)
    logger.info("Detect request completed: hazards=%s score=%s", len(hazards), driver_score.get("score"))
    return {
        "hazards": hazards,
        "total": len(hazards),
        "safe": len(hazards) == 0,
        "context": context,
        "driver_score": driver_score
    }

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    logger.info("Voice request received: filename=%s content_type=%s", file.filename, file.content_type)
    audio_bytes = await file.read()
    result = transcribe_audio(audio_bytes)
    logger.info("Voice request completed: success=%s", result.get("success"))
    return result


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    logger.info("Upload request received: filename=%s", file.filename)
    upload_dir = os.path.join(os.getcwd(), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    logger.info("Saved uploaded video to %s", save_path)
    return {"success": True, "path": save_path}
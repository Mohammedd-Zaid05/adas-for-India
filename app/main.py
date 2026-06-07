from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.detector import detect_hazards

app = FastAPI(title="ADAS India API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ADAS India API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    hazards = detect_hazards(image_bytes)
    return {
        "hazards": hazards,
        "total": len(hazards),
        "safe": len(hazards) == 0
    }
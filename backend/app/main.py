from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from .predict import TumorDetector
from .models import PredictionResponse, HealthResponse

app = FastAPI(
    title="YOLOv11 Brain Tumor Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = TumorDetector()

# ===== API ROUTES (all prefixed with /api) =====
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    info = detector.get_model_info()
    return HealthResponse(status="healthy", model_loaded=True, device=info["device"], model_path=info["model_path"])

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    try:
        result = detector.predict(contents)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/classes", tags=["Info"])
async def get_classes():
    return detector.get_model_info()["classes"]

# ===== SERVE REACT FRONTEND =====
# This must be AFTER all API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

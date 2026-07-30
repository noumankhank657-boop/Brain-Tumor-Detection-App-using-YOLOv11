from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from .predict import TumorDetector
from .models import PredictionResponse, HealthResponse

app = FastAPI(
    title="YOLOv11 Brain Tumor Detection API",
    description="MRI/CT Brain Tumor Detection using Fine-Tuned YOLOv11",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
detector = TumorDetector(MODEL_PATH)

@app.get("/", tags=["Root"])
async def root():
    return {"message": "YOLOv11 Brain Tumor Detection API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    info = detector.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=info["device"],
        model_path=info["model_path"]
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
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

@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch")

    results = []
    for file in files:
        contents = await file.read()
        try:
            result = detector.predict(contents)
            results.append({"filename": file.filename, **result})
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    return {"results": results}

@app.get("/classes", tags=["Info"])
async def get_classes():
    return detector.get_model_info()["classes"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

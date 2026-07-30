from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class PredictionResponse(BaseModel):
    success: bool
    message: str
    detections: List[DetectionResult]
    detection_count: int
    image_width: int
    image_height: int
    annotated_image: Optional[str] = None
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import os
import zipfile

def extract_model_if_needed():
    zip_path = "weights/best.zip"
    pt_path = "weights/best.pt"
    
    # If .pt exists, nothing to do
    if os.path.exists(pt_path):
        return pt_path
    
    # If .zip exists but .pt doesn't, extract it
    if os.path.exists(zip_path):
        print(f"Extracting model from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("weights/")
        print(f"Model extracted to {pt_path}")
        return pt_path
    
    # If neither exists, check env var for direct path
    env_path = os.getenv("MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    raise FileNotFoundError("No model found. Expected weights/best.pt or weights/best.zip")

class TumorDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = None
        
        # Auto-extract if needed
        actual_path = extract_model_if_needed() if model_path is None else model_path
        self.model_path = actual_path
        self._load_model()

    def _load_model(self):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def predict(self, image_bytes: bytes) -> dict:
        start_time = time.time()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        h, w = img.shape[:2]

        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        detections = []
        annotated_b64 = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in [x1, y1, x2, y2]]
                })

            annotated = r.plot(line_width=2, font_size=0.6)
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "message": f"Detected {len(detections)} tumor(s)" if len(detections) > 0 else "No tumor detected",
            "detections": detections,
            "detection_count": len(detections),
            "image_width": w,
            "image_height": h,
            "annotated_image": annotated_b64,
            "processing_time_ms": round(processing_time, 2)
        }

    def get_model_info(self) -> dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "classes": dict(self.model.names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
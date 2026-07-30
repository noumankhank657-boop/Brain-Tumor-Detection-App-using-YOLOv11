from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json

def predict_on_test(model_path, test_dir, output_dir, conf=0.25):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    image_paths = list(Path(test_dir).glob("*.jpg")) + \
                  list(Path(test_dir).glob("*.png")) + \
                  list(Path(test_dir).glob("*.jpeg"))

    results_data = []

    for img_path in image_paths:
        results = model(str(img_path), conf=conf, iou=0.45)

        for r in results:
            out_path = os.path.join(output_dir, img_path.name)
            r.save(filename=out_path)

            boxes = r.boxes
            result_info = {
                "image": img_path.name,
                "detections": len(boxes),
                "classes": [model.names[int(cls)] for cls in boxes.cls] if len(boxes) > 0 else [],
                "confidences": [float(conf) for conf in boxes.conf] if len(boxes) > 0 else [],
            }
            results_data.append(result_info)
            print(f"{img_path.name}: {result_info['detections']} detections")

    with open(os.path.join(output_dir, "predictions_summary.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}")

if __name__ == "__main__":
    predict_on_test(
        model_path="runs/detect/brain_tumor/weights/best.pt",
        test_dir="../dataset/test/images",   # <-- CHANGE if your test path is different
        output_dir="predictions_output",
        conf=0.25,
    )

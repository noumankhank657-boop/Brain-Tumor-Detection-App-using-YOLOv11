from ultralytics import YOLO
import torch
import os

def train_brain_tumor_detector():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv11
    model = YOLO("yolo11n.pt")  # or yolo11s.pt, yolo11m.pt for better accuracy

    # Training
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=8,
        patience=20,
        save=True,
        project="runs/detect",
        name="brain_tumor",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        val=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        plots=True,
    )

    # Export to ONNX for deployment
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

    print(f"Training complete. Best model: {results.best}")
    print(f"Final mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Final mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

    return results

if __name__ == "__main__":
    train_brain_tumor_detector()

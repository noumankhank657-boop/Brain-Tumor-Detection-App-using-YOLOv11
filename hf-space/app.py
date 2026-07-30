import os
import zipfile
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# ========== AUTO-EXTRACT MODEL ==========
ZIP_PATH = "best.zip"
PT_PATH = "best.pt"

if not os.path.exists(PT_PATH) and os.path.exists(ZIP_PATH):
    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(".")
    print("Model extracted!")

if not os.path.exists(PT_PATH):
    raise FileNotFoundError("best.pt not found. Upload best.zip to the Files tab.")

# ========== LOAD MODEL ==========
print("Loading YOLOv11 model...")
model = YOLO(PT_PATH)
print("Model loaded!")

# ========== INFERENCE FUNCTION ==========
def detect_tumor(image):
    if image is None:
        return None, "Please upload an image."
    
    # Convert PIL to numpy
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(img_bgr, conf=0.25, iou=0.45, verbose=False)
    
    # Plot results
    annotated = results[0].plot(line_width=2, font_size=0.6)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(annotated_rgb)
    
    # Build stats
    boxes = results[0].boxes
    detections = []
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        detections.append(f"• {cls_name}: {conf*100:.1f}%")
    
    if len(detections) == 0:
        stats = "✅ No tumor detected in this scan."
    else:
        stats = f"⚠️ {len(detections)} Detection(s) Found:\n\n" + "\n".join(detections)
    
    return output_image, stats

# ========== GRADIO UI ==========
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 100%) !important;
}
.main-title {
    text-align: center;
    font-size: 2.5rem !important;
    font-weight: 700;
    background: linear-gradient(90deg, #d4a853, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
.input-box {
    border: 2px dashed rgba(212,168,83,0.3) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.02) !important;
}
.output-box {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="display:inline-flex; align-items:center; gap:0.5rem; padding:0.4rem 1rem; 
                        background:rgba(212,168,83,0.1); border:1px solid rgba(212,168,83,0.2); 
                        border-radius:999px; font-size:0.75rem; color:#d4a853; font-weight:600; 
                        letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1rem;">
                <span style="width:6px;height:6px;background:#d4a853;border-radius:50%;display:inline-block;"></span>
                YOLOv11 Powered
            </div>
            <h1 class="main-title">Brain Tumor Detection</h1>
            <p class="subtitle">Upload an MRI or CT scan to detect tumors using deep learning</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                type="pil",
                label="Upload Medical Scan",
                elem_classes=["input-box"]
            )
            btn = gr.Button("🔍 Analyze Scan", variant="primary", size="lg")
        
        with gr.Column():
            output_img = gr.Image(
                label="Detection Result",
                elem_classes=["output-box"]
            )
            output_text = gr.Textbox(
                label="Analysis Summary",
                lines=6,
                elem_classes=["output-box"]
            )
    
    btn.click(fn=detect_tumor, inputs=input_img, outputs=[output_img, output_text])
    
    gr.Markdown("""
        <div style="text-align:center; margin-top:2rem; color:#64748b; font-size:0.8rem;">
            Built with YOLOv11 | Fine-tuned on Brain Tumor Dataset
        </div>
    """)

if __name__ == "__main__":
    demo.launch()
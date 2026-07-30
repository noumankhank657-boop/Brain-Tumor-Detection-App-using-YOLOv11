import streamlit as st
import os
import zipfile
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #d4a853, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #d4a853, #b8923e) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stSpinner > div {
        border-top-color: #d4a853 !important;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(212,168,83,0.4) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: rgba(255,255,255,0.02) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced MRI & CT scan analysis powered by YOLOv11</div>', unsafe_allow_html=True)

# ========== AUTO-EXTRACT MODEL ==========
ZIP_PATH = "best.zip"
PT_PATH = "best.pt"

if not os.path.exists(PT_PATH) and os.path.exists(ZIP_PATH):
    with st.spinner("Extracting model for first use..."):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(".")
        st.success("Model ready!")

if not os.path.exists(PT_PATH):
    st.error("Model not found. Please ensure `best.zip` is in the repository.")
    st.stop()

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    return YOLO(PT_PATH)

with st.spinner("Loading YOLOv11 model..."):
    model = load_model()

st.success("✅ Model loaded successfully")

# ========== UPLOAD ==========
uploaded_file = st.file_uploader("📤 Drop your MRI/CT scan here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    
    with col1:
        st.markdown("**📷 Original Scan**")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Analyze Scan"):
        with st.spinner("Running YOLOv11 inference..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.25, iou=0.45, verbose=False)
            
            annotated = results[0].plot(line_width=2, font_size=0.6)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("**🎯 Detection Result**")
                st.image(annotated_rgb, use_container_width=True)
            
            boxes = results[0].boxes
            st.markdown("---")
            st.markdown("### 📊 Analysis Summary")
            
            if len(boxes) > 0:
                st.error(f"⚠️ {len(boxes)} Tumor(s) Detected")
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id]
                    st.write(f"• **{name}**: `{conf*100:.1f}%` confidence")
            else:
                st.success("✅ No tumor detected in this scan")

st.markdown("---")
st.caption("Built with YOLOv11 | Fine-tuned on Brain Tumor Dataset")
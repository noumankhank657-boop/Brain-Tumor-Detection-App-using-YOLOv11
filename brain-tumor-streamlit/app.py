import streamlit as st
import os
import zipfile
from PIL import Image
import numpy as np

# ========== DEBUG: Show what's happening ==========
st.write("🔧 Debug Info:")
st.write("Current working directory:", os.getcwd())
st.write("__file__ path:", os.path.abspath(__file__))

app_dir = os.path.dirname(os.path.abspath(__file__))
st.write("App directory:", app_dir)

try:
    files_in_app_dir = os.listdir(app_dir)
    st.write("Files in app dir:", files_in_app_dir)
except Exception as e:
    st.write("Error listing app dir:", str(e))

try:
    files_in_cwd = os.listdir(os.getcwd())
    st.write("Files in cwd:", files_in_cwd)
except Exception as e:
    st.write("Error listing cwd:", str(e))

# ========== PATH SETUP ==========
# Try every possible location
possible_zip_paths = [
    os.path.join(app_dir, "best.zip"),
    os.path.join(os.getcwd(), "best.zip"),
    os.path.join(os.getcwd(), "brain-tumor-streamlit", "best.zip"),
    "best.zip",
    "brain-tumor-streamlit/best.zip",
]

ZIP_PATH = None
for p in possible_zip_paths:
    st.write(f"Checking: {p} → Exists: {os.path.exists(p)}")
    if os.path.exists(p):
        ZIP_PATH = p
        break

PT_PATH = os.path.join(app_dir, "best.pt")

# ========== EXTRACT MODEL ==========
if not os.path.exists(PT_PATH):
    if ZIP_PATH:
        st.info(f"Extracting model from: {ZIP_PATH}")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as z:
                z.extractall(app_dir)
            st.success("Model extracted successfully!")
        except Exception as e:
            st.error(f"Failed to extract: {str(e)}")
            st.stop()
    else:
        st.error("❌ best.zip not found in any location!")
        st.stop()

if not os.path.exists(PT_PATH):
    st.error("❌ best.pt not found after extraction!")
    st.stop()

st.success(f"✅ Model found at: {PT_PATH}")

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(PT_PATH)

with st.spinner("Loading YOLOv11 model..."):
    model = load_model()

st.success("✅ Model loaded!")

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #d4a853, #22d3ee);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
    .stButton>button { background: linear-gradient(90deg, #d4a853, #b8923e) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important; font-size: 1.1rem !important; font-weight: 600 !important; width: 100%; }
    div[data-testid="stFileUploader"] { border: 2px dashed rgba(212,168,83,0.4) !important;
        border-radius: 16px !important; padding: 2rem !important; background: rgba(255,255,255,0.02) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced MRI & CT scan analysis powered by YOLOv11</div>', unsafe_allow_html=True)

# ========== UPLOAD ==========
uploaded_file = st.file_uploader("📤 Drop your MRI/CT scan here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.markdown("**📷 Original Scan**")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Analyze Scan"):
        with st.spinner("Running YOLOv11 inference..."):
            img_np = np.array(image)
            results = model(img_np, conf=0.25, iou=0.45, verbose=False)
            annotated = results[0].plot(line_width=2, font_size=0.6)
            
            with col2:
                st.markdown("**🎯 Detection Result**")
                st.image(annotated, use_container_width=True)
            
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
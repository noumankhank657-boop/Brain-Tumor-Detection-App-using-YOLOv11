import streamlit as st
import os
import zipfile
from PIL import Image
import numpy as np

# ========== FORCE DARK THEME ==========
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS — LUXURY MEDICAL THEME ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

/* ─── GLOBAL RESET & THEME ─── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #070a12 0%, #0d1117 50%, #0a0e1a 100%) !important;
    color: #e2e8f0 !important;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none !important;}

/* ─── ANIMATED BACKGROUND ORBS ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%;
    right: -10%;
    width: 800px;
    height: 800px;
    background: radial-gradient(circle, rgba(212,168,83,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orbFloat1 20s ease-in-out infinite;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -20%;
    left: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(34,211,238,0.05) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orbFloat2 25s ease-in-out infinite;
}

@keyframes orbFloat1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(-40px, 30px) scale(1.1); }
    66% { transform: translate(20px, -20px) scale(0.95); }
}

@keyframes orbFloat2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(30px, -40px) scale(1.15); }
}

/* ─── GLASSMORPHISM CARD ─── */
.glass-card {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

/* ─── HERO SECTION ─── */
.hero-container {
    text-align: center;
    padding: 2rem 0 3rem 0;
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    background: rgba(212, 168, 83, 0.08);
    border: 1px solid rgba(212, 168, 83, 0.2);
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #d4a853;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    animation: fadeInDown 0.8s ease-out;
}

.hero-badge .pulse-dot {
    width: 6px;
    height: 6px;
    background: #d4a853;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(212,168,83,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(212,168,83,0); }
}

.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    margin-bottom: 1rem !important;
    background: linear-gradient(135deg, #f8fafc 0%, #d4a853 40%, #22d3ee 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: fadeInDown 0.8s ease-out 0.1s both;
}

.hero-subtitle {
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    max-width: 500px !important;
    margin: 0 auto !important;
    line-height: 1.7 !important;
    animation: fadeInDown 0.8s ease-out 0.2s both;
}

/* ─── UPLOAD ZONE ─── */
.upload-zone {
    border: 2px dashed rgba(212, 168, 83, 0.25) !important;
    border-radius: 24px !important;
    padding: 3.5rem 2rem !important;
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.8s ease-out 0.3s both;
}

.upload-zone::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 1.5px;
    background: linear-gradient(135deg, transparent 30%, rgba(212,168,83,0.3) 50%, transparent 70%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0;
    transition: opacity 0.5s;
}

.upload-zone:hover::before {
    opacity: 1;
}

.upload-zone:hover {
    border-color: rgba(212, 168, 83, 0.5) !important;
    background: rgba(212, 168, 83, 0.03) !important;
    transform: translateY(-3px);
    box-shadow: 0 12px 48px rgba(212, 168, 83, 0.1), 0 0 80px rgba(212, 168, 83, 0.05) !important;
}

.upload-icon-box {
    width: 72px;
    height: 72px;
    margin: 0 auto 1.25rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(212,168,83,0.12), rgba(34,211,238,0.08));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.upload-zone:hover .upload-icon-box {
    transform: scale(1.15) rotate(-8deg);
}

.upload-title {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    margin-bottom: 0.4rem !important;
}

.upload-hint {
    font-size: 0.85rem !important;
    color: #64748b !important;
}

/* ─── BUTTON ─── */
.analyze-btn {
    background: linear-gradient(135deg, #d4a853, #b8923e) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 3rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 24px rgba(212, 168, 83, 0.25) !important;
    width: 100% !important;
}

.analyze-btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 20%, rgba(255,255,255,0.25) 50%, transparent 80%);
    transform: translateX(-100%);
    transition: transform 0.7s;
}

.analyze-btn:hover::before {
    transform: translateX(100%);
}

.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(212, 168, 83, 0.4) !important;
}

.analyze-btn:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ─── IMAGE CARDS ─── */
.image-card {
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 20px !important;
    padding: 1.25rem !important;
    transition: transform 0.3s, box-shadow 0.3s !important;
}

.image-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
}

.image-card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.875rem;
}

.image-card-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.image-card-icon.gold {
    background: rgba(212, 168, 83, 0.12);
}

.image-card-icon.cyan {
    background: rgba(34, 211, 238, 0.12);
}

.image-card-label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
}

.image-card img {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    width: 100% !important;
}

/* ─── STATS PANEL ─── */
.stats-panel {
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 20px !important;
    padding: 1.75rem !important;
    animation: fadeInUp 0.6s ease-out;
}

.stats-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
}

.stats-header-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: rgba(52, 211, 153, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.stats-header-label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
}

/* ─── BADGES ─── */
.badge-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}

.lux-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
    backdrop-filter: blur(8px);
}

.lux-badge.danger {
    background: rgba(244, 63, 95, 0.06);
    border-color: rgba(244, 63, 95, 0.2);
    color: #f43f5e;
}

.lux-badge.success {
    background: rgba(52, 211, 153, 0.06);
    border-color: rgba(52, 211, 153, 0.2);
    color: #34d399;
}

.lux-badge.info {
    background: rgba(34, 211, 238, 0.06);
    border-color: rgba(34, 211, 238, 0.2);
    color: #22d3ee;
}

.lux-badge.gold {
    background: rgba(212, 168, 83, 0.06);
    border-color: rgba(212, 168, 83, 0.2);
    color: #d4a853;
}

/* ─── DETECTION ITEMS ─── */
.detection-list {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.detection-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.25rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.detection-item:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(212, 168, 83, 0.15);
    transform: translateX(6px);
}

.detection-class {
    font-weight: 600;
    font-size: 0.95rem;
    text-transform: capitalize;
    color: #f1f5f9;
    margin-bottom: 0.35rem;
}

.confidence-track {
    width: 180px;
    height: 5px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.confidence-fill::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2.5s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.detection-confidence {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1;
}

.detection-bbox {
    font-size: 0.65rem;
    color: #475569;
    margin-top: 0.2rem;
    font-family: 'Courier New', monospace;
}

/* ─── LOADING ─── */
.loading-container {
    text-align: center;
    padding: 3rem;
    animation: fadeIn 0.4s;
}

.loading-ring {
    width: 56px;
    height: 56px;
    margin: 0 auto 1.25rem;
    position: relative;
}

.loading-ring::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 3px solid transparent;
    border-top-color: #d4a853;
    animation: spin 1s linear infinite;
}

.loading-ring::after {
    content: '';
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    border: 3px solid rgba(212, 168, 83, 0.1);
}

.loading-text {
    color: #94a3b8;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

/* ─── EMPTY STATE ─── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #475569;
}

.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.4;
}

/* ─── FOOTER ─── */
.lux-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #334155;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

/* ─── ANIMATIONS ─── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(212, 168, 83, 0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212, 168, 83, 0.3); }

/* ─── STREAMLIT OVERRIDES ─── */
.stFileUploader > div > div {
    background: transparent !important;
    border: none !important;
}

.stFileUploader > div > div > div {
    background: transparent !important;
}

.stSpinner > div {
    border-top-color: #d4a853 !important;
}

.stAlert {
    background: rgba(244, 63, 95, 0.06) !important;
    border: 1px solid rgba(244, 63, 95, 0.15) !important;
    border-radius: 14px !important;
    color: #f43f5e !important;
}

.stSuccess {
    background: rgba(52, 211, 153, 0.06) !important;
    border: 1px solid rgba(52, 211, 153, 0.15) !important;
    border-radius: 14px !important;
    color: #34d399 !important;
}
</style>
""", unsafe_allow_html=True)

# ========== PATH SETUP ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "best.zip")
PT_PATH = os.path.join(BASE_DIR, "best.pt")

# ========== EXTRACT MODEL ==========
if not os.path.exists(PT_PATH) and os.path.exists(ZIP_PATH):
    with st.spinner("Extracting model for first use..."):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(BASE_DIR)

if not os.path.exists(PT_PATH):
    st.error("Model not found. Please ensure `best.zip` is in the repository.")
    st.stop()

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(PT_PATH)

with st.spinner(""):
    model = load_model()

# ========== HERO SECTION ==========
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        YOLOv11 Powered
    </div>
    <h1 class="hero-title">Brain Tumor Detection</h1>
    <p class="hero-subtitle">Advanced MRI & CT scan analysis powered by deep learning. Upload a medical scan to detect tumors with precision.</p>
</div>
""", unsafe_allow_html=True)

# ========== UPLOAD SECTION ==========

col_u1, col_u2, col_u3 = st.columns([1, 3, 1])
with col_u2:
    st.markdown("""
    <div style="text-align:center;">
        <div class="upload-icon-box">📤</div>
        <div class="upload-title">Upload Medical Scan</div>
        <div class="upload-hint">Drag & drop an MRI or CT image here, or click to browse</div>
        <div class="upload-hint" style="margin-top:0.3rem; font-size:0.75rem;">Supports JPG, PNG, JPEG — Maximum 10MB</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ========== RESULTS ==========
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Image comparison row
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("""
        <div class="image-card">
            <div class="image-card-header">
                <div class="image-card-icon gold">📷</div>
                <span class="image-card-label">Original Scan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with img_col2:
        st.markdown("""
        <div class="image-card">
            <div class="image-card-header">
                <div class="image-card-icon cyan">🎯</div>
                <span class="image-card-label">Detection Result</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Placeholder until analysis
        if "analyzed" not in st.session_state:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#334155;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem; opacity:0.3;">🔍</div>
                <div style="font-size:0.9rem;">Click "Analyze Scan" to see detection results</div>
            </div>
            """, unsafe_allow_html=True)

    # Analyze button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        analyze_clicked = st.button("🔍 Analyze Scan", key="analyze", use_container_width=True)

    if analyze_clicked:
        with st.spinner(""):
            img_np = np.array(image)
            results = model(img_np, conf=0.25, iou=0.45, verbose=False)
            annotated = results[0].plot(line_width=2, font_size=0.6)
            st.session_state.analyzed = True
            st.session_state.annotated = annotated
            st.session_state.boxes = results[0].boxes
            st.session_state.names = model.names
            st.rerun()

    # Show results after analysis
    if st.session_state.get("analyzed"):
        with img_col2:
            st.image(st.session_state.annotated, use_container_width=True)

        # Stats panel
        boxes = st.session_state.boxes
        names = st.session_state.names

        st.markdown("""
        <div class="stats-panel">
            <div class="stats-header">
                <div class="stats-header-icon">📊</div>
                <span class="stats-header-label">Analysis Summary</span>
            </div>
            <div class="badge-row">
        """, unsafe_allow_html=True)

        if len(boxes) > 0:
            st.markdown(f'<span class="lux-badge danger">⚠️ {len(boxes)} Tumor(s) Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="lux-badge success">✅ No Tumor Detected</span>', unsafe_allow_html=True)

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(boxes) > 0:
            st.markdown('<div class="detection-list">', unsafe_allow_html=True)
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = names[cls_id]

                if conf >= 0.8:
                    color = "#34d399"
                    label = "High Confidence"
                elif conf >= 0.5:
                    color = "#d4a853"
                    label = "Medium Confidence"
                else:
                    color = "#f43f5e"
                    label = "Low Confidence"

                st.markdown(f"""
                <div class="detection-item">
                    <div>
                        <div class="detection-class">{name}</div>
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width: {conf*100}%; background: {color};"></div>
                        </div>
                        <div style="font-size:0.7rem; color:#475569; margin-top:0.2rem;">{label}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="detection-confidence" style="color: {color};">{conf*100:.1f}%</div>
                        <div class="detection-bbox">[{', '.join([str(int(v)) for v in box.xyxy[0].cpu().numpy().tolist()])}]</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("""
<div class="lux-footer">
    Built with YOLOv11 · Fine-tuned on Brain Tumor Dataset · Medical AI Detection System
</div>
""", unsafe_allow_html=True)

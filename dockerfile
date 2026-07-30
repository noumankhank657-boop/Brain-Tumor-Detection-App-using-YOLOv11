# ===== STAGE 1: Build React Frontend =====
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ===== STAGE 2: Python Backend =====
FROM python:3.11-slim
WORKDIR /app

# System dependencies for OpenCV + YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app/ ./app/
COPY backend/weights/ ./weights/

# Copy built frontend into static folder
COPY --from=frontend-builder /app/frontend/dist ./static

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=weights/best.pt
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "python -c \"import os, zipfile; z='weights/best.zip'; p='weights/best.pt'; os.path.exists(p) or (os.path.exists(z) and (__import__('zipfile').ZipFile(z).extractall('weights/'), print('Model extracted')))\" && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
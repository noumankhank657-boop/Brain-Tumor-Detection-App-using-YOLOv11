# YOLOv11 Brain Tumor Detection - Full Stack

Complete full-stack application for brain tumor detection using fine-tuned YOLOv11.

## Project Structure

```
brain-tumor-yolo11/
├── training/          # Training scripts & data config
├── backend/           # FastAPI inference API
├── frontend/          # React web app
├── docker-compose.yml # Docker orchestration
└── render.yaml        # Render deployment config
```

## Quick Start

### 1. Update `training/data.yaml`
Edit the file and set your dataset's absolute path and class names.

### 2. Place your fine-tuned model
```bash
cp your_trained_model.pt backend/weights/best.pt
```

### 3. Run with Docker
```bash
docker-compose up --build
```
Open http://localhost

### 4. Or run locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Training (if you haven't trained yet)

```bash
cd training
pip install ultralytics torch
python train.py
```

## Deployment

### Render
1. Push to GitHub
2. Connect repo to Render
3. Upload `best.pt` to persistent disk or GitHub Releases
4. Deploy using `render.yaml`

### Railway
```bash
cd backend && railway up
```

## What to Submit

| Field | Value |
|-------|-------|
| GitHub Repo | `https://github.com/YOUR_USERNAME/brain-tumor-yolo11` |
| Deployed App | Your Render/Railway URL |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Ensure `best.pt` is in `backend/weights/` |
| CUDA OOM | Reduce batch size in `train.py` |
| CORS errors | Check `allow_origins` in `backend/app/main.py` |
| Frontend can't reach backend | Update `VITE_API_URL` in deployment env |

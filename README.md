# Emotion Suppression Detection — Backend

A FastAPI backend that detects **emotional suppression** from video. It combines facial Action Unit (AU) analysis, emotion detection (DeepFace), speech latency analysis, and an LSTM model to produce a suppression score.

## Overview

Upload a video → the pipeline extracts facial AUs (OpenFace), detects emotions (DeepFace), analyzes speech patterns, and runs an LSTM model to predict how much the subject is suppressing their emotions.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **ML:** PyTorch (LSTM), DeepFace, MediaPipe
- **Audio:** Librosa, MoviePy
- **Database:** MongoDB Atlas
- **Auth:** JWT (python-jose) + bcrypt

## Project Structure

```
├── Emotion-suppression-api/
│   └── backend/
│       ├── main.py                  # FastAPI app & routes
│       ├── requirements.txt
│       └── suppression/
│           └── logic.py             # Pipeline orchestration
│
├── NEW Emotion_Suppression_Project-main/
│   └── Emotion_Suppression_Project-main/
│       ├── emotion/                 # DeepFace emotion detection
│       ├── inference/               # Model inference
│       ├── models/                  # Trained LSTM weights (.pth)
│       ├── preprocessing/           # AU extraction & feature building
│       ├── speech/                  # Audio extraction & speech events
│       └── training/               # LSTM training scripts
│
├── requirements.txt                 # Consolidated deps (for deployment)
└── render.yaml                      # Render deployment config
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/auth/signup` | Create account |
| POST | `/auth/login` | Login, returns JWT |
| GET | `/me` | Current user (auth required) |
| POST | `/register-user` | Save subject details |
| POST | `/analyze-video` | Upload video, run pipeline |
| GET | `/my-results` | Get analysis results (auth required) |
| GET | `/my-results/full/{index}` | Full result with timeline |
| GET | `/users` | All users + results (auth required) |
| GET | `/users/{email}/results` | Results for specific user |

## Local Setup

### Prerequisites
- Python 3.10+
- OpenFace (for AU extraction)

### Install & Run

```bash
cd Emotion-suppression-api/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | Atlas cluster URI |
| `JWT_SECRET` | Secret key for JWT tokens | Built-in default |

## Deployment (Render)

The project includes a `render.yaml` for one-click deploy on Render:

1. Push to GitHub
2. Create a **Web Service** on [Render](https://render.com)
3. Connect the repo — Render auto-detects `render.yaml`
4. Set environment variables (`MONGO_URI`, `JWT_SECRET`)
5. Deploy

## Analyze Video — Response Shape

```json
{
  "suppression_score": 2.34,
  "normalized_score": 0.45,
  "level": "Moderate",
  "dominant_emotion": "happy",
  "suppressed_emotion": "sad",
  "timeline": [...],
  "latency_events": [...],
  "files_processed": 1
}
```

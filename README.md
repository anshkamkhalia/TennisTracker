# 🎾 TennisTracker &nbsp; ![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

End-to-end tennis video analysis with a FastAPI backend, a React/Vite frontend, and AI-assisted shot and motion metrics.

## Overview

TennisTracker processes uploaded tennis videos, runs camera-aware analysis, and returns an annotated result video plus structured stats for the dashboard and result pages.

The current app includes:

- A modern dashboard with the most recent video, global stats, and big charts
- A session history view with edit and delete actions
- A result page with video playback, charts, and a pose demo
- A three.js / React Three Fiber pose viewer with smoother playback and interactive rotation
- Supabase session storage and Clerk authentication on the frontend

## Features

- Shot classification from pose sequences
- Neutral-vs-swing gating for court-level clips
- Ball detection and trail visualization
- Court detection and mini-court overlay
- Wrist velocity estimates for court-level footage
- Player movement heatmap generation
- Top-view ball speed estimation
- Annotated MP4 output uploaded to Cloudflare R2
- Dashboard with global stats and recent sessions
- Session deletion from history and result screens
- Interactive pose replay in the result page

## Tech Stack

- Backend: FastAPI, OpenCV, NumPy, SciPy, MediaPipe, YOLO
- Frontend: React, Vite, TypeScript, Tailwind CSS
- Auth: Clerk
- Storage: Supabase for sessions, Cloudflare R2 for processed video
- Charts: ApexCharts
- Pose viewer: React Three Fiber, Drei, Three.js

## Project Structure

```text
api/                  # FastAPI backend
  app.py              # Main processing pipeline and /process-video endpoint
  configs.py          # Shared runtime state
  serialized_models/  # Runtime model artifacts
frontend/             # React/Vite frontend
  src/pages/          # Dashboard, history, record, result, etc.
  src/components/     # Layout and shared UI
src/                  # Training and experimentation code
  shot_classification/
  ball_tracking/
  court_model/
  metrics/
```

## Frontend Pages

- `Home`:
  - shows the most recent processed video
  - displays global stats across sessions
  - renders larger charts for shot mix and wrist speed
- `History`:
  - lists all sessions
  - supports rename/edit
  - supports delete
- `Record`:
  - uploads a video to the API
  - stores the returned analysis payload in Supabase
- `Result`:
  - plays back the processed video
  - shows charts and stats from the returned analysis
  - includes the pose replay demo

## Backend Pipeline

The `/process-video` endpoint in [`api/app.py`](api/app.py) performs:

1. file validation and size checks
2. camera-angle detection
3. court detection or court-view processing
4. pose extraction and shot classification for court-level clips
5. ball tracking and smoothing
6. wrist velocity calculation
7. movement heatmap generation
8. video annotation and transcoding
9. upload to Cloudflare R2

## API Response

The current backend response includes the following fields:

```json
{
  "message": "video processed successfully",
  "url": "<public_r2_url>",
  "video_type": "court",
  "n_shots_by_POI": 12,
  "total_shots": 21,
  "forehand_percent": 52.1,
  "backhand_percent": 34.4,
  "serve_overhead_percent": 13.5,
  "right_wrist_avg": 38.2,
  "right_wrist_v": [31.2, 32.8, 35.1],
  "pose_landmarks_3d": [[[...]]],
  "heatmap": "<image_or_url>",
  "ball_speeds": [72.1, 76.8, 81.4]
}
```

Notes:

- `video_type` is typically `court` or `top`
- `right_wrist_v` and `pose_landmarks_3d` are used by the frontend pose/result UI
- `heatmap` and `ball_speeds` are returned for top-view clips

## Setup

### Backend

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

### Backend

```env
R2_ENDPOINT=...
R2_KEY=...
R2_SECRET=...
R2_BUCKET=...
R2_PUBLIC_URL=...
```

### Frontend

```env
VITE_API_URL=http://localhost:5000
VITE_SUPABASE_URL=...
VITE_SUPABASE_ANON_KEY=...
VITE_CLERK_PUBLISHABLE_KEY=...
```

## Notes

- The frontend uses Supabase for session persistence and Clerk for auth.
- The pose viewer is intentionally kept as a wireframe-style Three.js scene for readability.
- The result page now uses the API payload directly so the latest analysis appears immediately after upload.
- The dashboard focuses on the most recent session and aggregated stats instead of only a raw history list.

## Training & Experimentation

The `src/` folder contains the original experimentation code for:

- shot classification
- ball tracking
- court detection
- wrist velocity
- movement heatmaps

Those scripts and assets are kept for model training and reproducibility.


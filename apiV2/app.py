# modularized api

import numpy as np
import librosa
import subprocess
from tqdm import tqdm
import cv2 as cv
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from werkzeug.utils import secure_filename
from apiV2.subsystems.dataclasses.models import Models
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
from apiV2.subsystems.dataclasses.consts import PipelineConstants
from apiV2.subsystems.dataclasses.state import PipelineState
import os
import apiV2.subsystems.utils as utils

app = FastAPI()

# rate limiting (slowapi)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# for rn connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8081", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/process-video")
@limiter.limit("1/minute")
async def main(request: Request, video: UploadFile = File(...)):
    
    # load classes
    models = Models()
    state = PipelineState()
    consts = PipelineConstants()

    # security checks
    content_length = request.headers.get("content-length")
    if content_length is None or int(content_length) > consts.MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="Video too large or missing")

    video_file = video

    if video_file.content_type not in state.ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = secure_filename(video_file.filename)
    os.makedirs("api/temp_videos", exist_ok=True)
    
    # create input and output path
    state.input_path = os.path.join("api/temp_videos", filename)
    state.output_path = os.path.join("api/temp_videos", f"output_{filename}")
    with open(state.input_path, "wb") as f:
        f.write(await video_file.read())

    if not utils.allowed_file(filename):
        raise HTTPException(status_code=400, detail="invalid file extension")

    del video_file # immediate removal

    state.cap = cv.VideoCapture(state.input_path) # load video
    if not state.cap.isOpened():
        raise HTTPException(status_code=400, detail="failed to open video file")
    
    state.total_frames = int(state.cap.get(cv.CAP_PROP_FRAME_COUNT)) # get total frames
    pbar = tqdm(total=state.total_frames, desc="processing video", unit="frame")

    state.input_fps = state.cap.get(cv.CAP_PROP_FPS)
    state.dt = 1.0 / state.input_fps if state.input_fps > 0 else 1/60 # for wrist velocity

    if state.input_fps <= 0:
        state.input_fps = 30
    
    state.speed_buffer_size = int(state.input_fps)

    state.out = cv.VideoWriter(
        state.output_path.replace(".mp4", ".avi"),
        cv.VideoWriter_fourcc(*"MJPG"),
        int(state.input_fps),
        (1280, 720),
    )

    if not state.out.isOpened():
        raise HTTPException(status_code=400, detail="failed to open video writer")

    state.video_path = state.input_path

    # load audio
    try:
        if not os.path.exists(state.audio_path):
            subprocess.run([
                "ffmpeg",
                "-i", state.video_path,
                "-vn",              
                "-acodec", "pcm_s16le",
                "-ar", "16000",     
                "-ac", "1", "-y",
                state.audio_path
            ], check=True)
    except:
        pass
        
    try:
        state.audio, state.sr = librosa.load(state.audio_path, sr=16_000) # load .wav audio with librosa
        state.time = np.arange(len(state.audio)) / state.sr
        state.avg_energy = np.mean(np.sqrt(state.audio**2))
        state.MIN_ENERGY = state.avg_energy + 0.03

    except:

        state.audio = None

    # sound based contact configs
    state.n_contacts = 0
    state.hit = False
    
    state.COOLDOWN_FRAMES = int(0.15 * state.input_fps)
    state.last_hit_frame = -state.COOLDOWN_FRAMES
    
    return {}
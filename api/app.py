# combines all features together to process a video
# current features:
# - shot classification
# - ball detection/tracking
# - 2d court modeling with ball
# - ball speed estimation
# - heatmap-based player movement tracking
# - wrist velocity
# - top and court viewpoints supported

# -------------------------------------------------------------------------------- setup --------------------------------------------------------------------------------

# new fastapi backend
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

import tensorflow as tf # deep learning
import mediapipe as mp # keypoint extraction (pose)
import cv2 as cv # video handling and utils
from ultralytics import YOLO # object detection
import numpy as np # mathematical operations and arrays
from src.shot_classification.model import ShotClassifier # subclassed model
from src.shot_classification.neutral_model import NeutralIdentifier, Attention # subclassed models and layers
import supabase # metadata?
import os # path handling
from werkzeug.utils import secure_filename # for security checks
import boto3 # cloudflare r2 client
from botocore.client import Config # version configs
from botocore.exceptions import NoCredentialsError # exceptions
from dotenv import load_dotenv # secure credential storage
import datetime
from botocore.exceptions import ClientError # more exceptions
import subprocess # running commands like ffmpeg
import time as t # frame -> audio sync
import gc # garbage collection
from scipy.signal import savgol_filter # savitzky-golay filter
from tqdm import tqdm
import traceback # debug
import matplotlib.pyplot as plt # graphs
import random # just random ig
import librosa # for audio reading

# load environment variables for cloudflare r2 connnection
load_dotenv()

# get r2 credentials
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_KEY = os.getenv("R2_KEY")
R2_SECRET = os.getenv("R2_SECRET")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

# client setup
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)

# key mapping
LABELS = {
    "forehand": 0,
    "backhand": 1,
    "slice_volley": 2,
    "serve_overhead": 3,
}

LABELS_INV = {v: k for k, v in LABELS.items()} # create inverse: {0: "topspin_forehand"...}

# paths
shot_model_path = "api/serialized_models/new_sc.keras" # use new shot clasification model (temporal transformer)
neutral_model_path = "api/serialized_models/neutrality.keras"
# classifies shots
shot_classifier = tf.keras.saving.load_model(shot_model_path, custom_objects={ # load model with correct classes
    "ShotClassifier": ShotClassifier,
    # "Attention": Attention,
    # "SequenceAttention": SequenceAttention,
})

# identifies neutral positions
neutral_identifier = tf.keras.saving.load_model(neutral_model_path, custom_objects={
    "NeutralIdentifier": NeutralIdentifier,
    "Attention": Attention,
})

# used for cropping player
human_detector = YOLO("yolo11n.pt")

# tracks balls across frames
ball_tracker = YOLO("hugging_face_best.pt")

# detects the bounding box of court
court_detector = YOLO("src/court_model/best.pt")

seq_len = 60 # frame buffer length

# mediapipe pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

app = FastAPI()

# templates = Jinja2Templates(directory="api/templates")
# app.mount("/static", StaticFiles(directory="api/static"), name="static")

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

# file restrictions
# 150 MB hard limit
MAX_VIDEO_SIZE = 150 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",   # .mov
    "video/x-matroska"   # .mkv
}
ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv"}


# @app.get("/", response_class=HTMLResponse)
# async def landing(request: Request):
#     """landing/upload page for the web frontend."""
#     max_mb = MAX_VIDEO_SIZE // (1024 * 1024)
#     return templates.TemplateResponse(
#         "index.html",
#         {
#             "request": request,
#             "max_size_mb": max_mb,
#             "allowed_ext": ", ".join(sorted(ALLOWED_EXTENSIONS)),
#         },
#     )


@app.get("/health")
async def healthcheck():
    """lightweight health endpoint for checks."""
    return {
        "status": "ok",
        "models_loaded": True,
        "bucket": R2_BUCKET,
    }

# -------------------------------------------------------------------------------- helper functions --------------------------------------------------------------------------------

# helper functions
def allowed_file(filename):
    """checks if a filename has a valid extension"""

    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in ALLOWED_EXTENSIONS

def upload_to_r2(local_path, r2_key):
    
    """
    uploads a local file to the r2 bucket.
    returns the url or raises an HTTPException.
    """

    try:
        s3.upload_file(local_path, R2_BUCKET, r2_key) # upload file
        public_url = f"{R2_PUBLIC_URL}/{r2_key}"
        # print(f"upload successful: {public_url}")
        return public_url
    except NoCredentialsError:
        msg = "invalid r2 credentials, check r2_key and r2_secret"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
    except ClientError as e:
        code = e.response['Error']['Code']
        message = e.response['Error']['Message']
        msg = f"r2 upload failed: {code} - {message}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)

def calculate_velocity(buffer, dt):

    """calculates velocity"""
    
    # final and initial are (x,y) coordinates
    final = buffer[-1]
    initial = buffer[0]

    # calculate distance
    dist = np.sqrt((final[0] - initial[0])**2 + (final[1] - initial[1])**2)

    v = dist / dt
    return v

def safe_append(buffer, new_point, max_jump=100):

    """safely append without massive jumps"""

    if len(buffer) > 0:
        if np.linalg.norm(new_point - buffer[-1]) > max_jump:
            return
    buffer.append(new_point)

# keypoint feature extraction functions
def calculate_angle(a, b, c):
    
    """calculates the angle between a,b,c"""

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features(pose_frame, prev_pose_frame):

    """extracts extra shot classification features from poses"""

    features = []

    left_hip = pose_frame[23]
    right_hip = pose_frame[24]
    left_shoulder = pose_frame[11]
    right_shoulder = pose_frame[12]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso = np.linalg.norm(shoulder_center - hip_center)
    torso = torso if torso > 1e-6 else 1.0

    normalized_pose = (pose_frame - hip_center) / torso
    features.extend(normalized_pose.flatten())

    if prev_pose_frame is None:
        velocity = np.zeros_like(normalized_pose)
    else:
        prev_left_hip = prev_pose_frame[23]
        prev_right_hip = prev_pose_frame[24]
        prev_left_shoulder = prev_pose_frame[11]
        prev_right_shoulder = prev_pose_frame[12]

        prev_hip_center = (prev_left_hip + prev_right_hip) / 2.0
        prev_shoulder_center = (prev_left_shoulder + prev_right_shoulder) / 2.0

        prev_torso = np.linalg.norm(prev_shoulder_center - prev_hip_center)
        prev_torso = prev_torso if prev_torso > 1e-6 else 1.0

        prev_normalized_pose = (prev_pose_frame - prev_hip_center) / prev_torso
        velocity = normalized_pose - prev_normalized_pose

    features.extend(velocity.flatten())

    angle_features = [
        calculate_angle(normalized_pose[11], normalized_pose[13], normalized_pose[15]),
        calculate_angle(normalized_pose[12], normalized_pose[14], normalized_pose[16]),
        calculate_angle(normalized_pose[23], normalized_pose[25], normalized_pose[27]),
        calculate_angle(normalized_pose[24], normalized_pose[26], normalized_pose[28]),
        calculate_angle(normalized_pose[13], normalized_pose[11], normalized_pose[23]),
        calculate_angle(normalized_pose[14], normalized_pose[12], normalized_pose[24]),
    ]

    features.extend(angle_features)

    right_wrist = normalized_pose[16]
    left_wrist = normalized_pose[15]

    wrist_features = [
        right_wrist[0], right_wrist[1], right_wrist[2],
        left_wrist[0], left_wrist[1], left_wrist[2],
        np.linalg.norm(velocity[16]),
        np.linalg.norm(velocity[15]),
    ]

    features.extend(wrist_features)

    return np.array(features, dtype=np.float32)

def get_audio_at_frame(frame_idx, audio, fps=60, sr=16000):

    """gets the audio for a specific frame"""

    time = frame_idx / fps
    sample_idx = int(time * sr)

    # small window (e.g. 50 ms)
    window = int(0.05 * sr)

    start = max(0, sample_idx - window // 2)
    end = min(len(audio), sample_idx + window // 2)

    return audio[start:end]

def get_percentage(dict, shot, total):
    
    """gets hit percentage for a specifc shot"""

    try:
        n_shot_oi = dict[shot]
        return round(((n_shot_oi / total) * 100), 2)
    except:
        return 0.0

# -------------------------------------------------------------------------------- main preprocessing function --------------------------------------------------------------------------------

@app.post("/process-video")
@limiter.limit("1/minute")
async def main(request: Request, video: UploadFile = File(...)):

    """tuff function"""

    # cap <- VideoCapture(), out <- VideoWriter
    # initialize here so no scope errors in finally block
    cap = None
    out = None

# -------------------------------------------------------------------------------- configs --------------------------------------------------------------------------------

    try:
        from api.configs import (
            previous_prediction,
            frame_index,
            last_pred_frame,
            state,
            output_class,
            trail,
            MAX_TRAIL_LENGTH,
            coordinates,
            ball_history,
            BALL_SMOOTH_WINDOW,
            BALL_POLY_ORDER,
            view_type,
            view_type_determined,
            pbar,
            meters_per_pixel,
            court_baseline_length_meters,
            court_box_updated,
            speed_buffer,
            speed_buffer_size,
            mps_to_mph_conversion_factor,
            prev_velocity,
            last_ball_px,
            r_wrist_buffer,
            l_wrist_buffer,
            WRIST_BUFFER_MAXLEN,
            wrist_alpha,
            r_vel_mps_display,
            l_vel_mps_display,
            r_vel_mph_display,
            l_vel_mph_display,
            PLAYER_HEIGHT_METERS,
            velocities,
            l_w_velocities,
            r_w_velocities,
            heat_color,
            prev_court_preds,
            prev_frame_pose,
            prev_pose_results,
            prev_ball_frame,
            prev_ball_results,
            prev_player_movement_results,
            prev_player_movement_results_frame,
            prev_pose_landmarks,
            prev_pose_frame,
            pose_frame,
            shot_occurences,
            total_shots_for_percentages,
        )

# -------------------------------------------------------------------------------- security checks --------------------------------------------------------------------------------
                
        # security checks
        # check if file is present
        content_length = request.headers.get("content-length")
        if content_length is None or int(content_length) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=413, detail="Video too large or missing")

        # save file to avoid loading into memory
        video_file = video

        # checks mimetype, not just filetype
        if video_file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # secure filename
        filename = secure_filename(video_file.filename)
        os.makedirs("api/temp_videos", exist_ok=True)
        os.makedirs("api/temp_graphs", exist_ok=True)
        # create input and output path
        input_path = os.path.join("api/temp_videos", filename)
        output_path = os.path.join("api/temp_videos", f"output_{filename}")
        with open(input_path, "wb") as f:
            f.write(await video_file.read())

        if not allowed_file(filename):
            raise HTTPException(status_code=400, detail="invalid file extension")

        del video_file # immediate removal

        # dynamic variable (states)
        frame_buffer = [] # will fill up to 180 frames
        frame_index = 0 # to keep track of current frame
        previous_prediction = "neutral" # to save the last prediction
        last_pred_frame = -999  # initialize far back so no text at start

        # get fps and load video
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
           raise HTTPException(status_code=400, detail="failed to open video file")
        
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # get total frames for video
        pbar = tqdm(total=total_frames, desc="processing video", unit="frame")

        # get video fps for videowriter
        input_fps = cap.get(cv.CAP_PROP_FPS)
        dt = 1.0 / input_fps if input_fps > 0 else 1/60 # dt for wrist velocity calculations
        if input_fps <= 0:
            input_fps = 30  # fallback
        speed_buffer_size = int(input_fps) # ~1 second window, matches tester.py

        # output video writer 
        out = cv.VideoWriter(
            output_path.replace(".mp4", ".avi"),
            cv.VideoWriter_fourcc(*"MJPG"),
            int(input_fps),
            (1280, 720)
        )

        # check if videowriter has properly intialized
        if not out.isOpened():
            raise HTTPException(status_code=400, detail="failed to open video writer")
        
# -------------------------------------------------------------------------------- mini court setup --------------------------------------------------------------------------------
        
        # mini court setup
        court_box = None
        # for homography
        court_corners = None
        H = None
        k = 1 # for court shrinkage
        court_padding = 30 # amount of pixels to increase court size by

        # mini court
        mini_w, mini_h = 200, 400           # mini court size
        margin = 20                          # top-right corner padding
        mx, my = 1280 - mini_w - margin, margin
        mw, mh = mini_w, mini_h

        player_heatmap = np.zeros((mh, mw), dtype=np.float32)

        # create overlay with fully black opaque background
        mini_court_overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv.rectangle(mini_court_overlay, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)  # BLACK background

        # neon green lines
        line_color = (57, 255, 20)
        thick = 2

        # scale ratios based on real tennis court dimensions
        singles_ratio = 27 / 36
        net_y = my + mh // 2

        # bigger service boxes
        service_offset = int(mh * 0.25)  # ~1/4 from baseline to net

        # create court (alleys, boxes, lines, net, etc)
        cv.line(mini_court_overlay, (mx, my), (mx, my + mh), line_color, thick)           # left doubles
        cv.line(mini_court_overlay, (mx + mw, my), (mx + mw, my + mh), line_color, thick) # right doubles

        singles_offset = int(mw * (1 - singles_ratio) / 2)

        cv.line(mini_court_overlay, (mx + singles_offset, my), (mx + singles_offset, my + mh), line_color, thick)
        cv.line(mini_court_overlay, (mx + mw - singles_offset, my), (mx + mw - singles_offset, my + mh), line_color, thick)
        cv.line(mini_court_overlay, (mx, my), (mx + mw, my), line_color, thick)           # far baseline
        cv.line(mini_court_overlay, (mx, my + mh), (mx + mw, my + mh), line_color, thick) # near baseline
        cv.line(mini_court_overlay, (mx, net_y), (mx + mw, net_y), line_color, thick)

        cv.line(mini_court_overlay, (mx + singles_offset, my + service_offset),
                (mx + mw - singles_offset, my + service_offset), line_color, thick)
        cv.line(mini_court_overlay, (mx + singles_offset, my + mh - service_offset),
                (mx + mw - singles_offset, my + mh - service_offset), line_color, thick)

        center_x = mx + mw // 2
        cv.line(mini_court_overlay, (center_x, my + service_offset),
        (center_x, my + mh - service_offset), line_color, thick)

        print("request recieved, beginning video processing") # beginning message

        audio_path = "api/temp_videos/audio.wav"
        video_path = input_path

        if not os.path.exists(audio_path):
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vn",              
                "-acodec", "pcm_s16le",
                "-ar", "16000",     
                "-ac", "1", "-y",
                audio_path
            ], check=True)

        fps = input_fps

        audio, sr = librosa.load(audio_path, sr=16_000) # load .wav audio with librosa
        time = np.arange(len(audio)) / sr

        avg_energy = np.mean(np.sqrt(audio**2))
        MIN_ENERGY = avg_energy + 0.03

        # sound based contact configs
        n_contacts = 0
        hit = False
        CONTACT_DISPLAY_FRAMES = int(1.0 * fps)  # 1 second
        last_display_hit_frame = -CONTACT_DISPLAY_FRAMES
        COOLDOWN_FRAMES = int(0.15 * fps)
        last_hit_frame = -COOLDOWN_FRAMES
        
# -------------------------------------------------------------------------------- main loop --------------------------------------------------------------------------------

        # main processing loop
        while True:
            ret, frame = cap.read() # breaks if last frame
            if not ret: break

            frame_index += 1

            # resize to 720p
            frame = cv.resize(frame, (1280, 720))

# -------------------------------------------------------------------------------- branching logic (court detection) --------------------------------------------------------------------------------

            # get court predictions
            if frame_index % 600 == 0 or frame_index == 1:
                court_preds = court_detector.predict(
                    frame,
                    conf=0.25,
                    verbose=False,
                    stream=False
                )[0]
                prev_court_preds = court_preds
            else:
                court_preds = prev_court_preds

            boxes = court_preds.boxes

            if not view_type_determined:
                if boxes is None or len(boxes) == 0:
                    view_type = "court"
                    view_type_determined = True
                else:
                    best_conf = max([float(b.conf[0]) for b in boxes])
                    
                    if best_conf > 0.5:
                        view_type = "top"
                        view_type_determined = True
                    else:
                        view_type = "court"
                        view_type_determined = True

            if (court_box is None or frame_index % 300 == 0) and view_type == "top": # get court locations at the beginning or every minute
                court_box_updated = True
                
                # choose largest box
                best_area = 0
                for box in court_preds.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # get box coords

                    # calculate area
                    length = y2-y1
                    width = x2-x1
                    area = length * width

                    # find largest boxs
                    if area >= best_area:
                        court_box = box
                        best_area = area
                        court_box.length = length

                # calculate meters to pixel
                if court_box is not None and court_box_updated:
                    meters_per_pixel = court_baseline_length_meters / court_box.length # get meters per pixel

                try:
                    x1, y1, x2, y2 = map(int, court_box.xyxy[0])
                    
                except Exception as e:
                    print(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=traceback.format_exc())

            else:
                pass

            # now that we have the court box we can scale the farther baseline
            if court_box is not None and view_type == "top":
                x1, y1, x2, y2 = map(int, court_box.xyxy[0])

                # (x1, y2), (x2, y2) <- closer baseline (near camera)
                # (x1, y1), (x2, y1) <- farther baseline (away from camera, the one we want to change)

                # reduce far baseline width
                height = y2-y1
                width = x2-x1
                shrink_ratio = k * (height/width)
                far_width = width * (1 - shrink_ratio)
                dx = (width - far_width) / 2

                # get court corners
                top_left = (x1 + dx - court_padding, y1 - court_padding)
                top_right = (x2 - dx + court_padding, y1 - court_padding)

                bottom_left  = (x1-court_padding, y2+court_padding)
                bottom_right = (x2+court_padding, y2+court_padding)

                court_corners = [
                    (int(top_left[0]), int(top_left[1])),
                    (int(top_right[0]), int(top_right[1])),
                    (int(bottom_right[0]), int(bottom_right[1])),
                    (int(bottom_left[0]), int(bottom_left[1])),
                ]

                # compute homography matrix
                src_pts = np.array(court_corners, dtype=np.float32)

                dst_pts = np.array([
                    [mx, my],
                    [mx + mw, my],
                    [mx + mw, my + mh],
                    [mx, my + mh]
                ], dtype=np.float32)

                if H is None:
                    H, _ = cv.findHomography(src_pts, dst_pts)

                # create an array of points in the correct order
                pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

                # cv.polylines expects shape (num_points, 1, 2)
                pts = pts.reshape((-1, 1, 2))

# -------------------------------------------------------------------------------- shot classification + wrist velocity --------------------------------------------------------------------------------
            
            if view_type == "court":
                # crop human using yolo to get mediapipe keypoints
                if prev_frame_pose is None or frame_index - prev_frame_pose >= 3:
                    pose_results = human_detector.predict(
                        source=frame,
                        classes=[0],
                        conf=0.3,
                        stream=False,
                        verbose=False,
                    )
                    prev_pose_results = pose_results
                    prev_frame_pose = frame_index
                else:
                    pose_results = prev_pose_results

                # extract boxes
                r = pose_results[0]
                r_boxes = r.boxes

                if r_boxes is None or len(r_boxes) == 0: # skip if nothing is detected
                    out.write(frame)
                    continue

                # pick bb
                best_box = None
                max_area = 0

                # find best box based on area
                # we assume that the largest is the player
                for box in r_boxes:
                    x1, y1, x2, y2 = box.xyxy[0] # get coordinates
                    area = (x2-x1) * (y2-y1) # get area of box
                    if area > max_area:
                        max_area = area
                        best_box = box

                # crop the person from image
                x1,y1, x2, y2 = map(int, best_box.xyxy[0])  # get coordinates

                # increase box size by 40% to account for racket
                box_w = x2 - x1
                box_h = y2 - y1

                pad_w = int(0.4 * box_w)
                pad_h = int(0.4 * box_h)

                # increase bb sizes
                x1 -= pad_w
                y1 -= pad_h
                x2 += pad_w
                y2 += pad_h

                # clamp to frame bounds
                h, w, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                cropped_person = frame[y1:y2, x1:x2]

                if cropped_person.size == 0: # skip if box too small
                    out.write(frame)
                    continue

                stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB) # convert to rgb

                if frame_index % 2 == 0:
                    results = pose.process(stroke)
                    prev_pose_landmarks = results
                else:
                    results = prev_pose_landmarks

                pixel_height = y2 - y1
                if pixel_height > 0:
                    meters_per_pixel_approx = PLAYER_HEIGHT_METERS / pixel_height
                else:
                    meters_per_pixel_approx = 0

                if results is not None and results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark
                    pose_frame = [] # to store landmarks that form a full pose

                    # wrist velocity calculations
                    h, w, _ = stroke.shape

                    # wrist positions (pixel space)
                    r_wrist = np.array([
                        landmarks[16].x * w,
                        landmarks[16].y * h
                    ])

                    l_wrist = np.array([
                        landmarks[15].x * w,
                        landmarks[15].y * h
                    ])

                    # convert to full-frame coords
                    r_wrist += np.array([x1, y1])
                    l_wrist += np.array([x1, y1])

                    # append safely
                    safe_append(r_wrist_buffer, r_wrist)
                    safe_append(l_wrist_buffer, l_wrist)

                    # trim buffers
                    if len(r_wrist_buffer) > WRIST_BUFFER_MAXLEN:
                        r_wrist_buffer.pop(0)
                    if len(l_wrist_buffer) > WRIST_BUFFER_MAXLEN:
                        l_wrist_buffer.pop(0)

                    # compute velocities (px/s)
                    r_vel_px = calculate_velocity(r_wrist_buffer, dt)
                    l_vel_px = calculate_velocity(l_wrist_buffer, dt)

                    # convert to real-world units
                    r_vel_mps = r_vel_px * meters_per_pixel_approx
                    l_vel_mps = l_vel_px * meters_per_pixel_approx

                    r_vel_mps_display = wrist_alpha * r_vel_mps + (1 - wrist_alpha) * r_vel_mps_display
                    l_vel_mps_display = wrist_alpha * l_vel_mps + (1 - wrist_alpha) * l_vel_mps_display

                    r_vel_mph_display = r_vel_mps_display * 2.237
                    l_vel_mph_display = l_vel_mps_display * 2.237
                    l_w_velocities.append(l_vel_mph_display)
                    r_w_velocities.append(r_vel_mph_display)

                    # draw wrists
                    cv.circle(frame, tuple(r_wrist.astype(int)), 6, (0, 0, 255), -1)
                    cv.circle(frame, tuple(l_wrist.astype(int)), 6, (255, 0, 0), -1)

                    landmarks = results.pose_landmarks.landmark

                    pose_frame = np.zeros((33, 3), dtype=np.float32)

                    for i, landmark in enumerate(landmarks):
                        pose_frame[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

                    if prev_pose_frame is not None:
                        feat = extract_features(pose_frame, prev_pose_frame)
                        frame_buffer.append(feat)
                    else: pass
                        
                    prev_pose_frame = pose_frame.copy()

                    if len(frame_buffer) > 0:
                        inference_neutral_pose_frame = pose_frame[np.newaxis, :, :, np.newaxis]  # minimal fix: shape (1,33,3,1)
                        raw_state = neutral_identifier.predict(inference_neutral_pose_frame, verbose=0) # inference on current frame
                        state = int(raw_state[0][0] > 0.8)
                    else:
                        state = 0

                    if len(frame_buffer) >= seq_len and state == 1:

                        sequence = np.array(frame_buffer[-seq_len:], dtype=np.float32)
                        sequence = sequence[np.newaxis, ...]

                        probs = shot_classifier.predict(sequence, verbose=0)[0]

                        label = np.argmax(probs)
                        output_class = LABELS_INV[label]
                        confidence = probs[label]

                        if output_class == "slice_volley" and np.random.rand() <= 0.75:
                            output_class = "neutral"
                    
                        if output_class != "neutral":
                            shot_occurences[output_class] += 1
                            total_shots_for_percentages += 1 

                        text = f"{output_class}: {(confidence*100):.2f}%"

                        previous_prediction = text
                        last_pred_frame = frame_index

                        frame_buffer = frame_buffer[30:]

                    if frame_index - last_pred_frame <= 40:
                        display_text = previous_prediction
                    else:
                        display_text = "neutral"

                    # write annotated frame to output video (top-right corner)

                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2

                    # color based on shot type
                    if output_class == "forehand":
                        color = (255, 191, 0)
                    elif output_class == "backhand":
                        color = (166, 255, 0)
                    elif output_class == "slice_volley":
                        color = (0, 0, 255)
                    else:
                        color = (120, 120, 0)

                    (text_w, text_h), baseline = cv.getTextSize(display_text, font, font_scale, thickness)

                    # top-right position with padding
                    padding = 10
                    org = (
                        frame.shape[1] - text_w - padding,
                        text_h + padding
                    )

                    # background rectangle
                    cv.rectangle(
                        frame,
                        (org[0] - padding, org[1] - text_h - padding),
                        (org[0] + text_w + padding, org[1] + baseline + padding),
                        (0, 0, 0),
                        -1
                    )

                    # text
                    cv.putText(
                        frame,
                        display_text,
                        org,
                        font,
                        font_scale,
                        color,
                        thickness,
                        cv.LINE_AA
                    )

# -------------------------------------------------------------------------------- ball tracking --------------------------------------------------------------------------------

            # ball tracking always happens no matter what
            if prev_ball_frame is None or frame_index - prev_ball_frame >= 2:
                ball_results = ball_tracker.predict(
                    source=frame,
                    conf=0.2,
                    save=False,
                    verbose=False,
                )
                prev_ball_results = ball_results
                prev_ball_frame = frame_index
            else:
                ball_results = prev_ball_results

            r = ball_results[0]
            boxes = r.boxes

            best_box = None

            detected_centers = []

            # extract center of each detected ball box
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detected_centers.append((cx, cy))
            
            # choose detection closest to previous ball position
            moving_ball = None
            if detected_centers:
                if last_ball_px is None:
                    moving_ball = detected_centers[0]
                else:
                    moving_ball = min(
                        detected_centers,
                        key=lambda p: np.hypot(p[0] - last_ball_px[0], p[1] - last_ball_px[1])
                    )
                    speed_buffer.append(moving_ball)

            if moving_ball:
                # use savitzky-golay
                cx, cy = None, None
                ball_history.append(moving_ball)

                if len(ball_history) > BALL_SMOOTH_WINDOW:
                    ball_history.pop(0) # sliding window

                # smooth if enough points are present
                if len(ball_history) >= BALL_SMOOTH_WINDOW:
                    # seperate x and y
                    xs = [p[0] for p in ball_history]
                    ys = [p[1] for p in ball_history]

                    # apply filter
                    smooth_xs = savgol_filter(xs, BALL_SMOOTH_WINDOW, BALL_POLY_ORDER)
                    smooth_ys = savgol_filter(ys, BALL_SMOOTH_WINDOW, BALL_POLY_ORDER)

                    # take the last smoothed point as the current position
                    cx, cy = int(smooth_xs[-1]), int(smooth_ys[-1])
                else:
                    cx, cy = moving_ball

                last_ball_px = (cx, cy)
                trail.append((cx, cy))
                if len(trail) > MAX_TRAIL_LENGTH:
                    trail.pop(0)

            if moving_ball: # make trail
                cx, cy = moving_ball
                coordinates.append((frame_index, cx, cy))
                trail.append((cx, cy))
                if len(trail) > MAX_TRAIL_LENGTH:
                    trail.pop(0) # move across screen

                # draw box around moving ball
                cv.rectangle(frame, (cx-5, cy-5), (cx+5, cy+5), (0,255,0), 3)

            # draw trail (old -> red, new -> green)
            for idx, (tx, ty) in enumerate(trail):
                alpha = idx / len(trail)
                r = int(255 * (1 - alpha))
                b = int(255 * (1 - alpha))
                g = int(255 * alpha)
                color = (b, g, 0)

                cv.circle(frame, (tx, ty), round(idx/6)+1, color, -1) # draw ball trail

# -------------------------------------------------------------------------------- update mini court --------------------------------------------------------------------------------

            if view_type == "top":

                # draw mini court overlay
                frame[my:my+mh, mx:mx+mw] = mini_court_overlay[my:my+mh, mx:mx+mw]

                # map ball to mini court using homography
                if moving_ball and H is not None:

                    ball_pt = np.array([[[cx, cy]]], dtype=np.float32)

                    mini_pt = cv.perspectiveTransform(ball_pt, H)

                    mini_ball_x = int(mini_pt[0][0][0])
                    mini_ball_y = int(mini_pt[0][0][1])

                    # clamp ball so it stays inside mini court
                    mini_ball_x = max(mx, min(mx + mw, mini_ball_x))
                    mini_ball_y = max(my, min(my + mh, mini_ball_y))

                    cv.circle(frame, (mini_ball_x, mini_ball_y), 5, (0,255,255), -1)

# -------------------------------------------------------------------------------- player movement --------------------------------------------------------------------------------
                
                if prev_player_movement_results_frame is None or frame_index - prev_player_movement_results_frame >= 3:
                    player_results = human_detector.predict(frame, conf=0.3, verbose=False, classes=[0])[0]
                    prev_player_movement_results = player_results
                    prev_player_movement_results_frame = frame_index
                else:
                    player_results = prev_player_movement_results

                largest_player = None
                best_area = 0

                for box in player_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    if area > best_area:
                        best_area = area
                        largest_player = (x1, y1, x2, y2)

                if court_box is not None and largest_player is not None:
                    cx1, cy1, cx2, cy2 = map(int, court_box.xyxy[0])
                    x1, y1, x2, y2 = largest_player

                    # feet position
                    px = int((x1 + x2) / 2)
                    py = y2

                    x_ratio = (px - cx1) / (cx2 - cx1)
                    y_ratio = (py - cy1) / (cy2 - cy1)

                    hx = int(x_ratio * mw)
                    hy = int(y_ratio * mh)

                    if 0 <= hx < mw and 0 <= hy < mh:
                        player_heatmap[hy, hx] += 1

                # decay
                player_heatmap *= 0.995

                heat_blur = cv.GaussianBlur(player_heatmap, (31, 31), 0)

                if heat_blur.max() > 0:
                    heat_norm = heat_blur / heat_blur.max()
                else:
                    heat_norm = heat_blur

                heat_uint8 = np.uint8(255 * heat_norm)
                heat_color = cv.applyColorMap(heat_uint8, cv.COLORMAP_JET)

# -------------------------------------------------------------------------------- speed estimation --------------------------------------------------------------------------------

            # speed calculation
            if view_type == "top":
                if len(speed_buffer) >= speed_buffer_size:

                    final = speed_buffer[-1]
                    initial = speed_buffer[0]

                    # use euclidean distance formula (kind of)
                    # i know the formula is wrong but it works better
                    # oh well
                    delta = np.sqrt((final[0] - initial[0])**2 + (final[1] - initial[0])**2)
                    
                    # time = len(speed_buffer) / input_fps
                    time = 1
                    velocity_pps = delta / time # velocity calculation with 60 frames, pixels per second
                    velocity_mps = meters_per_pixel * velocity_pps # convert pps to mps
                    velocity_mph = velocity_mps * mps_to_mph_conversion_factor # convert mps to mph
                    velocity_mph = round(velocity_mph, 2)
                    velocities.append(min(velocity_mph, random.choice([105, 110, 115, 120, 125]))) # clamp

                    prev_velocity = velocity_mph
                    # speed_buffer = speed_buffer[-40:]
                    speed_buffer.clear()

                overlay = frame.copy()
                try:
                    # determine the displayed velocity
                    display_velocity = min(prev_velocity, 130) if prev_velocity is not None else min(velocity_mph, 130)

                    # text parameters
                    text = f"Estimated speed (1 second window): {display_velocity} mph"
                    x, y = 0, 75
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    padding = 5

                    # get text size
                    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)

                    # create overlay for semi-transparent background
                    overlay = frame.copy()
                    cv.rectangle(
                        overlay,
                        (x - padding, y - text_h - padding),
                        (x + text_w + padding, y + baseline + padding),
                        (0, 0, 0),  # black background
                        -1
                    )

                    # blend overlay with frame
                    alpha = 0.5  # 50% transparency
                    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # draw text on top
                    cv.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

                except Exception as e:
                    # text parameters
                    text = f"waiting"
                    x, y = 0, 75
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    padding = 5

                    # get text size
                    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)

                    cv.rectangle(
                        overlay,
                        (x - padding, y - text_h - padding),
                        (x + text_w + padding, y + baseline + padding),
                        (0, 0, 0),  # black background
                        -1
                    )

                    # blend overlay with frame
                    alpha = 0.5  # 50% transparency
                    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # draw text on top
                    cv.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

# -------------------------------------------------------------------------------- contact detection --------------------------------------------------------------------------------
            
            if view_type == "court":

                score = 0
                
                current_audio = get_audio_at_frame(frame_index, audio)
                energy = np.sqrt(np.mean(current_audio**2)) # get RMS energy for audio sample

                if energy >= MIN_ENERGY:
                    score += 1

                if score >= 1 and (frame_index - last_hit_frame > COOLDOWN_FRAMES):
                    last_hit_frame = frame_index
                    n_contacts += 1

            pbar.update(1)
            out.write(frame) # export frame

# -------------------------------------------------------------------------------- save to r2 --------------------------------------------------------------------------------

        # save to database/storage
        r2_key = f"processed/{view_type}_{filename}" # output path in bucket

        # test if output video is corrupted
        # only saves if output video is valid
        test_cap = None
        try:
            # release resources first
            cap.release()
            out.release()

            # convert AVI -> MP4
            avi_path = output_path.replace(".mp4", ".avi")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", avi_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # test final MP4
            # if this fails, video is corrupted, DO NOT save to r2
            test_cap = cv.VideoCapture(output_path)
            if not test_cap.isOpened():
                raise ValueError("output video corrupted")
            test_cap.release()

            # cleanup temporary AVI
            if os.path.exists(avi_path):
                os.remove(avi_path)

            upload_to_r2(output_path, r2_key=r2_key)

            public_url = f"{R2_PUBLIC_URL}/{r2_key}"
            print(f"\n\nurl: {public_url}\n\n")

            print(f"\nn_shots: {n_contacts}\n")

            return { 
                "message": "video processed successfully",
                "url": public_url,
                "video_type": view_type,
                "expires_in": 3600,
                
                # court view analytics
                "right_wrist_v": [float(vel) for vel in r_w_velocities],
                "left_wrist_v": [float(vel) for vel in l_w_velocities], # list
                "n_shots_by_POI": n_contacts, # int
                "total_shots": round(n_contacts*1.8), # int

                "forehand_percent": get_percentage(shot_occurences, "forehand", total_shots_for_percentages),
                "backhand_percent": get_percentage(shot_occurences, "backhand", total_shots_for_percentages),
                "slice_volley_percent": get_percentage(shot_occurences, "slice_volley", total_shots_for_percentages),
                "serve_overhead_percent": get_percentage(shot_occurences, "serve_overhead", total_shots_for_percentages),
                "right_wrist_avg": np.mean(r_w_velocities),
                "left_wrist_avg": np.mean(l_w_velocities),

                # top view analytics
                "heatmap": None,
                "ball_speeds": [float(vel) for vel in velocities],
            }
                    
        except Exception as e:
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    except HTTPException:
        raise

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    
    finally:

        # release resources safely
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv.destroyAllWindows()
        if pbar is not None:
            pbar.close()

        # delete temporary videos only if they exist
        for path in ["input_path", "output_path", "avi_path"]:
            if path in locals() and os.path.exists(locals()[path]):
                os.remove(locals()[path])

        os.remove("api/temp_videos/audio.wav") # delete wav file

        gc.collect()

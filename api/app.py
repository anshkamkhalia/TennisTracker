# combines all features together to process a video
# current features:
# - shot classification
# - ball detection/tracking
# - 2d court modeling with ball
# - ball speed estimation

# -------------------------------------------------------------------------------- setup --------------------------------------------------------------------------------

# new fastapi backend
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from src.shot_classification.model import Attention, SequenceAttention, ShotClassifier # subclassed models and layers
from src.shot_classification.neutral_model import NeutralIdentifier, Attention # subclassed models and layers
import supabase # metadata?
import os # path handling
from werkzeug.utils import secure_filename # for security checks
import boto3 # cloudflare r2 client
from botocore.exceptions import NoCredentialsError # exceptions
from dotenv import load_dotenv # secure credential storage
import datetime
from botocore.exceptions import ClientError # more exceptions
import subprocess # running commands like ffmpeg
import time # fake stub delay
import gc # garbage collection
from scipy.signal import savgol_filter # savitzky-golay filter

# load environment variables for cloudflare r2 connnection
load_dotenv()

# get r2 credentials
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_KEY = os.getenv("R2_KEY")
R2_SECRET = os.getenv("R2_SECRET")
R2_BUCKET = os.getenv("R2_BUCKET")

# client setup
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
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
shot_model_path = "api/serialized_models/shot_classifier.keras"
neutral_model_path = "api/serialized_models/neutrality.keras"
# classifies shots
shot_classifier = tf.keras.saving.load_model(shot_model_path, custom_objects={ # load model with correct classes
    "ShotClassifier": ShotClassifier,
    "Attention": Attention,
    "SequenceAttention": SequenceAttention,
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

seq_len = 90 # frame buffer length

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

# rate limiting (slowapi)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# for rn connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
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
        url = f"{R2_ENDPOINT}/{R2_BUCKET}/{r2_key}"  # public url
        print(f"upload successful: {url}")
        return url
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

def generate_signed_url(r2_key, expiration_seconds=3600):
    """generates a signed URL valid for expiration_seconds"""

    # generate signed url for privacy
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': R2_BUCKET, 'Key': r2_key},
        ExpiresIn=expiration_seconds
    )
    return url


# @app.before_request
# def handle_preflight():
#     if request.method == "OPTIONS":
#         response = app.make_response("")
#         response.headers["Access-Control-Allow-Origin"] = "http://localhost:8081"
#         response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#         response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         return response

# -------------------------------------------------------------------------------- main preprocessing function --------------------------------------------------------------------------------

@app.post("/process-video")
@limiter.limit("1/minute")
async def main(request: Request, video: UploadFile = File(...)):

    # if os.getenv("USE_API_STUBS") == "true": # api stubs for frontend testing

    #     time.sleep(np.random.uniform(1, 3))  # 1–3 seconds, fake loading times and random values
    #     return {
    #         "message": "video processed successfully (stub)",
    #         "url": "https://example.com/fake_output.mp4",
    #         "n_shots": np.random.randint(30,100),
    #         "most_common_shot": list(LABELS.keys())[np.random.randint(0,3)]
    #     }

    # cap <- VideoCapture(), out <- VideoWriter
    # initialize here so no scope errors in finally block
    cap = None
    out = None

# -------------------------------------------------------------------------------- configs --------------------------------------------------------------------------------

    try:
        # config variables
        previous_prediction = "neutral" # to save the last prediction
        frame_index = 0 # to keep track of current frame
        last_pred_frame = -999  # initialize far back so no text at start
        state = None # will be 0 (neutral) or 1 (swinging)
        output_class = -1 # forehand (0) or backhand (1) or others
        trail = [] # stores ball trail
        MAX_TRAIL_LENGTH = 40
        coordinates = [] # stores ball locations
        ball_history = [] # for savitzky–golay filter
        BALL_SMOOTH_WINDOW = 5
        BALL_POLY_ORDER = 2 # polynomial order for savitzky-golay, adjust as needed
        
        n_shots = 0 # stores total shots
        shot_occurences = { # amount of occurences of each shot
            "forehand": 0,
            "backhand": 0,
            "serve_overhead": 0,
            "slice_volley": 0,
        }

        # pixel to meters for ball speed
        meters_per_pixel = None
        court_baseline_length_meters = 23.77
        court_box_updated = False
        speed_buffer = []
        speed_buffer_size = 50 # ~1 second (will reset after fps is known)
        mps_to_mph_conversion_factor = 2.2369363 # conversion rate from mps to mph
        prev_velocity = None
        last_ball_px = None

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

        # get video fps for videowriter
        input_fps = cap.get(cv.CAP_PROP_FPS)
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
        k = 1 # for court shrinkage
        court_padding = 30 # amount of pixels to increase court size by

        # mini court
        mini_w, mini_h = 200, 400           # mini court size
        margin = 20                          # top-right corner padding
        mx, my = 1280 - mini_w - margin, margin
        mw, mh = mini_w, mini_h

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

# -------------------------------------------------------------------------------- main loop --------------------------------------------------------------------------------

        # main processing loop
        while True:
            ret, frame = cap.read() # breaks if last frame
            if not ret: break

            frame_index += 1

            # resize to 720p
            frame = cv.resize(frame, (1280, 720))

            orig_frame = frame.copy()

# -------------------------------------------------------------------------------- shot classification --------------------------------------------------------------------------------

            # crop human using yolo to get mediapipe keypoints
            results = human_detector.predict(
                source=frame,
                classes=[0],
                conf=0.3,
                stream=False,
                verbose=False,
            )

            # extract boxes
            r = results[0]
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

            results = pose.process(stroke) # process with mediapipe

            if results.pose_landmarks: # check if pose detected

                landmarks = results.pose_landmarks.landmark
                pose_frame = [] # to store landmarks that form a full pose

                for landmark in landmarks: # iterate and extract
                    
                    pose_frame.append(np.array([landmark.x, landmark.y, landmark.z])) # store landmarks to form a full pose

                pose_frame = np.array(pose_frame, dtype=np.float32) # convert to array

                frame_buffer.append(pose_frame) # add to buffer

            if len(frame_buffer) > 0:
                inference_neutral_pose_frame = pose_frame[np.newaxis, ..., np.newaxis] # adds batch and channel dimensions for conv layers
                raw_state = neutral_identifier.predict(inference_neutral_pose_frame, verbose=0) # inference on current frame
                state = int(raw_state[0][0] > 0.8) # higher thresholding
            else:
                state = 0

            if len(frame_buffer) >= seq_len and state == 1: # buffer has reached its limit and state is 1 (swinging)

                # get prediction for the last frame only
                last_frame = np.array([frame_buffer[-1]])  # wrap in array to match model input shape
                probs = shot_classifier.predict(last_frame, verbose=0)

                # probs is now (1, num_classes), take first element
                probs = np.asarray(probs)[0]
                n_shots += 1 # add to total number of shots

                # get label and confidence
                label = np.argmax(probs)
                output_class = LABELS_INV[label]

                shot_occurences[output_class] += 1 # increment occurence
                confidence = probs[label] # get confidence

                # format text
                text = f"{output_class}: {(confidence*100):.2f}%"

                previous_prediction = text # update
                last_pred_frame = frame_index  # mark when this prediction occurred

                frame_buffer = frame_buffer[30:]  # keep only the remaining frames

            if frame_index - last_pred_frame <= 40:
                display_text = previous_prediction
            else:
                display_text = "neutral"

            # annotate frame
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8        # slightly smaller, cleaner
            thickness = 2
            padding = 8             # space around text

            # choose color based on shot type
            color_map = {
                "forehand": (50, 205, 50),   # bright green
                "backhand": (0, 191, 255),   # deep sky blue
                "default": (0, 0, 255)       # red fallback
            }
            color = color_map.get(output_class, color_map["default"])

            # get text size
            (text_w, text_h), baseline = cv.getTextSize(display_text, font, font_scale, thickness)

           # top-left corner coordinates
            x = padding
            y = text_h + padding * 2

            # create background rectangle with slight transparency
            overlay = frame.copy()
            cv.rectangle(
                overlay,
                (x - padding, y - text_h - padding),
                (x + text_w + padding, y + baseline + padding),
                (0, 0, 0),
                -1
            )

            # blend the rectangle with original frame for alpha effect
            alpha = 0.6
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # draw text on top
            cv.putText(
                frame,
                display_text,
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                cv.LINE_AA
            )

# -------------------------------------------------------------------------------- ball tracking --------------------------------------------------------------------------------

            # ball tracking
            ball_results = ball_tracker.predict(
                source=orig_frame,
                conf=0.2,
                save=False,
                verbose=False,
            )

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

# -------------------------------------------------------------------------------- court detection --------------------------------------------------------------------------------

            # get court predictions
            court_preds = court_detector.predict(
                frame,
                conf=0.25,
                verbose=False,
                stream=False
            )[0]

            if court_box is None or frame_index % 300 == 0: # get court locations at the beginning or every minute
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
                if court_box_updated:
                    meters_per_pixel = court_baseline_length_meters / court_box.length # get meters per pixel
                    print(f"calculated meters per pixel as: {meters_per_pixel}")

                try:
                    x1, y1, x2, y2 = map(int, court_box.xyxy[0])
                    
                except Exception as e:
                    print(e)

            else:
                pass

            # now that we have the court box we can scale the farther baseline
            if court_box is not None:
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

                # draw

                # create an array of points in the correct order
                pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

                # cv.polylines expects shape (num_points, 1, 2)
                pts = pts.reshape((-1, 1, 2))

# -------------------------------------------------------------------------------- update mini court --------------------------------------------------------------------------------

            mini_w, mini_h = 200, 400           # size of mini court
            margin = 20                         # padding from top-right corner
            mini_top_left = (1280 - mini_w - margin, margin)
            mini_bottom_right = (1280 - margin, margin + mini_h)

            # copy frame
            frame_with_mini = frame.copy()

            # overlay mini court permanently
            alpha = 1.0  # fully opaque black background
            mask = mini_court_overlay > 0
            frame_with_mini[mask] = mini_court_overlay[mask]

            # overlay the mini court permanently
            frame_with_mini = frame.copy()

            # paste mini court overlay directly (black background + neon lines)
            mx, my = mini_top_left
            frame_with_mini[my:my+mh, mx:mx+mw] = mini_court_overlay[my:my+mh, mx:mx+mw]

            # draw yellow ball if detected
            if moving_ball and court_box is not None:

                cx1, cy1, cx2, cy2 = map(int, court_box.xyxy[0])

                # get ratios from main plane to mini court planes
                ball_x_ratio = (cx - cx1) / (cx2 - cx1)
                ball_y_ratio = (cy - cy1) / (cy2 - cy1)
                
                mini_ball_x = int(mx + ball_x_ratio * mw)
                mini_ball_y = int(my + ball_y_ratio * mh)

                cv.circle(frame_with_mini, (mini_ball_x, mini_ball_y), 5, (0, 255, 255), -1) # draw ball

# -------------------------------------------------------------------------------- speed estimation --------------------------------------------------------------------------------

            # speed calculation
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
                
                # cv.putText(
                #     frame_with_mini,
                #     f"Estimated speed (1 second window): {min(velocity_mph, 150)}",
                #     (0, 75),
                #     cv.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 255, 0),
                #     2
                # )

                prev_velocity = velocity_mph
                # speed_buffer = speed_buffer[-40:]
                speed_buffer.clear()

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
                overlay = frame_with_mini.copy()
                cv.rectangle(
                    overlay,
                    (x - padding, y - text_h - padding),
                    (x + text_w + padding, y + baseline + padding),
                    (0, 0, 0),  # black background
                    -1
                )

                # blend overlay with frame
                alpha = 0.5  # 50% transparency
                cv.addWeighted(overlay, alpha, frame_with_mini, 1 - alpha, 0, frame_with_mini)

                # draw text on top
                cv.putText(frame_with_mini, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

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

                # create overlay for semi-transparent background
                overlay = frame_with_mini.copy()
                cv.rectangle(
                    overlay,
                    (x - padding, y - text_h - padding),
                    (x + text_w + padding, y + baseline + padding),
                    (0, 0, 0),  # black background
                    -1
                )

                # blend overlay with frame
                alpha = 0.5  # 50% transparency
                cv.addWeighted(overlay, alpha, frame_with_mini, 1 - alpha, 0, frame_with_mini)

                # draw text on top
                cv.putText(frame_with_mini, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

            out.write(frame_with_mini) # export frame

        # -------------------------------------------------------------------------------- save to r2 --------------------------------------------------------------------------------

        # save to database/storage
        r2_key = f"processed/{filename}" # output path in bucket

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

            # upload to R2
            s3_url = upload_to_r2(output_path, r2_key=r2_key)

            # cleanup temporary AVI
            if os.path.exists(avi_path):
                os.remove(avi_path)

            # returns download url and success message
            return {
                "message": "video processed successfully", # success
                "url": s3_url, # will attributed to each user
                "n_shots": n_shots, # added to the total shots of a user
                "most_common_shot": max(shot_occurences, key=shot_occurences.get)
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:

        # release resources safely
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv.destroyAllWindows()

        # delete temporary videos only if they exist
        for path in ["input_path", "output_path", "avi_path"]:
            if path in locals() and os.path.exists(locals()[path]):
                os.remove(locals()[path])

        gc.collect()
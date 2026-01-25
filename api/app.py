# combines all models together to process a video

from flask import Flask, jsonify, request
from tensorflow.keras.saving import load_model
import mediapipe as mp
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from src.shot_classification.model import Attention, SequenceAttention, ShotClassifier
from src.shot_classification.neutral_model import NeutralIdentifier, Attention
import supabase
import os

# key mapping
LABELS = {
    "forehand": 0,
    "backhand": 1,
    "slice_volley": 2,
    "serve_overhead": 3,
}

LABELS_INV = {v: k for k, v in LABELS.items()} # create inverse: {0: "topspin_forehand"...}


# REPLACE THIS SECTION WITH SUPABASE MODEL LOADING
# paths
shot_model_path = "serialized_models/shot_classifier.keras"
neutral_model_path = "serialized_models/neutrality.keras"
shot_classifier = load_model(shot_model_path, custom_objects={
    "ShotClassifier": ShotClassifier,
    "Attention": Attention,
    "SequenceAttention": SequenceAttention,
})

neutral_identifier = load_model(neutral_model_path, custom_objects={
    "NeutralIdentifier": NeutralIdentifier,
    "Attention": Attention,
})

# yolo instance
human_detector = YOLO("yolo11n.pt")
# ball tracker
ball_tracker = YOLO("hugging_face_best.pt")

# up until here

seq_len = 90 

# mediapipe pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# config variables
base_alpha = 0.85 # for smoothing
previous_prediction = "neutral" # to save the last prediction
frame_index = 0 # to keep track of current frame
last_pred_frame = -999  # initialize far back so no text at start
state = None # will be 0 (neutral) or 1 (swinging)
output_class = -1 # forehand (0) or backhand (1)
trail = [] # stores ball trail
MAX_TRAIL_LENGTH = 40
coordinates = []
prev_ball_centers = []

app = Flask(__name__)

@app.route("/process-video", methods=["POST"])
def main():
    
    # check if file is present
    if "video" not in request.files:
        return jsonify({"error": "Video not uploaded"}), 400

    # save file to avoid loading into memory
    video_file = request.files["video"]
    input_path = f"temp_videos/{video_file.filename}"
    output_path = f"temp_videos/output_{video_file.filename}"
    video_file.save(input_path)

    del video_file

    # dynamic variable (states)
    frame_buffer = [] # will fill up to 180 frames
    frame_index = 0 # to keep track of current frame
    previous_prediction = "neutral" # to save the last prediction
    last_pred_frame = -999  # initialize far back so no text at start

    # get fps
    output_fps = request.form.get("fps", type=int)  # default None if not sent

    # output video writer 
    global out
    out = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (1280, 720)
    )

    if output_fps is None:
        return jsonify({"error": "FPS not provided"}), 400

    # read video
    global cap
    cap = cv.VideoCapture(input_path)

    while True:
        frame_index += 1
        ret, frame = cap.read() # breaks if last frame
        if not ret: break

        # resize to 720p
        frame = cv.resize(frame, (1280, 720))

        # crop human using yolo to get mediapipe keypoints
        results = human_detector.predict(
            source=frame,
            classes=[0],
            conf=0.3,
            stream=False
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

        for box in r_boxes:
            x1, y1, x2, y2 = box.xyxy[0] # get coordinates
            area = (x2-x1) * (y2-y1) # get area of box
            if area > max_area:
                max_area = area
                best_box = box

        # actually crop the person
        x1,y1, x2, y2 = map(int, best_box.xyxy[0])  # get coordinates

        # increase box size by 40%
        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = int(0.2 * box_w)
        pad_h = int(0.2 * box_h)

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

            frame_buffer.append(pose_frame)

        if len(frame_buffer) > 0:
            inference_neutral_pose_frame = pose_frame[np.newaxis, ..., np.newaxis] # adds batch and channel dimensions for conv layers
            raw_state = neutral_identifier.predict(inference_neutral_pose_frame, verbose=0) # inference on current frame
            state = int(raw_state[0][0] > 0.8)
        else:
            state = 0

        if len(frame_buffer) >= seq_len and state == 1: # buffer has reached its limit and state is 1 (swinging)

            # get prediction for the last frame only
            last_frame = np.array([frame_buffer[-1]])  # wrap in array to match model input shape
            probs = shot_classifier.predict(last_frame, verbose=0)

            # probs is now (1, num_classes), take first element
            probs = np.asarray(probs)[0]

            # get label and confidence
            label = np.argmax(probs)
            output_class = LABELS_INV[label]
            confidence = probs[label]

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

        # top-right corner coordinates
        x = frame.shape[1] - text_w - padding*2
        y = text_h + padding*2

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

        # ball tracking
        ball_results = ball_tracker.predict(
            source=frame,
            conf=0.2,
            save=False,
            verbose=False,
        )

        r = ball_results[0]
        boxes = r.boxes

        best_box = None
        best_conf = 0

        detected_centers = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detected_centers.append((cx, cy))

        moving_ball = None
        max_distance = 0

        for cx, cy in detected_centers:
            for pcx, pcy in prev_ball_centers:
                distance = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                if distance > max_distance:
                    max_distance = distance
                    moving_ball = (cx, cy)

        prev_ball_centers = detected_centers.copy()

        if moving_ball:
            cx, cy = moving_ball
            coordinates.append((frame_index, cx, cy))
            trail.append((cx, cy))
            if len(trail) > MAX_TRAIL_LENGTH:
                trail.pop(0)

            # draw box around moving ball
            cv.rectangle(frame, (cx-5, cy-5), (cx+5, cy+5), (0,255,0), 3)

        # draw trail (old → red, new → green)
        for idx, (tx, ty) in enumerate(trail):
            alpha = idx / len(trail)
            r = int(255 * (1 - alpha))
            b = int(255 * (1 - alpha))
            g = int(255 * alpha)
            color = (b, g, 0)

            cv.circle(frame, (tx, ty), round(idx/6)+1, color, -1)

        # save to database/storage
        # TODO: implement S3 stuff

    # release resources
    cap.release()
    out.release()
    cv.destroyAllWindows()

    # delete temporary videos
    os.remove(input_path)
    os.remove(output_path)

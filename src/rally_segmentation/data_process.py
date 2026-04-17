# preprocesses video and audio data for contact detection model
# each saved sample is a flattened 15-frame sliding window of features

import librosa
import cv2 as cv
import numpy as np
import mediapipe as mp
import subprocess
import os
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler

os.makedirs("src/rally_segmentation/processed_data", exist_ok=True)

# models
yolo = YOLO("yolo11n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
ball_detector = YOLO("hugging_face_best.pt")

# constants
FPS = 50
SR = 16000
MIN_ENERGY = 0.05
COOLDOWN_FRAMES = int(0.3 * FPS)
BUFFER_SIZE = 15


def get_audio_at_frame(audio, frame_idx, fps=FPS, sr=SR):
    time = frame_idx / fps
    sample_idx = int(time * sr)
    window = int(0.05 * sr)  # 50 ms
    start = max(0, sample_idx - window // 2)
    end = min(len(audio), sample_idx + window // 2)
    return audio[start:end]


def angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_theta = np.dot(ba, bc) / denom
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


videos = sorted([
    f for f in os.listdir("src/rally_segmentation/raw_data")
    if not f.startswith(".")
])

feature_names = [
    "audio_rms", "audio_zcr", "audio_centroid",
    "wrist_x", "wrist_y",
    "elbow_x", "elbow_y",
    "elbow_angle",
    "wrist_speed", "elbow_speed",
    "ball_x", "ball_y",
    "ball_vx", "ball_vy",
    "ball_speed", "ball_accel",
    "wrist_ball_dist", "elbow_ball_dist"
]

for video in videos:

    X_curr, y_curr = [], []

    last_hit_frame = -COOLDOWN_FRAMES

    # convert audio to wav
    wav_path = f"src/rally_segmentation/sound_data/{video.split('.')[0]}.wav"

    subprocess.run([
        "ffmpeg",
        "-i", f"src/rally_segmentation/raw_data/{video}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SR),
        "-ac", "1",
        "-y",
        wav_path
    ], check=True)

    # load audio
    audio, _ = librosa.load(wav_path, sr=SR)

    # read video
    cap = cv.VideoCapture(f"src/rally_segmentation/raw_data/{video}")
    if not cap.isOpened():
        continue

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    # previous state for temporal features
    prev_ball_center = None
    prev_ball_vel = np.array([0.0, 0.0])
    prev_wrist = None
    prev_elbow = None

    # sliding window buffer: holds up to BUFFER_SIZE feature dicts.
    # each slot is either a real features dict or None (zero-padded later).
    frame_buffer = deque(maxlen=BUFFER_SIZE)

    def make_sequence(buf, names):
        rows = []

        for slot in buf:
            if slot is None:
                rows.append(np.zeros(len(names)))
            else:
                rows.append(np.array([slot.get(name, 0.0) for name in names]))

        while len(rows) < BUFFER_SIZE:
            rows.insert(0, np.zeros(len(names)))

        return np.array(rows)

    with tqdm(total=total_frames, desc=f"{video}", leave=False) as pbar:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            features = dict.fromkeys(feature_names, 0.0)

            current_audio = get_audio_at_frame(audio, frame_index)

            if len(current_audio) == 0:
                frame_index += 1
                pbar.update(1)
                print("skipped audio too small")
                frame_buffer.append(None)   # keep timeline consistent
                continue

            rms = np.sqrt(np.mean(current_audio ** 2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(current_audio))
            spec_centroid = np.mean(
                librosa.feature.spectral_centroid(y=current_audio, sr=SR)
            )

            features["audio_rms"] = rms
            features["audio_zcr"] = zcr
            features["audio_centroid"] = spec_centroid

            hit = False
            if rms >= MIN_ENERGY and frame_index - last_hit_frame > COOLDOWN_FRAMES:
                hit = True
                last_hit_frame = frame_index

            candidates = yolo.predict(
                source=frame,
                conf=0.3,
                stream=False,
                verbose=False,
                classes=[0]
            )

            boxes = candidates[0].boxes

            best_box = None
            largest_area = 0

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    best_box = box

            if best_box is None:
                frame_index += 1
                pbar.update(1)
                print("skipped box too small")
                frame_buffer.append(None)
                continue

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = int(0.4 * box_w)
            pad_h = int(0.4 * box_h)
            x1 -= pad_w
            y1 -= pad_h
            x2 += pad_w
            y2 += pad_h

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_person = frame[y1:y2, x1:x2]

            if cropped_person.size == 0:
                frame_index += 1
                pbar.update(1)
                print("skipped cropped person too small")
                frame_buffer.append(None)
                continue

            stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB)
            results = pose.process(stroke)

            if not results.pose_landmarks:
                frame_index += 1
                pbar.update(1)
                print("skipped pose not there")
                frame_buffer.append(None)
                continue

            landmarks = results.pose_landmarks.landmark

            def get_landmark(idx):
                px = landmarks[idx].x * (x2 - x1) + x1
                py = landmarks[idx].y * (y2 - y1) + y1
                return np.array([px, py])

            shoulder = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            elbow    = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            wrist    = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)

            elbow_angle = angle(shoulder, elbow, wrist)

            wrist_speed = 0.0
            elbow_speed = 0.0
            if prev_wrist is not None:
                wrist_speed = np.linalg.norm(wrist - prev_wrist)
            if prev_elbow is not None:
                elbow_speed = np.linalg.norm(elbow - prev_elbow)

            features["wrist_x"]      = wrist[0]
            features["wrist_y"]      = wrist[1]
            features["elbow_x"]      = elbow[0]
            features["elbow_y"]      = elbow[1]
            features["elbow_angle"]  = elbow_angle
            features["wrist_speed"]  = wrist_speed
            features["elbow_speed"]  = elbow_speed

            ball_locations = ball_detector.predict(
                frame,
                stream=False,
                conf=0.3,
                verbose=False,
            )

            best_conf     = 0
            best_ball_box = None

            for box in ball_locations[0].boxes:
                conf = float(box.conf.item())
                if conf > best_conf:
                    best_ball_box = box
                    best_conf     = conf

            if best_ball_box is not None:
                bx1, by1, bx2, by2 = map(int, best_ball_box.xyxy[0])
                ball_center = np.array([
                    (bx1 + bx2) / 2,
                    (by1 + by2) / 2
                ])

                ball_vel   = np.array([0.0, 0.0])
                ball_accel = np.array([0.0, 0.0])

                if prev_ball_center is not None:
                    ball_vel   = ball_center - prev_ball_center
                    ball_accel = ball_vel - prev_ball_vel

                ball_speed     = np.linalg.norm(ball_vel)
                ball_accel_mag = np.linalg.norm(ball_accel)

                wrist_ball_dist = np.linalg.norm(ball_center - wrist)
                elbow_ball_dist = np.linalg.norm(ball_center - elbow)

                features["ball_x"]           = ball_center[0]
                features["ball_y"]           = ball_center[1]
                features["ball_vx"]          = ball_vel[0]
                features["ball_vy"]          = ball_vel[1]
                features["ball_speed"]       = ball_speed
                features["ball_accel"]       = ball_accel_mag
                features["wrist_ball_dist"]  = wrist_ball_dist
                features["elbow_ball_dist"]  = elbow_ball_dist

                prev_ball_center = ball_center
                prev_ball_vel    = ball_vel

            # push this frame's features into the rolling buffer
            frame_buffer.append(features)
            
            # sampling decision (same logic as before, based on current frame)
            keep_frame = hit

            if not hit:
                wrist_ball_dist = features.get("wrist_ball_dist", 9999)
                elbow_ball_dist = features.get("elbow_ball_dist", 9999)
                wrist_speed_v   = features.get("wrist_speed", 0)
                audio_rms       = features.get("audio_rms", 0)

                hard_negative = (
                    wrist_ball_dist < 120 or
                    elbow_ball_dist < 150 or
                    wrist_speed_v   > 10  or
                    audio_rms       > 0.03
                )

                keep_frame = random.random() < (0.35 if hard_negative else 0.05)

            if keep_frame and feature_names is not None:
                # flatten the full buffer (zero-pads any None/missing slots)
                row = make_sequence(frame_buffer, feature_names)
                X_curr.append(row)
                y_curr.append(int(hit))

            prev_wrist = wrist
            prev_elbow = elbow

            frame_index += 1
            pbar.update(1)

        cap.release()

        if len(X_curr) > 0:
            X_array = np.array(X_curr) # shape: (n_samples, BUFFER_SIZE * n_features)
            y_array = np.array(y_curr)

            print(X_array.shape)
            print(y_array.shape)

            base_name = video.split(".")[0]

            n_samples, seq_len, n_features = X_array.shape

            scaler = StandardScaler()
            X_reshaped = X_array.reshape(-1, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, seq_len, n_features)

            np.save(
                f"src/rally_segmentation/processed_data/{base_name}_X.npy",
                X_array
            )
            np.save(
                f"src/rally_segmentation/processed_data/{base_name}_y.npy",
                y_array
            )
# uses yolo to automatically label sequences based on rapid ball direction changes
# stores only coordinates and labels, no raw frames to save memory

import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_DIR = "src/shot_scoring/ball_speed/videos"
OUTPUT_DIR = "src/shot_scoring/ball_speed/labels"

SEQ_LEN = 21              # must be odd (10 back, center, 10 forward)
MIN_START_FRAME = 20      # skip first frames
BALL_CONF = 0.25
MODEL_PATH = "hugging_face_best.pt"
DIRECTION_THRESHOLD = 15   # pixels/frame change to count as "contact"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_videos(video_dir):
    return [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.endswith((".mp4", ".mov", ".avi"))
    ]

def detect_ball(model, frame):
    """return first ball center detected or None"""
    results = model.predict(source=frame, conf=BALL_CONF, save=False, verbose=True)
    boxes = results[0].boxes
    if boxes:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return (cx, cy)
    return None

def main():
    model = YOLO(MODEL_PATH)
    videos = load_videos(VIDEO_DIR)

    for video_path in videos:
        print(f"\nprocessing: {video_path}")
        cap = cv2.VideoCapture(video_path)

        coords_buffer = []       # sliding window of coordinates
        velocities_buffer = []   # sliding window of velocities
        labels_buffer = []       # center-frame contact labels

        frame_idx = 0
        skipped_no_ball = 0

        X_train = []
        y_train = []

        prev_coord = None
        prev_velocity = (0, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx < MIN_START_FRAME:
                continue

            coord = detect_ball(model, frame)
            if coord is None:
                skipped_no_ball += 1
                coord = None
            # calculate velocity
            if prev_coord is not None and coord is not None:
                dx = coord[0] - prev_coord[0]
                dy = coord[1] - prev_coord[1]
                velocity = (dx, dy)
            else:
                velocity = (0, 0)

            # detect rapid direction change
            delta_dx = abs(velocity[0] - prev_velocity[0])
            delta_dy = abs(velocity[1] - prev_velocity[1])
            contact_label = int(delta_dx > DIRECTION_THRESHOLD or delta_dy > DIRECTION_THRESHOLD)

            coords_buffer.append(coord if coord is not None else (-1, -1))
            velocities_buffer.append(velocity)
            labels_buffer.append(contact_label)

            prev_coord = coord
            prev_velocity = velocity

            # slide window when buffer is full
            if len(coords_buffer) >= SEQ_LEN:
                # save sequence
                X_train.append(np.array(coords_buffer[-SEQ_LEN:], dtype=np.int16))
                y_train.append(np.array(labels_buffer[-SEQ_LEN:], dtype=np.int8))

                # pop oldest to slide window
                coords_buffer.pop(0)
                velocities_buffer.pop(0)
                labels_buffer.pop(0)

        cap.release()
        base = os.path.splitext(os.path.basename(video_path))[0]

        np.save(os.path.join(OUTPUT_DIR, f"{base}_X.npy"), np.array(X_train, dtype=np.int16))
        np.save(os.path.join(OUTPUT_DIR, f"{base}_y.npy"), np.array(y_train, dtype=np.int8))

        print(f"saved {len(X_train)} sequences")
        print(f"skipped frames (no ball): {skipped_no_ball}")

        del X_train, y_train, coords_buffer, velocities_buffer, labels_buffer

if __name__ == "__main__":
    main()

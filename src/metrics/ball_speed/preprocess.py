# uses yolo to automatically label sequences based on rapid ball direction changes
# stores only coordinates + velocity + window-level labels

import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_DIR = "src/shot_scoring/ball_speed/videos"
OUTPUT_DIR = "src/shot_scoring/ball_speed/labels"

SEQ_LEN = 21                  # must be odd
MIN_START_FRAME = 20
BALL_CONF = 0.25
MODEL_PATH = "hugging_face_best.pt"
DIRECTION_THRESHOLD = 15      # pixels/frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_ball(model, frame):
    results = model.predict(frame, conf=BALL_CONF, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def main():
    model = YOLO(MODEL_PATH)

    for video_name in sorted(os.listdir(VIDEO_DIR)):
        if not video_name.endswith(".mp4"):
            continue

        path = os.path.join(VIDEO_DIR, video_name)
        print(f"\nprocessing: {path}")

        cap = cv2.VideoCapture(path)

        coords = []
        velocities = []
        contact_flags = []

        X, y = [], []

        prev_coord = None
        prev_velocity = (0, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx < MIN_START_FRAME:
                continue

            coord = detect_ball(model, frame)

            if coord is not None and prev_coord is not None:
                dx = coord[0] - prev_coord[0]
                dy = coord[1] - prev_coord[1]
                velocity = (dx, dy)
            else:
                velocity = (0, 0)

            # detect sharp direction change
            delta_dx = abs(velocity[0] - prev_velocity[0])
            delta_dy = abs(velocity[1] - prev_velocity[1])

            contact = int(
                delta_dx > DIRECTION_THRESHOLD or
                delta_dy > DIRECTION_THRESHOLD
            )

            if coord is None:
                coords.append((-1, -1))
                velocities.append((0, 0))
            else:
                coords.append(coord)
                velocities.append(velocity)

            contact_flags.append(contact)

            prev_coord = coord
            prev_velocity = velocity

            if len(coords) >= SEQ_LEN:
                window_coords = coords[-SEQ_LEN:]
                window_vels = velocities[-SEQ_LEN:]
                window_labels = contact_flags[-SEQ_LEN:]

                # combine (x, y, dx, dy)
                seq = np.hstack([
                    np.array(window_coords, dtype=np.float32),
                    np.array(window_vels, dtype=np.float32),
                ])

                X.append(seq)

                # window-level label
                y.append(int(any(window_labels)))

                coords.pop(0)
                velocities.pop(0)
                contact_flags.pop(0)

        cap.release()

        base = os.path.splitext(video_name)[0]
        np.save(os.path.join(OUTPUT_DIR, f"{base}_X.npy"), np.array(X))
        np.save(os.path.join(OUTPUT_DIR, f"{base}_y.npy"), np.array(y))

        print(f"saved {len(X)} sequences")

if __name__ == "__main__":
    main()

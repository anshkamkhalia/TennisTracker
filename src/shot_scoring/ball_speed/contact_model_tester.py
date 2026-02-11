# tests the trained contact detection model (2-pass, window-level)

import cv2 as cv
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from model import ContactDetector, Attention
from trainer import binary_focal_loss

model = tf.keras.models.load_model(
    "serialized_models/contact_detector.keras",
    custom_objects={
        "ContactDetector": ContactDetector,
        "Attention": Attention,
        "binary_focal_loss": binary_focal_loss,
    },
)

ball_model = YOLO("hugging_face_best.pt")

VIDEO_PATH = "src/shot_scoring/ball_speed/videos/videoplayback5.mp4"
SEQ_LEN = 21
CENTER_IDX = SEQ_LEN // 2
threshold = 0.6

def detect_ball_center(frame, conf=0.25):
    res = ball_model.predict(frame, conf=conf, verbose=False)
    boxes = res[0].boxes
    if len(boxes) == 0:
        return None
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# pass 1: detect contact windows
cap = cv.VideoCapture(VIDEO_PATH)
coords = []
vels = []
prev = None
frame_idx = 0

# output video
out = cv.VideoWriter(
    f"outputs/run_contact_tester5.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    60,
    (1280, 720)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 720))
    frame_idx += 1

    # detect ball
    c = detect_ball_center(frame)

    if c is not None and prev is not None:
        v = (c[0] - prev[0], c[1] - prev[1])
    else:
        v = (0, 0)

    if c is None:
        coords.append((-1, -1))
        vels.append((0, 0))
    else:
        coords.append(c)
        vels.append(v)

    prev = c

    if len(coords) > SEQ_LEN:
        coords.pop(0)
        vels.pop(0)

    if len(coords) < SEQ_LEN:
        out.write(frame)
        continue

    # prepare input for model
    seq = np.hstack([
        np.array(coords, dtype=np.float32),
        np.array(vels, dtype=np.float32),
    ])
    seq[:, :2] /= np.array([frame.shape[1], frame.shape[0]])

    # predict contact
    p = model.predict(seq[None], verbose=0)[0][0]

    # overlay CONTACT if above threshold
    if p > threshold:
        cv.putText(
            frame,
            "CONTACT",
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 255),
            4,
            cv.LINE_AA
        )

    # draw detected ball for visualization
    if c is not None:
        cv.circle(frame, c, 7, (0, 255, 0), -1)

    out.write(frame)
    print(frame_idx, "CONTACT" if p > threshold else "")

cap.release()
out.release()
print("Processing complete.")

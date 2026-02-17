# tennis ball 3d reconstruction exporter
# detects ball, reconstructs z with sliding window,
# saves trajectory to csv for unity

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import tensorflow as tf
from model import Reconstructor, PositionalEncoding

# config
video_path = "api/videoplayback8.mp4"
model_path = "serialized_models/reconstructor.keras"
# output_path = "unity_visualizer/Assets/StreamingAssets/trajectory.csv"
output_path = "trajectory.csv"

window_size = 60
smoothing_alpha = 0.4
scale_xy = 0.01  # scale pixels down for unity
scale_z = 1.0    # keep z as meters

# load models
ball_tracker = YOLO("hugging_face_best.pt")

model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "Reconstructor": Reconstructor,
        "PositionalEncoding": PositionalEncoding
    }
)

# video setup
cap = cv.VideoCapture(video_path)

# tracking state
last_ball = None
smoothed_ball = None

# reconstruction state
recon_buffer = []
trajectory_points = []

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 720))

    # detect ball
    results = ball_tracker.predict(frame, conf=0.25, verbose=False)[0]

    detected_centers = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detected_centers.append((cx, cy))

    moving_ball = None

    if detected_centers:
        if last_ball is None:
            moving_ball = detected_centers[0]
        else:
            moving_ball = min(
                detected_centers,
                key=lambda p: np.hypot(
                    p[0] - last_ball[0],
                    p[1] - last_ball[1]
                )
            )

    if moving_ball is None:
        continue

    cx, cy = moving_ball

    # smooth position
    if smoothed_ball is None:
        smoothed_ball = (cx, cy)
    else:
        sx, sy = smoothed_ball
        smoothed_ball = (
            int(smoothing_alpha * cx + (1 - smoothing_alpha) * sx),
            int(smoothing_alpha * cy + (1 - smoothing_alpha) * sy)
        )

    last_ball = smoothed_ball
    cx, cy = smoothed_ball

    # add to sliding window buffer
    recon_buffer.append([cx, cy])

    if len(recon_buffer) >= window_size:
        input_seq = np.array(
            recon_buffer[-window_size:],
            dtype=np.float32
        )
        input_seq = np.expand_dims(input_seq, axis=0)

        z_pred = model.predict(input_seq, verbose=0)[0]
        z_latest = float(np.squeeze(z_pred[-1]))

        trajectory_points.append([cx, cy, z_latest])

cap.release()

# save trajectory for unity
if len(trajectory_points) == 0:
    print("no trajectory reconstructed")
    exit()

trajectory = np.asarray(trajectory_points, dtype=np.float32)

# scale for unity
trajectory[:, 0] *= scale_xy
trajectory[:, 1] *= scale_xy
trajectory[:, 2] *= scale_z

np.savetxt(
    output_path,
    trajectory,
    delimiter=",",
    header="x,y,z",
    comments=""
)

print("trajectory saved successfully")
print("shape:", trajectory.shape)

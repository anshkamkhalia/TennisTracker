# test wrist velocity
# only for court level videos

import numpy as np
import cv2 as cv
from ultralytics import YOLO
import mediapipe as mp

# test video path
test_video = "data/court-level-videos/videoplayback5.mp4"

# load yolo model
yolo = YOLO("yolo11n.pt")

# open video
cap = cv.VideoCapture(test_video)
if not cap.isOpened():
    print("video capture failed to open")

fps = cap.get(cv.CAP_PROP_FPS)
dt = 1.0 / fps if fps > 0 else 1/60

PLAYER_HEIGHT_METERS = 1.73  # 5'8"

# initialize videowriter
out = cv.VideoWriter(
    "outputs/wrist_velocity_test5.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (1280, 720)
)

r_wrist_buffer = []
l_wrist_buffer = []
WRIST_BUFFER_MAXLEN = 60

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

alpha = 0.75
r_vel_mps_display = 0.0
l_vel_mps_display = 0.0
r_vel_mph_display = 0.0
l_vel_mph_display = 0.0

def smooth_velocity(buffer, dt, window=5):
    if len(buffer) < window:
        return 0.0
    
    velocities = []
    for i in range(-window, -1):
        p1 = buffer[i]
        p2 = buffer[i + 1]
        velocities.append(np.linalg.norm(p2 - p1) / dt)
    
    return np.mean(velocities)

# prevent insane jumps
def safe_append(buffer, new_point, max_jump=100):
    if len(buffer) > 0:
        if np.linalg.norm(new_point - buffer[-1]) > max_jump:
            return
    buffer.append(new_point)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 2 == 0:
        continue

    frame = cv.resize(frame, (1280, 720))

    r_boxes = yolo.predict(frame, verbose=False, stream=False, classes=[0])

    if r_boxes is None or len(r_boxes[0].boxes) == 0:
        out.write(frame)
        continue

    largest_box = None
    largest_area = 0

    for box in r_boxes[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h

        if box_area > largest_area:
            largest_box = (x1, y1, x2, y2)
            largest_area = box_area

    if largest_box is None:
        out.write(frame)
        continue

    x1, y1, x2, y2 = largest_box
    cropped = frame[y1:y2, x1:x2]

    # compute scale (meters per pixel)
    pixel_height = y2 - y1
    if pixel_height > 0:
        meters_per_pixel = PLAYER_HEIGHT_METERS / pixel_height
    else:
        meters_per_pixel = 0

    results = pose.process(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark
        h, w, _ = cropped.shape

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
        r_vel_px = smooth_velocity(r_wrist_buffer, dt)
        l_vel_px = smooth_velocity(l_wrist_buffer, dt)
 
        # convert to real-world units
        r_vel_mps = r_vel_px * meters_per_pixel
        l_vel_mps = l_vel_px * meters_per_pixel

        r_vel_mps_display = alpha * r_vel_mps + (1 - alpha) * r_vel_mps_display
        l_vel_mps_display = alpha * l_vel_mps + (1 - alpha) * l_vel_mps_display

        r_vel_mph_display = r_vel_mps_display * 2.237
        l_vel_mph_display = l_vel_mps_display * 2.237

        # draw wrists
        cv.circle(frame, tuple(r_wrist.astype(int)), 6, (0, 0, 255), -1)
        cv.circle(frame, tuple(l_wrist.astype(int)), 6, (255, 0, 0), -1)

        # display velocity
        cv.putText(frame, f"R: {r_vel_mps_display:.2f} m/s ({r_vel_mph_display:.1f} mph)", (20, 40),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.putText(frame, f"L: {l_vel_mps_display:.2f} m/s ({l_vel_mph_display:.1f} mph)", (20, 80),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv.destroyAllWindows()
# tests the court modeling

from ultralytics import YOLO
import cv2 as cv
from tensorflow.keras.saving import load_model
from model import CourtDetector
import numpy as np

i = 7  # video index
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instance
model = YOLO("hugging_face_best.pt")

# load trained cnn for court detection
court_detector = load_model("serialized_models/court_keypoint_detector.keras", custom_objects={
    "CourtDetector": CourtDetector
})

# load video
cap = cv.VideoCapture(video_path)

# output video writer
out = cv.VideoWriter(
    f"outputs/run_court_model{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (1280, 720)
)

videos60fps = [5,6,7]

coordinates = []

# trail storage
trail = []
MAX_TRAIL_LENGTH = 20

frame_index = 0
keypoints = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % 2 == 0 and i in videos60fps:
        continue

    # get frame keypoints
    if keypoints is None or frame_index % 60 == 0:  # predict every ~2 seconds
        input_frame = frame.astype("float32") / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)
        kps_raw = court_detector.predict(input_frame)
        
        # convert to float32 and reshape
        keypoints = np.array(kps_raw, dtype=np.float32).reshape(-1, 14, 2)

        # denormalize ONCE
        h, w = frame.shape[:2]
        keypoints[..., 0] = (keypoints[..., 0] * w).astype(np.int32)
        keypoints[..., 1] = (keypoints[..., 1] * h).astype(np.int32)
        print(keypoints)

    # convert to integer for opencv
    keypoints = keypoints.astype(np.int32)
    print(keypoints)

    # predict on frame
    results = model.predict(
        source=frame,
        conf=0.20,
        save=False,
        verbose=False
    )

    r = results[0]
    boxes = r.boxes

    best_box = None
    best_conf = 0

    # select highest confidence box
    for box in boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    # if a ball was detected
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        coordinates.append([x1, y1, x2, y2])
        
        # draw bounding box
        cv.rectangle(
            frame,
            (x1 + 5, y1 + 5),
            (x2 + 5, y2 + 5),
            (0, 255, 0),
            3
        )

        # compute center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        trail.append((cx, cy))
        if len(trail) > MAX_TRAIL_LENGTH:
            trail.pop(0)

    # draw trail (old → red, new → green)
    for idx, (tx, ty) in enumerate(trail):
        alpha = idx / len(trail)
        g = int(255 * (1 - alpha))
        r = int(255 * alpha)
        color = (0, g, r)

        cv.circle(frame, (tx, ty), round(idx/2), color, -1)

    # draw court keypoints
    for (x, y) in keypoints[0]:  # 0 because batch size = 1
        cv.circle(frame, (x,y), 5, (0, 255, 0), -1)

    # write frame
    out.write(cv.resize(frame, (1280, 720)))

cap.release()
out.release()
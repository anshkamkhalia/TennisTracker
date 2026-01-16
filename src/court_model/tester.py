# tests the court modeling

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import sys

i = sys.argv[2]  # video index
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instance
model = YOLO("hugging_face_best.pt")

# load video
cap = cv.VideoCapture(video_path)

# output video writer
out = cv.VideoWriter(
    f"outputs/run_court_model{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (1280, 720)
)

videos60fps = [5,6,7,8] # videos that are more than 60fps
verbose = False if int(sys.argv[1]) == 0 else True # verbose flag
coordinates = [] # stores ball locations

# trail storage
trail = []
MAX_TRAIL_LENGTH = 50

frame_index = 0
keypoints = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % 2 == 0 and i in videos60fps:
        continue

    # predict on frame
    results = model.predict(
        source=frame,
        conf=0.20,
        save=False,
        verbose=verbose
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
        r = int(255 * (1 - alpha))
        b = int(255 * (1 - alpha))
        g = int(255 * alpha)
        color = (b, g, 0)

        cv.circle(frame, (tx, ty), round(idx/6)+1, color, -1)

    # keypoint detections

    frame = cv.resize(frame, (1280, 720)) # resize frame
    frame = cv.GaussianBlur(frame, (5, 5), 0) # gaussian blur to clear noise

    # convert to hsv <- low saturation = white = court lines
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # create mask
    lower = np.array([0, 0, 100])  # H, S, V
    upper = np.array([180, 55, 255]) 
    mask = cv.inRange(hsv_frame, lower, upper)

    # apply mask
    masked_frame = cv.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    edges = cv.Canny(
        masked_frame,
        threshold1=50,
        threshold2=120,
        apertureSize=3
    )

    # write frame
    # out.write(cv.resize(frame))

    # show frame - for testing
    cv.imshow("frame", masked_frame)
    cv.waitKey(1)

cap.release()
out.release()
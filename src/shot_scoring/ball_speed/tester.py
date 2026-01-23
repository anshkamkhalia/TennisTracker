# uses approximate player height to estimate ball speed (across 3-5 frames)
# this file has the latest version of ball tracking

import numpy as np
import cv2 as cv
from ultralytics import YOLO
import sys

player_height = (6.0) * 0.3048 # will be provided by user in app, in feet or meters (meters preferred)

i = 5  # video index
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# ball tracking model
tracker = YOLO("hugging_face_best.pt")

# player detector
detector = YOLO("yolo11n.pt")

# load video
cap = cv.VideoCapture(video_path)

video_fps = int(sys.argv[1])

# output video writer
out = cv.VideoWriter(
    f"outputs/run_ball_speed{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (1280, 720)
)

coordinates = [] # for ball

# trail storage
trail = []
MAX_TRAIL_LENGTH = 40

# speed estimation
n_frames = 15
heights = []
meters_per_pixel = None
meters_per_pixel_calculated = False
speed_buffer = 3
display_mph = None
display_timer = 0 
DISPLAY_DURATION = int(video_fps * 1.2)  # ~1.2 seconds
prev_speed_mps = 0
sliding_window_size = 10
max_px_jump = 100

frame_index = 0

prev_ball_centers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    # get player box for first 10-15 frames
    if frame_index <= n_frames:
        # run inference on frame
        player_boxes = detector.predict(
            source=frame,
            conf=0.20,
            save=False,
            verbose=False,
        )

        # extract iterable boxes
        pb = player_boxes[0]
        boxes = pb.boxes

        best_player_box = None
        best_player_conf = 0
        
        # get box with highest confidence
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_player_conf:
                best_player_conf = conf
                best_player_box = box

        # now that we have the best box, we can get the height in pixels
        x1, y1, x2, y2 = best_player_box.xyxy[0] # extract coords
        pixel_height = y2-y1 # find height
        heights.append(pixel_height)

    # calculate meters per pixel    
    if len(heights) >= n_frames and not meters_per_pixel_calculated:
        meters_per_pixel = player_height / np.mean(heights) # uses mean for smoothing
        meters_per_pixel = meters_per_pixel * 2.1 # k factor for horizontal scaling
        meters_per_pixel_calculated = True

    # predict on frame
    results = tracker.predict(
        source=frame,
        conf=0.20,
        save=False,
        verbose=False,
    )

    r = results[0]
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

    # speed estimation
    if len(coordinates) >= speed_buffer and meters_per_pixel_calculated:

        speeds = []
        for i in range(len(coordinates) - 1):
            dt = (coordinates[i+1][0] - coordinates[i][0]) / video_fps
            if dt <= 0 or dt > 2.0:  # skip impossible jumps
                print("skipped")
                continue
            dx = coordinates[i+1][1] - coordinates[i][1]
            dy = coordinates[i+1][2] - coordinates[i][2]
            dx = np.clip(dx, -max_px_jump, max_px_jump)
            dy = np.clip(dy, -max_px_jump, max_px_jump)
            speeds.append(np.sqrt(dx**2 + dy**2) * meters_per_pixel / dt)

        speed_mps = np.median(speeds)  # smooth over sliding window

        if len(speeds) == 0:
            speed_mps = prev_speed_mps
        else:
            speed_mps = np.median(speeds)

        # save as previous
        prev_speed_mps = speed_mps
        global mph
        mph = speed_mps * 2.23694 # convert to miles per hour
        print(mph)
        mph = 145 if mph >= 145 else mph
        if 'speed_history' not in globals():
            speed_history = []

        speed_history.append(speed_mps)
        if len(speed_history) > 3:
            speed_history.pop(0)

        smooth_speed = np.median(speed_history)
        display_mph = min(smooth_speed * 2.23694, 145)
        display_timer = DISPLAY_DURATION

        # slide window
        if len(coordinates) >= sliding_window_size:
            coordinates = coordinates[sliding_window_size:]
        
    if display_timer > 0 and display_mph is not None:
        text = f"estimated ball speed: {int(display_mph)} mph"

        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 1.3
        thickness = 3

        (tw, th), _ = cv.getTextSize(text, font, scale, thickness)

        x, y = 40, 60  # top-left corner
        padding = 10

        # background box
        cv.rectangle(
            frame,
            (x - padding, y - th - padding),
            (x + tw + padding, y + padding),
            (0, 0, 0),
            -1
        )

        # text
        cv.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv.LINE_AA
        )

        display_timer -= 1
    print("\n\n")
    # write frame
    out.write(cv.resize(frame, (1280, 720)))

out.release()
cap.release()
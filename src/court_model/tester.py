# tests the court modeling
# has latest version of ball tracking

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import sys

i = sys.argv[1]  # video index
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instances
ball_tracker = YOLO("hugging_face_best.pt")
court_detector = YOLO("src/court_model/best.pt")

# load video
cap = cv.VideoCapture(video_path)

# output video writer
out = cv.VideoWriter(
    f"outputs/run_court_model{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    50,
    (1280, 720)
)

coordinates = [] # stores ball locations

# trail storage
trail = []
MAX_TRAIL_LENGTH = 40

frame_index = 0
prev_ball = None
vels = []

# load video
cap = cv.VideoCapture(video_path)

ball_history = [] # stores (cx, cy)
last_ball_px = None
court_box = None
k = 1 # for court shrinkage
court_padding = 30 # amount of pixels to increase court size by

# mini court
mini_w, mini_h = 200, 400           # mini court size
margin = 20                         # top-right corner padding
mx, my = 1280 - mini_w - margin, margin
mw, mh = mini_w, mini_h

# create overlay with fully black opaque background
mini_court_overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
cv.rectangle(mini_court_overlay, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)  # BLACK background

# neon green lines
line_color = (57, 255, 20)
thick = 2

# scale ratios based on real tennis court dimensions
singles_ratio = 27 / 36
net_y = my + mh // 2

# bigger service boxes
service_offset = int(mh * 0.25)  # ~1/4 from baseline to net

# LEFT AND RIGHT SIDELINES
cv.line(mini_court_overlay, (mx, my), (mx, my + mh), line_color, thick)           # left doubles
cv.line(mini_court_overlay, (mx + mw, my), (mx + mw, my + mh), line_color, thick) # right doubles

# LEFT AND RIGHT SINGLES LINES
singles_offset = int(mw * (1 - singles_ratio) / 2)
cv.line(mini_court_overlay, (mx + singles_offset, my), (mx + singles_offset, my + mh), line_color, thick)
cv.line(mini_court_overlay, (mx + mw - singles_offset, my), (mx + mw - singles_offset, my + mh), line_color, thick)

# BASELINES
cv.line(mini_court_overlay, (mx, my), (mx + mw, my), line_color, thick)           # far baseline
cv.line(mini_court_overlay, (mx, my + mh), (mx + mw, my + mh), line_color, thick) # near baseline

# NET
cv.line(mini_court_overlay, (mx, net_y), (mx + mw, net_y), line_color, thick)

# SERVICE LINES (parallel to net) – bigger boxes
cv.line(mini_court_overlay, (mx + singles_offset, my + service_offset),
        (mx + mw - singles_offset, my + service_offset), line_color, thick)
cv.line(mini_court_overlay, (mx + singles_offset, my + mh - service_offset),
        (mx + mw - singles_offset, my + mh - service_offset), line_color, thick)

# CENTER SERVICE LINES (perpendicular to net)
center_x = mx + mw // 2
cv.line(mini_court_overlay, (center_x, my + service_offset),
        (center_x, my + mh - service_offset), line_color, thick)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 720))
    frame_index += 1

    # detect ball
    res = ball_tracker.predict(
        source=frame,
        conf=0.25,
        save=False,
        verbose=False
    )[0]

    detected_centers = []

    # extract center of each detected ball box
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detected_centers.append((cx, cy))
    
    # choose detection closest to previous ball position
    moving_ball = None
    if detected_centers:
        if last_ball_px is None:
            moving_ball = detected_centers[0]
        else:
            moving_ball = min(
                detected_centers,
                key=lambda p: np.hypot(p[0] - last_ball_px[0], p[1] - last_ball_px[1])
            )

    if moving_ball:
        cx, cy = moving_ball
        last_ball_px = (cx, cy)

        trail.append((cx, cy))
        if len(trail) > MAX_TRAIL_LENGTH:
            trail.pop(0)

    # draw fading trail
    for j, (tx, ty) in enumerate(trail):
        alpha = j / max(1, len(trail))
        cv.circle(
            frame,
            (tx, ty),
            max(1, j // 6),
            (int(255 * (1 - alpha)), int(255 * alpha), 0),
            -1
        )

    # get court predictions
    court_preds = court_detector.predict(
        frame,
        conf=0.25,
        verbose=False,
        stream=False
    )[0]

    if court_box is None or frame_index % 300 == 0: # get court locations at the beginning or every minute
        # choose largest box
        best_area = 0
        for box in court_preds.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # get box coords

            # calculate area
            length = y2-y1
            width = x2-x1
            area = length * width

            if area >= best_area:
                court_box = box
                best_area = area

        try:
            x1, y1, x2, y2 = map(int, court_box.xyxy[0])
            
        except Exception as e:
            print(e)

    else:
        pass

    # now that we have the court box we can scale the farther baseline
    if court_box is not None:
        x1, y1, x2, y2 = map(int, court_box.xyxy[0])

        # (x1, y2), (x2, y2) <- closer baseline (near camera)
        # (x1, y1), (x2, y1) <- farther baseline (away from camera, the one we want to change)

        # reduce far baseline width
        height = y2-y1
        width = x2-x1
        shrink_ratio = k * (height/width)
        far_width = width * (1 - shrink_ratio)
        dx = (width - far_width) / 2

        top_left = (x1 + dx - court_padding, y1 - court_padding)
        top_right = (x2 - dx + court_padding, y1 - court_padding)

        bottom_left  = (x1-court_padding, y2+court_padding)
        bottom_right = (x2+court_padding, y2+court_padding)

        # draw

        # create an array of points in the correct order
        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        # cv.polylines expects shape (num_points, 1, 2)
        pts = pts.reshape((-1, 1, 2))

        # draw the trapezoid
        cv.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    mini_w, mini_h = 200, 400           # size of mini court
    margin = 20                         # padding from top-right corner
    mini_top_left = (1280 - mini_w - margin, margin)
    mini_bottom_right = (1280 - margin, margin + mini_h)

    # copy frame
    frame_with_mini = frame.copy()

    # overlay mini court permanently
    alpha = 1.0  # fully opaque black background
    mask = mini_court_overlay > 0
    frame_with_mini[mask] = mini_court_overlay[mask]

   # overlay the mini court permanently
    frame_with_mini = frame.copy()

    # paste mini court overlay directly (black background + neon lines)
    mx, my = mini_top_left
    frame_with_mini[my:my+mh, mx:mx+mw] = mini_court_overlay[my:my+mh, mx:mx+mw]

    # draw yellow ball if detected
    if moving_ball and court_box is not None:
        cx1, cy1, cx2, cy2 = map(int, court_box.xyxy[0])
        ball_x_ratio = (cx - cx1) / (cx2 - cx1)
        ball_y_ratio = (cy - cy1) / (cy2 - cy1)
        
        mini_ball_x = int(mx + ball_x_ratio * mw)
        mini_ball_y = int(my + ball_y_ratio * mh)

        cv.circle(frame_with_mini, (mini_ball_x, mini_ball_y), 5, (0, 255, 255), -1)

    out.write(frame_with_mini)
        
cap.release()
out.release()
cv.destroyAllWindows()
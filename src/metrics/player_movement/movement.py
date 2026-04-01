from ultralytics import YOLO
import cv2 as cv
import numpy as np
import sys

i = sys.argv[1]
video_path = f"api/videoplayback{i}.mp4"

ball_tracker = YOLO("hugging_face_best.pt")
court_detector = YOLO("src/court_model/best.pt")
yolo = YOLO("yolo11n.pt")

cap = cv.VideoCapture(video_path)

out = cv.VideoWriter(
    f"outputs/run_player_movement{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    50,
    (1280, 720)
)

frame_index = 0
last_ball_px = None
court_box = None

# mini court size (used only for heatmap space)
mw, mh = 200, 400

player_heatmap = np.zeros((mh, mw), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 720))
    frame_index += 1

    court_preds = court_detector.predict(frame, conf=0.25, verbose=False)[0]

    if court_box is None or frame_index % 300 == 0:
        best_area = 0
        for box in court_preds.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area >= best_area:
                court_box = box
                best_area = area

    player_results = yolo.predict(frame, conf=0.3, verbose=False, classes=[0])[0]

    largest_player = None
    best_area = 0

    for box in player_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            largest_player = (x1, y1, x2, y2)

    if court_box is not None and largest_player is not None:
        cx1, cy1, cx2, cy2 = map(int, court_box.xyxy[0])
        x1, y1, x2, y2 = largest_player

        # feet position
        px = int((x1 + x2) / 2)
        py = y2

        x_ratio = (px - cx1) / (cx2 - cx1)
        y_ratio = (py - cy1) / (cy2 - cy1)

        hx = int(x_ratio * mw)
        hy = int(y_ratio * mh)

        if 0 <= hx < mw and 0 <= hy < mh:
            player_heatmap[hy, hx] += 1

        # debug box
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # decay
    player_heatmap *= 0.995

    out.write(frame)

heat_blur = cv.GaussianBlur(player_heatmap, (31, 31), 0)

if heat_blur.max() > 0:
    heat_norm = heat_blur / heat_blur.max()
else:
    heat_norm = heat_blur

heat_uint8 = np.uint8(255 * heat_norm)
heat_color = cv.applyColorMap(heat_uint8, cv.COLORMAP_JET)

cv.imwrite(f"outputs/player_heatmap{i}.png", heat_color)

cap.release()
out.release()
cv.destroyAllWindows()
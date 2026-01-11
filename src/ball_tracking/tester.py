# tests the fine-tuned yolo model on actual videos

from ultralytics import YOLO
import cv2 as cv

i = 5 # video" index to predict on
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instance
model = YOLO("runs/detect/train/weights/best.pt")

# load video
cap = cv.VideoCapture(video_path)

# output video writer 
out = cv.VideoWriter(
    f"outputs/run_ball_tracking{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (1280, 720)
)

# current frame
frame_index = 0

# main loop
while True:
    ret, frame = cap.read()
    if not ret: break # break if end of video

    # 60fps -> 30 fps
    frame_index += 1
    if frame_index % 2 == 0 and i == 5:
        continue

    # predict on current video
    results = model.predict(
        source=frame,        # input frame
        conf=0.20,           # confidence threshold
        save=False,          # dont save output
    )

    # extract boxes
    r = results[0]
    boxes = r.boxes

    # draw on frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # get coordinates
        color = (0,255,0) # box color
        cv.rectangle(frame, (x1+5,y1+5), (x2+5, y2+5), color, 3) # draw rectangle, add 5 to make slightly larger
    
    # write frame to video
    out.write(cv.resize(frame, (1280, 720)))
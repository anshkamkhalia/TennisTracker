import cv2 as cv
import os
from datetime import datetime
import pyttsx3
import time

# set label
label = "serve_overhead"

# output directories
base_dir = "data/shot-classification"
dirs = {"forehand": os.path.join(base_dir, "forehand"),
        "backhand": os.path.join(base_dir, "backhand"),
        "slice_volley": os.path.join(base_dir, "slice_volley"),
        "serve_overhead": os.path.join(base_dir, "serve_overhead"),
    }               
out_dir = dirs.get(label)
if out_dir is None:
    raise ValueError("bad label")
os.makedirs(out_dir, exist_ok=True)

# open camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera fail")

# read one frame to get size
ret, frame = cap.read()
if not ret:
    raise RuntimeError("no frame")
h, w = frame.shape[:2]

# fps & delay
fps = 30
delay = 26  # ms per frame

# initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

video_counter = 1  # count of recorded videos

print("press space to start recording a clip, press 'q' to quit")

while True:
    # idle loop
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("camera fail")

    cv.putText(frame, f"idle: press space to start (video {video_counter})", (50, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("recording...", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord(" "):
        # generate filename
        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(out_dir, filename)

        # setup video writer
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(filepath, fourcc, fps, (w, h))

        # visual prep time (5 seconds)
        prep_start = time.time()
        while time.time() - prep_start < 5:
            ret, frame = cap.read()
            if not ret:
                break
            cv.putText(frame, f"recording starting in {int(5 - (time.time() - prep_start))}", 
                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv.imshow("recording...", frame)
            cv.waitKey(1)

        # announce recording started
        engine.stop()
        engine.say("recording started")
        engine.runAndWait()

        # record loop (3 seconds)
        num_frames = 90
        count = 0
        while count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            cv.putText(frame, f"recording... (video {video_counter})", (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(frame)
            cv.imshow("recording...", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            count += 1

        writer.release()

        # announce recording finished
        engine.stop()
        engine.say("recording finished")
        engine.runAndWait()

        print(f"saved: {filepath}")
        video_counter += 1

    elif key == ord("q"):
        break

# cleanup
cap.release()
cv.destroyAllWindows()
print("all done")

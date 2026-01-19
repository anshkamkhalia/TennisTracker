# uses a pretrained yolo model to create labels for a custom spatiotemporal model

import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import gc
import torch

source_video_path = "src/ball_tracking/videos" # data path
output_path = "src/ball_tracking/videos/labels" # output path (.npy files will be stored here)
videos = os.listdir(source_video_path) # list of all video names

detector = YOLO("hugging_face_best.pt") # loads ball detector (teacher model)

max_jump = 80          # max pixels ball can move between frames
conf_gap_thresh = 0.15 # confidence separation threshold
SEQUENCE_LEN = 8
TARGET_FPS = 25
BATCH_SIZE = 4 # chunking videos to avoid memory

# helper function
# def select_valid_box(boxes, last_center):
#     """returns the best box based on a list of candidates and the previous valid box"""

#     if len(boxes) == 0:
#         return None
    
#     # sort boxes by confidence
#     boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)

#     # if multiple detections
#     if len(boxes) > 1:
#         c0 = float(boxes[0].conf[0])
#         c1 = float(boxes[1].conf[0])
#         if c0-c1 < conf_gap_thresh:
#             return None # ambiguous frame
    
#     best = boxes[0]
#     x1, y1, x2, y2 = map(int, best.xyxy[0].numpy())
#     cx = (x1+x2) // 2
#     cy = y2

#     if last_center is not None:
#         px, py = last_center
#         dist = np.hypot(cx-px, cy-py)
#         if dist > max_jump:
#             return None # motion too large
        
#     return (x1, y1, x2, y2, cx, cy)

# helper function
def select_valid_box(boxes, last_center):
    """returns the best box based on confidence; very permissive for max frames"""

    if len(boxes) == 0:
        return None

    # just take the box with highest confidence
    best = max(boxes, key=lambda b: float(b.conf[0]))
    
    x1, y1, x2, y2 = map(int, best.xyxy[0].cpu().numpy())
    cx = (x1 + x2) // 2
    cy = y2

    return (x1, y1, x2, y2, cx, cy)

def save_and_clear_batch(X_buf, y_buf, batch_idx, video_name):
    """helper function for batch saving to save colabs puny 12gb of sys ram"""
    X = np.asarray(X_buf, dtype=np.uint8)
    y = np.asarray(y_buf, dtype=np.float32)

    np.save(os.path.join(output_path, f"X_train_batch{batch_idx}_{video_name}.npy"), X)
    np.save(os.path.join(output_path, f"y_train_batch{batch_idx}_{video_name}.npy"), y)

    X_buf.clear()
    y_buf.clear()
    gc.collect()

# processing loop
for filename in videos:
    print(filename)
    if filename == "label":
        continue
    else: pass
    cap = cv.VideoCapture(os.path.join(source_video_path, filename)) # load new video
    fps = cap.get(cv.CAP_PROP_FPS) # get fps
    X_train_local, y_train_local = [], [] # initialize to be empty every time a new video is loaded
    video_name = filename.replace(".mp4", "") # for npy naming purposes
    sequence = [] # for lstm timesteps
    stride = max(1, round(fps / TARGET_FPS))

    frame_index = 0
    last_center = None
    batch_idx = 0  # initialize before the while loop

    # video writer to save videos with discarded frames
    # out = cv.VideoWriter(
    #     f"{source_video_path}/edited_videos/edited_{filename}",
    #     cv.VideoWriter_fourcc(*"mp4v"),
    #     30,
    #     (1280, 720)
    # )

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_index += 1
        if frame_index % stride != 0:
            continue

        if frame_index % 100 == 0:
            print(f"on frame {frame_index}")

        frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LANCZOS4) # resize frame to 1280x720 and apply interpolation

        results = detector.predict(
            source=frame,
            conf=0.15,
            save=False,
            verbose=False
        )
        torch.cuda.empty_cache()

        boxes = results[0].boxes

        selected = select_valid_box(boxes, last_center)

        # discard invalid frames completely
        if selected is None:
            continue

        x1, y1, x2, y2, cx, cy = selected

        h, w, _ = frame.shape
        x1 = np.clip(x1, 0, w-1)
        x2 = np.clip(x2, 0, w-1)
        y1 = np.clip(y1, 0, h-1)
        y2 = np.clip(y2, 0, h-1)

        if x2 <= x1 or y2 <= y1:
            continue

        # now we trust this frame
        sequence.append(frame.astype(np.uint8))
        last_center = (cx, cy)

        if len(sequence) < SEQUENCE_LEN:
            continue

        if len(sequence) >= SEQUENCE_LEN:
            X_train_local.append(np.stack(sequence, axis=0))
            y_train_local.append([x1, y1, x2, y2])
            sequence.clear()

        if len(X_train_local) == BATCH_SIZE:
            batch_idx += 1
            print(f"saved batch {batch_idx}")
            save_and_clear_batch(X_train_local, y_train_local, batch_idx, video_name)

        del frame, results, boxes, selected
        gc.collect()

    # save any remaining sequences for THIS video (final partial batch)
    if len(X_train_local) > 0:
        X_batch = np.asarray(X_train_local, dtype=np.uint8)
        y_batch = np.asarray(y_train_local, dtype=np.float32)

        batch_idx += 1
        print(f"saved batch {batch_idx} (final)")

        np.save(
            os.path.join(output_path, f"X_train_batch{batch_idx}_{video_name}.npy"),
            X_batch
        )
        np.save(
            os.path.join(output_path, f"y_train_batch{batch_idx}_{video_name}.npy"),
            y_batch
        )

        # CRITICAL: clear in-place to free RAM
        X_train_local.clear()
        y_train_local.clear()
        del X_batch, y_batch
        gc.collect()

# uses a pretrained yolo model to create labels for TrackNet

import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import gc
import torch
import tensorflow.keras.mixed_precision as mp
mp.set_global_policy("mixed_float16") # 50% reduction in memory

source_video_path = "src/ball_tracking/ball_tracking_data" # data path
output_path = "src/ball_tracking/videos/labels" # output path (.npy files will be stored here)
videos = os.listdir(source_video_path) # list of all video names

detector = YOLO("hugging_face_best.pt") # loads ball detector (teacher model)
sigma = 10 # spread 

BATCH_SIZE = 2 # chunking videos to avoid memory failure; n frames per batch
batch_idx = 1

# load videos
for filename in videos:

    if filename.endswith(".mp4"): # discard DS_Store or directories, only accept mp4 files

        X_train_local = []
        y_train_local = []

        cap = cv.VideoCapture(os.path.join(source_video_path, filename))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (1280, 720))
            results = detector.predict(
                source=frame,
                conf=0.3,
                save=False,
                verbose=False
            )[0]

            best_box = None
            best_conf = 0

            # get box with higheset confidence
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                # add this frame to X_train_local
                X_train_local.append(frame)

                # create gaussian blur for TrackNet
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # get center (x,y)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                H, W = frame.shape[:2]

                y = np.arange(H)[:, None]
                x = np.arange(W)[None, :]

                heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                heatmap = heatmap[..., np.newaxis]
                heatmap = heatmap.astype(np.float16)  # 50% reduction
                y_train_local.append(heatmap) # append heatmap
                del heatmap
                del frame
                del results

            else:
                # add frame to X_train_local
                X_train_local.append(frame)

                # create an empty heatmap
                H, W = frame.shape[:2]  # use actual frame shape
                heatmap = np.zeros((H, W, 1), dtype=np.float16)
                y_train_local.append(heatmap) # append heatmap
                del heatmap
                del frame

            # save to .npy file
            if len(X_train_local) >= BATCH_SIZE:
                print(np.array(y_train_local).shape)

                np.save(f"src/ball_tracking/ball_tracking_data/labels/X_train_batch_{batch_idx}.npy", np.array(X_train_local))
                np.save(f"src/ball_tracking/ball_tracking_data/labels/y_train_batch_{batch_idx}.npy", np.array(y_train_local))
                X_train_local, y_train_local = [], []
                print(f"saved batch {batch_idx}")
                batch_idx += 1

                gc.collect()

    else:
        continue
# creates a different version of preprocess.py as the ready position data consists of images not videos
# will be used for an image based model

import cv2 as cv                                        # image handling
import mediapipe as mp                                  # extract keypoints
import os                                               # paths
from ultralytics import YOLO                            # bounding boxes
import numpy as np                                      # math
from sklearn.model_selection import train_test_split    # splitting

# data path
ROOT = "data/shot-classification/neutral"

# key map
LABELS = {
    "neutral": 0,
    "not_neutral": 1,
}

# 26 ms delay (optimal video speed)
DELAY = 26

# augmentation function

def augment_keypoints(
    keypoints: np.ndarray,
    noise_std: float = 0.003,
    translate_std: float = 0.01,
    scale_range: tuple = (0.93, 1.06)
) -> np.ndarray:
    """
    mediapipe keypoint augmentations

    - gaussian noise
    - translation
    - scaaling
    """

    kp = keypoints.copy()

    # gaussian noise
    kp += np.random.normal(0, noise_std, kp.shape)

    # translate skeleton
    translation = np.random.normal(0, translate_std, size=(1, 3))
    kp += translation

    # scale evenly
    center = kp.mean(axis=0, keepdims=True)
    scale = np.random.uniform(*scale_range)
    kp = (kp - center) * scale + center

    return kp

dirs = ["backhand", "forehand", "ready_position", "serve"]

# 0 -> neutral, 1 -> not neutral
X_train_0, y_train_0 = [], []
X_train_1, y_train_1 = [], []

# create mediapipe pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,       
    model_complexity=1,            
    enable_segmentation=False,     
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5
)

# load YOLO11n instance
detector = YOLO("yolo11n.pt")

# main loop
for dir in dirs:

    if dir == ".DS_Store":
        break # skips if unwanted file found

    for image in os.listdir(os.path.join(ROOT, dir)):
        img = cv.imread(os.path.join(ROOT, dir, image)) # fixed image reading

        results = detector.predict(
            source=img,
            classes=[0],
            conf=0.3,
            stream=False
        )

        r = results[0]
        r_boxes = r.boxes

        if r_boxes is None or len(r_boxes) == 0: # skip if nothing is detected
            continue

        # pick bb
        best_box = None
        max_area = 0

        for box in r_boxes:
            x1, y1, x2, y2 = box.xyxy[0] # get coords
            area = (x2-x1) * (y2-y1) # get area
            if area > max_area:
                max_area = area
                best_box = box

        # actually crop the person
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # get coordinates

        # increase box size by 40%
        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = int(0.2 * box_w)
        pad_h = int(0.2 * box_h)

        x1 -= pad_w
        y1 -= pad_h
        x2 += pad_w
        y2 += pad_h

        # clamp to frame bounds
        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cropped_person = img[y1:y2, x1:x2]

        del img # clear from memory manually

        if cropped_person.size == 0: # skip if box is too small
            continue

        stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB) # convert to rgb

        results = pose.process(stroke) # process with mediapipe

        if results.pose_landmarks: # check if pose was detected
            landmarks = results.pose_landmarks.landmark # extract landmark object

            pose_frame = [] # to store landmarks that form a full pose

            for landmark in landmarks: # iterate and extract
                
                pose_frame.append(np.array([landmark.x, landmark.y, landmark.z])) # store landmarks to form a full pose

            pose_frame = np.array(pose_frame, dtype=np.float32) # convert to array

            # add original copy
            if dir == "ready_position":
                X_train_0.append(pose_frame)
                y_train_0.append(0)
            else:
                X_train_1.append(pose_frame)
                y_train_1.append(1)

            # create augmented copies
            if dir == "ready_position":
                for _ in range(0,65):
                    X_train_0.append(augment_keypoints(pose_frame))
                    y_train_0.append(0)
            else:
                for _ in range(0, 10):
                    X_train_1.append(augment_keypoints(pose_frame))
                    y_train_1.append(1)

        else:
            continue

# combine both datasets into one
X_train_0.extend(X_train_1)
y_train_0.extend(y_train_1)

X_train = np.array(X_train_0)
y_train = np.array(y_train_0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

# save files to .npy
np.save(f"data/shot-classification/neutral_landmarks/X_train_shot_classification_neutral.npy", X_train)
np.save(f"data/shot-classification/neutral_landmarks/X_test_shot_classification_neutral.npy", X_test)
np.save(f"data/shot-classification/neutral_landmarks/y_train_shot_classification_neutral.npy", y_train)
np.save(f"data/shot-classification/neutral_landmarks/y_test_shot_classification_neutral.npy", y_test)
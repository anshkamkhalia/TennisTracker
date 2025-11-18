# loads and preprocesses videos -> saves keypoints to .npy files

import cv2 as cv # video handling
import numpy as np # computing
from concurrent.futures import ThreadPoolExecutor, as_completed # for parallelization
from sklearn.model_selection import train_test_split # split data
from typing import Tuple, List # function dtype annotation
from tqdm import tqdm # progress bar
import os # paths and directories
import mediapipe as mp # drawing/extracting keypoints

# establish root dir
ROOT = "data"

# str -> int
label_map = {
    'backhand': 0,
    'forehand': 1,
}

TARGET_FRAMES = 90

def apply_augmentations(keypoints):
    """applies moderate augmentations to increase data"""

    keypoints = np.asarray(keypoints, dtype=np.float32)  # guarantees float32 array
    keypoints = keypoints + np.random.normal(0, 0.01, keypoints.shape).astype(np.float32)
    if np.random.rand() > 0.5:
        keypoints[::3] = 1 - keypoints[::3]  # Flip X

    keypoints += np.random.normal(0, 0.01, keypoints.shape)  # jitter

    scale = np.random.uniform(0.98, 1.02)
    offset_x = np.random.uniform(-0.01, 0.01)
    offset_y = np.random.uniform(-0.01, 0.01)

    keypoints[::3] = keypoints[::3] * scale + offset_x
    keypoints[1::3] = keypoints[1::3] * scale + offset_y
    return keypoints.tolist()

def process_video(filename: str, target_idx: int, repeats: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    preprocess a single video
    """

    mp_pose = mp.solutions.pose
    # use 'with' so resources are properly released per thread
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        X_local = []
        y_local = []

        # determine path
        folder = 'BACKHAND' if target_idx == 0 else 'FOREHAND'
        cap = cv.VideoCapture(f"{ROOT}/{folder}/{filename}")

        sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # bgr -> rgb
            results = pose.process(frame_rgb)

            # keypoints
            if results.pose_landmarks:
                keypoints = [lm.x for lm in results.pose_landmarks.landmark] + \
                            [lm.y for lm in results.pose_landmarks.landmark] + \
                            [lm.z for lm in results.pose_landmarks.landmark]
                keypoints = np.array(keypoints, dtype=np.float32)
            else:
                keypoints = np.zeros(99, dtype=np.float32)

            sequence.append(keypoints)

        cap.release()

        # pad/truncate to TARGET_FRAMES
        if len(sequence) > TARGET_FRAMES:
            sequence = sequence[:TARGET_FRAMES]
        else:
            pad_amount = TARGET_FRAMES - len(sequence)
            sequence += [np.zeros(99, dtype=np.float32)] * pad_amount

        # append original sequence
        X_local.append(np.array(sequence, dtype=np.float32))
        y_local.append(target_idx)

        # augmentations
        for _ in range(repeats):
            seq_aug = [apply_augmentations(f) for f in sequence]
            X_local.append(np.array(seq_aug, dtype=np.float32))
            y_local.append(target_idx)

        return np.array(X_local, dtype=np.float32), np.array(y_local, dtype=np.uint8)

X_train = []
y_train = []

# forehands
with ThreadPoolExecutor(max_workers=5) as executor: # parallel processing

    futures = [executor.submit(process_video, f, 1, 10) for f in os.listdir(f"{ROOT}/FOREHAND")]

    for future in tqdm(futures, colour="RED"):
        X_tr, y_tr = future.result()
        X_train.extend(X_tr)
        y_train.extend(y_tr)

# backhands
with ThreadPoolExecutor(max_workers=5) as executor: # parallel processing

    futures = [executor.submit(process_video, f, 0, 10) for f in os.listdir(f"{ROOT}/BACKHAND")]

    for future in tqdm(futures, colour="RED"):
        X_tr, y_tr = future.result()
        X_train.extend(X_tr)
        y_train.extend(y_tr)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.uint8)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.15,
                                                    random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# save as .npy files
np.save("keypoints/X_train.npy", X_train)
np.save("keypoints/X_test.npy", X_test)
np.save("keypoints/y_train.npy", y_train)
np.save("keypoints/y_test.npy", y_test)
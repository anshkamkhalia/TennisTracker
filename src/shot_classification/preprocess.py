# loads and preprocesses data for shot classification

import mediapipe as mp                                  # extract keypoints
import cv2 as cv                                        # video handling
import os                                               # paths
from ultralytics import YOLO                            # bounding boxes
from concurrent.futures import ProcessPoolExecutor      # parallel computing - higher speeds
from typing import Tuple                                # function notation
import numpy as np                                      # math
from sklearn.model_selection import train_test_split    # splitting
from tqdm import tqdm                                   # progress bat
from tensorflow.keras import mixed_precision

# Set global policy to float16 (mixed precision)
mixed_precision.set_global_policy("mixed_float16")

# establish paths
ROOT = "data/shot-classification"

# 26 ms delay (for optimal video speed)
DELAY = 26

# key mapping
LABELS = {
    "forehand": 0,
    "backhand": 1,
    "slice_volley": 2,
    "serve_overhead": 3,
}

# copy map: number of times each class will be duplicated
COPY_MAP = {
    "forehand": 50,
    "backhand": 50,
    "slice_volley": 10,
    "serve_overhead": 25, 
}

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

# preprocess function

def preprocess(
    path: str,
    shot_type:str, 
    n_copies: int=10, 
    save: bool=True, 
    split: bool=True,
) -> Tuple[list, list, list, list]:
    
    """
    preprocesses a batch of videos for a specific shot type
    saves to .npy files

    args:
    - path: path to the video files
    - shot_type: (forehand, backhand, serve, slice, volley, overhead)
    - n_copies: the amount of times to copy the video (with augementations applied)
    - save: determines whether it is saved into a .npy file
    - split: determines whether data is split 

    """
    
    # check if invalid shot_type exists
    if shot_type not in LABELS.keys() and shot_type is not None:
        raise ValueError("invalid shot type")

    filenames = None

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

    # load the filenames of the selected shot
    if not path.startswith(ROOT): # check if path begins with ROOT
        filenames = os.listdir(os.path.join(ROOT, path))
    else:
        filenames = os.listdir(path)

    X_train_local, y_train_local = [], [] # will be converted into arrays later + split into testing

    # loop over all files
    for filename in tqdm(filenames, desc="processing videos"): # wrap in tqdm for a progress bar

        cap = cv.VideoCapture(os.path.join(os.path.join(ROOT, path), filename)) # uses opencv to read video data
        
        NUM_LANDMARKS = 33
        SEQUENCE_LENGTH = 75  # timesteps for LSTM

        sequence_buffer = []  # store frames temporarily

        while True:
            ret, frame = cap.read()
            if not ret: 
                break  # end of video

            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LANCZOS4)
            pose_frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

            # crop human out using YOLO
            results = detector.predict(
                source=frame,
                classes=[0],
                conf=0.3,
                stream=False
            )

            r = results[0]
            r_boxes = r.boxes

            if r_boxes is None or len(r_boxes) == 0:
                pass
            else:
                # pick largest bounding box
                best_box = max(r_boxes, key=lambda box: (box.xyxy[0][2]-box.xyxy[0][0]) * (box.xyxy[0][3]-box.xyxy[0][1]))

                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                box_w, box_h = x2 - x1, y2 - y1
                pad_w, pad_h = int(0.35 * box_w), int(0.35 * box_h)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(frame.shape[1], x2 + pad_w), min(frame.shape[0], y2 + pad_h)

                cropped_person = frame[y1:y2, x1:x2]
                del frame
                if cropped_person.size != 0:
                    stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB)
                    pose_results = pose.process(stroke)

                    if pose_results.pose_landmarks:
                        # create frame array with guaranteed 33 landmarks
                        # if some landmarks are not detected, empty spots are 0s
                        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            pose_frame[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

            # flatten to 99 features (zeros if no pose detected)
            pose_frame = pose_frame.flatten()
            sequence_buffer.append(pose_frame)

            # if we have a full sequence of 33 frames
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                sequence = np.stack(sequence_buffer)  # shape (33, 99)
                X_train_local.append(sequence)
                if shot_type is not None:
                    y_train_local.append(LABELS[shot_type])

                # augmentation for the sequence
                for _ in range(n_copies):
                    augmented_sequence = np.array([augment_keypoints(f.reshape(NUM_LANDMARKS,3)).flatten() for f in sequence])
                    X_train_local.append(augmented_sequence)
                    if shot_type is not None:
                        y_train_local.append(LABELS[shot_type])

                # remove first frame to slide window
                sequence_buffer.pop(0)

            else:
                continue # skips is no one was detected (should not happen due to YOLO boxes)

        cap.release()
        cv.destroyAllWindows()
        
    # splits data if needed
    X_test_local, y_test_local = [], []
    if split:
        X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(X_train_local, y_train_local, # split data
                                                                                    test_size=0.2,
                                                                                    random_state=42)
    else:
        pass

    # convert lists to arrays
    X_train_local = np.array(X_train_local, dtype=np.float32)
    y_train_local = np.array(y_train_local, dtype=np.int64)
    X_test_local = np.array(X_test_local, dtype=np.float32)
    y_test_local = np.array(y_test_local, dtype=np.int64)

    # saves datasets in .npy
    if save:
        np.save(f"{ROOT}/landmarks/X_train_shot_classification_{shot_type}.npy", X_train_local)
        np.save(f"{ROOT}/landmarks/y_train_shot_classification_{shot_type}.npy", y_train_local)
        if split:
            np.save(f"{ROOT}/landmarks/X_test_shot_classification_{shot_type}.npy", X_test_local)
            np.save(f"{ROOT}/landmarks/y_test_shot_classification_{shot_type}.npy", y_test_local)

    return X_train_local, y_train_local, X_test_local, y_test_local

# utilize functions to preprocess data 
# wrap in main() to prevent bootstrapping
def main():
    shots = ["forehand", "backhand", "slice_volley", "serve_overhead"]

    # use processpoolexecutor to speed up
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(preprocess,
                                   path=shot,
                                   shot_type=shot,
                                   n_copies=COPY_MAP[shot],
                                   save=True,
                                   split=True)
                   for shot in shots]

        results = [f.result() for f in futures]  # wait for results
    print("finished preprocessing:", results)

if __name__ == "__main__":
    main()
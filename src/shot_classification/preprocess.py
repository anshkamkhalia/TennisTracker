# loads and preprocesses data for shot classification

import mediapipe as mp                                  # extract keypoints
import cv2 as cv                                        # video handling
import os                                               # paths
from ultralytics import YOLO                            # bounding boxes
from concurrent.futures import ProcessPoolExecutor      # parallel computing - higher speeds
from typing import Tuple                                # function notation
import numpy as np                                      # math
from sklearn.model_selection import train_test_split    # splitting
from tqdm import tqdm                                   # progress bar

# establish paths
ROOT = "data/shot-classification"

# 26 ms delay (for optimal video speed)
DELAY = 26

# key mapping - will add more later
LABELS = {
    "forehand": 0,
    "backhand": 1,
    "slice": 2,
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
        
        # loop through frames of current video
        while True:

            ret, frame = cap.read()
            if not ret: break # ends if no more frames

            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LANCZOS4) # resize with linear interpolation for higher quality

            # crop human out using yolo
            results = detector.predict(
                source=frame,
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
                x1, y1, x2, y2 = box.xyxy[0] # get coordinates
                area = (x2-x1) * (y2-y1) # get area of box
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
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_person = frame[y1:y2, x1:x2]

            del frame # clear from memory to avoid crashing (ProcessPoolExecutor)

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

                # add original points
                X_train_local.append(pose_frame) # append keypoints to list in the form of an array
                if shot_type is not None:
                    y_train_local.append(LABELS[shot_type]) # numerical representation (0,1,2 ...)

                # augment and copy
                for _ in range(n_copies):
                    X_train_local.append(augment_keypoints(pose_frame)) # append augmented keypoints to list in the form of an array
                    if shot_type is not None:
                        y_train_local.append(LABELS[shot_type]) # numerical representation (0,1,2 ...)

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
    shots = ["forehand", "backhand", "slice"]

    # use processpoolexecutor to speed up
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(preprocess,
                                   path=shot,
                                   shot_type=shot,
                                   n_copies=30 if "slice" not in shot or "volley" not in shot else 15,
                                   save=True,
                                   split=True)
                   for shot in shots]

        results = [f.result() for f in futures]  # wait for results
    print("finished preprocessing:", results)

if __name__ == "__main__":
    main()
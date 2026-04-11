# tests the model on actual videos

from tensorflow.keras.saving import load_model                            # load serialized model
from model import ShotClassifier            # custom classes
from neutral_model import Attention, NeutralIdentifier                    # more custom classes
import mediapipe as mp                                                    # keypoint extraction
import cv2 as cv                                                          # video handling
from ultralytics import YOLO                                              # bounding boxes
import numpy as np                                                        # computations

i = 5 # index of video to predict on

# key mapping - will add more later
LABELS = {
    "forehand": 0,
    "backhand": 1,
    "slice_volley": 2,
    "serve_overhead": 3,
}

LABELS_INV = {v: k for k, v in LABELS.items()} # create inverse: {0: "topspin_forehand"...}

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features(pose_frame, prev_pose_frame):
    features = []

    left_hip = pose_frame[23]
    right_hip = pose_frame[24]
    left_shoulder = pose_frame[11]
    right_shoulder = pose_frame[12]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso = np.linalg.norm(shoulder_center - hip_center)
    torso = torso if torso > 1e-6 else 1.0

    normalized_pose = (pose_frame - hip_center) / torso
    features.extend(normalized_pose.flatten())

    if prev_pose_frame is None:
        velocity = np.zeros_like(normalized_pose)
    else:
        prev_left_hip = prev_pose_frame[23]
        prev_right_hip = prev_pose_frame[24]
        prev_left_shoulder = prev_pose_frame[11]
        prev_right_shoulder = prev_pose_frame[12]

        prev_hip_center = (prev_left_hip + prev_right_hip) / 2.0
        prev_shoulder_center = (prev_left_shoulder + prev_right_shoulder) / 2.0

        prev_torso = np.linalg.norm(prev_shoulder_center - prev_hip_center)
        prev_torso = prev_torso if prev_torso > 1e-6 else 1.0

        prev_normalized_pose = (prev_pose_frame - prev_hip_center) / prev_torso
        velocity = normalized_pose - prev_normalized_pose

    features.extend(velocity.flatten())

    angle_features = [
        calculate_angle(normalized_pose[11], normalized_pose[13], normalized_pose[15]),
        calculate_angle(normalized_pose[12], normalized_pose[14], normalized_pose[16]),
        calculate_angle(normalized_pose[23], normalized_pose[25], normalized_pose[27]),
        calculate_angle(normalized_pose[24], normalized_pose[26], normalized_pose[28]),
        calculate_angle(normalized_pose[13], normalized_pose[11], normalized_pose[23]),
        calculate_angle(normalized_pose[14], normalized_pose[12], normalized_pose[24]),
    ]

    features.extend(angle_features)

    right_wrist = normalized_pose[16]
    left_wrist = normalized_pose[15]

    wrist_features = [
        right_wrist[0], right_wrist[1], right_wrist[2],
        left_wrist[0], left_wrist[1], left_wrist[2],
        np.linalg.norm(velocity[16]),
        np.linalg.norm(velocity[15]),
    ]

    features.extend(wrist_features)

    return np.array(features, dtype=np.float32)

# paths
shot_model_path = "serialized_models/new_sc.keras"
neutral_model_path = "serialized_models/neutrality.keras"
video_path = f"api/videoplayback5.mp4"

# load models
shot_classifier = load_model(shot_model_path, custom_objects={
    "ShotClassifier": ShotClassifier,
    # "Attention": Attention,
    # "SequenceAttention": SequenceAttention,
})

neutral_identifier = load_model(neutral_model_path, custom_objects={
    "NeutralIdentifier": NeutralIdentifier,

})

# yolo instance
detector = YOLO("yolo11n.pt")

seq_len = 60  # updated to match training

# mediapipe pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# for drawing keypoints
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# read video
cap = cv.VideoCapture(video_path)

# output video writer 
out = cv.VideoWriter(
    f"outputs/run{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    60,
    (1280, 720)
)

# config variables
frame_buffer = []
prev_pose_frame = None
base_alpha = 0.85 # for smoothing
previous_prediction = "neutral" # to save the last prediction
frame_index = 0 # to keep track of current frame
last_pred_frame = -999  # initialize far back so no text at start
fps = 30
state = None # will be 0 (neutral) or 1 (swinging)
output_class = -1 # forehand (0) or backhand (1)

while True:
    frame_index += 1
    ret, frame = cap.read()
    if not ret: break # breaks if last frame

    if i != 5:
        frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LANCZOS4) # resize frame to 720p
    else:
        frame = cv.resize(frame, (1280, 720)) # without interpolation

    # crop human out using yolo
    results = detector.predict(
        source=frame,
        classes=[0],
        conf=0.3,
        stream=False
    )
    
    # extract boxes
    r = results[0]
    r_boxes = r.boxes

    if r_boxes is None or len(r_boxes) == 0: # skip if nothing is detected
        out.write(frame)
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
    x1,y1, x2, y2 = map(int, best_box.xyxy[0])  # get coordinates

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

    if cropped_person.size == 0: # skip if box is too small
        out.write(frame)
        continue

    stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB) # convert to rgb

    results = pose.process(stroke) # process with mediapipe
    landmarks = results.pose_landmarks.landmark

    pose_frame = np.zeros((33, 3), dtype=np.float32)

    for i, landmark in enumerate(landmarks):
        pose_frame[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    feat = extract_features(pose_frame, prev_pose_frame)
    prev_pose_frame = pose_frame.copy()

    frame_buffer.append(feat)

    if len(frame_buffer) > 0:
        inference_neutral_pose_frame = pose_frame[np.newaxis, :, :, np.newaxis]  # minimal fix: shape (1,33,3,1)
        raw_state = neutral_identifier.predict(inference_neutral_pose_frame, verbose=0) # inference on current frame
        state = int(raw_state[0][0] > 0.8)
    else:
        state = 0

    if len(frame_buffer) >= seq_len and state == 1:

        sequence = np.array(frame_buffer[-seq_len:], dtype=np.float32)
        sequence = sequence[np.newaxis, ...]

        probs = shot_classifier.predict(sequence, verbose=0)[0]

        label = np.argmax(probs)
        output_class = LABELS_INV[label]
        confidence = probs[label]

        text = f"{output_class}: {(confidence*100):.2f}%"

        previous_prediction = text
        last_pred_frame = frame_index

        frame_buffer = frame_buffer[30:]

    if frame_index - last_pred_frame <= 40:
        display_text = previous_prediction
    else:
        display_text = "neutral"

    # write annotated frame to output video (top-right corner)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # color based on shot type
    if output_class == "forehand":
        color = (255, 191, 0)      # greenish
    elif output_class == "backhand":
        color = (166, 255, 0)      # blueish
    else:
        color = (0, 0, 255)      # red / fallback

    (text_w, text_h), baseline = cv.getTextSize(display_text, font, font_scale, thickness)

    # top-right position with padding
    padding = 10
    org = (
        frame.shape[1] - text_w - padding,
        text_h + padding
    )

    # background rectangle
    cv.rectangle(
        frame,
        (org[0] - padding, org[1] - text_h - padding),
        (org[0] + text_w + padding, org[1] + baseline + padding),
        (0, 0, 0),
        -1
    )

    # text
    cv.putText(
        frame,
        display_text,
        org,
        font,
        font_scale,
        color,
        thickness,
        cv.LINE_AA
    )

    out.write(frame)

    del frame # clear from memory

cap.release()
cv.destroyAllWindows()
out.release()
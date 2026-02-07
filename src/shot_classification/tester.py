# tests the model on actual videos

from tensorflow.keras.saving import load_model                            # load serialized model
from model import Attention, ShotClassifier, SequenceAttention            # custom classes
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

# paths
shot_model_path = "serialized_models/shot_classifier.keras"
neutral_model_path = "serialized_models/neutrality.keras"
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load models
shot_classifier = load_model(shot_model_path, custom_objects={
    "ShotClassifier": ShotClassifier,
    "Attention": Attention,
    "SequenceAttention": SequenceAttention,
})

neutral_identifier = load_model(neutral_model_path, custom_objects={
    "NeutralIdentifier": NeutralIdentifier,
    "Attention": Attention,
})

# yolo instance
detector = YOLO("yolo11n.pt")

seq_len = 90 

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
frame_buffer = [] # will fill up to 180 frames
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

    if results.pose_landmarks: # check if pose was detected

        landmarks = results.pose_landmarks.landmark

        pose_frame = [] # to store landmarks that form a full pose

        for landmark in landmarks: # iterate and extract
            
            pose_frame.append(np.array([landmark.x, landmark.y, landmark.z])) # store landmarks to form a full pose

        pose_frame = np.array(pose_frame, dtype=np.float32) # convert to array

        frame_buffer.append(pose_frame)

    if len(frame_buffer) > 0:
        inference_neutral_pose_frame = pose_frame[np.newaxis, ..., np.newaxis] # adds batch and channel dimensions for conv layers
        raw_state = neutral_identifier.predict(inference_neutral_pose_frame, verbose=0) # inference on current frame
        state = int(raw_state[0][0] > 0.8)
    else:
        state = 0

    if len(frame_buffer) >= seq_len and state == 1: # buffer has reached its limit and state is 1 (swinging)

        # get prediction for the last frame only
        last_frame = np.array([frame_buffer[-1]])  # wrap in array to match model input shape
        probs = shot_classifier.predict(last_frame, verbose=0)

        # probs is now (1, num_classes), take first element
        probs = np.asarray(probs)[0]

        # get label and confidence
        label = np.argmax(probs)
        output_class = LABELS_INV[label]
        confidence = probs[label]

        # format text
        text = f"{output_class}: {(confidence*100):.2f}%"

        previous_prediction = text # update
        last_pred_frame = frame_index  # mark when this prediction occurred

        frame_buffer = frame_buffer[30:]  # keep only the remaining frames

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


# # tests the model on actual videos

# from tensorflow.keras.saving import load_model          # load serialized model
# from model import Attention, ShotClassifier             # custom classes
# from neutral_model import Attention, NeutralIdentifier  # more custom classes
# import mediapipe as mp                                  # keypoint extraction
# import cv2 as cv                                        # video handling
# from ultralytics import YOLO                            # bounding boxes
# import numpy as np                                      # computations

# # key mapping - will add more later
# LABELS = {
#     "forehand": 0,
#     "backhand": 1,
# }

# LABELS_INV = {v: k for k, v in LABELS.items()} # create inverse: {0: "topspin_forehand"...}

# # paths
# shot_model_path = "serialized_models/shot_classifier.keras"
# neutral_model_path = "serialized_models/neutrality.keras"
# video_path = "data/court-level-videos/videoplayback.mp4"

# # load models
# shot_classifier = load_model(shot_model_path, custom_objects={
#     "ShotClassifier": ShotClassifier,
#     "Attention": Attention,
# })

# neutral_identifier = load_model(neutral_model_path, custom_objects={
#     "NeutralIdentifier": NeutralIdentifier,
#     "Attention": Attention,
# })

# # yolo instance
# detector = YOLO("yolo11n.pt")

# seq_len = 90 

# # mediapipe pose instance
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=2,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )

# # for drawing keypoints
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # read video
# cap = cv.VideoCapture(video_path)

# # output video writer 
# out = cv.VideoWriter(
#     "outputs/run.mp4",
#     cv.VideoWriter_fourcc(*"mp4v"),
#     30,
#     (1280, 720)
# )

# # config variables
# frame_buffer = [] # will fill up to 180 frames
# base_alpha = 0.85 # for smoothing
# previous_prediction = None # to save the last prediction
# frame_index = 0 # to keep track of current frame
# last_pred_frame = -999  # initialize far back so no text at start
# fps = 30
# state = None # will be 0 (neutral) or 1 (swinging)

# while True:
#     frame_index += 1
#     ret, frame= cap.read()
#     if not ret: break # breaks if last frame

#     frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LANCZOS4) # resize frame to 720p

#     # crop human out using yolo
#     results = detector.predict(
#         source=frame,
#         classes=[0],
#         conf=0.3,
#         stream=False
#     )
    
#     # extract boxes
#     r = results[0]
#     r_boxes = r.boxes

#     if r_boxes is None or len(r_boxes) == 0: # skip if nothing is detected
#         continue

#     # pick bb 
#     best_box = None
#     max_area = 0

#     for box in r_boxes:
#         x1, y1, x2, y2 = box.xyxy[0] # get coordinates
#         area = (x2-x1) * (y2-y1) # get area of box
#         if area > max_area:
#             max_area = area
#             best_box = box

#     # actually crop the person
#     x1,y1, x2, y2 = map(int, best_box.xyxy[0])  # get coordinates

#     # increase box size by 40%
#     box_w = x2 - x1
#     box_h = y2 - y1

#     pad_w = int(0.2 * box_w)
#     pad_h = int(0.2 * box_h)

#     x1 -= pad_w
#     y1 -= pad_h
#     x2 += pad_w
#     y2 += pad_h

#     # clamp to frame bounds
#     h, w, _ = frame.shape
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(w, x2)
#     y2 = min(h, y2)

#     cropped_person = frame[y1:y2, x1:x2]

#     if cropped_person.size == 0: # skip if box is too small
#         continue

#     stroke = cv.cvtColor(cropped_person, cv.COLOR_BGR2RGB) # convert to rgb

#     results = pose.process(stroke) # process with mediapipe

#     if results.pose_landmarks: # check if pose was detected

#         landmarks = results.pose_landmarks.landmark

#         pose_frame = [] # to store landmarks that form a full pose

#         for landmark in landmarks: # iterate and extract
            
#             pose_frame.append(np.array([landmark.x, landmark.y, landmark.z])) # store landmarks to form a full pose

#         pose_frame = np.array(pose_frame, dtype=np.float32) # convert to array

#         frame_buffer.append(pose_frame)

#     inference_neutral_pose_frame = pose_frame[np.newaxis, ..., np.newaxis] # adds batch and channel dimensions for conv layers
#     state = neutral_identifier.predict(inference_neutral_pose_frame) # inference on current frame
#     state = int(state[0][0] > 0.5)

#     if len(frame_buffer) >= seq_len and state == 1: # buffer has reached its limit and state is 1 (swinging)

#         # probs = shot_classifier.predict(np.array(frame_buffer)) # get predictions

#         # # get confidence
#         # probs = np.asarray(probs)
#         # conf = probs.max(axis=1)

#         # # smoothen'

#         # ema = probs[0].copy()
#         # for i in range(1, len(probs)):
#         #     alpha = base_alpha + (1 - base_alpha) * conf[i]
#         #     ema = alpha * ema + (1 - alpha) * probs[i]

#         # # calculate final
#         # mean_probs = probs.mean(axis=0)
#         # final_probs = 0.5 * ema + 0.5 * mean_probs
        
#         # # convert to text
#         # label = np.argmax(final_probs)
#         # output_class = LABELS_INV[label]
#         # confidence = final_probs[label]

#         # get prediction for the last frame only
#         last_frame = np.array([frame_buffer[-1]])  # wrap in array to match model input shape
#         probs = shot_classifier.predict(last_frame)

#         # probs is now (1, num_classes), take first element
#         probs = np.asarray(probs)[0]

#         # get label and confidence
#         label = np.argmax(probs)
#         output_class = LABELS_INV[label]
#         confidence = probs[label]

#         # format text
#         text = f"{output_class}: {(confidence*100):.3f}%"

#         previous_prediction = text # update
#         last_pred_frame = frame_index  # mark when this prediction occurred

#         if previous_prediction is not None and (frame_index - last_pred_frame) <= 40:
#             # draw the text
#             org = (x1, max(20, y1 - 10))
#             font = cv.FONT_HERSHEY_SIMPLEX
#             font_scale = 1.0
#             color = (0, 255, 0) if label == 0 else (255, 0, 0)  # green
#             thickness = 2

#             (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
#             cv.rectangle(
#                 frame,
#                 (org[0] - 10, org[1] - text_h - 10),
#                 (org[0] + text_w + 10, org[1] + baseline + 10),
#                 (0, 0, 0),
#                 -1
#             )

#             cv.putText(
#                 frame,
#                 text,
#                 org,
#                 font,
#                 font_scale,
#                 color,
#                 thickness,
#                 cv.LINE_AA
#             )

#         frame_buffer = frame_buffer[5:]  # keep only the remaining frames
#         # frame_buffer.pop(0) # remove last frame

#     else:
#         if previous_prediction is not None:
#             previous_prediction = "neutral"

#             # text position + style
#             org = (x1, max(20, y1 - 10))  # 10px above the cropped person
#             font = cv.FONT_HERSHEY_SIMPLEX
#             font_scale = 1.2
#             color = (0, 255, 0) if label == 0 else (255, 0, 0)  # green
#             thickness = 3

#             # optional: background box for readability
#             (text_w, text_h), baseline = cv.getTextSize(previous_prediction, font, font_scale, thickness)
#             cv.rectangle(
#                 frame,
#                 (org[0] - 10, org[1] - text_h - 10),
#                 (org[0] + text_w + 10, org[1] + baseline + 10),
#                 (0, 0, 0),
#                 -1
#             )

#             # draw text
#             cv.putText(
#                 frame,
#                 previous_prediction,
#                 org,
#                 font,
#                 font_scale,
#                 color,
#                 thickness,
#                 cv.LINE_AA
#             ) 
#         else: pass # continues if frame buffer is not ready

#     # write annotated frame to output video
#     out.write(frame)

#     del frame # clear from memory

# cap.release()
# cv.destroyAllWindows()
# out.release()
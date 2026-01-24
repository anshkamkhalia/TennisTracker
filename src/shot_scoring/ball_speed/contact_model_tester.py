# tests the trained contact detection model

import cv2 as cv
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from model import ContactDetector, Attention
from trainer import binary_focal_loss

# Load your trained model
model = tf.keras.models.load_model("serialized_models/contact_detector.keras", custom_objects={
    "ContactDetector": ContactDetector,
    "Attention": Attention,
    "binary_focal_loss": binary_focal_loss
})

# Load YOLO for ball detection
ball_model = YOLO("hugging_face_best.pt")  # same as used for labeling

cap = cv.VideoCapture("data/court-level-videos/videoplayback5.mp4")

SEQ_LEN = 21  # must match training
CENTER_IDX = SEQ_LEN // 2

frames_buffer = []
coords_buffer = []
frame_idx = 0
video_fps = int(cap.get(cv.CAP_PROP_FPS))

# write output video with contacts drawn
out = cv.VideoWriter(
    "outputs/contact_test.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
)

def detect_ball_center(frame, conf_thresh=0.25):
    results = ball_model.predict(frame, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

trail = []
MAX_TRAIL = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frames_buffer.append(frame)

    # detect ball
    center = detect_ball_center(frame)
    coords_buffer.append(center)

    # maintain buffers
    if len(frames_buffer) > SEQ_LEN:
        frames_buffer.pop(0)
        coords_buffer.pop(0)

    # skip until we have a full sequence
    if len(frames_buffer) < SEQ_LEN:
        continue

    # If any frame in the sequence has no detection, skip sequence
    if None in coords_buffer:
        continue

    # Prepare model input: normalized coordinates
    seq_array = np.array(coords_buffer, dtype=np.float32)
    seq_array = seq_array / np.array([frame.shape[1], frame.shape[0]])  # normalize by width,height
    model_input = np.expand_dims(seq_array, axis=0)  # batch dimension

    # Run inference
    preds = model.predict(model_input, verbose=0)[0]  # shape (SEQ_LEN,)

    # check center frame
    prob = preds[CENTER_IDX]
    if prob > 0.5:
        text = f"contact! prob={prob:.2f}"
        cv.putText(frame, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        print(f"frame {frame_idx}: contact! prob={prob:.2f}")

    # Draw trail
    if center is not None:
        trail.append(center)
        if len(trail) > MAX_TRAIL:
            trail.pop(0)

        for i, (tx, ty) in enumerate(trail):
            alpha = i / len(trail)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv.circle(frame, (tx, ty), 5, color, -1)

    # Show & write
    cv.imshow("contact", frame)
    out.write(frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
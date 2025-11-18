import cv2
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture('test.mp4')

print("Press 'q' to quit")

fps = 60
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)  # mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose detection
    results = pose.process(rgb_frame)

    # Draw keypoints if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Test", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
